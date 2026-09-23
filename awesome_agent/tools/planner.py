from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from awesome_agent.contracts import EvidenceBundle, RelationQuery, ToolOutput, ToolTrace
from awesome_agent.memory.stores import GraphMemory
from awesome_agent.tools.base import MemoryTool


@dataclass
class ToolPlan:
    conflict_type: str
    signature: str
    tools: List[str]


class ToolPlanner:
    """Conflict-aware router for deterministic memory tools."""

    def __init__(self, memory: GraphMemory):
        self.memory = memory

    @staticmethod
    def _max(evidence: EvidenceBundle, key: str) -> float:
        return max((float(getattr(ev, key, 0.0)) for ev in evidence.by_id.values()), default=0.0)

    def classify_relation(self, evidence: EvidenceBundle, top1_prob: float, best_by_mem: int) -> str:
        max_direct = self._max(evidence, "direct")
        max_prior = self._max(evidence, "prior")
        top1 = evidence.candidate_set.top1
        if max_direct < 1.0 and max_prior < 0.30:
            return "local_sparse"
        if int(best_by_mem) != int(top1) and max_direct >= 1.0:
            return "local_global_conflict"
        if int(evidence.prev_rel) >= 0 and float(top1_prob) < 0.84 and int(best_by_mem) != int(top1):
            return "confounder_risk"
        if float(top1_prob) < 0.76:
            return "multi_ambiguity"
        return "stable"

    def _tool_value(self, task: str, sig: str, name: str, chosen: Optional[List[str]] = None) -> float:
        chosen = chosen or []
        global_util = float(self.memory.tool_utility(name, task))
        sig_util = float(self.memory.tool_utility_sig(name, task, sig))
        pair_util = sum(0.22 * float(self.memory.tool_pair_utility(task, sig, name, other)) for other in chosen)
        return float(0.45 * global_util + 0.95 * sig_util + pair_util)

    def plan_relation(self, evidence: EvidenceBundle, top1_prob: float, best_by_mem: int) -> ToolPlan:
        conflict = self.classify_relation(evidence, top1_prob=top1_prob, best_by_mem=best_by_mem)
        sig = f"rel|{conflict}|prev={int(evidence.prev_rel) if int(evidence.prev_rel) >= 0 else 'none'}"
        if conflict == "stable":
            return ToolPlan(conflict_type=conflict, signature=sig, tools=[])

        tools = ["pair_local", "counterfactual_verifier"]
        if conflict == "local_sparse":
            pool = ["entity_high_order", "event_centric", "causal_probe"]
            min_score = -0.04
        elif conflict == "local_global_conflict":
            pool = ["event_centric", "causal_probe", "entity_high_order"]
            min_score = 0.00
        elif conflict == "confounder_risk":
            pool = ["causal_probe", "event_centric", "entity_high_order"]
            min_score = -0.02
        else:
            pool = ["event_centric", "entity_high_order", "causal_probe"]
            min_score = -0.01

        severity = max(0.0, min(1.0, (0.82 - float(top1_prob)) / 0.22))
        if int(best_by_mem) != int(evidence.candidate_set.top1):
            severity += 0.25
        if conflict in {"local_global_conflict", "confounder_risk"}:
            severity += 0.18
        max_optional = 3 if severity >= 0.42 else 2

        ranked = sorted(pool, key=lambda name: self._tool_value("rel", sig, name, chosen=tools), reverse=True)
        picked: List[str] = []
        for name in ranked:
            score = self._tool_value("rel", sig, name, chosen=tools + picked)
            if score < float(min_score):
                continue
            picked.append(name)
            if len(picked) >= max_optional:
                break
        if not picked and ranked:
            picked.append(ranked[0])
        for name in picked:
            if name not in tools:
                tools.append(name)
        return ToolPlan(conflict_type=conflict, signature=sig, tools=tools[:4])


class ToolRunner:
    """Runs selected tools and normalizes their outputs for downstream policy."""

    def __init__(self, tools: Mapping[str, MemoryTool], planner: ToolPlanner):
        self.tools = dict(tools)
        self.planner = planner

    @staticmethod
    def _normalize_outputs(candidate_ids: List[int], outputs: Dict[str, ToolOutput]) -> Dict[str, Dict[int, float]]:
        normalized: Dict[str, Dict[int, float]] = {}
        cands = [int(x) for x in candidate_ids]
        for name, out in outputs.items():
            vals = [float(out.net(cid)) for cid in cands]
            if not vals:
                normalized[name] = {}
                continue
            raw_strength = max(abs(v) for v in vals)
            if raw_strength < 0.015:
                normalized[str(name)] = {int(cid): 0.0 for cid in cands}
                continue
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
            sd = variance ** 0.5 + 1e-6
            vmin = min(vals)
            vmax = max(vals)
            span = max(1e-6, vmax - vmin)
            if span < 0.012:
                normalized[str(name)] = {int(cid): 0.0 for cid in cands}
                continue
            if str(name) == "pair_local":
                confidence = min(1.0, raw_strength / 0.30)
            elif str(name) == "causal_probe":
                confidence = min(1.0, raw_strength / 0.16)
            elif str(name) == "counterfactual_verifier":
                confidence = min(1.0, raw_strength / 0.22)
            else:
                confidence = min(0.70, raw_strength / 0.18)
            local: Dict[int, float] = {}
            for cid, val in zip(cands, vals):
                z = (float(val) - mean) / sd
                mm = (float(val) - vmin) / span - 0.5
                local[int(cid)] = float(confidence * max(-1.2, min(1.2, 0.45 * z + 0.70 * mm)))
            normalized[str(name)] = local
        return normalized

    def run_relation(self, query: RelationQuery, evidence: EvidenceBundle, top1_prob: float, best_by_mem: int) -> ToolTrace:
        plan = self.planner.plan_relation(evidence, top1_prob=top1_prob, best_by_mem=best_by_mem)
        outputs: Dict[str, ToolOutput] = {}
        for name in plan.tools:
            tool = self.tools.get(name)
            if tool is None:
                continue
            try:
                outputs[name] = tool.run_relation(query, evidence)
            except Exception as exc:
                outputs[name] = ToolOutput(name=name, summary=[f"tool_error={type(exc).__name__}"])
        normalized = self._normalize_outputs(evidence.candidate_set.ids, outputs)
        return ToolTrace(
            task="rel",
            conflict_type=plan.conflict_type,
            signature=plan.signature,
            selected_tools=list(outputs.keys()),
            outputs=outputs,
            normalized_scores=normalized,
        )
