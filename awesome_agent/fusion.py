from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from awesome_agent.contracts import CandidateSet, RelationQuery, ToolTrace, dedup_ints
from awesome_agent.mapping import NLMapper
from awesome_agent.memory.evidence import EvidenceBuilder
from awesome_agent.memory.stores import GraphMemory
from awesome_agent.tools import CausalProbeTool, CounterfactualVerifierTool, EntityHighOrderTool, EventCentricTool, PairLocalTool, ToolPlanner, ToolRunner

def REPORT_DEFAULTS():
    return {
        'candidate_count_sum': 0,
        'det_rule_guard_applied': 0,
        'det_rule_guard_candidates': 0,
        'fails': 0,
        'fixes': 0,
        'hits10_down': 0,
        'hits10_up': 0,
        'hits3_down': 0,
        'hits3_up': 0,
        'hits5_down': 0,
        'hits5_up': 0,
        'llm_reflection_hypothesis_trace_path': '',
        'llm_reflection_hypothesis_trace_records': 0,
        'mrr_delta_sum': 0.0,
        'neutral_flips': 0,
        'nonzero_delta_rows': 0,
        'proposal_rows': 0,
        'rank_changed_rows': 0,
        'rank_gain_sum': 0.0,
        'rank_improved': 0,
        'rank_worsened': 0,
        'rows': 0,
        'strong_agent_counterfactual_sum': 0.0,
        'strong_agent_high_counterfactual_rows': 0,
        'strong_agent_reflection_policy_rows': 0,
        'strong_agent_rows': 0,
        'strong_agent_semantic_sum': 0.0,
        'strong_agent_verifier_tool_rows': 0,
        'target_selection_no_rule': 0,
        'target_selection_rows': 0,
        'target_selection_rule_backed': 0,
        'target_selection_rule_candidate_rows': 0,
        'top1_flip_rows': 0,
    }

@dataclass
class FusionConfig:
    base_topn: int = 12
    expand_add: int = 10
    relation_upper_bound: Optional[int] = None
    max_delta: float = 2.4
    min_delta: float = -0.35
    commit_weight: float = 1.05
    tool_weight: float = 0.62
    memory_total_weight: float = 0.30
    recall_weight: float = 0.16
    reflection_weight: float = 0.24
    contradiction_penalty: float = 0.45
    top1_anchor: float = 0.05
    min_report_delta: float = 0.05
    llm_reflection_hypothesis_trace_path: str = "src/logs/llm_reflection_hypothesis_traces.jsonl"
    llm_reflection_hypothesis_trace_limit: int = 512
    llm_reflection_hypothesis_top_depth: int = 8
    llm_choice_delta: float = 0.95
    precision_min_commit: float = 0.12
    precision_min_commit_gap: float = 0.045
    precision_min_tool_gap: float = 0.12
    recall_only_max_delta: float = 0.08
    target_min_score: float = 0.48
    target_min_precision: float = 0.58
    target_fail_penalty: float = 2.0
    target_neutral_penalty: float = 0.04
    target_min_valid_fixes: int = 6
    target_contract_margin: float = 0.02
    target_use_rule_calibrator: bool = False
    target_rule_min_support: int = 10
    target_rule_min_fixes: int = 3
    target_rule_min_precision: float = 0.68
    target_rule_max_neutral_rate: float = 0.82
    target_rule_min_reward: float = 1.0
    trace_enabled: bool = False
    llm_framework: str = "llm_trace_distill"
    llm_reflection_hypothesis_enabled: bool = False
    llm_reflection_hypothesis_path: str = ""


@dataclass
class FusionResult:
    query: RelationQuery
    top1: int
    candidates: List[int]
    deltas: Dict[int, float] = field(default_factory=dict)
    ranked_ids: List[int] = field(default_factory=list)
    tool_trace: Optional[ToolTrace] = None
    used_llm: bool = False
    llm_choice: int = -1
    llm_confidence: float = 0.0
    llm_action: str = "none"
    llm_ranked_ids: List[int] = field(default_factory=list)
    llm_explanation: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)
    target_candidate: int = -1
    target_score: float = 0.0
    target_features: Dict[str, float] = field(default_factory=dict)

    def top_delta_candidate(self) -> int:
        if not self.deltas:
            return int(self.top1)
        return max(self.deltas, key=lambda rid: self.deltas[int(rid)])


class RelationScoreFusionAgent:
    def __init__(
        self,
        memory: Optional[GraphMemory] = None,
        mapper: Optional[Any] = None,
        config: Optional[FusionConfig] = None,
        llm_client: Optional[Any] = None,
        model_name: str = "",
        self_consistency: int = 1,
    ):
        self.memory = memory or GraphMemory()
        self.mapper = mapper or NLMapper()
        self.config = config or FusionConfig()
        self._sync_relation_space()
        self.evidence_builder = EvidenceBuilder(self.memory)
        self.tool_planner = ToolPlanner(self.memory)
        self.tool_runner = ToolRunner(
            {
                "pair_local": PairLocalTool(self.memory, self.mapper),
                "counterfactual_verifier": CounterfactualVerifierTool(self.memory, self.mapper),
                "entity_high_order": EntityHighOrderTool(self.memory, self.mapper),
                "event_centric": EventCentricTool(self.memory, self.mapper),
                "causal_probe": CausalProbeTool(self.memory, self.mapper),
            },
            self.tool_planner,
        )
        self.reset_target_calibrator()
        self.reset_report()

    def _sync_relation_space(self) -> None:
        base_count = 0
        try:
            base_count = int(self.mapper.base_relation_count())
        except Exception:
            base_count = 0
        if base_count <= 0:
            try:
                upper = int(getattr(self.config, "relation_upper_bound", 0) or 0)
                if upper > 0 and upper % 2 == 0:
                    base_count = upper // 2
            except Exception:
                base_count = 0
        if base_count > 0 and hasattr(self.memory, "set_relation_base_count"):
            try:
                self.memory.set_relation_base_count(base_count)
            except Exception:
                pass

    def reset_report(self):
        sink = getattr(self, '_target_trace_sink', None)
        if sink is not None:
            sink.close()
        self._target_trace_sink = None
        self.llm_reflection_hypothesis_trace_records = []
        self.report = REPORT_DEFAULTS()

    def reset_target_calibrator(self) -> None:
        self.target_samples: List[Dict[str, Any]] = []
        self.target_ready = False
        self.target_threshold = float("inf")
        self.target_summary: Dict[str, Any] = {
            "samples": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "threshold": float("inf"),
            "selected": 0,
            "selected_fix": 0,
            "selected_fail": 0,
            "selected_neutral": 0,
            "precision": 0.0,
            "reward": 0.0,
            "ready": False,
        }
        self.target_rules_ready = False
        self.target_rules: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.target_rule_summary: Dict[str, Any] = {
            "rules": 0,
            "best_precision": 0.0,
            "best_reward": 0.0,
            "best_rule": "",
        }

    @staticmethod
    def _tanh_norm(x: float, scale: float) -> float:
        if scale <= 0:
            return 0.0
        return float(math.tanh(float(x) / float(scale)))

    @staticmethod
    def _lookup_score(score_lookup: Any, candidate_id: int, fallback: float) -> float:
        if score_lookup is None:
            return float(fallback)
        cid = int(candidate_id)
        try:
            if isinstance(score_lookup, dict):
                return float(score_lookup.get(cid, fallback))
            value = score_lookup[cid]
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return float(fallback)

    @staticmethod
    def _rank_of_gt(order: Sequence[int], gt: Sequence[int]) -> Optional[int]:
        gt_set = {int(x) for x in gt}
        for idx, rid in enumerate(order, start=1):
            if int(rid) in gt_set:
                return idx
        return None

    def _uncertainty(self, candidate_set: CandidateSet) -> float:
        margin = candidate_set.margin()
        prob = candidate_set.top1_probability()
        u_margin = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, 2.0 * (margin - 0.6)))))
        u_prob = 1.0 - max(0.0, min(1.0, prob))
        return float(max(0.0, min(1.0, 0.60 * u_margin + 0.40 * u_prob)))

    def _bump_report_counter(self, field: str, key: Any) -> None:
        bucket = self.report.setdefault(str(field), {})
        if not isinstance(bucket, dict):
            bucket = {}
            self.report[str(field)] = bucket
        label = str(key or "unknown")
        bucket[label] = int(bucket.get(label, 0)) + 1

    def _resolve_reflection_hypothesis_trace_path(self, raw_path: str) -> Path:
        configured = str(raw_path or "").strip() or "src/logs/llm_reflection_hypothesis_traces.jsonl"
        path = Path(configured)
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parents[1]
        return repo_root / path

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(parsed):
            return float(default)
        return float(parsed)

    def _record_llm_reflection_hypothesis_trace(
        self,
        result: FusionResult,
        before_order: Sequence[int],
        after_order: Sequence[int],
        gt: Sequence[int],
    ) -> None:
        if not self.config.trace_enabled:
            return
        limit = max(0, int(getattr(self.config, "llm_reflection_hypothesis_trace_limit", 512) or 0))
        sink = getattr(self, "_target_trace_sink", None)
        recorded = sink.count if sink is not None else len(self.llm_reflection_hypothesis_trace_records)
        if limit <= 0 or recorded >= limit:
            return
        gt_set = {int(x) for x in (gt or [])}
        if not gt_set or not before_order:
            return
        debug = dict(getattr(result, "debug", {}) or {})
        components_raw = dict(debug.get("components", {}) or {})
        score_context: Dict[int, float] = {}
        for raw_key, raw_value in dict(debug.get("score_context", {}) or {}).items():
            try:
                score_context[int(raw_key)] = float(raw_value)
            except Exception:
                continue
        components: Dict[int, Dict[str, Any]] = {}
        for raw_key, raw_value in components_raw.items():
            try:
                rid = int(raw_key)
            except Exception:
                try:
                    rid = int(float(raw_key))
                except Exception:
                    continue
            if isinstance(raw_value, dict):
                components[int(rid)] = dict(raw_value)
        if not components:
            return
        before_top = int(before_order[0])
        after_top = int(after_order[0]) if after_order else before_top
        before_rank = self._rank_of_gt(before_order, gt_set)
        after_rank = self._rank_of_gt(after_order, gt_set)
        base_order = [int(x) for x in list(after_order or [])]
        if not base_order:
            base_order = [int(x) for x in list(getattr(result, "ranked_ids", []) or [])]
        if not base_order:
            base_order = [int(x) for x in list(before_order or [])]
        base_top = int(base_order[0]) if base_order else int(after_top)
        base_rank = self._rank_of_gt(base_order, gt_set)
        raw_rank_by_id = {int(rid): int(idx) for idx, rid in enumerate(list(before_order or []), start=1)}
        base_rank_by_id = {int(rid): int(idx) for idx, rid in enumerate(list(base_order or []), start=1)}
        depth = max(1, int(getattr(self.config, "llm_reflection_hypothesis_top_depth", 8) or 8))
        ranked = dedup_ints(
            [int(x) for x in list(before_order or [])[:depth]]
            + [int(x) for x in list(base_order or [])[:depth]],
            lower=0,
            upper=getattr(self.config, "relation_upper_bound", None),
        )
        ranked = [int(x) for x in ranked if int(x) in components][: max(depth, min(len(ranked), depth * 2))]
        candidates: List[Dict[str, Any]] = []
        feature_names = [
            "commit",
            "commit_gap",
            "tool",
            "tool_gap",
            "total",
            "total_gap",
            "semantic_fit",
            "semantic_gap",
            "verifier_gap",
            "counterfactual_risk",
            "negative_penalty",
            "pop_penalty",
            "reflection",
            "reflection_policy",
            "baseline_protected",
            "direct",
            "pair_transition",
            "prior",
        ]
        for rank, rid in enumerate(ranked, start=1):
            comp = dict(components.get(int(rid), {}) or {})
            if int(before_top) not in gt_set and int(rid) in gt_set:
                outcome = "fix"
            elif int(before_top) in gt_set and int(rid) not in gt_set:
                outcome = "fail"
            else:
                outcome = "neutral"
            features = {
                name: round(self._safe_float(comp.get(name, 0.0), 0.0), 6)
                for name in feature_names
                if name in comp
            }
            raw_rank = int(raw_rank_by_id.get(int(rid), rank))
            current_base_rank = int(base_rank_by_id.get(int(rid), raw_rank))
            score = self._safe_float(score_context.get(int(rid), 0.0), 0.0)
            prev_score = score
            top_score = score
            if current_base_rank > 1 and current_base_rank - 2 < len(base_order):
                prev_score = self._safe_float(score_context.get(int(base_order[current_base_rank - 2]), score), score)
            if base_order:
                top_score = self._safe_float(score_context.get(int(base_order[0]), score), score)
            gap_prev = max(0.0, float(prev_score) - float(score)) if score_context else 0.0
            gap_top = max(0.0, float(top_score) - float(score)) if score_context else 0.0
            step_after_rank = base_rank
            step_mrr_delta = 0.0
            if base_rank is not None and base_rank > 0 and current_base_rank > 1:
                if int(rid) in gt_set:
                    step_after_rank = min(int(base_rank), max(1, int(current_base_rank) - 1))
                elif int(current_base_rank) <= len(base_order):
                    above_rid = int(base_order[int(current_base_rank) - 2])
                    if int(above_rid) in gt_set and int(base_rank) == int(current_base_rank) - 1:
                        step_after_rank = int(current_base_rank)
                if step_after_rank is not None and step_after_rank > 0:
                    step_mrr_delta = float((1.0 / float(step_after_rank)) - (1.0 / float(base_rank)))
            step_outcome = "fix" if step_mrr_delta > 1e-12 else "fail" if step_mrr_delta < -1e-12 else "neutral"
            features.update(
                {
                    "rank": float(raw_rank),
                    "current_rank": float(current_base_rank),
                    "score_gap_to_prev": round(float(gap_prev), 6),
                    "score_gap_to_top": round(float(gap_top), 6),
                    "needed_delta_to_prev": round(float(gap_prev), 6),
                    "needed_delta_to_top": round(float(gap_top), 6),
                    "step_mrr_delta": round(float(step_mrr_delta), 8),
                }
            )
            llm_provenance = None
            candidates.append(
                {
                    "rid": int(rid),
                    "rank": int(raw_rank),
                    "base_rank": int(current_base_rank),
                    "is_gt": bool(int(rid) in gt_set),
                    "outcome_if_promoted": outcome,
                    "outcome_if_step_lift": step_outcome,
                    "step_after_rank": int(step_after_rank) if step_after_rank is not None else None,
                    "step_mrr_delta": round(float(step_mrr_delta), 8),
                    "score": round(float(score), 6),
                    "score_gap_to_prev": round(float(gap_prev), 6),
                    "score_gap_to_top": round(float(gap_top), 6),
                    "needed_delta_to_prev": round(float(gap_prev), 6),
                    "needed_delta_to_top": round(float(gap_top), 6),
                    "features": features,
                    **({"llm_provenance": llm_provenance} if llm_provenance is not None else {}),
                }
            )
        trace = {
            "query": {"s": int(result.query.s), "o": int(result.query.o), "t": int(result.query.t)},
            "target_relation_id": debug.get("target_relation_id"),
            "signature": str(getattr(getattr(result, "tool_trace", None), "signature", "") or ""),
            "selected_tools": list(getattr(getattr(result, "tool_trace", None), "selected_tools", []) or []),
            "prev_rel": int(debug.get("prev_rel", -1)),
            "before_top": int(before_top),
            "after_top": int(after_top),
            "base_top": int(base_top),
            "gt": sorted(int(x) for x in gt_set),
            "before_rank": int(before_rank) if before_rank is not None else None,
            "after_rank": int(after_rank) if after_rank is not None else None,
            "base_rank": int(base_rank) if base_rank is not None else None,
            "base_order": [int(x) for x in list(base_order or [])[: max(8, depth)]],
            "row_outcome": "fix"
            if (before_top not in gt_set and after_top in gt_set)
            else "fail"
            if (before_top in gt_set and after_top not in gt_set)
            else "neutral",
            "candidates": candidates,
        }
        if os.environ.get("AGENT_LLM_TRACE_STREAM") == "1":
            from awesome_agent.trace_storage import TargetTraceSink, validate_collector
            if sink is None:
                validate_collector(self.config)
                path = self._resolve_reflection_hypothesis_trace_path(self.config.llm_reflection_hypothesis_trace_path)
                sink = self._target_trace_sink = TargetTraceSink(path)
            sink.append(trace)
            self.report["llm_reflection_hypothesis_trace_records"] = sink.count
            return
        self.llm_reflection_hypothesis_trace_records.append(trace)
        self.report["llm_reflection_hypothesis_trace_records"] = int(
            len(self.llm_reflection_hypothesis_trace_records)
        )

    def _write_llm_reflection_hypothesis_traces(self) -> str:
        sink = getattr(self, "_target_trace_sink", None)
        if sink is not None:
            sink.close()
            self.report["llm_reflection_hypothesis_trace_path"] = str(sink.path)
            print(f"[TargetTraceStream] records={sink.count} actions={sink.actions} unrecoverable={sink.dropped}", flush=True)
            return str(sink.path)
        records = list(getattr(self, "llm_reflection_hypothesis_trace_records", []) or [])
        if not records:
            return ""
        path = self._resolve_reflection_hypothesis_trace_path(
            str(getattr(self.config, "llm_reflection_hypothesis_trace_path", "") or "")
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for item in records:
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            self.report["llm_reflection_hypothesis_trace_path"] = str(path)
            return str(path)
        except Exception as exc:
            self._bump_report_counter("llm_reflection_hypothesis_load_errors", f"trace_write_{type(exc).__name__}")
            return ""

    def observe_llm_reflection_hypothesis_outcome(self, result, before_order, after_order, gt):
        self._record_llm_reflection_hypothesis_trace(result, before_order, after_order, gt)

    @staticmethod
    def _bin_value(value: float, cuts: Sequence[float]) -> int:
        val = float(value)
        for idx, cut in enumerate(cuts):
            if val < float(cut):
                return int(idx)
        return int(len(cuts))

    def _target_rule_keys(self, features: Dict[str, float]) -> List[Tuple[Any, ...]]:
        f = features or {}
        score = float(f.get("score", 0.0))
        commit = float(f.get("commit", 0.0))
        commit_gap = float(f.get("commit_gap", 0.0))
        tool_gap = float(f.get("tool_gap", 0.0))
        total_gap = float(f.get("total_gap", 0.0))
        reflection = float(f.get("reflection", 0.0))
        direct = float(f.get("direct", 0.0))
        pair_transition = float(f.get("pair_transition", 0.0))
        prior = float(f.get("prior", 0.0))
        needed = float(f.get("needed_delta", 0.0))
        uncertainty = float(f.get("uncertainty", 0.0))
        recall_only = int(float(f.get("recall_only", 0.0)) > 0.5)
        weak_source = int(float(f.get("weak_source", 0.0)) > 0.5)
        contradiction = int(float(f.get("contradiction", 0.0)) > 0.5)

        source_count = (
            int(commit >= 0.08)
            + int(commit_gap >= 0.04)
            + int(tool_gap >= 0.08)
            + int(total_gap >= 0.04)
            + int(reflection >= 0.04)
            + int(direct >= 0.10)
            + int(pair_transition >= 0.05)
            + int(prior >= 0.06)
        )
        source_count = min(5, int(source_count))
        score_bin = self._bin_value(score, [0.35, 0.55, 0.75, 0.95, 1.15, 1.35])
        need_bin = self._bin_value(needed, [0.08, 0.20, 0.45, 0.80, 1.20])
        commit_bin = self._bin_value(max(commit, commit_gap), [0.03, 0.07, 0.12, 0.20, 0.32])
        tool_bin = self._bin_value(tool_gap, [0.00, 0.05, 0.12, 0.24, 0.42])
        direct_bin = int(direct >= 0.10) + int(direct >= 0.24)
        trans_bin = int(pair_transition >= 0.04) + int(pair_transition >= 0.10)
        prior_bin = int(prior >= 0.05) + int(prior >= 0.12)
        reflection_bin = int(reflection >= 0.04) + int(reflection >= 0.12)
        uncertainty_bin = self._bin_value(uncertainty, [0.25, 0.45, 0.65, 0.82])
        risk = (recall_only, weak_source, contradiction)

        keys = [
            ("score_src_need", score_bin, source_count, need_bin, risk),
            ("commit_tool_need", commit_bin, tool_bin, need_bin, risk),
            ("sources_need", source_count, direct_bin, trans_bin, prior_bin, need_bin, risk),
            ("commit_sources", commit_bin, source_count, direct_bin, trans_bin, risk),
            ("tool_reflect_need", tool_bin, reflection_bin, need_bin, risk),
            ("score_commit_tool", score_bin, commit_bin, tool_bin, risk),
            ("uncertain_sources", uncertainty_bin, source_count, need_bin, risk),
        ]
        if direct_bin or trans_bin or prior_bin:
            keys.append(("hard_source_combo", direct_bin, trans_bin, prior_bin, commit_bin, need_bin, risk))
        if tool_bin >= 2 and commit_bin >= 2:
            keys.append(("tool_commit_strong", tool_bin, commit_bin, source_count, need_bin, risk))
        return keys

    def _match_target_rule(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "target_rules_ready", False)):
            return None
        best: Optional[Dict[str, Any]] = None
        for key in self._target_rule_keys(features):
            item = self.target_rules.get(tuple(key))
            if not item:
                continue
            if best is None or (
                float(item.get("priority", 0.0)),
                float(item.get("precision", 0.0)),
                int(item.get("fix", 0)),
            ) > (
                float(best.get("priority", 0.0)),
                float(best.get("precision", 0.0)),
                int(best.get("fix", 0)),
                ):
                best = dict(item)
        return best

    def _target_feature_score(
        self,
        comp: Dict[str, float],
        uncertainty: float,
        needed_delta: float,
        max_delta: float,
        conflict: float,
    ) -> Tuple[float, Dict[str, float]]:
        commit = max(0.0, float(comp.get("commit", 0.0)))
        commit_gap = max(0.0, float(comp.get("commit_gap", 0.0)))
        tool_gap = max(0.0, float(comp.get("tool_gap", 0.0)))
        total_gap = max(0.0, float(comp.get("total_gap", 0.0)))
        recall_gap = max(0.0, float(comp.get("recall_gap", 0.0)))
        reflection = max(0.0, float(comp.get("reflection", 0.0)))
        negative_penalty = max(0.0, float(comp.get("negative_penalty", comp.get("neg_penalty", 0.0))))
        pop_penalty = max(0.0, float(comp.get("pop_penalty", 0.0)))
        direct = max(0.0, float(comp.get("direct", 0.0)))
        pair_transition = max(0.0, float(comp.get("pair_transition", 0.0)))
        prior = max(0.0, float(comp.get("prior", 0.0)))
        semantic_fit = max(0.0, float(comp.get("semantic_fit", 0.0)))
        semantic_gap = float(comp.get("semantic_gap", 0.0))
        counterfactual_risk = max(0.0, float(comp.get("counterfactual_risk", 0.0)))
        verifier_gap = float(comp.get("verifier_gap", 0.0))
        reflection_policy = float(comp.get("reflection_policy", 0.0))

        commit_norm = max(0.0, min(1.0, commit / 0.30))
        commit_gap_norm = max(0.0, min(1.0, commit_gap / 0.16))
        tool_gap_norm = max(0.0, min(1.0, tool_gap / 0.42))
        total_gap_norm = max(0.0, min(1.0, total_gap / 0.55))
        reflection_norm = max(0.0, min(1.0, reflection / 0.22))
        direct_norm = max(0.0, min(1.0, direct / 0.35))
        transition_norm = max(0.0, min(1.0, pair_transition / 0.16))
        prior_norm = max(0.0, min(1.0, prior / 0.18))
        semantic_norm = max(0.0, min(1.0, semantic_fit / 0.22))
        semantic_gap_norm = max(0.0, min(1.0, semantic_gap / 0.16))
        verifier_gap_norm = max(0.0, min(1.0, verifier_gap / 0.18))
        counterfactual_norm = max(0.0, min(1.0, counterfactual_risk / 0.34))
        need_norm = max(0.0, min(1.0, needed_delta / max(1e-6, float(self.config.llm_choice_delta) + 0.20)))
        close_norm = 1.0 - need_norm
        max_delta_norm = max(0.0, min(1.0, max_delta / max(1e-6, float(self.config.max_delta))))

        recall_only = bool(commit < 0.07 and commit_gap < 0.035 and tool_gap < 0.22 and recall_gap > 0.10)
        weak_source = bool(direct < 0.05 and pair_transition < 0.03 and prior < 0.04 and tool_gap < 0.06 and semantic_fit < 0.12)
        contradiction = bool(
            negative_penalty > 0.12
            or float(comp.get("commit_gap", 0.0)) < -0.10
            or float(comp.get("tool_gap", 0.0)) < -0.24
            or counterfactual_risk >= 0.34
        )

        score = (
            0.18 * max(0.0, min(1.0, float(uncertainty)))
            + 0.20 * commit_norm
            + 0.24 * commit_gap_norm
            + 0.20 * tool_gap_norm
            + 0.10 * total_gap_norm
            + 0.08 * reflection_norm
            + 0.08 * direct_norm
            + 0.07 * transition_norm
            + 0.05 * prior_norm
            + 0.09 * semantic_norm
            + 0.08 * semantic_gap_norm
            + 0.07 * verifier_gap_norm
            + 0.12 * close_norm
            + 0.08 * max_delta_norm
            + 0.06 * max(0.0, min(1.0, float(conflict)))
            - 0.16 * need_norm
            - 0.26 * counterfactual_norm
            - 0.20 * negative_penalty
            - 0.08 * pop_penalty
        )
        if reflection_policy < -0.05:
            score += 0.25 * reflection_policy
        if recall_only:
            score -= 0.20
        if weak_source:
            score -= 0.14
        if contradiction:
            score -= 0.30
        features = {
            "score": float(score),
            "commit": float(commit),
            "commit_gap": float(commit_gap),
            "tool_gap": float(tool_gap),
            "total_gap": float(total_gap),
            "recall_gap": float(recall_gap),
            "reflection": float(reflection),
            "negative_penalty": float(negative_penalty),
            "pop_penalty": float(pop_penalty),
            "direct": float(direct),
            "pair_transition": float(pair_transition),
            "prior": float(prior),
            "semantic_fit": float(semantic_fit),
            "semantic_gap": float(semantic_gap),
            "counterfactual_risk": float(counterfactual_risk),
            "verifier_gap": float(verifier_gap),
            "reflection_policy": float(reflection_policy),
            "needed_delta": float(max(0.0, needed_delta)),
            "max_delta": float(max_delta),
            "uncertainty": float(uncertainty),
            "recall_only": float(1.0 if recall_only else 0.0),
            "weak_source": float(1.0 if weak_source else 0.0),
            "contradiction": float(1.0 if contradiction else 0.0),
        }
        return float(max(0.0, score)), features

    def _select_target_candidate(
        self,
        evidence: Any,
        components: Dict[int, Dict[str, float]],
        deltas: Dict[int, float],
        ranked: Sequence[int],
        uncertainty: float,
        tool_trace: Optional[ToolTrace],
    ) -> Tuple[int, float, Dict[str, float]]:
        top1 = int(evidence.candidate_set.top1)
        current_top = int(ranked[0]) if ranked else top1
        current_score = float(evidence.candidate_set.logit_of(current_top) + deltas.get(current_top, 0.0))
        max_delta = max((float(v) for rid, v in deltas.items() if int(rid) != current_top), default=0.0)
        conflict = 0.0
        if tool_trace is not None and str(tool_trace.conflict_type) != "stable":
            conflict = 1.0

        candidates: List[Tuple[int, float, float, Dict[str, float], Optional[Dict[str, Any]]]] = []
        use_rules = bool(self.config.target_use_rule_calibrator) and bool(getattr(self, "target_rules_ready", False))
        for rid in ranked:
            rid = int(rid)
            if rid == current_top:
                continue
            comp = components.get(rid, {})
            if not comp:
                continue
            cand_score = float(evidence.candidate_set.logit_of(rid) + deltas.get(rid, 0.0))
            needed_delta = max(0.0, current_score - cand_score + float(self.config.target_contract_margin))
            target_score, features = self._target_feature_score(comp, uncertainty, needed_delta, max_delta, conflict)
            features["current_top"] = float(current_top)
            features["original_top1"] = float(top1)
            rule_match: Optional[Dict[str, Any]] = None
            adjusted_score = float(target_score)
            if use_rules:
                rule_match = self._match_target_rule(dict(features or {}))
                if rule_match:
                    precision = float(rule_match.get("precision", 0.0))
                    fix = float(rule_match.get("fix", 0.0))
                    fail = float(rule_match.get("fail", 0.0))
                    priority = float(rule_match.get("priority", 0.0))
                    adjusted_score += (
                        0.20 * max(0.0, min(1.0, precision))
                        + 0.035 * min(8.0, max(0.0, fix))
                        + 0.030 * max(0.0, min(4.0, priority))
                        - 0.060 * min(5.0, max(0.0, fail))
                    )
                    features = {
                        **dict(features or {}),
                        "rule_support": float(rule_match.get("support", 0)),
                        "rule_fix": float(rule_match.get("fix", 0)),
                        "rule_fail": float(rule_match.get("fail", 0)),
                        "rule_neutral": float(rule_match.get("neutral", 0)),
                        "rule_precision": float(rule_match.get("precision", 0.0)),
                        "rule_reward": float(rule_match.get("reward", 0.0)),
                        "rule_priority": float(rule_match.get("priority", 0.0)),
                        "rule_examples": " || ".join(str(x) for x in list(rule_match.get("examples", []) or [])[:4]),
                    }
                else:
                    adjusted_score -= 0.18
            features["selection_score"] = float(adjusted_score)
            candidates.append((int(rid), float(target_score), float(adjusted_score), features, rule_match))
        if not candidates:
            return -1, 0.0, {}
        rule_candidates = [item for item in candidates if item[4] is not None]
        if hasattr(self, "report"):
            self.report["target_selection_rows"] += 1
            if rule_candidates:
                self.report["target_selection_rule_candidate_rows"] += 1
        pool = rule_candidates if rule_candidates else candidates
        best_id, best_score, _, best_features, best_rule = max(
            pool,
            key=lambda item: (
                float(item[2]),
                float((item[4] or {}).get("precision", 0.0)),
                float(item[1]),
                -float(item[3].get("needed_delta", 0.0)),
            ),
        )
        if hasattr(self, "report"):
            if best_rule is not None:
                self.report["target_selection_rule_backed"] += 1
            else:
                self.report["target_selection_no_rule"] += 1
        return int(best_id), float(best_score), dict(best_features)

    def _apply_deterministic_rule_guard(
        self,
        evidence: Any,
        components: Dict[int, Dict[str, float]],
        deltas: Dict[int, float],
        uncertainty: float,
        tool_trace: Optional[ToolTrace],
    ) -> Dict[int, float]:
        """Damp high-risk deterministic flips that validation rules do not support."""

        if not (
            bool(self.config.target_use_rule_calibrator)
            and bool(getattr(self, "target_rules_ready", False))
            and bool(getattr(self, "target_rules", {}))
        ):
            return deltas
        guarded = dict(deltas)
        top1 = int(evidence.candidate_set.top1)
        if top1 not in set(int(x) for x in evidence.candidate_set.ids):
            return guarded
        top_score = float(evidence.candidate_set.logit_of(top1) + guarded.get(top1, 0.0))
        max_delta = max((float(v) for rid, v in guarded.items() if int(rid) != top1), default=0.0)
        conflict = 1.0 if tool_trace is not None and str(tool_trace.conflict_type) != "stable" else 0.0

        for rid in list(evidence.candidate_set.ids):
            rid = int(rid)
            if rid == top1:
                continue
            current_delta = float(guarded.get(rid, 0.0) or 0.0)
            if current_delta <= max(0.04, float(self.config.min_report_delta)):
                continue
            candidate_score = float(evidence.candidate_set.logit_of(rid) + current_delta)
            if candidate_score <= top_score:
                continue
            comp = dict(components.get(rid, {}) or {})
            target_score, features = self._target_feature_score(
                comp,
                uncertainty=float(uncertainty),
                needed_delta=max(0.0, top_score - candidate_score + float(self.config.target_contract_margin)),
                max_delta=max_delta,
                conflict=conflict,
            )
            features["current_top"] = float(top1)
            features["original_top1"] = float(top1)
            rule_match = self._match_target_rule(dict(features or {}))
            if rule_match:
                continue

            commit_gap = float(comp.get("commit_gap", 0.0) or 0.0)
            tool_gap = float(comp.get("tool_gap", 0.0) or 0.0)
            total_gap = float(comp.get("total_gap", 0.0) or 0.0)
            hard_source = max(
                float(comp.get("direct", 0.0) or 0.0),
                float(comp.get("pair_transition", 0.0) or 0.0),
                float(comp.get("prior", 0.0) or 0.0),
            )
            semantic_gap = float(comp.get("semantic_gap", 0.0) or 0.0)
            verifier_gap = float(comp.get("verifier_gap", 0.0) or 0.0)
            counterfactual_risk = float(comp.get("counterfactual_risk", 0.0) or 0.0)
            neg_penalty = float(comp.get("negative_penalty", comp.get("neg_penalty", 0.0)) or 0.0)
            pop_penalty = float(comp.get("pop_penalty", 0.0) or 0.0)
            baseline_protected = bool(float(comp.get("baseline_protected", 0.0) or 0.0) > 0.5)
            weak_ruleless_override = bool(
                baseline_protected
                and hard_source < 0.12
                and commit_gap < 0.18
                and (tool_gap < 0.55 or commit_gap < 0.08)
                and semantic_gap < 0.16
                and verifier_gap < 0.10
            )
            tool_dominant_weak_hard = bool(hard_source < 0.08 and commit_gap < 0.10 and tool_gap >= 0.12)
            contradicted = bool(counterfactual_risk >= 0.24 or neg_penalty > 0.14 or pop_penalty > 0.08)
            weak_total = bool(total_gap < 0.04 and commit_gap < 0.10 and hard_source < 0.10)
            reason = ""
            if weak_ruleless_override:
                reason = "det_rule_guard_weak_ruleless_override"
            elif tool_dominant_weak_hard:
                reason = "det_rule_guard_tool_dominant_weak_hard"
            elif contradicted and weak_total:
                reason = "det_rule_guard_contradicted_weak_total"
            if not reason:
                continue

            self.report["det_rule_guard_candidates"] += 1
            cap = float(top_score - evidence.candidate_set.logit_of(rid) - 1e-4)
            new_delta = max(float(self.config.min_delta), min(current_delta, cap))
            if new_delta + 1e-9 < current_delta:
                guarded[rid] = float(new_delta)
                self.report["det_rule_guard_applied"] += 1
                self._bump_report_counter("det_rule_guard_reasons", reason)
                top_score = float(evidence.candidate_set.logit_of(top1) + guarded.get(top1, 0.0))
        return guarded

    def _component_scores(
        self,
        query: RelationQuery,
        ids: List[int],
        logits: List[float],
        score_lookup: Any = None,
    ) -> Tuple[Any, ToolTrace, Dict[int, Dict[str, float]]]:
        base = CandidateSet.from_topk(ids[: self.config.base_topn], logits[: self.config.base_topn], source="gnn")
        expanded = self.evidence_builder.expand_relation_candidates(
            query,
            base,
            max_add=self.config.expand_add,
            upper_bound=self.config.relation_upper_bound,
        )
        if score_lookup is not None:
            expanded = CandidateSet.from_topk(
                expanded.ids,
                [self._lookup_score(score_lookup, rid, expanded.logit_of(int(rid))) for rid in expanded.ids],
                source=expanded.candidates[0].source if expanded.candidates else "gnn+memory",
            )
        evidence = self.evidence_builder.build_relation_evidence(query, expanded)
        top1_prob = expanded.top1_probability()
        best_by_mem = evidence.best_by_total()
        tool_trace = self.tool_runner.run_relation(query, evidence, top1_prob=top1_prob, best_by_mem=best_by_mem)
        if self.evidence_builder.audit_observer is not None:
            self.evidence_builder.audit_observer.tools(tool_trace)
        if tool_trace.suggestions:
            extra = dedup_ints(tool_trace.suggestions, lower=0, upper=self.config.relation_upper_bound)[:3]
            if extra:
                ids2 = expanded.ids + extra
                logits2 = [self._lookup_score(score_lookup, rid, expanded.logit_of(int(rid))) for rid in ids2]
                expanded2 = CandidateSet.from_topk(ids2, logits2, source="gnn+memory+tools")
                evidence = self.evidence_builder.build_relation_evidence(query, expanded2)
                best_by_mem = evidence.best_by_total()
                top1_prob = expanded2.top1_probability()
                tool_trace = self.tool_runner.run_relation(query, evidence, top1_prob=top1_prob, best_by_mem=best_by_mem)
                if self.evidence_builder.audit_observer is not None:
                    self.evidence_builder.audit_observer.tools(tool_trace)
        components: Dict[int, Dict[str, float]] = {}
        top1 = evidence.candidate_set.top1
        ev_top = evidence.evidence(top1)
        top_commit = EvidenceBuilder.commit_support(ev_top)
        top_recall = EvidenceBuilder.recall_support(ev_top)
        top_total = ev_top.total
        top_tool = sum(scores.get(top1, 0.0) for scores in tool_trace.normalized_scores.values())
        top_semantic = float(getattr(ev_top, "semantic_fit", 0.0) or 0.0)
        top_counterfactual = float(getattr(ev_top, "counterfactual_risk", 0.0) or 0.0)
        signature = str(tool_trace.signature)
        verifier_out = tool_trace.outputs.get("counterfactual_verifier") if tool_trace is not None else None
        top_verifier_support = float(verifier_out.scores.get(top1, 0.0)) if verifier_out is not None else 0.0
        top_verifier_penalty = float(verifier_out.penalties.get(top1, 0.0)) if verifier_out is not None else 0.0
        top_hard = max(float(ev_top.direct), float(ev_top.pair_transition), float(ev_top.prior))
        baseline_protected_flag = float(
            1.0
            if (
                float(top_commit) >= 0.24
                or float(top_total) >= 0.34
                or float(top_tool) >= 0.55
                or top_hard >= 0.20
            )
            else 0.0
        )
        for rid in evidence.candidate_set.ids:
            ev = evidence.evidence(int(rid))
            commit = EvidenceBuilder.commit_support(ev)
            recall = EvidenceBuilder.recall_support(ev)
            tool = sum(scores.get(int(rid), 0.0) for scores in tool_trace.normalized_scores.values())
            verifier_support = float(verifier_out.scores.get(int(rid), 0.0)) if verifier_out is not None else float(getattr(ev, "verifier_support", 0.0) or 0.0)
            verifier_penalty = float(verifier_out.penalties.get(int(rid), 0.0)) if verifier_out is not None else float(getattr(ev, "verifier_penalty", 0.0) or 0.0)
            semantic_fit = float(getattr(ev, "semantic_fit", 0.0) or 0.0)
            counterfactual_risk = float(getattr(ev, "counterfactual_risk", 0.0) or 0.0)
            preliminary = {
                "commit": float(commit),
                "commit_gap": float(commit - top_commit),
                "recall": float(recall),
                "recall_gap": float(recall - top_recall),
                "total": float(ev.total),
                "total_gap": float(ev.total - top_total),
                "tool": float(tool),
                "tool_gap": float(tool - top_tool),
                "semantic_fit": float(semantic_fit),
                "semantic_gap": float(semantic_fit - top_semantic),
                "counterfactual_risk": float(counterfactual_risk),
                "counterfactual_gap": float(counterfactual_risk - top_counterfactual),
                "verifier_support": float(verifier_support),
                "verifier_penalty": float(verifier_penalty),
                "verifier_gap": float((verifier_support - verifier_penalty) - (top_verifier_support - top_verifier_penalty)),
                "direct": float(ev.direct),
                "pair_transition": float(ev.pair_transition),
                "prior": float(ev.prior),
                "baseline_protected": float(baseline_protected_flag),
            }
            reflection_base = self.memory.commit_utility_rel(signature, evidence.prev_rel, int(rid), top1=top1)
            reflection_policy = self.memory.relation_policy_utility(signature, evidence.prev_rel, int(rid), top1=top1, features=preliminary)
            reflection = float(reflection_base + reflection_policy)
            components[int(rid)] = {
                "commit": float(commit),
                "commit_gap": float(commit - top_commit),
                "recall": float(recall),
                "recall_gap": float(recall - top_recall),
                "total": float(ev.total),
                "total_gap": float(ev.total - top_total),
                "tool": float(tool),
                "tool_gap": float(tool - top_tool),
                "reflection": float(reflection),
                "reflection_base": float(reflection_base),
                "reflection_policy": float(reflection_policy),
                "semantic_fit": float(semantic_fit),
                "semantic_gap": float(semantic_fit - top_semantic),
                "counterfactual_risk": float(counterfactual_risk),
                "counterfactual_gap": float(counterfactual_risk - top_counterfactual),
                "verifier_support": float(verifier_support),
                "verifier_penalty": float(verifier_penalty),
                "verifier_gap": float((verifier_support - verifier_penalty) - (top_verifier_support - top_verifier_penalty)),
                "baseline_protected": float(baseline_protected_flag),
                "neg_penalty": float(ev.neg_penalty + ev.pop_penalty),
                "negative_penalty": float(ev.neg_penalty),
                "pop_penalty": float(ev.pop_penalty),
                "direct": float(ev.direct),
                "pair_transition": float(ev.pair_transition),
                "prior": float(ev.prior),
                "commit_raw": float(ev.commit),
                "recall_raw": float(ev.recall),
            }
        return evidence, tool_trace, components

    def _score_component_delta(
        self,
        *,
        rid: int,
        top1: int,
        comp: Dict[str, Any],
        uncertainty: float,
        scale: float,
    ) -> float:
        rid = int(rid)
        top1 = int(top1)
        if rid == top1:
            return -float(self.config.top1_anchor) * max(0.0, float(uncertainty) - 0.35)
        commit_gap = float(comp.get("commit_gap", 0.0))
        commit = float(comp.get("commit", 0.0))
        tool_gap = float(comp.get("tool_gap", 0.0))
        recall_gap = float(comp.get("recall_gap", 0.0))
        total_gap = float(comp.get("total_gap", 0.0))
        reflection = float(comp.get("reflection", 0.0))
        neg_penalty = float(comp.get("neg_penalty", 0.0))
        semantic_gap = float(comp.get("semantic_gap", 0.0))
        counterfactual_risk = float(comp.get("counterfactual_risk", 0.0))
        verifier_gap = float(comp.get("verifier_gap", 0.0))

        recall_only = bool(commit < 0.07 and commit_gap < 0.035 and tool_gap < 0.25)
        contradiction = bool(tool_gap < -0.35 or commit_gap < -0.12 or reflection < -0.20 or counterfactual_risk >= 0.34)
        precision_evidence = bool(
            commit_gap >= float(self.config.precision_min_commit_gap)
            or commit >= float(self.config.precision_min_commit)
            or (tool_gap >= float(self.config.precision_min_tool_gap) and commit_gap >= 0.02)
            or (semantic_gap >= 0.10 and counterfactual_risk <= 0.08 and verifier_gap >= 0.04)
        )

        raw = (
            self.config.commit_weight * self._tanh_norm(commit_gap, 0.30)
            + self.config.tool_weight * self._tanh_norm(tool_gap, 0.85)
            + self.config.memory_total_weight * self._tanh_norm(total_gap, 0.75)
            + self.config.recall_weight * self._tanh_norm(recall_gap, 0.55)
            + self.config.reflection_weight * self._tanh_norm(reflection, 0.35)
            + 0.18 * self._tanh_norm(semantic_gap, 0.22)
            + 0.16 * self._tanh_norm(verifier_gap, 0.28)
            - self.config.contradiction_penalty * max(0.0, neg_penalty)
            - 0.42 * max(0.0, counterfactual_risk)
        )
        if recall_only:
            raw = min(raw, float(self.config.recall_only_max_delta) + 0.04 * max(0.0, recall_gap))
        if contradiction and not precision_evidence:
            raw -= 0.40
        if precision_evidence and tool_gap >= 0.0:
            raw += 0.10
        if not precision_evidence and commit_gap < 0.02:
            raw = min(raw, 0.05)
        delta = float(scale) * raw
        observer = getattr(self.evidence_builder, 'audit_observer', None)
        if observer is not None and hasattr(observer, 'component_delta'):
            observer.component_delta(delta, self.config.min_delta, self.config.max_delta)
        return float(max(self.config.min_delta, min(self.config.max_delta, delta)))

    def score_relation(
        self,
        s: int,
        o: int,
        t: int,
        gnn_topk_ids: Sequence[Any],
        gnn_logits: Sequence[Any],
        use_llm: bool = False,
        count_row: bool = True,
        score_lookup: Any = None,
        llm_intervention_mode: str = "",
    ) -> FusionResult:
        self._sync_relation_space()
        ids = dedup_ints(gnn_topk_ids, lower=0, upper=self.config.relation_upper_bound)
        logits = [float(x) for x in (list(gnn_logits) if gnn_logits is not None else [])]
        if not ids:
            return FusionResult(query=RelationQuery(int(s), int(o), int(t)), top1=-1, candidates=[])
        query = RelationQuery(int(s), int(o), int(t))
        evidence, tool_trace, components = self._component_scores(query, ids, logits, score_lookup=score_lookup)
        top1 = int(evidence.candidate_set.top1)
        uncertainty = self._uncertainty(evidence.candidate_set)
        scale = float(0.42 + 0.95 * uncertainty)
        deltas: Dict[int, float] = {}
        debug: Dict[str, Any] = {
            "uncertainty": uncertainty,
            "top1_prob": float(evidence.candidate_set.top1_probability()),
            "components": components,
            "prev_rel": int(evidence.prev_rel),
        }
        if bool(count_row):
            max_semantic = max((float(comp.get("semantic_fit", 0.0) or 0.0) for comp in components.values()), default=0.0)
            max_counterfactual = max((float(comp.get("counterfactual_risk", 0.0) or 0.0) for comp in components.values()), default=0.0)
            self.report["strong_agent_rows"] += 1
            self.report["strong_agent_semantic_sum"] += float(max_semantic)
            self.report["strong_agent_counterfactual_sum"] += float(max_counterfactual)
            if max_counterfactual >= 0.24:
                self.report["strong_agent_high_counterfactual_rows"] += 1
            if tool_trace is not None and "counterfactual_verifier" in set(tool_trace.selected_tools or []):
                self.report["strong_agent_verifier_tool_rows"] += 1
            if any(abs(float(comp.get("reflection_policy", 0.0) or 0.0)) >= 0.05 for comp in components.values()):
                self.report["strong_agent_reflection_policy_rows"] += 1
        for rid in evidence.candidate_set.ids:
            rid = int(rid)
            deltas[rid] = self._score_component_delta(
                rid=rid,
                top1=top1,
                comp=dict(components.get(rid, {}) or {}),
                uncertainty=uncertainty,
                scale=scale,
            )
        deltas = self._apply_deterministic_rule_guard(
            evidence=evidence,
            components=components,
            deltas=deltas,
            uncertainty=uncertainty,
            tool_trace=tool_trace,
        )
        ranked = sorted(evidence.candidate_set.ids, key=lambda rid: evidence.candidate_set.logit_of(int(rid)) + deltas.get(int(rid), 0.0), reverse=True)
        target_candidate, target_score, target_features = self._select_target_candidate(
            evidence=evidence,
            components=components,
            deltas=deltas,
            ranked=ranked,
            uncertainty=uncertainty,
            tool_trace=tool_trace,
        )
        matched_rule = None
        if (
            int(target_candidate) >= 0
            and bool(self.config.target_use_rule_calibrator)
            and bool(getattr(self, "target_rules_ready", False))
        ):
            matched_rule = self._match_target_rule(dict(target_features or {}))
            if matched_rule:
                target_features = {
                    **dict(target_features or {}),
                    "rule_support": float(matched_rule.get("support", 0)),
                    "rule_fix": float(matched_rule.get("fix", 0)),
                    "rule_fail": float(matched_rule.get("fail", 0)),
                    "rule_neutral": float(matched_rule.get("neutral", 0)),
                    "rule_precision": float(matched_rule.get("precision", 0.0)),
                    "rule_reward": float(matched_rule.get("reward", 0.0)),
                    "rule_priority": float(matched_rule.get("priority", 0.0)),
                    "rule_examples": " || ".join(str(x) for x in list(matched_rule.get("examples", []) or [])[:4]),
                }
        debug["target_candidate"] = int(target_candidate)
        debug["target_score"] = float(target_score)
        debug["target_features"] = dict(target_features)
        if matched_rule:
            debug["target_rule"] = {
                "support": int(matched_rule.get("support", 0)),
                "fix": int(matched_rule.get("fix", 0)),
                "fail": int(matched_rule.get("fail", 0)),
                "neutral": int(matched_rule.get("neutral", 0)),
                "precision": float(matched_rule.get("precision", 0.0)),
                "reward": float(matched_rule.get("reward", 0.0)),
                "priority": float(matched_rule.get("priority", 0.0)),
            }
        used_llm = False
        llm_choice = -1
        llm_confidence = 0.0
        llm_action = "none"
        llm_ranked: List[int] = []
        llm_explanation = ""
        result_target_candidate = int(target_candidate)
        result_target_score = float(target_score)
        result_target_features: Dict[str, Any] = dict(target_features)
        score_context: Dict[int, float] = {}
        if score_lookup is not None:
            try:
                if hasattr(score_lookup, "detach"):
                    raw_scores = score_lookup.detach().cpu().tolist()
                elif hasattr(score_lookup, "tolist"):
                    raw_scores = score_lookup.tolist()
                else:
                    raw_scores = list(score_lookup)
                upper = self.config.relation_upper_bound
                limit = int(upper) if upper is not None else len(raw_scores)
                for rid in range(max(0, min(len(raw_scores), limit))):
                    score_context[int(rid)] = float(raw_scores[int(rid)] + deltas.get(int(rid), 0.0))
            except Exception:
                score_context = {}
        if not score_context:
            score_context = {
                int(rid): float(evidence.candidate_set.logit_of(int(rid)) + deltas.get(int(rid), 0.0))
                for rid in evidence.candidate_set.ids
            }
        debug["score_context"] = score_context
        debug["score_order"] = [int(x) for x in list(ranked or [])[:16]]
        if bool(count_row):
            self.report["rows"] += 1
            self.report["candidate_count_sum"] += len(evidence.candidate_set.ids)
            if any(abs(v) >= self.config.min_report_delta for v in deltas.values()):
                self.report["nonzero_delta_rows"] += 1
            if ranked and int(ranked[0]) != top1:
                self.report["proposal_rows"] += 1
        return FusionResult(
            query=query,
            top1=top1,
            candidates=evidence.candidate_set.ids,
            deltas=deltas,
            ranked_ids=ranked,
            tool_trace=tool_trace,
            used_llm=used_llm,
            llm_choice=llm_choice,
            llm_confidence=llm_confidence,
            llm_action=llm_action,
            llm_ranked_ids=llm_ranked,
            llm_explanation=llm_explanation,
            target_candidate=int(result_target_candidate),
            target_score=float(result_target_score),
            target_features=dict(result_target_features),
            debug=debug,
        )

    def observe_target_outcome(self, result: FusionResult, current_order: Sequence[int], gt: Sequence[int]) -> None:
        """Collect valid-set labels for the calibrated LLM intervention target.

        The label is defined relative to the deterministic score-fusion ranking,
        because the LLM marginal report also measures gains from that point.
        """

        if result is None:
            return
        target = int(getattr(result, "target_candidate", -1))
        target_score = float(getattr(result, "target_score", 0.0))
        if target < 0 or target_score <= 0.0 or not current_order:
            return
        gt_set = {int(x) for x in gt}
        if not gt_set:
            return
        current_top = int(current_order[0])
        if int(target) == int(current_top):
            return
        current_ok = current_top in gt_set
        target_ok = target in gt_set
        label = 0
        if (not current_ok) and target_ok:
            label = 1
        elif current_ok and (not target_ok):
            label = -1
        features = dict(getattr(result, "target_features", {}) or {})
        self.target_samples.append(
            {
                "score": float(target_score),
                "label": int(label),
                "target": int(target),
                "current_top": int(current_top),
                "features": features,
            }
        )

    def finalize_target_calibrator(self) -> Dict[str, Any]:
        samples = list(getattr(self, "target_samples", []) or [])
        if not samples:
            self.target_ready = False
            self.target_threshold = float("inf")
            self.target_rules_ready = False
            self.target_rules = {}
            self.target_rule_summary = {"rules": 0, "best_precision": 0.0, "best_reward": 0.0, "best_rule": ""}
            self.target_summary = {
                "samples": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "threshold": float("inf"),
                "selected": 0,
                "selected_fix": 0,
                "selected_fail": 0,
                "selected_neutral": 0,
                "precision": 0.0,
                "reward": 0.0,
                "ready": False,
            }
            return dict(self.target_summary)

        positives = sum(1 for row in samples if int(row.get("label", 0)) > 0)
        negatives = sum(1 for row in samples if int(row.get("label", 0)) < 0)
        neutrals = len(samples) - positives - negatives
        best: Optional[Dict[str, Any]] = None
        fallback: Optional[Dict[str, Any]] = None
        minimum_score = float(self.config.target_min_score)
        eligible = []
        for row in samples:
            score = float(row.get("score", 0.0))
            if score > 0.0 and score >= minimum_score:
                eligible.append((score, int(row.get("label", 0))))
        eligible.sort(key=lambda item: item[0], reverse=True)
        if not eligible:
            eligible = [
                (minimum_score, int(row.get("label", 0)))
                for row in samples
                if float(row.get("score", 0.0)) >= minimum_score
            ]
        selected_count = 0
        fix = 0
        fail = 0
        neutral = 0
        offset = 0
        while offset < len(eligible):
            threshold = float(eligible[offset][0])
            end = offset
            while end < len(eligible) and float(eligible[end][0]) == threshold:
                label = int(eligible[end][1])
                selected_count += 1
                if label > 0:
                    fix += 1
                elif label < 0:
                    fail += 1
                else:
                    neutral += 1
                end += 1
            precision = float(fix / max(1, fix + fail))
            reward = float(fix - float(self.config.target_fail_penalty) * fail - float(self.config.target_neutral_penalty) * neutral)
            item = {
                "threshold": float(threshold),
                "selected": int(selected_count),
                "selected_fix": int(fix),
                "selected_fail": int(fail),
                "selected_neutral": int(neutral),
                "precision": float(precision),
                "reward": float(reward),
            }
            if fallback is None or (
                reward,
                fix,
                precision,
                -float(threshold),
            ) > (
                float(fallback.get("reward", -1e9)),
                int(fallback.get("selected_fix", 0)),
                float(fallback.get("precision", 0.0)),
                -float(fallback.get("threshold", 1e9)),
            ):
                fallback = item
            meets_precision = precision >= float(self.config.target_min_precision)
            meets_fix_floor = fix >= int(self.config.target_min_valid_fixes)
            if meets_precision and meets_fix_floor:
                if best is None or (
                    reward,
                    fix,
                    precision,
                    -float(threshold),
                ) > (
                    float(best.get("reward", -1e9)),
                    int(best.get("selected_fix", 0)),
                    float(best.get("precision", 0.0)),
                    -float(best.get("threshold", 1e9)),
                ):
                    best = item
            offset = end

        if best is None:
            best = fallback or {
                "threshold": float(self.config.target_min_score),
                "selected": 0,
                "selected_fix": 0,
                "selected_fail": 0,
                "selected_neutral": 0,
                "precision": 0.0,
                "reward": 0.0,
            }
        ready = bool(int(best.get("selected_fix", 0)) > 0 and float(best.get("reward", 0.0)) > 0.0)
        self.target_ready = bool(ready)
        self.target_threshold = float(best.get("threshold", float(self.config.target_min_score))) if ready else float("inf")
        self.target_summary = {
            "samples": int(len(samples)),
            "positive": int(positives),
            "negative": int(negatives),
            "neutral": int(neutrals),
            "ready": bool(ready),
            **best,
            "threshold": float(self.target_threshold if ready else best.get("threshold", float("inf"))),
        }
        self._finalize_target_rule_calibrator(samples)
        return dict(self.target_summary)

    def _finalize_target_rule_calibrator(self, samples: Sequence[Dict[str, Any]]) -> None:
        aggregates: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for row in samples:
            features = dict(row.get("features", {}) or {})
            label = int(row.get("label", 0))
            example = ""
            if label != 0:
                example = (
                    f"target={int(row.get('target', -1))}, current={int(row.get('current_top', -1))}, "
                    f"label={label}, score={float(row.get('score', 0.0)):.2f}, "
                    f"needed={float(features.get('needed_delta', 0.0)):.2f}, "
                    f"commit_gap={float(features.get('commit_gap', 0.0)):.2f}, "
                    f"tool_gap={float(features.get('tool_gap', 0.0)):.2f}, "
                    f"direct={float(features.get('direct', 0.0)):.2f}, "
                    f"transition={float(features.get('pair_transition', 0.0)):.2f}, "
                    f"prior={float(features.get('prior', 0.0)):.2f}"
                )
            for key in self._target_rule_keys(features):
                key = tuple(key)
                state = aggregates.setdefault(
                    key,
                    {
                        "support": 0,
                        "fix": 0,
                        "fail": 0,
                        "neutral": 0,
                        "positive_examples": [],
                        "negative_examples": [],
                    },
                )
                state["support"] += 1
                if label > 0:
                    state["fix"] += 1
                    if len(state["positive_examples"]) < 2:
                        state["positive_examples"].append(example)
                elif label < 0:
                    state["fail"] += 1
                    if len(state["negative_examples"]) < 2:
                        state["negative_examples"].append(example)
                else:
                    state["neutral"] += 1

        accepted: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        best_rule: Optional[Tuple[Any, ...]] = None
        best_item: Optional[Dict[str, Any]] = None
        for key, state in aggregates.items():
            support = int(state.get("support", 0))
            fix = int(state.get("fix", 0))
            fail = int(state.get("fail", 0))
            neutral = int(state.get("neutral", 0))
            if support < int(self.config.target_rule_min_support):
                continue
            if fix < int(self.config.target_rule_min_fixes):
                continue
            precision = float(fix / max(1, fix + fail))
            neutral_rate = float(neutral / max(1, support))
            reward = float(
                fix
                - float(self.config.target_fail_penalty) * fail
                - float(self.config.target_neutral_penalty) * neutral
            )
            if precision < float(self.config.target_rule_min_precision):
                continue
            if neutral_rate > float(self.config.target_rule_max_neutral_rate):
                continue
            if reward < float(self.config.target_rule_min_reward):
                continue
            priority = float(reward / max(1.0, math.sqrt(float(support))) + 0.55 * precision + 0.015 * fix)
            examples: List[str] = []
            pos_examples = list(state.get("positive_examples", []) or [])[:2]
            neg_examples = list(state.get("negative_examples", []) or [])[:2]
            if pos_examples:
                examples.extend("positive: " + line for line in pos_examples)
            if neg_examples:
                examples.extend("negative: " + line for line in neg_examples)
            item = {
                "key": key,
                "support": support,
                "fix": fix,
                "fail": fail,
                "neutral": neutral,
                "precision": precision,
                "neutral_rate": neutral_rate,
                "reward": reward,
                "priority": priority,
                "examples": examples[:4],
            }
            accepted[key] = item
            if best_item is None or (
                float(item["priority"]),
                float(item["precision"]),
                int(item["fix"]),
            ) > (
                float(best_item.get("priority", 0.0)),
                float(best_item.get("precision", 0.0)),
                int(best_item.get("fix", 0)),
            ):
                best_rule = key
                best_item = item

        self.target_rules = accepted
        self.target_rules_ready = bool(accepted)
        self.target_rule_summary = {
            "rules": int(len(accepted)),
            "best_precision": float(best_item.get("precision", 0.0)) if best_item else 0.0,
            "best_reward": float(best_item.get("reward", 0.0)) if best_item else 0.0,
            "best_rule": str(best_rule) if best_rule is not None else "",
        }

    def format_target_calibration(self) -> str:
        summary = dict(getattr(self, "target_summary", {}) or {})
        threshold = float(summary.get("threshold", float("inf")))
        threshold_text = "inf" if not math.isfinite(threshold) else f"{threshold:.4f}"
        return (
            "[TargetCalibrator] "
            f"ready={bool(summary.get('ready', False))} | samples={int(summary.get('samples', 0))} | "
            f"pos/fail/neutral={int(summary.get('positive', 0))}/{int(summary.get('negative', 0))}/{int(summary.get('neutral', 0))} | "
            f"threshold={threshold_text} | selected={int(summary.get('selected', 0))} | "
            f"selected_fix/fail/neutral={int(summary.get('selected_fix', 0))}/{int(summary.get('selected_fail', 0))}/{int(summary.get('selected_neutral', 0))} | "
            f"precision={float(summary.get('precision', 0.0)):.4f} | reward={float(summary.get('reward', 0.0)):.2f} | "
            f"rules={int((getattr(self, 'target_rule_summary', {}) or {}).get('rules', 0))} | "
            f"best_rule_precision={float((getattr(self, 'target_rule_summary', {}) or {}).get('best_precision', 0.0)):.4f}"
        )

    def update_report_with_outcome(self, before_order: Sequence[int], after_order: Sequence[int], gt: Sequence[int]) -> None:
        gt_set = {int(x) for x in gt}
        if not before_order or not after_order or not gt_set:
            return
        before_top = int(before_order[0])
        after_top = int(after_order[0])
        if after_top != before_top:
            self.report["top1_flip_rows"] += 1
            if before_top not in gt_set and after_top in gt_set:
                self.report["fixes"] += 1
            elif before_top in gt_set and after_top not in gt_set:
                self.report["fails"] += 1
            else:
                self.report["neutral_flips"] += 1
        if list(map(int, before_order[:10])) != list(map(int, after_order[:10])):
            self.report["rank_changed_rows"] += 1
        before_rank = self._rank_of_gt(before_order, gt_set)
        after_rank = self._rank_of_gt(after_order, gt_set)
        if before_rank is None or after_rank is None:
            return
        self.report["rank_gain_sum"] += float(before_rank - after_rank)
        self.report["mrr_delta_sum"] += float((1.0 / after_rank) - (1.0 / before_rank))
        if before_rank > 3 and after_rank <= 3:
            self.report["hits3_up"] += 1
        elif before_rank <= 3 and after_rank > 3:
            self.report["hits3_down"] += 1
        if before_rank > 5 and after_rank <= 5:
            self.report["hits5_up"] += 1
        elif before_rank <= 5 and after_rank > 5:
            self.report["hits5_down"] += 1
        if before_rank > 10 and after_rank <= 10:
            self.report["hits10_up"] += 1
        elif before_rank <= 10 and after_rank > 10:
            self.report["hits10_down"] += 1
        if after_rank < before_rank:
            self.report["rank_improved"] += 1
        elif after_rank > before_rank:
            self.report["rank_worsened"] += 1

    def observe_outcome(self, result: FusionResult, before_order: Sequence[int], after_order: Sequence[int], gt: Sequence[int]) -> None:
        """Write executable reflection back into memory after a row is evaluated.

        The old framework accumulated many natural-language reflections but almost
        never let them change ranking. Here the reflection is operational: tool
        credit and commit utility are updated from whether the fused ranking moved
        toward a valid relation.
        """

        self.update_report_with_outcome(before_order, after_order, gt)
        if result.tool_trace is None or not before_order or not after_order:
            return
        gt_set = {int(x) for x in gt}
        if not gt_set:
            return
        if hasattr(self, "observe_llm_reflection_hypothesis_outcome"):
            try:
                self.observe_llm_reflection_hypothesis_outcome(result, before_order, after_order, gt_set)
            except Exception:
                if os.environ.get("AGENT_LLM_TRACE_STREAM") == "1":
                    raise
                pass
        before_top = int(before_order[0])
        after_top = int(after_order[0])
        before_ok = before_top in gt_set
        after_ok = after_top in gt_set
        before_rank = self._rank_of_gt(before_order, gt_set)
        after_rank = self._rank_of_gt(after_order, gt_set)
        rank_better = before_rank is not None and after_rank is not None and after_rank < before_rank
        rank_worse = before_rank is not None and after_rank is not None and after_rank > before_rank
        policy_outcome = "neutral"
        if (not before_ok and after_ok) or rank_better:
            policy_outcome = "fix"
        elif (before_ok and not after_ok) or rank_worse:
            policy_outcome = "fail"
        result_label = "SUCCESS" if (after_ok or rank_better) else "FAIL"
        if after_ok and before_ok and not rank_worse:
            result_label = "SUCCESS"
        try:
            feedback = result.tool_trace.as_feedback_record(result.query, choice=after_top, top1=before_top)
            feedback["prev_rel"] = int(result.debug.get("prev_rel", -1)) if isinstance(result.debug, dict) else -1
            self.memory.record_tool_feedback_outcome(feedback, result_label, gt_set)
            if after_top != before_top and hasattr(self.memory, "record_relation_policy_outcome"):
                debug = dict(result.debug or {})
                components = dict(debug.get("components", {}) or {})
                choice_features = components.get(after_top, components.get(str(after_top), {})) if isinstance(components, dict) else {}
                base_features = components.get(before_top, components.get(str(before_top), {})) if isinstance(components, dict) else {}
                if isinstance(choice_features, dict):
                    base_hard = max(
                        float((base_features or {}).get("direct", 0.0) or 0.0),
                        float((base_features or {}).get("pair_transition", 0.0) or 0.0),
                        float((base_features or {}).get("prior", 0.0) or 0.0),
                    )
                    enriched = {
                        **dict(choice_features),
                        "baseline_protected": float(
                            1.0
                            if (
                                float((base_features or {}).get("commit", 0.0) or 0.0) >= 0.24
                                or float((base_features or {}).get("total", 0.0) or 0.0) >= 0.34
                                or float((base_features or {}).get("tool", 0.0) or 0.0) >= 0.55
                                or base_hard >= 0.20
                            )
                            else 0.0
                        ),
                    }
                    self.memory.record_relation_policy_outcome(
                        signature=str(result.tool_trace.signature),
                        prev_rel=int(debug.get("prev_rel", -1)),
                        top1=int(before_top),
                        choice=int(after_top),
                        outcome=str(policy_outcome),
                        features=enriched,
                    )
        except Exception:
            pass

    def print_report(self):
        print(f"[Memory] rows={self.report['rows']} changed={self.report['rank_changed_rows']} "
              f"improved={self.report['rank_improved']} worsened={self.report['rank_worsened']}")
