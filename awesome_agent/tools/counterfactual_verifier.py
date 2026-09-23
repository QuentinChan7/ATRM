from __future__ import annotations

from typing import List

from awesome_agent.contracts import EvidenceBundle, RelationQuery, ToolOutput
from awesome_agent.tools.base import MemoryTool


class CounterfactualVerifierTool(MemoryTool):
    """Verifier that asks whether replacing the baseline contradicts history."""

    name = "counterfactual_verifier"

    def run_relation(self, query: RelationQuery, evidence: EvidenceBundle) -> ToolOutput:
        scores = {}
        penalties = {}
        summary: List[str] = []
        baseline = int(evidence.candidate_set.top1)
        baseline_ev = evidence.evidence(baseline)
        baseline_hard = max(float(baseline_ev.direct), float(baseline_ev.pair_transition), float(baseline_ev.prior))
        baseline_guard = bool(
            float(baseline_ev.commit) >= 0.24
            or float(baseline_ev.total) >= 0.34
            or baseline_hard >= 0.20
        )

        for rid in evidence.candidate_set.ids:
            rid = int(rid)
            ev = evidence.evidence(rid)
            hard = max(float(ev.direct), float(ev.pair_transition), float(ev.prior))
            support = (
                0.55 * max(0.0, float(ev.semantic_fit))
                + 0.30 * max(0.0, hard)
                + 0.20 * max(0.0, float(ev.commit))
                + 0.12 * max(0.0, float(ev.reflection_policy))
            )
            risk = float(ev.counterfactual_risk)
            if rid != baseline and baseline_guard and hard < baseline_hard + 0.08:
                risk += 0.12
            if rid != baseline and float(ev.commit) < 0.10 and float(ev.semantic_fit) > 0.08:
                risk += 0.08
            scores[rid] = float(max(0.0, min(0.75, support)))
            penalties[rid] = float(max(0.0, min(0.90, risk)))

        ranked_risk = sorted(
            ((float(penalties.get(int(rid), 0.0)), int(rid)) for rid in evidence.candidate_set.ids),
            key=lambda item: item[0],
            reverse=True,
        )
        for risk, rid in ranked_risk[:3]:
            if risk <= 0.0:
                continue
            prov = dict(evidence.evidence(rid).provenance or {})
            exact_dom = int(float(prov.get("cf_exact_dom_rel", -1.0)))
            pair_dom = int(float(prov.get("cf_pair_dom_rel", -1.0)))
            trans_dom = int(float(prov.get("cf_trans_dom_rel", -1.0)))
            parts = [f"risk_id={rid} ({self.mapper.rel_name(rid)}) risk={risk:.2f}"]
            if exact_dom >= 0:
                parts.append(f"exact_pair_prefers={self.mapper.rel_name(exact_dom)}")
            if pair_dom >= 0:
                parts.append(f"cluster_prefers={self.mapper.rel_name(pair_dom)}")
            if trans_dom >= 0:
                parts.append(f"transition_prefers={self.mapper.rel_name(trans_dom)}")
            summary.append("; ".join(parts))

        ranked_support = sorted(
            ((float(scores.get(int(rid), 0.0)) - float(penalties.get(int(rid), 0.0)), int(rid)) for rid in evidence.candidate_set.ids),
            key=lambda item: item[0],
            reverse=True,
        )
        suggestions = [rid for net, rid in ranked_support if net > 0.18 and rid != baseline][:2]
        if baseline_guard:
            summary.append(
                "baseline_guard=1 hard={:.2f} commit={:.2f} semantic={:.2f}".format(
                    baseline_hard,
                    float(baseline_ev.commit),
                    float(baseline_ev.semantic_fit),
                )
            )
        return ToolOutput(name=self.name, scores=scores, penalties=penalties, suggestions=suggestions, summary=summary[:4])
