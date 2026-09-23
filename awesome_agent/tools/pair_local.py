from __future__ import annotations

from typing import List

from awesome_agent.contracts import EvidenceBundle, RelationQuery, ToolOutput
from awesome_agent.tools.base import MemoryTool


class PairLocalTool(MemoryTool):
    """Commit-oriented tool for exact pair history and local pair transitions."""

    name = "pair_local"

    def run_relation(self, query: RelationQuery, evidence: EvidenceBundle) -> ToolOutput:
        scores = {}
        for rid in evidence.candidate_set.ids:
            ev = evidence.evidence(int(rid))
            scores[int(rid)] = float(max(0.0, ev.commit + 0.05 * ev.recall - ev.neg_penalty - 0.5 * ev.pop_penalty))

        ctx, injected = self.memory.get_context_and_injection(query.s, query.o, query.t, self.mapper)
        summary: List[str] = []
        for key in ("Layer2_Episodic_Direct_and_Recip", "Layer2_Episodic_TwoHop"):
            for line in ctx.get(key, [])[:2]:
                summary.append(str(line))
        prior = ctx.get("Layer2_Historical_Prior", {})
        if isinstance(prior, dict) and prior:
            summary.append("pair_prior=" + ", ".join(f"{name}:{cnt}" for name, cnt in list(prior.items())[:3]))

        top_commit = sorted(
            [(float(evidence.evidence(rid).commit), int(rid)) for rid in evidence.candidate_set.ids],
            key=lambda item: item[0],
            reverse=True,
        )[:3]
        if top_commit:
            summary.append(
                "commit_evidence_top="
                + ", ".join(f"{self.mapper.rel_name(rid)}:{score:.2f}" for score, rid in top_commit)
            )
        suggestions = [int(injected)] if int(injected) >= 0 else []
        return ToolOutput(name=self.name, scores=scores, suggestions=suggestions, summary=summary[:4])
