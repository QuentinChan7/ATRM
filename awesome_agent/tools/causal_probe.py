from __future__ import annotations

from collections import Counter

from awesome_agent.contracts import EvidenceBundle, RelationQuery, ToolOutput
from awesome_agent.tools.base import MemoryTool


class CausalProbeTool(MemoryTool):
    """Temporal transition tool that separates motif support from confound risk."""

    name = "causal_probe"

    def run_relation(self, query: RelationQuery, evidence: EvidenceBundle) -> ToolOutput:
        prev_rel = int(evidence.prev_rel)
        trans = self.memory.transition_counts(prev_rel) if prev_rel >= 0 else Counter()
        trans_total = float(sum(trans.values()))
        global_counter = self.memory.transition_target_counts()
        global_total = float(sum(global_counter.values()))

        cs = self.memory.get_cluster_id(query.s)
        co = self.memory.get_cluster_id(query.o)
        pair_counter = self.memory.group_relation_counts(query.s, query.o)
        pair_total = float(sum(pair_counter.values()))

        scores = {}
        penalties = {}
        for rid in evidence.candidate_set.ids:
            rid = int(rid)
            base_r, inverse = self.memory.relation_view(rid)
            active_s, active_o = (query.o, query.s) if inverse else (query.s, query.o)
            active_prev, _ = self.memory.get_recent_relation_between(active_s, active_o, current_t=query.t, include_reciprocal=False)
            active_trans = self.memory.transition_counts(active_prev) if active_prev >= 0 else Counter()
            active_trans_total = float(sum(active_trans.values()))
            active_cs = self.memory.get_cluster_id(active_s)
            active_co = self.memory.get_cluster_id(active_o)
            active_pair_counter = self.memory.group_relation_counts(active_s, active_o)
            active_pair_total = float(sum(active_pair_counter.values()))
            p_local = float(active_trans.get(base_r, 0.0) / (active_trans_total + 1.0))
            p_bg = float(global_counter.get(base_r, 0.0) / (global_total + 1.0))
            trans_count = float(active_trans.get(base_r, 0.0))
            support = 0.0
            if active_trans_total >= 5.0 and trans_count >= 2.0:
                support = min(1.0, trans_count / 5.0) * max(0.0, p_local - p_bg)
            confound = 0.0
            if active_pair_total > 0:
                dom_rel, dom_cnt = active_pair_counter.most_common(1)[0]
                dom_p = float(dom_cnt / active_pair_total)
                if int(dom_rel) != base_r and dom_p > 0.45 and support < 0.20:
                    confound = 0.25 * dom_p
            scores[rid] = float(support)
            penalties[rid] = float(confound)

        summary = []
        if prev_rel >= 0 and trans_total > 0:
            summary.append(
                "causal_motif="
                + ", ".join(f"{self.mapper.rel_name(int(r))}:{int(c)}" for r, c in trans.most_common(3))
            )
        if pair_total > 0:
            summary.append(
                "cluster_background="
                + ", ".join(f"{self.mapper.rel_name(int(r))}:{int(c)}" for r, c in pair_counter.most_common(2))
            )
        candidate_ids = set(evidence.candidate_set.ids)
        suggestions = [
            int(rid)
            for rid, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if float(score) > 0.0 and int(rid) not in candidate_ids
        ]
        return ToolOutput(name=self.name, scores=scores, penalties=penalties, suggestions=suggestions[:2], summary=summary[:3])
