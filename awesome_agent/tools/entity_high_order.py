from __future__ import annotations

from collections import Counter

from awesome_agent.contracts import EvidenceBundle, RelationQuery, ToolOutput
from awesome_agent.tools.base import MemoryTool


class EntityHighOrderTool(MemoryTool):
    """Recall-oriented tool over recent entity clusters and shared high-order neighbors."""

    name = "entity_high_order"

    def run_relation(self, query: RelationQuery, evidence: EvidenceBundle) -> ToolOutput:
        s, o, t = query.as_key()
        cs = self.memory.get_cluster_id(s)
        co = self.memory.get_cluster_id(o)
        rcs = self.memory.get_cluster_id(o)
        rco = self.memory.get_cluster_id(s)
        pair_counter = self.memory.group_relation_counts(s, o)
        reverse_pair_counter = self.memory.group_relation_counts(o, s)

        neigh_s = set(self.memory.get_recent_neighbors(s, t, limit=25))
        neigh_o = set(self.memory.get_recent_neighbors(o, t, limit=25))
        overlap = neigh_s.intersection(neigh_o)
        role_rel = Counter()
        for event in list(self.memory.recent_events)[-800:]:
            if int(event.get("t", -1)) >= int(t):
                continue
            pass
            if int(event.get("s", -1)) in overlap or int(event.get("o", -1)) in overlap:
                role_rel[int(event.get("r", -1))] += 1

        scores = {}
        penalties = {}
        for rid in evidence.candidate_set.ids:
            rid = int(rid)
            base_r, inverse = self.memory.relation_view(rid)
            active_pair_counter = reverse_pair_counter if inverse else pair_counter
            ev = evidence.evidence(rid)
            raw = float(self.norm_counter_score(active_pair_counter, base_r) + 0.12 * self.norm_counter_score(role_rel, base_r))
            scores[rid] = float(raw if raw >= 0.06 else 0.0)
            penalties[rid] = 0.035 if raw > 0.0 and float(ev.commit) < 0.05 else 0.0

        candidate_ids = set(evidence.candidate_set.ids)
        suggestions = []
        for rid, _ in pair_counter.most_common(3):
            cand = int(rid)
            if cand not in candidate_ids and cand not in suggestions:
                suggestions.append(cand)
        for rid, _ in reverse_pair_counter.most_common(3):
            cand = self.memory.candidate_relation_id(int(rid), inverse=True)
            if cand not in candidate_ids and cand not in suggestions:
                suggestions.append(cand)
        summary = [
            "cluster_pair=({},{}) pair_rel_modes={}".format(
                cs,
                co,
                ", ".join(f"{self.mapper.rel_name(int(r))}:{int(c)}" for r, c in pair_counter.most_common(3))
                or "None",
            ),
            "reverse_cluster_pair=({},{}) pair_rel_modes={}".format(
                rcs,
                rco,
                ", ".join(f"{self.mapper.rel_name(self.memory.candidate_relation_id(int(r), inverse=True))}:{int(c)}" for r, c in reverse_pair_counter.most_common(3))
                or "None",
            ),
            f"shared_high_order_neighbors={len(overlap)}",
        ]
        return ToolOutput(name=self.name, scores=scores, penalties=penalties, suggestions=suggestions[:2], summary=summary)
