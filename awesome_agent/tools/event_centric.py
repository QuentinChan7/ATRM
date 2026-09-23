from __future__ import annotations

from collections import Counter

from awesome_agent.contracts import EvidenceBundle, RelationQuery, ToolOutput
from awesome_agent.tools.base import MemoryTool


class EventCentricTool(MemoryTool):
    """Analogical tool over similar historical events."""

    name = "event_centric"

    def run_relation(self, query: RelationQuery, evidence: EvidenceBundle) -> ToolOutput:
        sims_forward = self.memory.retrieve_similar_events_rel(
            query.s,
            query.o,
            query.t,
            prev_rel=evidence.prev_rel,
            limit=6,
        )
        reverse_prev, _ = self.memory.get_recent_relation_between(query.o, query.s, current_t=query.t, include_reciprocal=False)
        sims_reverse = self.memory.retrieve_similar_events_rel(
            query.o,
            query.s,
            query.t,
            prev_rel=reverse_prev,
            limit=6,
        )
        rel_counter = Counter()
        reverse_rel_counter = Counter()
        summary = []
        for event in sims_forward:
            rid = int(event.get("r", -1))
            rel_counter[rid] += float(event.get("sim_score", 0.0))
            summary.append(
                "sim_event score={:.2f}: {} -[{}]-> {}".format(
                    float(event.get("sim_score", 0.0)),
                    self.mapper.ent_name(int(event.get("s", -1))),
                    self.mapper.rel_name(rid),
                    self.mapper.ent_name(int(event.get("o", -1))),
                )
            )
        for event in sims_reverse:
            rid = int(event.get("r", -1))
            reverse_rel_counter[rid] += float(event.get("sim_score", 0.0))
            summary.append(
                "reverse_sim_event score={:.2f}: {} -[{}]-> {}".format(
                    float(event.get("sim_score", 0.0)),
                    self.mapper.ent_name(int(event.get("s", -1))),
                    self.mapper.rel_name(self.memory.candidate_relation_id(rid, inverse=True)),
                    self.mapper.ent_name(int(event.get("o", -1))),
                )
            )
        denom = float(sum(rel_counter.values()) + sum(reverse_rel_counter.values()) + 1.0)
        scores = {}
        penalties = {}
        for rid in evidence.candidate_set.ids:
            rid = int(rid)
            base_r, inverse = self.memory.relation_view(rid)
            active_counter = reverse_rel_counter if inverse else rel_counter
            ev = evidence.evidence(rid)
            raw = float(active_counter.get(base_r, 0.0) / denom)
            scores[rid] = float(raw if raw >= 0.08 else 0.0)
            penalties[rid] = 0.04 if raw > 0.0 and float(ev.commit) < 0.06 else 0.0
        candidate_ids = set(evidence.candidate_set.ids)
        suggestions = []
        for rid, _ in rel_counter.most_common(3):
            cand = int(rid)
            if cand not in candidate_ids and cand not in suggestions:
                suggestions.append(cand)
        for rid, _ in reverse_rel_counter.most_common(3):
            cand = self.memory.candidate_relation_id(int(rid), inverse=True)
            if cand not in candidate_ids and cand not in suggestions:
                suggestions.append(cand)
        return ToolOutput(name=self.name, scores=scores, penalties=penalties, suggestions=suggestions[:2], summary=summary[:4])
