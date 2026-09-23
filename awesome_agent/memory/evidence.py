from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

from awesome_agent.contracts import CandidateSet, EvidenceBundle, RelationEvidence, RelationQuery, dedup_ints
from awesome_agent.memory.stores import GraphMemory


class EvidenceBuilder:
    """Builds memory-grounded evidence for relation candidates."""

    def __init__(self, memory: GraphMemory):
        self.memory = memory
        self.audit_observer = None

    def expand_relation_candidates(
        self,
        query: RelationQuery,
        base: CandidateSet,
        max_add: int = 8,
        upper_bound: Optional[int] = None,
    ) -> CandidateSet:
        base_ids = dedup_ints(base.ids, lower=0, upper=upper_bound)
        seen = set(base_ids)
        additions: List[int] = []
        if self.audit_observer is not None:
            self.audit_observer.expansion(base_ids, int(max_add))

        def add(candidate_id: int, source: str) -> None:
            if self.audit_observer is None and len(additions) >= int(max_add):
                return
            rid = int(candidate_id)
            reason = ("invalid" if rid < 0 or (upper_bound is not None and rid >= int(upper_bound))
                      else "duplicate" if rid in seen else "budget" if len(additions) >= int(max_add)
                      else "accepted")
            if self.audit_observer is not None:
                self.audit_observer.proposal(rid, source, reason)
            if reason != "accepted":
                return
            seen.add(rid)
            additions.append(rid)

        s, o, t = query.as_key()

        add(self.memory.episodic_memory.retrieve_most_frequent(s, o), "interaction")
        recip_r = self.memory.episodic_memory.retrieve_most_frequent(o, s)
        if int(recip_r) >= 0:
            add(self.memory.candidate_relation_id(int(recip_r), inverse=True), "interaction")

        prev_base_rel, _, prev_is_reciprocal = self.memory.get_recent_relation_view(s, o, current_t=t)
        if int(prev_base_rel) >= 0:
            for rid, _ in self.memory.global_transitions.get(int(prev_base_rel), Counter()).most_common(4):
                add(self.memory.candidate_relation_id(int(rid), inverse=prev_is_reciprocal), "regularity")
            for rid, _ in self.memory.event_transition_counts.get(int(prev_base_rel), Counter()).most_common(4):
                add(self.memory.candidate_relation_id(int(rid), inverse=prev_is_reciprocal), "regularity")

        for rid, _ in self.memory.pair_frequencies(s, o).most_common(5):
            add(int(rid), "interaction")
        for rid, _ in self.memory.pair_frequencies(o, s).most_common(3):
            add(self.memory.candidate_relation_id(int(rid), inverse=True), "interaction")

        try:
            for rid, _ in self.memory.group_relation_counts(s, o).most_common(4):
                add(int(rid), "regularity")
        except Exception:
            if self.audit_observer is not None:
                raise
            pass

        for event in self.memory.retrieve_similar_events_rel(s, o, t, prev_rel=prev_base_rel, limit=8):
            event_s = int(event.get("s", -1))
            event_o = int(event.get("o", -1))
            event_inverse = bool(event_s == o and event_o == s)
            add(self.memory.candidate_relation_id(int(event.get("r", -1)), inverse=event_inverse), "configuration")

        if not additions:
            return base

        extended = CandidateSet.from_topk(base_ids + additions, base.logits + [0.0] * len(additions), source="gnn+memory")
        return extended

    def build_relation_evidence(self, query: RelationQuery, candidate_set: CandidateSet) -> EvidenceBundle:
        prev_base_rel, prev_t, prev_is_reciprocal = self.memory.get_recent_relation_view(
            query.s,
            query.o,
            current_t=query.t,
        )
        prev_rel = self.memory.candidate_relation_id(prev_base_rel, inverse=prev_is_reciprocal)
        by_id: Dict[int, RelationEvidence] = {}
        for rid in candidate_set.ids:
            try:
                raw = self.memory.score_rel_candidate(query.s, query.o, query.t, int(rid))
            except Exception:
                if self.audit_observer is not None:
                    raise
                raw = {}
            by_id[int(rid)] = RelationEvidence.from_mapping(int(rid), raw)
        bundle = EvidenceBundle(
            query=query,
            candidate_set=candidate_set,
            by_id=by_id,
            prev_rel=int(prev_rel),
            prev_t=int(prev_t),
        )
        if self.audit_observer is not None:
            self.audit_observer.evidence(bundle)
        return bundle

    @staticmethod
    def commit_support(evidence: RelationEvidence) -> float:
        return float(evidence.direct + 0.85 * evidence.pair_transition + 0.55 * evidence.prior)

    @staticmethod
    def recall_support(evidence: RelationEvidence) -> float:
        return float(
            0.45 * evidence.reciprocal
            + 0.45 * evidence.event
            + 0.35 * evidence.cluster
            + 0.25 * evidence.work
            + 0.20 * evidence.twohop
        )

    @staticmethod
    def rank_by_total(candidate_ids: Iterable[int], bundle: EvidenceBundle) -> List[int]:
        ids = [int(x) for x in candidate_ids]
        return sorted(ids, key=lambda rid: bundle.evidence(rid).total, reverse=True)
