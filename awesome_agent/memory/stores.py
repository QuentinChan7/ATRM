from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Set, Tuple



class EpisodicGraphStore:
    """Indexed temporal graph history for local TKG retrieval."""

    def __init__(self, max_capacity: int = 6000):
        self.max_capacity = int(max_capacity)
        self.adj_out: Dict[int, Deque[Dict[str, int]]] = defaultdict(lambda: deque(maxlen=self.max_capacity))
        self.adj_in: Dict[int, Deque[Dict[str, int]]] = defaultdict(lambda: deque(maxlen=self.max_capacity))
        self._adj_out_pair: Dict[Tuple[int, int], Deque[Dict[str, int]]] = defaultdict(deque)
        self.pair_hist: Dict[Tuple[int, int], Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=self.max_capacity))
        self.pair_counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self._recent_entity_cache: Dict[Tuple[Tuple[int, ...], int, int], Tuple[Dict[str, int], ...]] = {}
        self._direct_recip_cache: Dict[Tuple[int, int, int, int], Tuple[Dict[str, Any], ...]] = {}
        self._two_hop_cache: Dict[Tuple[int, int, int, int], Tuple[Dict[str, Any], ...]] = {}

    def clear(self) -> None:
        self.adj_out.clear()
        self.adj_in.clear()
        self._adj_out_pair.clear()
        self.pair_hist.clear()
        self.pair_counts.clear()
        self._recent_entity_cache.clear()
        self._direct_recip_cache.clear()
        self._two_hop_cache.clear()

    def update(self, s: int, r: int, o: int, t: int) -> None:
        s = int(s)
        r = int(r)
        o = int(o)
        t = int(t)
        if self._recent_entity_cache:
            self._recent_entity_cache.clear()
        if self._direct_recip_cache:
            self._direct_recip_cache.clear()
        if self._two_hop_cache:
            self._two_hop_cache.clear()
        record = {"s": s, "r": r, "o": o, "t": t}
        outgoing = self.adj_out[s]
        if outgoing.maxlen and len(outgoing) == outgoing.maxlen:
            evicted = outgoing[0]
            evicted_key = (s, int(evicted["o"]))
            indexed = self._adj_out_pair.get(evicted_key)
            if indexed:
                if indexed[0] is evicted:
                    indexed.popleft()
                else:
                    self._adj_out_pair[evicted_key] = deque(
                        row for row in indexed if row is not evicted
                    )
                if not self._adj_out_pair[evicted_key]:
                    del self._adj_out_pair[evicted_key]
        outgoing.append(record)
        if outgoing.maxlen != 0:
            self._adj_out_pair[(s, o)].append(record)
        self.adj_in[o].append(record)
        self.pair_hist[(s, o)].append((r, t))
        self.pair_counts[(s, o)][r] += 1

    def retrieve_recent_entity_events(
        self,
        entity_ids: Iterable[int],
        current_t: int,
        k: int = 4,
    ) -> List[Dict[str, int]]:
        """Return recent incident events without depending on snapshot density."""

        pass

        current_t = int(current_t)
        entity_key = tuple(sorted({int(value) for value in entity_ids}))
        cache_key = (entity_key, current_t, max(0, int(k)))
        cached = self._recent_entity_cache.get(cache_key)
        if cached is not None:
            return [dict(row) for row in cached]
        events: Dict[Tuple[int, int, int, int], Dict[str, int]] = {}
        for entity_id in entity_key:
            for records in (self.adj_out.get(entity_id, []), self.adj_in.get(entity_id, [])):
                for record in reversed(records):
                    timestamp = int(record.get("t", -1))
                    if timestamp >= current_t:
                        continue
                    key = (
                        int(record.get("s", -1)),
                        int(record.get("r", -1)),
                        int(record.get("o", -1)),
                        timestamp,
                    )
                    events.setdefault(key, record)
        ordered = sorted(
            events.values(),
            key=lambda row: (
                int(row.get("t", -1)),
                int(row.get("s", -1)),
                int(row.get("r", -1)),
                int(row.get("o", -1)),
            ),
            reverse=True,
        )
        result = tuple(dict(row) for row in ordered[: max(0, int(k))])
        self._recent_entity_cache[cache_key] = result
        return [dict(row) for row in result]

    def retrieve_direct_and_recip(self, s: int, o: int, current_t: int, k: int = 5) -> List[Dict[str, Any]]:
        pass
        s = int(s)
        o = int(o)
        current_t = int(current_t)
        cache_key = (s, o, current_t, max(0, int(k)))
        cached = self._direct_recip_cache.get(cache_key)
        if cached is not None:
            return [dict(row) for row in cached]
        direct = [
            {"r": int(r), "gap": current_t - int(t), "is_recip": False}
            for r, t in self.pair_hist.get((s, o), [])
            if int(t) < current_t
        ]
        recip = [
            {"r": int(r), "gap": current_t - int(t), "is_recip": True}
            for r, t in self.pair_hist.get((o, s), [])
            if int(t) < current_t
        ]
        combined = direct + recip
        combined.sort(key=lambda row: int(row.get("gap", 10**9)))
        result = tuple(dict(row) for row in combined[: max(0, int(k))])
        self._direct_recip_cache[cache_key] = result
        return [dict(row) for row in result]

    def retrieve_most_frequent(self, s: int, o: int) -> int:
        pass
        counts = self.pair_counts.get((int(s), int(o)), None)
        if not counts:
            return -1
        return int(counts.most_common(1)[0][0])

    def retrieve_two_hop(self, s: int, o: int, current_t: int, max_paths: int = 2) -> List[Dict[str, Any]]:
        pass
        s = int(s)
        o = int(o)
        current_t = int(current_t)
        cache_key = (s, o, current_t, max(0, int(max_paths)))
        cached = self._two_hop_cache.get(cache_key)
        if cached is not None:
            return [dict(row) for row in cached]
        paths: List[Dict[str, Any]] = []
        seen_mids: Set[int] = set()
        for e1 in reversed(self.adj_out.get(s, [])):
            if int(e1.get("t", -1)) >= current_t:
                continue
            mid = int(e1.get("o", -1))
            if mid in seen_mids:
                continue
            # The pair index mirrors the records retained by the bounded
            # adjacency deque, so this is equivalent to filtering adj_out.
            for e2 in reversed(self._adj_out_pair.get((mid, o), [])):
                if int(e2.get("t", -1)) >= current_t:
                    continue
                if int(e1.get("t", -1)) > int(e2.get("t", -1)):
                    continue
                paths.append(
                    {
                        "mid": mid,
                        "r1": int(e1.get("r", -1)),
                        "gap1": current_t - int(e1.get("t", current_t)),
                        "r2": int(e2.get("r", -1)),
                        "gap2": current_t - int(e2.get("t", current_t)),
                    }
                )
                seen_mids.add(mid)
                break
            if len(paths) >= int(max_paths):
                break
        result = tuple(dict(row) for row in paths)
        self._two_hop_cache[cache_key] = result
        return [dict(row) for row in result]


class GraphMemory:
    """Three-layer TKG memory plus procedural traces for tool/reflection learning.

    Conceptual layers:
      1. Working memory: recent snapshot events.
      2. Episodic memory: indexed historical events and motifs.
      3. Semantic/procedural memory: distilled rules, constraints, tool and commit experience.
    """

    def __init__(self, working_size: int = 500, recent_size: int = 8000):
        self.working_memory: Deque[List[int]] = deque(maxlen=int(working_size))
        self.episodic_memory = EpisodicGraphStore(max_capacity=6000, )
        self.semantic_memory: List[str] = []
        self.negative_constraints: List[str] = []
        self.cognitive_feedbacks: Deque[str] = deque(maxlen=200)
        self.decision_traces: Deque[Dict[str, Any]] = deque(maxlen=5000)
        self.recent_events: Deque[Dict[str, int]] = deque(maxlen=int(recent_size))

        self.global_transitions: Dict[int, Counter] = defaultdict(Counter)
        self.recent_snapshots: Deque[List[Tuple[int, int, int, int]]] = deque(maxlen=10)
        self.event_cases: Deque[Dict[str, int]] = deque(maxlen=6000)
        self.event_transition_counts: Dict[int, Counter] = defaultdict(Counter)
        self.pair_transition_counts: Dict[Tuple[int, int, int], Counter] = defaultdict(Counter)
        self.rel_global_counts: Counter = Counter()
        self.relation_base_count = 0

        self.memory_version = 0
        self._cluster_cache_version = -1
        self._cluster_assign: Dict[int, int] = {}
        self._cluster_pair_rel_counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self._cluster_rel_tail_counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self._cluster_pair_sizes: Dict[Tuple[int, int], int] = defaultdict(int)
        self._cluster_rel_sizes: Dict[Tuple[int, int], int] = defaultdict(int)
        self._similar_event_cache: Dict[Tuple[int, int, int, int, int], Tuple[Dict[str, Any], ...]] = {}
        self._recent_neighbor_cache: Dict[Tuple[int, int, int], Tuple[int, ...]] = {}
        self._relation_score_cache: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
        self._transition_target_cache: Optional[Counter] = None

        self.tool_experience: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "fix": 0.0, "fail": 0.0, "score": 0.0}
        )
        self.tool_experience_sig: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "fix": 0.0, "fail": 0.0, "score": 0.0}
        )
        self.tool_pair_experience: Dict[Tuple[str, str, str, str], Dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "fix": 0.0, "fail": 0.0, "score": 0.0}
        )
        self.commit_experience_rel: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "fix": 0.0, "fail": 0.0, "score": 0.0}
        )
        self.relation_policy_experience: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "fix": 0.0, "fail": 0.0, "neutral": 0.0, "score": 0.0}
        )
        self.tool_reflections: Deque[Dict[str, Any]] = deque(maxlen=2000)
        self.commit_reflections_rel: Deque[Dict[str, Any]] = deque(maxlen=2000)

    def reset_dynamic_state(self, clear_reflection: bool = False) -> None:
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.recent_events.clear()
        self.global_transitions.clear()
        self.recent_snapshots.clear()
        self.event_cases.clear()
        self.event_transition_counts.clear()
        self.pair_transition_counts.clear()
        self.rel_global_counts.clear()
        self.memory_version = 0
        self._cluster_cache_version = -1
        self._cluster_assign = {}
        self._cluster_pair_rel_counts = defaultdict(Counter)
        self._cluster_rel_tail_counts = defaultdict(Counter)
        self._cluster_pair_sizes = defaultdict(int)
        self._cluster_rel_sizes = defaultdict(int)
        self._similar_event_cache.clear()
        self._recent_neighbor_cache.clear()
        self._relation_score_cache.clear()
        self._transition_target_cache = None
        if clear_reflection:
            self.semantic_memory.clear()
            self.negative_constraints.clear()
            self.cognitive_feedbacks.clear()
            self.decision_traces.clear()
            self.tool_experience.clear()
            self.tool_experience_sig.clear()
            self.tool_pair_experience.clear()
            self.commit_experience_rel.clear()
            self.relation_policy_experience.clear()
            self.tool_reflections.clear()
            self.commit_reflections_rel.clear()

    @staticmethod
    def _iter_snapshot(snapshot: Any) -> Iterable[Tuple[int, int, int, int]]:
        if snapshot is None:
            return []
        out: List[Tuple[int, int, int, int]] = []
        for row in snapshot:
            try:
                s, r, o, t = row[:4]
                out.append((int(s), int(r), int(o), int(t)))
            except Exception:
                continue
        return out

    def update_snapshot(self, snapshot: Any) -> None:
        rows = list(self._iter_snapshot(snapshot))
        if not rows:
            return

        self._similar_event_cache.clear()
        self._recent_neighbor_cache.clear()
        self._relation_score_cache.clear()
        self._transition_target_cache = None

        prevs: List[Tuple[int, int]] = []
        for s, _, o, t in rows:
            prevs.append(self._history_relation_between(s, o, current_t=t - 1, include_reciprocal=False))

        for (s, r, o, t), (prev_rel, _) in zip(rows, prevs):
            # `prevs` is captured before this snapshot is inserted, so only a
            # genuine earlier-time relation can create a transition. The old
            # working-deque scan both lost dense-dataset history and counted
            # same-timestamp rows as temporal transitions.
            if int(prev_rel) >= 0:
                self.global_transitions[int(prev_rel)][int(r)] += 1
            self.working_memory.append([s, r, o, t])
            self.episodic_memory.update(s, r, o, t)
            self.recent_events.append({"s": s, "r": r, "o": o, "t": t})

        self.recent_snapshots.append(rows)
        self.memory_version += 1

        for (s, r, o, t), (prev_rel, prev_t) in zip(rows, prevs):
            self.event_cases.append({"s": s, "r": r, "o": o, "t": t, "prev_rel": int(prev_rel), "prev_t": int(prev_t)})
            self.rel_global_counts[int(r)] += 1
            if int(prev_rel) >= 0:
                self.event_transition_counts[int(prev_rel)][int(r)] += 1
            exact_prev = None
            for rr_prev, tt_prev in reversed(self.episodic_memory.pair_hist.get((s, o), [])):
                if int(tt_prev) < int(t):
                    exact_prev = int(rr_prev)
                    break
            if exact_prev is not None:
                self.pair_transition_counts[(int(s), int(o), int(exact_prev))][int(r)] += 1

    def add_distilled_rule(self, rule: str) -> None:
        self.semantic_memory.append(str(rule))

    def add_negative_constraint(self, constraint: str) -> None:
        self.negative_constraints.append(str(constraint))

    def add_decision_trace(self, trace: Mapping[str, Any]) -> None:
        if isinstance(trace, Mapping):
            self.decision_traces.append(dict(trace))

    def add_cognitive_feedback(self, context: str, action: str, result: str, correct_ans: Any) -> None:
        if isinstance(correct_ans, (list, tuple, set)):
            preview = ", ".join(str(x) for x in list(correct_ans)[:10])
            correct = f"[{preview}]"
        else:
            correct = str(correct_ans)
        self.cognitive_feedbacks.append(
            f"Temporal Context: {context} | Action: {action} | Result: {result} | CorrectSet: {correct}"
        )

    def get_recent_relation_between(
        self, s: int, o: int, current_t: Optional[int] = None, include_reciprocal: bool = True,
    ) -> Tuple[int, int]:
        pass
        return self._history_relation_between(s, o, current_t, include_reciprocal)

    def pair_history(self, s: int, o: int):
        return self.episodic_memory.pair_hist.get((int(s), int(o)), ()) if True else ()

    def pair_frequencies(self, s: int, o: int) -> Counter:
        return self.episodic_memory.pair_counts.get((int(s), int(o)), Counter()) if True else Counter()

    def transition_counts(self, previous: int) -> Counter:
        return self.event_transition_counts.get(int(previous), Counter()) if True else Counter()

    def group_relation_counts(self, s: int, o: int) -> Counter:
        pass
        cs, co = self.get_cluster_id(s), self.get_cluster_id(o)
        return self._cluster_pair_rel_counts.get((cs, co), Counter())

    def _history_relation_between(
        self,
        s: int,
        o: int,
        current_t: Optional[int] = None,
        include_reciprocal: bool = True,
    ) -> Tuple[int, int]:
        s = int(s)
        o = int(o)

        def _latest(pair: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            for relation, timestamp in reversed(self.episodic_memory.pair_hist.get(pair, [])):
                if current_t is not None and int(timestamp) > int(current_t):
                    continue
                return int(relation), int(timestamp)
            return None

        direct = _latest((s, o))
        if direct is not None:
            return direct
        if bool(include_reciprocal):
            reciprocal = _latest((o, s))
            if reciprocal is not None:
                return reciprocal
        return -1, -1

    def get_recent_direct_relation_between(self, s: int, o: int, current_t: Optional[int] = None) -> Tuple[int, int]:
        return self.get_recent_relation_between(s, o, current_t=current_t, include_reciprocal=False)

    def get_recent_relation_view(
        self,
        s: int,
        o: int,
        current_t: Optional[int] = None,
    ) -> Tuple[int, int, bool]:
        """Return a recent base relation and its orientation for this query."""

        direct_rel, direct_t = self.get_recent_direct_relation_between(s, o, current_t=current_t)
        if int(direct_rel) >= 0:
            return int(direct_rel), int(direct_t), False
        reciprocal_rel, reciprocal_t = self.get_recent_direct_relation_between(o, s, current_t=current_t)
        if int(reciprocal_rel) >= 0:
            return int(reciprocal_rel), int(reciprocal_t), True
        return -1, -1, False

    def get_context_and_injection(self, s: int, o: int, current_t: int, mapper: Any) -> Tuple[Dict[str, Any], int]:
        s = int(s)
        o = int(o)
        current_t = int(current_t)
        working_context: List[str] = []
        last_base_rel, _, last_is_reciprocal = self.get_recent_relation_view(
            s,
            o,
            current_t=current_t - 1,
        )
        last_wr = self.candidate_relation_id(last_base_rel, inverse=last_is_reciprocal)
        for event in self.episodic_memory.retrieve_recent_entity_events((s, o), current_t=current_t, k=4):
            ws = int(event.get("s", -1))
            wr = int(event.get("r", -1))
            wo = int(event.get("o", -1))
            wt = int(event.get("t", -1))
            pass
            age = current_t - wt
            working_context.append(
                f"[t-{age}] {mapper.ent_name(ws)} -[{mapper.rel_name(wr)}]-> {mapper.ent_name(wo)}"
            )

        direct_context: List[str] = []
        for edge in self.episodic_memory.retrieve_direct_and_recip(s, o, current_t, k=5):
            direction = "Reciprocal (o->s)" if edge.get("is_recip", False) else "Direct (s->o)"
            rel_id = self.candidate_relation_id(int(edge.get("r", -1)), inverse=bool(edge.get("is_recip", False)))
            try:
                rel_text = mapper.rel_phrase(rel_id, subject="S", object_="O")
            except Exception:
                rel_text = mapper.rel_name(rel_id)
            direct_context.append(f"[t-{int(edge.get('gap', 0))}] {direction}: {rel_text}")

        twohop_context: List[str] = []
        for path in self.episodic_memory.retrieve_two_hop(s, o, current_t, max_paths=2):
            twohop_context.append(
                f"via {mapper.ent_name(int(path['mid']))}: "
                f"[t-{int(path['gap1'])}] {mapper.rel_name(int(path['r1']))} -> "
                f"[t-{int(path['gap2'])}] {mapper.rel_name(int(path['r2']))}"
            )

        pair_counts = self.pair_frequencies(s, o)
        reverse_pair_counts = self.pair_frequencies(o, s)
        hist_prior = {mapper.rel_name(int(r)): int(cnt) for r, cnt in pair_counts.items()}
        for r, cnt in reverse_pair_counts.items():
            hist_prior[mapper.rel_name(self.candidate_relation_id(int(r), inverse=True))] = int(cnt)
        motif_injected_r = -1
        if last_base_rel != -1 and self.global_transitions.get(last_base_rel):
            next_base_rel = int(self.global_transitions[last_base_rel].most_common(1)[0][0])
            motif_injected_r = self.candidate_relation_id(next_base_rel, inverse=last_is_reciprocal)

        return (
            {
                "last_wr": int(last_wr),
                "Layer1_Working": working_context,
                "Layer2_Episodic_Direct_and_Recip": direct_context,
                "Layer2_Episodic_TwoHop": twohop_context,
                "Layer2_Historical_Prior": hist_prior,
                "Layer3_Semantic_Motifs": self.semantic_memory[-3:],
                "Layer3_Negative_Constraints": self.negative_constraints[-3:],
                "Agent_Cognitive_Feedbacks": list(self.cognitive_feedbacks)[-3:],
            },
            motif_injected_r,
        )

    @staticmethod
    def _constraint_mentions_id(text: str, value: int) -> bool:
        try:
            return re.search(rf"(?<!\d){int(value)}(?!\d)", str(text)) is not None
        except Exception:
            return False

    def check_negative_constraint_violation(self, s: int, o: int, t: int, r: int) -> bool:
        rid = int(r)
        return any(self._constraint_mentions_id(text, rid) for text in list(self.negative_constraints)[-200:])

    def _rebuild_cluster_cache(self) -> None:
        if self._cluster_cache_version == self.memory_version:
            return
        parent: Dict[int, int] = {}
        rank: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(int(x), int(x))
            rank.setdefault(int(x), 0)
            while parent[int(x)] != int(x):
                parent[int(x)] = parent[parent[int(x)]]
                x = parent[int(x)]
            return int(x)

        def union(a: int, b: int) -> None:
            ra, rb = find(int(a)), find(int(b))
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        for snap in list(self.recent_snapshots)[-6:]:
            for s, _, o, _ in snap:
                union(int(s), int(o))

        root2cid: Dict[int, int] = {}
        assign: Dict[int, int] = {}
        for ent in list(parent.keys()):
            root = find(ent)
            if root not in root2cid:
                root2cid[root] = len(root2cid)
            assign[int(ent)] = int(root2cid[root])

        pair_rel: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        rel_tail: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        pair_sizes: Dict[Tuple[int, int], int] = defaultdict(int)
        rel_sizes: Dict[Tuple[int, int], int] = defaultdict(int)
        for event in list(self.recent_events)[-3000:]:
            s = int(event["s"])
            r = int(event["r"])
            o = int(event["o"])
            cs = assign.get(s, 1000000 + s)
            co = assign.get(o, 1000000 + o)
            pair_rel[(cs, co)][r] += 1
            rel_tail[(cs, r)][o] += 1
            pair_sizes[(cs, co)] += 1
            rel_sizes[(cs, r)] += 1

        self._cluster_assign = assign
        self._cluster_pair_rel_counts = pair_rel
        self._cluster_rel_tail_counts = rel_tail
        self._cluster_pair_sizes = pair_sizes
        self._cluster_rel_sizes = rel_sizes
        self._cluster_cache_version = self.memory_version

    def get_cluster_id(self, entity_id: int) -> int:
        self._rebuild_cluster_cache()
        eid = int(entity_id)
        return int(self._cluster_assign.get(eid, 1000000 + eid))

    def get_cluster_pair_rel_score(self, s: int, o: int, cand_r: int) -> float:
        pass
        self._rebuild_cluster_cache()
        cs = self.get_cluster_id(int(s))
        co = self.get_cluster_id(int(o))
        cnt = float(self._cluster_pair_rel_counts.get((cs, co), {}).get(int(cand_r), 0.0))
        total = float(self._cluster_pair_sizes.get((cs, co), 0))
        return float(cnt / (total + 1.0))

    def set_relation_base_count(self, count: int) -> None:
        try:
            value = int(count)
        except Exception:
            value = 0
        if value > 0:
            self.relation_base_count = value

    def relation_view(self, candidate_r: int) -> Tuple[int, bool]:
        """Return the base relation id and whether candidate_r is inverse.

        Relation prediction in the legacy RGCN uses ids [0, R) for S->O and
        [R, 2R) for O->S. The memory store itself only observes base triples, so
        inverse candidates must be scored against the reversed entity pair.
        """

        rid = int(candidate_r)
        base_count = int(getattr(self, "relation_base_count", 0) or 0)
        if base_count > 0 and base_count <= rid < 2 * base_count:
            return int(rid - base_count), True
        return int(rid), False

    def candidate_relation_id(self, base_r: int, inverse: bool = False) -> int:
        base_r = int(base_r)
        base_count = int(getattr(self, "relation_base_count", 0) or 0)
        if bool(inverse) and base_count > 0 and 0 <= base_r < base_count:
            return int(base_count + base_r)
        return base_r

    def get_recent_neighbors(self, ent_id: int, current_t: int, limit: int = 50) -> List[int]:
        pass
        ent_id = int(ent_id)
        current_t = int(current_t)
        cache_key = (ent_id, current_t, max(0, int(limit)))
        cached = self._recent_neighbor_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        seen: Set[int] = set()
        out: List[int] = []
        for event in reversed(self.recent_events):
            if int(event.get("t", -1)) >= current_t:
                continue
            nbr = None
            if int(event.get("s", -1)) == ent_id:
                nbr = int(event.get("o", -1))
            elif int(event.get("o", -1)) == ent_id:
                nbr = int(event.get("s", -1))
            if nbr is None or nbr in seen:
                continue
            seen.add(nbr)
            out.append(nbr)
            if len(out) >= int(limit):
                break
        result = tuple(out)
        self._recent_neighbor_cache[cache_key] = result
        return list(result)

    def retrieve_similar_events_rel(self, s: int, o: int, t: int, prev_rel: int = -1, limit: int = 5) -> List[Dict[str, Any]]:
        pass
        pass
        s = int(s)
        o = int(o)
        t = int(t)
        prev_rel = int(prev_rel)
        cache_key = (s, o, t, prev_rel, max(1, int(limit)))
        cached = self._similar_event_cache.get(cache_key)
        if cached is not None:
            return [dict(event) for event in cached]
        cs = self.get_cluster_id(s)
        co = self.get_cluster_id(o)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for event in reversed(self.event_cases):
            try:
                if int(event.get("t", -1)) >= t:
                    continue
                score = 0.0
                es = int(event.get("s", -1))
                eo = int(event.get("o", -1))
                pass
                if es == s and eo == o:
                    score += 3.0
                elif es == s or eo == o:
                    score += 1.4
                if self.get_cluster_id(es) == cs and self.get_cluster_id(eo) == co:
                    score += 1.2
                if prev_rel >= 0 and int(event.get("prev_rel", -2)) == prev_rel:
                    score += 1.0
                gap = max(1, t - int(event.get("t", t)))
                score = score / (1.0 + gap / 8.0)
                if score > 0.15:
                    scored.append((float(score), dict(event)))
            except Exception:
                continue
            if len(scored) >= 80:
                break
        scored.sort(key=lambda item: item[0], reverse=True)
        out = []
        for score, event in scored[: max(1, int(limit))]:
            event["sim_score"] = float(score)
            pass
            out.append(event)
        result = tuple(dict(event) for event in out)
        self._similar_event_cache[cache_key] = result
        return [dict(event) for event in result]

    def transition_target_counts(self) -> Counter:
        """Return the snapshot-stable aggregate used by the causal tool."""

        pass
        if self._transition_target_cache is None:
            aggregate = Counter()
            for counter in self.event_transition_counts.values():
                aggregate.update(counter)
            self._transition_target_cache = aggregate
        return self._transition_target_cache

    @staticmethod
    def _bounded_ratio(num: float, den: float, prior: float = 1.0) -> float:
        try:
            return float(max(0.0, num) / (max(0.0, den) + float(prior)))
        except Exception:
            return 0.0

    def relation_semantic_profile(self, s: int, o: int, t: int, cand_r: int) -> Dict[str, float]:
        """Map relation semantics to structural role compatibility.

        The LLM sees natural-language relation names, but the model needs an
        executable signal. This profile estimates whether the candidate relation
        historically fits the subject/object cluster roles of the current pair.
        """

        s = int(s)
        o = int(o)
        cand_r = int(cand_r)
        base_r, inverse_candidate = self.relation_view(cand_r)
        active_s, active_o = (o, s) if inverse_candidate else (s, o)
        cs = self.get_cluster_id(active_s)
        co = self.get_cluster_id(active_o)
        pair_counter = self.group_relation_counts(active_s, active_o)
        pair_total = float(sum(pair_counter.values()))
        pair_count = float(pair_counter.get(base_r, 0.0))
        pair_fit = self._bounded_ratio(pair_count, pair_total, 1.0)

        rel_tail_counter = self._cluster_rel_tail_counts.get((cs, base_r), Counter()) if True else Counter()
        rel_tail_total = float(self._cluster_rel_sizes.get((cs, base_r), 0.0)) if True else 0.0
        tail_fit = self._bounded_ratio(float(rel_tail_counter.get(active_o, 0.0)), rel_tail_total, 1.0)

        exact_counter = self.pair_frequencies(active_s, active_o)
        exact_total = float(sum(exact_counter.values()))
        exact_fit = self._bounded_ratio(float(exact_counter.get(base_r, 0.0)), exact_total, 1.0)

        global_total = float(sum(self.rel_global_counts.values())) if True else 0.0
        global_count = float(self.rel_global_counts.get(base_r, 0.0)) if True else 0.0
        global_fit = self._bounded_ratio(global_count, global_total, 1.0)
        rel_role_fit = max(0.0, pair_fit - 0.35 * global_fit) + 0.35 * tail_fit + 0.25 * exact_fit
        subj_fit = pair_fit
        obj_fit = tail_fit
        endpoint_fit = exact_fit
        reliability = min(1.0, (pair_total + rel_tail_total + exact_total) / 18.0)
        semantic_fit = reliability * (
            0.10 * subj_fit
            + 0.10 * obj_fit
            + 0.28 * pair_fit
            + 0.18 * rel_role_fit
            + 0.14 * endpoint_fit
        )
        return {
            "semantic_fit": float(max(0.0, min(0.45, semantic_fit))),
            "semantic_subj_fit": float(subj_fit),
            "semantic_obj_fit": float(obj_fit),
            "semantic_pair_fit": float(pair_fit),
            "semantic_role_fit": float(rel_role_fit),
            "semantic_reliability": float(reliability),
            "semantic_rel_total": global_count,
        }

    def counterfactual_profile(self, s: int, o: int, t: int, cand_r: int) -> Dict[str, float]:
        """Estimate how strongly local history argues against a candidate."""

        s = int(s)
        o = int(o)
        t = int(t)
        cand_r = int(cand_r)
        base_r, inverse_candidate = self.relation_view(cand_r)
        active_s, active_o = (o, s) if inverse_candidate else (s, o)
        exact_counter = self.pair_frequencies(active_s, active_o)
        exact_total = float(sum(exact_counter.values()))
        exact_candidate = float(exact_counter.get(base_r, 0.0))
        exact_dom_rel = -1
        exact_dom_p = 0.0
        if exact_total > 0.0:
            exact_dom_rel, exact_dom_cnt = exact_counter.most_common(1)[0]
            exact_dom_p = float(exact_dom_cnt / exact_total)

        last_rel = -1
        last_t = -1
        for rr, tt in reversed(self.pair_history(active_s, active_o)):
            if int(tt) < t:
                last_rel = int(rr)
                last_t = int(tt)
                break

        active_prev, _ = self.get_recent_relation_between(active_s, active_o, current_t=t, include_reciprocal=False)
        trans = self.transition_counts(active_prev) if int(active_prev) >= 0 else Counter()
        trans_total = float(sum(trans.values()))
        trans_dom_rel = -1
        trans_dom_p = 0.0
        trans_candidate_p = 0.0
        if trans_total > 0.0:
            trans_dom_rel, trans_dom_cnt = trans.most_common(1)[0]
            trans_dom_p = float(trans_dom_cnt / trans_total)
            trans_candidate_p = float(trans.get(base_r, 0.0) / (trans_total + 1.0))

        cs = self.get_cluster_id(active_s)
        co = self.get_cluster_id(active_o)
        pair_counter = self.group_relation_counts(active_s, active_o)
        pair_total = float(sum(pair_counter.values()))
        pair_dom_rel = -1
        pair_dom_p = 0.0
        pair_candidate_p = 0.0
        if pair_total > 0.0:
            pair_dom_rel, pair_dom_cnt = pair_counter.most_common(1)[0]
            pair_dom_p = float(pair_dom_cnt / pair_total)
            pair_candidate_p = float(pair_counter.get(base_r, 0.0) / (pair_total + 1.0))

        risk = 0.0
        if exact_dom_rel >= 0 and int(exact_dom_rel) != base_r and exact_dom_p >= 0.45 and exact_candidate <= 1.0:
            risk += 0.34 * exact_dom_p
        if last_rel >= 0 and int(last_rel) != base_r and last_t >= 0 and (t - last_t) <= 4 and exact_candidate <= 1.0:
            risk += 0.18 / (1.0 + max(1, t - last_t) / 3.0)
        if pair_dom_rel >= 0 and int(pair_dom_rel) != base_r and pair_dom_p >= 0.50 and pair_candidate_p < 0.18:
            risk += 0.24 * pair_dom_p
        if trans_dom_rel >= 0 and int(trans_dom_rel) != base_r and trans_dom_p >= 0.45 and trans_candidate_p < 0.12:
            risk += 0.20 * trans_dom_p
        return {
            "counterfactual_risk": float(max(0.0, min(0.75, risk))),
            "cf_exact_dom_rel": float(exact_dom_rel),
            "cf_exact_dom_p": float(exact_dom_p),
            "cf_pair_dom_rel": float(pair_dom_rel),
            "cf_pair_dom_p": float(pair_dom_p),
            "cf_trans_dom_rel": float(trans_dom_rel),
            "cf_trans_dom_p": float(trans_dom_p),
            "cf_candidate_exact_count": float(exact_candidate),
        }

    @staticmethod
    def _policy_bin(value: float, lo: float, hi: float) -> int:
        value = float(value or 0.0)
        return int(value >= lo) + int(value >= hi)

    def _relation_policy_keys(
        self,
        signature: str,
        prev_rel: int,
        top1: int,
        choice: int,
        features: Mapping[str, Any],
    ) -> List[Tuple[Any, ...]]:
        f = dict(features or {})
        cf_bin = self._policy_bin(float(f.get("counterfactual_risk", 0.0) or 0.0), 0.12, 0.28)
        sem_bin = self._policy_bin(float(f.get("semantic_fit", 0.0) or 0.0), 0.08, 0.18)
        hard_bin = self._policy_bin(
            max(
                float(f.get("direct", 0.0) or 0.0),
                float(f.get("pair_transition", 0.0) or 0.0),
                float(f.get("prior", 0.0) or 0.0),
            ),
            0.08,
            0.24,
        )
        weak = int(float(f.get("weak_source", 0.0) or 0.0) > 0.5)
        baseline_guard = int(float(f.get("baseline_protected", 0.0) or 0.0) > 0.5)
        sig = str(signature or "rel|global")
        return [
            ("policy_choice", int(choice)),
            ("policy_pair", int(top1), int(choice)),
            ("policy_prev_pair", int(prev_rel), int(top1), int(choice)),
            ("policy_sig_mode", sig, cf_bin, sem_bin, hard_bin, weak, baseline_guard),
            ("policy_global_mode", cf_bin, sem_bin, hard_bin, weak, baseline_guard),
        ]

    def relation_policy_utility(
        self,
        signature: str,
        prev_rel: int,
        choice: int,
        top1: int,
        features: Mapping[str, Any],
    ) -> float:
        keys = self._relation_policy_keys(signature, prev_rel, top1, choice, features)
        weights = [0.18, 0.34, 0.58, 0.82, 0.48]
        num = 0.0
        den = 0.0
        for weight, key in zip(weights, keys):
            state = self.relation_policy_experience.get(key)
            if not state:
                continue
            calls = float(state.get("calls", 0.0) or 0.0)
            if calls <= 0.0:
                continue
            util = float((state.get("fix", 0.0) - 1.75 * state.get("fail", 0.0) - 0.04 * state.get("neutral", 0.0)) / (calls + 2.0))
            conf = min(1.0, calls / 5.0)
            num += float(weight) * conf * util
            den += float(weight) * conf
        return float(max(-0.55, min(0.35, num / den))) if den > 0.0 else 0.0

    def record_relation_policy_outcome(
        self,
        signature: str,
        prev_rel: int,
        top1: int,
        choice: int,
        outcome: str,
        features: Mapping[str, Any],
    ) -> None:
        if int(choice) < 0 or int(choice) == int(top1):
            return
        label = str(outcome or "neutral").lower()
        if label not in {"fix", "fail", "neutral", "success"}:
            label = "neutral"
        if label == "success":
            label = "fix"
        for key in self._relation_policy_keys(str(signature), int(prev_rel), int(top1), int(choice), dict(features or {})):
            state = self.relation_policy_experience[key]
            state["calls"] += 1.0
            if label == "fix":
                state["fix"] += 1.0
                state["score"] += 1.0
            elif label == "fail":
                state["fail"] += 1.0
                state["score"] -= 1.75
            else:
                state["neutral"] += 1.0
                state["score"] -= 0.04

    @staticmethod
    def _copy_relation_score(result: Mapping[str, Any]) -> Dict[str, Any]:
        copied = dict(result)
        provenance = copied.get("provenance")
        if isinstance(provenance, Mapping):
            copied["provenance"] = dict(provenance)
        return copied

    def score_rel_candidate(self, s: int, o: int, t: int, cand_r: int) -> Dict[str, Any]:
        s = int(s)
        o = int(o)
        t = int(t)
        cand_r = int(cand_r)
        cache_key = (s, o, t, cand_r)
        cached = self._relation_score_cache.get(cache_key)
        if cached is not None:
            return self._copy_relation_score(cached)
        base_r, inverse_candidate = self.relation_view(cand_r)
        active_s, active_o = (o, s) if inverse_candidate else (s, o)
        reciprocal_s, reciprocal_o = (s, o) if inverse_candidate else (o, s)

        def recency_w(gap: int) -> float:
            return float(1.0 / (1.0 + max(1, int(gap)) / 6.0))

        exact_recent = 0.0
        reciprocal = 0.0
        exact_count = 0.0
        pair_total = 0.0
        last_exact_rel = -1
        last_exact_t = -1
        cand_last_t = -1
        for rr, tt in self.pair_history(active_s, active_o):
            rr = int(rr)
            tt = int(tt)
            if tt >= t:
                continue
            pair_total += 1.0
            if tt > last_exact_t:
                last_exact_t = tt
                last_exact_rel = rr
            if rr == base_r:
                exact_count += 1.0
                cand_last_t = max(cand_last_t, tt)
                exact_recent += 1.85 * recency_w(t - tt)
        for rr, tt in self.pair_history(reciprocal_s, reciprocal_o):
            rr = int(rr)
            tt = int(tt)
            if tt < t and rr == base_r:
                reciprocal += 0.42 * recency_w(t - tt)

        pair_prior = 0.0
        if pair_total > 0:
            last_gap = max(1, t - int(cand_last_t)) if cand_last_t >= 0 else 10**6
            recency_factor = recency_w(last_gap) if cand_last_t >= 0 else 0.0
            pair_prior = 0.82 * (exact_count / (pair_total + 2.0)) + 0.06 * math.log1p(min(exact_count, 6.0))
            pair_prior *= float(0.55 + 0.45 * recency_factor)

        pair_transition = 0.0
        if last_exact_rel >= 0:
            trans = self.pair_transition_counts.get((active_s, active_o, last_exact_rel), Counter())
            trans_total = float(sum(trans.values()))
            if trans_total > 0:
                trans_count = float(trans.get(base_r, 0.0))
                reliability = min(1.0, trans_total / 4.0) * min(1.0, trans_count / 2.0)
                pair_transition = 1.05 * reliability * float(trans_count / (trans_total + 1.0))

        global_transition = 0.0
        prev_rel, _ = self.get_recent_relation_between(active_s, active_o, current_t=t, include_reciprocal=False)
        if int(prev_rel) >= 0:
            trans = self.transition_counts(prev_rel)
            trans_total = float(sum(trans.values()))
            global_total = float(sum(self.rel_global_counts.values()))
            trans_count = float(trans.get(base_r, 0.0))
            if trans_total >= 5.0 and trans_count >= 2.0:
                p_local = float(trans_count / (trans_total + 1.0))
                p_bg = float(self.rel_global_counts.get(base_r, 0.0) / (global_total + 1.0))
                reliability = min(1.0, trans_count / 5.0)
                global_transition = 0.32 * reliability * max(0.0, p_local - p_bg)

        cluster = 0.22 * float(self.get_cluster_pair_rel_score(active_s, active_o, base_r))

        event_analog = 0.0
        sims = self.retrieve_similar_events_rel(active_s, active_o, t, prev_rel=last_exact_rel, limit=8)
        denom = 1.0 + sum(float(ev.get("sim_score", 0.0)) for ev in sims)
        num = sum(float(ev.get("sim_score", 0.0)) for ev in sims if int(ev.get("r", -1)) == base_r)
        if denom > 0:
            event_analog = 0.30 * float(num / denom)

        endpoint_weak = 0.0
        legacy_work = 0.0
        for event in (list(self.recent_events)[-420:] if True else ()):
            if int(event.get("t", -1)) >= t or int(event.get("r", -1)) != base_r:
                continue
            es = int(event.get("s", -1))
            eo = int(event.get("o", -1))
            if (es == active_s or eo == active_o) and not (es == active_s and eo == active_o):
                endpoint_weak += 0.018 * recency_w(t - int(event.get("t", t)))
                legacy_work += 0.08 * recency_w(t - int(event.get("t", t)))
        endpoint_weak = min(0.08, endpoint_weak)
        legacy_work = min(0.36, legacy_work)

        legacy_twohop = 0.0
        for path in self.episodic_memory.retrieve_two_hop(active_s, active_o, t, max_paths=4):
            if base_r in (int(path.get("r1", -2)), int(path.get("r2", -3))):
                legacy_twohop += 0.12 * recency_w(int(path.get("gap1", 0)) + int(path.get("gap2", 0)))
        legacy_twohop = min(0.32, legacy_twohop)

        pop_penalty = 0.0
        global_total = float(sum(self.rel_global_counts.values()))
        if global_total > 0:
            popularity = float(self.rel_global_counts.get(base_r, 0.0) / (global_total + 1.0))
            if exact_recent + pair_prior + pair_transition < 0.18:
                pop_penalty = 0.30 * popularity

        neg_penalty = 0.35 if self.check_negative_constraint_violation(s, o, t, cand_r) else 0.0
        semantic_profile = self.relation_semantic_profile(s, o, t, cand_r)
        counterfactual_profile = self.counterfactual_profile(s, o, t, cand_r)
        semantic_fit = float(semantic_profile.get("semantic_fit", 0.0))
        counterfactual_risk = float(counterfactual_profile.get("counterfactual_risk", 0.0))
        direct = float(exact_recent)
        prior = float(pair_prior)
        twohop = float(0.35 * event_analog + 0.25 * cluster + legacy_twohop)
        work = float(endpoint_weak + global_transition)
        stale_penalty = 0.0
        if last_exact_rel >= 0 and base_r != int(last_exact_rel) and last_exact_t >= 0:
            recent_last_gap = max(1, t - int(last_exact_t))
            if recent_last_gap <= 3 and exact_count <= 1.0 and pair_transition < 0.08:
                stale_penalty = 0.10 * recency_w(recent_last_gap)
        commit_total = max(0.0, direct + prior + pair_transition + 0.16 * semantic_fit - pop_penalty - neg_penalty - stale_penalty - 0.22 * counterfactual_risk)
        recall_total = max(0.0, 0.55 * reciprocal + twohop + work + legacy_work)
        total = max(0.0, commit_total + 0.28 * recall_total + 0.10 * semantic_fit - 0.18 * counterfactual_risk)
        result = {
            "work": float(work),
            "direct": float(direct),
            "reciprocal": float(reciprocal),
            "twohop": float(twohop),
            "prior": float(prior),
            "transition": float(pair_transition + global_transition),
            "pair_transition": float(pair_transition),
            "global_transition": float(global_transition),
            "cluster": float(cluster),
            "event": float(event_analog),
            "commit": float(commit_total),
            "recall": float(recall_total),
            "semantic_fit": float(semantic_fit),
            "counterfactual_risk": float(counterfactual_risk),
            "verifier_support": float(max(0.0, semantic_fit + 0.20 * direct + 0.15 * pair_transition)),
            "verifier_penalty": float(counterfactual_risk),
            "pop_penalty": float(pop_penalty),
            "neg_penalty": float(neg_penalty),
            "stale_penalty": float(stale_penalty),
            "total": float(max(0.0, total)),
            "provenance": {**semantic_profile, **counterfactual_profile},
        }
        self._relation_score_cache[cache_key] = self._copy_relation_score(result)
        return self._copy_relation_score(result)

    @staticmethod
    def _normalize_gt_set(correct_ans: Any) -> Set[int]:
        if isinstance(correct_ans, (list, tuple, set)):
            out = set()
            for item in correct_ans:
                try:
                    out.add(int(item))
                except Exception:
                    continue
            return out
        try:
            return {int(correct_ans)}
        except Exception:
            return set()

    def record_tool_feedback(
        self,
        tool_name: str,
        task: str,
        helpful: bool = False,
        harmful: bool = False,
        sig: Optional[str] = None,
        selected_tools: Optional[List[str]] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        task = str(task)
        tool_name = str(tool_name)
        sig = str(sig) if sig is not None else f"{task}|global"
        delta_fix = 1.0 if helpful else 0.0
        delta_fail = 1.0 if harmful else 0.0
        for key in ((task, tool_name),):
            state = self.tool_experience[key]
            state["calls"] += 1.0
            state["fix"] += delta_fix
            state["fail"] += delta_fail
            state["score"] += delta_fix - 1.25 * delta_fail
        state_sig = self.tool_experience_sig[(task, sig, tool_name)]
        state_sig["calls"] += 1.0
        state_sig["fix"] += delta_fix
        state_sig["fail"] += delta_fail
        state_sig["score"] += delta_fix - 1.25 * delta_fail

        selected = [str(x) for x in (selected_tools or [])]
        if tool_name in selected and len(selected) >= 2:
            for other in selected:
                if other == tool_name:
                    continue
                a, b = sorted([tool_name, other])
                state_pair = self.tool_pair_experience[(task, sig, a, b)]
                state_pair["calls"] += 1.0
                state_pair["fix"] += delta_fix
                state_pair["fail"] += delta_fail
                state_pair["score"] += delta_fix - 1.15 * delta_fail
        if meta:
            self.tool_reflections.append(
                {"task": task, "tool": tool_name, "meta": dict(meta), "helpful": bool(helpful), "harmful": bool(harmful)}
            )

    def tool_utility(self, tool_name: str, task: str) -> float:
        state = self.tool_experience.get((str(task), str(tool_name)), None)
        if not state:
            return 0.0
        calls = float(state.get("calls", 0.0))
        return float((float(state.get("fix", 0.0)) - 1.2 * float(state.get("fail", 0.0))) / (calls + 2.0))

    def tool_utility_sig(self, tool_name: str, task: str, sig: Optional[str]) -> float:
        sig = str(sig) if sig is not None else f"{task}|global"
        state = self.tool_experience_sig.get((str(task), sig, str(tool_name)), None)
        if not state:
            return self.tool_utility(tool_name, task)
        calls = float(state.get("calls", 0.0))
        return float((float(state.get("fix", 0.0)) - 1.25 * float(state.get("fail", 0.0))) / (calls + 2.0))

    def tool_pair_utility(self, task: str, sig: Optional[str], a: str, b: str) -> float:
        sig = str(sig) if sig is not None else f"{task}|global"
        a, b = sorted([str(a), str(b)])
        state = self.tool_pair_experience.get((str(task), sig, a, b), None)
        if not state:
            return 0.0
        calls = float(state.get("calls", 0.0))
        return float((float(state.get("fix", 0.0)) - 1.10 * float(state.get("fail", 0.0))) / (calls + 2.0))

    def commit_utility_rel(self, sig: Optional[str], prev_rel: int, choice: int, top1: Optional[int] = None) -> float:
        sig = str(sig) if sig is not None else "rel|global"
        keys: List[Tuple[float, Tuple[Any, ...]]] = [
            (0.20, ("global", int(choice))),
            (0.34, ("sig", sig, int(choice))),
            (0.56, ("sig_prev", sig, int(prev_rel), int(choice))),
        ]
        if top1 is not None:
            keys.append((0.74, ("sig_prev_pair", sig, int(prev_rel), int(top1), int(choice))))
        num = 0.0
        den = 0.0
        for weight, key in keys:
            state = self.commit_experience_rel.get(key)
            if not state:
                continue
            calls = float(state.get("calls", 0.0))
            if calls <= 0.0:
                continue
            util = float((state.get("fix", 0.0) - 1.45 * state.get("fail", 0.0)) / (calls + 2.0))
            conf = min(1.0, calls / 4.0)
            num += float(weight) * conf * util
            den += float(weight) * conf
        return float(num / den) if den > 0 else 0.0

    def record_commit_outcome_rel(
        self,
        signature: str,
        prev_rel: int,
        choice: int,
        top1: int,
        success: bool,
    ) -> None:
        if int(choice) < 0 or int(choice) == int(top1):
            return
        result = "SUCCESS" if success else "FAIL"
        keys = [
            ("global", int(choice)),
            ("sig", str(signature), int(choice)),
            ("sig_prev", str(signature), int(prev_rel), int(choice)),
            ("sig_prev_pair", str(signature), int(prev_rel), int(top1), int(choice)),
        ]
        for key in keys:
            state = self.commit_experience_rel[key]
            state["calls"] += 1.0
            if success:
                state["fix"] += 1.0
                state["score"] += 1.0
            else:
                state["fail"] += 1.0
                state["score"] -= 1.45
        self.commit_reflections_rel.append(
            {"signature": str(signature), "prev_rel": int(prev_rel), "top1": int(top1), "choice": int(choice), "result": result}
        )

    def record_tool_feedback_outcome(self, fb: Optional[Mapping[str, Any]], result: str, correct_ans: Any) -> None:
        if not isinstance(fb, Mapping):
            return
        result_key = str(result).upper().strip()
        if result_key not in {"SUCCESS", "FAIL"}:
            return
        gt = self._normalize_gt_set(correct_ans)
        task = str(fb.get("task", "rel"))
        choice = int(fb.get("choice", -1))
        top1 = int(fb.get("top1", -1))
        signature = str(fb.get("signature", f"{task}|global"))
        final_action = str(fb.get("final_action", "override" if choice != top1 else "keep"))
        selected_tools = [str(x) for x in (fb.get("selected_tools", []) or [])]
        per_tool_scores = fb.get("per_tool_scores", {}) or {}
        per_tool_penalties = fb.get("per_tool_penalties", {}) or {}

        def _net(score_map: Mapping[Any, Any], penalty_map: Mapping[Any, Any], cid: int) -> float:
            return float(score_map.get(int(cid), score_map.get(str(int(cid)), 0.0))) - float(
                penalty_map.get(int(cid), penalty_map.get(str(int(cid)), 0.0))
            )

        for tool_name in selected_tools:
            score_map = per_tool_scores.get(tool_name, {}) or {}
            penalty_map = per_tool_penalties.get(tool_name, {}) or {}
            score_choice = _net(score_map, penalty_map, choice)
            score_top1 = _net(score_map, penalty_map, top1)
            score_gt = max([_net(score_map, penalty_map, gid) for gid in gt], default=max(score_choice, score_top1))
            helpful = False
            harmful = False
            if result_key == "SUCCESS":
                if final_action == "keep":
                    helpful = score_top1 >= score_choice - 1e-6 and score_top1 >= score_gt - 1e-6
                    harmful = score_choice > max(score_top1, score_gt) + 1e-6
                else:
                    helpful = score_choice >= score_top1 - 1e-6 and score_choice >= score_gt - 1e-6
                    harmful = score_top1 > score_choice + 1e-6
            else:
                helpful = score_gt > max(score_choice, score_top1) + 1e-6
                harmful = score_choice > max(score_gt, score_top1) + 1e-6
            self.record_tool_feedback(
                tool_name=tool_name,
                task=task,
                helpful=helpful,
                harmful=harmful,
                sig=signature,
                selected_tools=selected_tools,
                meta={
                    "choice": choice,
                    "top1": top1,
                    "score_choice": float(score_choice),
                    "score_top1": float(score_top1),
                    "score_gt": float(score_gt),
                    "signature": signature,
                    "final_action": final_action,
                    "result": result_key,
                },
            )
        if task == "rel" and choice != top1:
            self.record_commit_outcome_rel(
                signature=signature,
                prev_rel=int(fb.get("prev_rel", -1)),
                choice=choice,
                top1=top1,
                success=(result_key == "SUCCESS"),
            )

    def summarize_tool_experience(self, task: str, max_items: int = 4) -> List[str]:
        rows = []
        for (tk, tool), state in list(self.tool_experience.items()):
            if tk != str(task):
                continue
            calls = int(state.get("calls", 0.0))
            if calls <= 0:
                continue
            util = self.tool_utility(tool, task)
            rows.append(
                (
                    float(util),
                    calls,
                    f"{tool}: calls={calls}, helpful={int(state.get('fix', 0.0))}, harmful={int(state.get('fail', 0.0))}, utility={util:.2f}",
                )
            )
        rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [line for _, _, line in rows[: max(0, int(max_items))]]

    def summarize_tool_experience_for_signature(self, task: str, sig: str, max_items: int = 3) -> List[str]:
        rows = []
        for (tk, sg, tool), state in list(self.tool_experience_sig.items()):
            if tk != str(task) or sg != str(sig):
                continue
            calls = int(state.get("calls", 0.0))
            if calls <= 0:
                continue
            util = self.tool_utility_sig(tool, task, sig)
            rows.append((float(util), calls, f"{tool}: utility={util:.2f}, calls={calls}"))
        rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [line for _, _, line in rows[: max(0, int(max_items))]]

    def summarize_commit_experience_rel(self, sig: Optional[str], prev_rel: int, max_items: int = 3) -> List[str]:
        sig = str(sig) if sig is not None else "rel|global"
        rows = []
        seen = set()
        for key, state in list(self.commit_experience_rel.items()):
            choice = None
            if key[0] == "sig_prev" and len(key) == 4 and key[1] == sig and int(key[2]) == int(prev_rel):
                choice = int(key[3])
            elif key[0] == "sig" and len(key) == 3 and key[1] == sig:
                choice = int(key[2])
            if choice is None or choice in seen:
                continue
            seen.add(choice)
            calls = int(state.get("calls", 0.0))
            if calls <= 0:
                continue
            util = self.commit_utility_rel(sig, int(prev_rel), choice)
            rows.append(
                (
                    float(util),
                    calls,
                    f"relation_id={choice}: commits={calls}, success={int(state.get('fix', 0.0))}, fail={int(state.get('fail', 0.0))}, utility={util:.2f}",
                )
            )
        rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [line for _, _, line in rows[: max(0, int(max_items))]]
