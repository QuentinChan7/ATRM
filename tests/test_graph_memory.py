from __future__ import annotations

import unittest

from awesome_agent.memory.stores import EpisodicGraphStore, GraphMemory


class GraphMemoryTemporalIndexTest(unittest.TestCase):
    @staticmethod
    def _brute_two_hop(store, s, o, current_t, max_paths):
        paths = []
        seen_mids = set()
        for e1 in reversed(store.adj_out.get(s, [])):
            if int(e1.get("t", -1)) >= current_t:
                continue
            mid = int(e1.get("o", -1))
            if mid in seen_mids:
                continue
            for e2 in reversed(store.adj_out.get(mid, [])):
                if int(e2.get("o", -1)) != o:
                    continue
                if int(e2.get("t", -1)) >= current_t:
                    continue
                if int(e1.get("t", -1)) > int(e2.get("t", -1)):
                    continue
                paths.append({
                    "mid": mid,
                    "r1": int(e1.get("r", -1)),
                    "gap1": current_t - int(e1.get("t", current_t)),
                    "r2": int(e2.get("r", -1)),
                    "gap2": current_t - int(e2.get("t", current_t)),
                })
                seen_mids.add(mid)
                break
            if len(paths) >= max_paths:
                break
        return paths

    def test_recent_pair_lookup_is_not_limited_by_working_event_capacity(self) -> None:
        memory = GraphMemory(working_size=2, recent_size=4)
        memory.update_snapshot([[1, 7, 2, 1]])
        memory.update_snapshot([[10, 3, 11, 2], [12, 4, 13, 2], [14, 5, 15, 2]])

        self.assertEqual(
            memory.get_recent_relation_between(1, 2, current_t=3),
            (7, 1),
        )

    def test_same_timestamp_rows_do_not_create_temporal_transitions(self) -> None:
        memory = GraphMemory()
        memory.update_snapshot([[1, 3, 2, 5], [1, 4, 2, 5]])
        self.assertEqual(sum(sum(values.values()) for values in memory.global_transitions.values()), 0)

        memory.update_snapshot([[1, 5, 2, 6]])
        self.assertEqual(memory.global_transitions[4][5], 1)

    def test_context_and_motif_are_not_limited_by_working_event_capacity(self) -> None:
        class Mapper:
            @staticmethod
            def ent_name(value: int) -> str:
                return f"e{value}"

            @staticmethod
            def rel_name(value: int) -> str:
                return f"r{value}"

            @staticmethod
            def rel_phrase(value: int, subject: str, object_: str) -> str:
                return f"{subject}-r{value}-{object_}"

        memory = GraphMemory(working_size=2, recent_size=4)
        memory.update_snapshot([[1, 8, 2, 1], [3, 8, 4, 1]])
        memory.update_snapshot([[3, 9, 4, 2]])
        memory.update_snapshot([[10, 1, 11, 2], [12, 2, 13, 2], [14, 3, 15, 2]])

        context, motif = memory.get_context_and_injection(1, 2, 3, Mapper())

        self.assertEqual(context["last_wr"], 8)
        self.assertEqual(motif, 9)
        self.assertTrue(any("e1" in row and "e2" in row for row in context["Layer1_Working"]))

    def test_reciprocal_context_maps_transition_to_inverse_relation_space(self) -> None:
        class Mapper:
            @staticmethod
            def ent_name(value: int) -> str:
                return f"e{value}"

            @staticmethod
            def rel_name(value: int) -> str:
                return f"r{value}"

            @staticmethod
            def rel_phrase(value: int, subject: str, object_: str) -> str:
                return f"{subject}-r{value}-{object_}"

        memory = GraphMemory(working_size=2, recent_size=4)
        memory.set_relation_base_count(10)
        memory.update_snapshot([[1, 8, 2, 1], [3, 8, 4, 1]])
        memory.update_snapshot([[3, 9, 4, 2]])

        context, motif = memory.get_context_and_injection(2, 1, 3, Mapper())

        self.assertEqual(context["last_wr"], 18)
        self.assertEqual(motif, 19)

    def test_query_caches_are_exact_and_invalidated_after_snapshot_update(self) -> None:
        memory = GraphMemory()
        memory.set_relation_base_count(10)
        memory.update_snapshot([[1, 2, 4, 1], [4, 3, 7, 2]])

        first = memory.episodic_memory.retrieve_two_hop(1, 7, 3, max_paths=2)
        self.assertEqual(first[0]["mid"], 4)
        first[0]["mid"] = 999
        second = memory.episodic_memory.retrieve_two_hop(1, 7, 3, max_paths=2)
        self.assertEqual(second[0]["mid"], 4)
        self.assertEqual(len(memory.episodic_memory._two_hop_cache), 1)

        score_before = memory.score_rel_candidate(1, 4, 3, 2)
        expected_score = memory.score_rel_candidate(1, 4, 3, 2)
        self.assertEqual(score_before, expected_score)
        score_before["provenance"]["semantic_fit"] = 999.0
        self.assertEqual(memory.score_rel_candidate(1, 4, 3, 2), expected_score)
        self.assertEqual(len(memory._relation_score_cache), 1)
        aggregate = memory.transition_target_counts()
        self.assertIs(aggregate, memory.transition_target_counts())

        memory.update_snapshot([[1, 2, 4, 3]])
        self.assertEqual(len(memory.episodic_memory._two_hop_cache), 0)
        self.assertEqual(len(memory._relation_score_cache), 0)
        self.assertIsNone(memory._transition_target_cache)
        score_after = memory.score_rel_candidate(1, 4, 4, 2)
        self.assertGreater(score_after["direct"], expected_score["direct"])

    def test_two_hop_pair_index_matches_bounded_adjacency_scan(self) -> None:
        store = EpisodicGraphStore(max_capacity=3)
        events = [
            [1, 1, 4, 1], [4, 2, 7, 1], [4, 3, 8, 2],
            [1, 4, 5, 2], [5, 5, 7, 3], [4, 6, 9, 3],
            [4, 7, 7, 4], [1, 8, 4, 5], [4, 9, 10, 5],
            [4, 10, 7, 6],
        ]
        for event in events:
            store.update(*event)

        for source in (1, 4, 5):
            for target in (7, 8, 9, 10):
                for current_t in (4, 6, 8):
                    for max_paths in (1, 2, 4):
                        expected = self._brute_two_hop(
                            store, source, target, current_t, max_paths
                        )
                        actual = store.retrieve_two_hop(
                            source, target, current_t, max_paths=max_paths
                        )
                        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
