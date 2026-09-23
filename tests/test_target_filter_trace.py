import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from awesome_agent.target_trace import TARGET_SCHEMA, label_record, step_reward


class TargetRewardTest(unittest.TestCase):
    def test_exhaustive_adjacent_swaps_match_filtered_target_ranking(self):
        for order in itertools.permutations(range(4)):
            positions = dict(enumerate(order, 1))
            for mask in range(1, 16):
                answers = {r for r in range(4) if mask & (1 << r)}
                for target in answers:
                    before_order = [r for r in order if r not in answers or r == target]
                    before = before_order.index(target) + 1
                    for rank in range(1, 5):
                        after_order = list(order)
                        if rank > 1:
                            after_order[rank-1], after_order[rank-2] = after_order[rank-2], after_order[rank-1]
                        after_order = [r for r in after_order if r not in answers or r == target]
                        after = after_order.index(target) + 1
                        actual = step_reward(positions, answers, target, rank)
                        self.assertAlmostEqual(actual[0], 1/after - 1/before)

    def test_other_valid_answer_is_filtered_not_used_as_the_target(self):
        order = {1: 0, 2: 9, 3: 2}
        self.assertEqual(step_reward(order, {0, 2}, 2, 3), (.5, 2, 1))
        self.assertEqual(step_reward(order, {0, 2}, 0, 3)[0], 0)

    def test_unknown_ranks_are_not_silently_labelled_neutral(self):
        self.assertIsNone(step_reward({1: 0, 3: 2}, {2}, 2, 3))
        self.assertIsNone(step_reward({1: 0, 3: 8, 4: 2}, {2}, 2, 4))
        self.assertEqual(step_reward({1: 0, 3: 8, 4: 2}, {5}, 5, 4)[0], 0)

    def test_label_preserves_original_and_explicitly_excludes_unknown_actions(self):
        raw = {"base_order": [0, 9], "gt": [0, 2], "candidates": [
            {"rid": 2, "base_rank": 3, "step_mrr_delta": 0.0, "features": {}},
            {"rid": 7, "base_rank": 20, "step_mrr_delta": 0.0, "features": {}},
        ]}
        saved = copy.deepcopy(raw)
        labelled, dropped = label_record(raw, 2)
        self.assertEqual(raw, saved)
        self.assertEqual(dropped, 1)
        self.assertEqual(labelled["reward_schema"], TARGET_SCHEMA)
        self.assertEqual(labelled["target_relation_id"], 2)
        self.assertEqual(labelled["candidates"][0]["step_mrr_delta"], .5)
        self.assertEqual(labelled["candidates"][0]["step_source_relation_id"], 9)




if __name__ == "__main__":
    unittest.main()
