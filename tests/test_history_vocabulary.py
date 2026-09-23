import unittest

import numpy as np
import torch

from awesome_agent.history_vocabulary import TemporalHistoryVocabulary


class TemporalHistoryVocabularyTests(unittest.TestCase):
    def test_forward_inverse_and_temporal_updates_match_sparse_protocol(self):
        history = TemporalHistoryVocabulary(num_nodes=4, num_rels=2)
        history.update(np.array([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 2, 0],
        ]))
        queries = np.array([
            [0, 0, 1, 1],
            [1, 2, 0, 1],
            [1, 1, 2, 1],
            [2, 3, 1, 1],
            [0, 1, 3, 1],
        ])
        tails, relations = history.vocabularies(queries)
        expected_tails = torch.zeros(5, 4)
        expected_tails[0, 1] = 1
        expected_tails[1, 0] = 1
        expected_tails[2, 2] = 1
        expected_tails[3, 1] = 1
        expected_relations = torch.zeros(5, 4)
        expected_relations[0, 0] = 1
        expected_relations[1, 2] = 1
        expected_relations[2, 1] = 1
        expected_relations[3, 3] = 1
        self.assertTrue(torch.equal(tails, expected_tails))
        self.assertTrue(torch.equal(relations, expected_relations))

        before, _ = history.vocabularies(np.array([[0, 1, 3, 2]]))
        self.assertEqual(before.sum().item(), 0)
        history.update(np.array([[0, 1, 3, 1]]))
        after, relation_after = history.vocabularies(np.array([[0, 1, 3, 2]]))
        self.assertEqual(after[0, 3].item(), 1)
        self.assertEqual(relation_after[0, 1].item(), 1)
        self.assertEqual(history.summary()["snapshots"], 2)
        self.assertEqual(history.summary()["facts"], 4)

    def test_rejects_invalid_raw_and_directional_ids(self):
        history = TemporalHistoryVocabulary(num_nodes=3, num_rels=2)
        with self.assertRaises(ValueError):
            history.update(np.array([[0, 2, 1, 0]]))
        with self.assertRaises(ValueError):
            history.vocabularies(np.array([[0, 4, 1, 0]]))


if __name__ == "__main__":
    unittest.main()
