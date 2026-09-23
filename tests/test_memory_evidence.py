from __future__ import annotations

import unittest

from awesome_agent.contracts import CandidateSet, RelationQuery
from awesome_agent.memory.evidence import EvidenceBuilder
from awesome_agent.memory.stores import GraphMemory


class EvidenceBuilderInverseRelationTest(unittest.TestCase):
    def test_reciprocal_transition_expands_inverse_candidate(self) -> None:
        memory = GraphMemory()
        memory.set_relation_base_count(10)
        memory.update_snapshot([[1, 8, 2, 1], [3, 8, 4, 1]])
        memory.update_snapshot([[3, 9, 4, 2]])
        builder = EvidenceBuilder(memory)

        expanded = builder.expand_relation_candidates(
            RelationQuery(s=2, o=1, t=3),
            CandidateSet.from_topk([0], [1.0], source="test"),
            max_add=8,
            upper_bound=20,
        )

        self.assertIn(19, expanded.ids)


if __name__ == "__main__":
    unittest.main()
