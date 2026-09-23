from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from awesome_agent.llm_hypothesis_provenance import LLMHypothesisProvenance


def _record(time_id: int, delta: float) -> dict:
    return {
        "query": {"s": 1, "o": 2, "t": time_id},
        "signature": "rel|local_sparse|prev=7",
        "prev_rel": 7,
        "base_top": 3,
        "base_order": [3, 2, 4],
        "candidates": [
            {
                "rid": 2,
                "step_mrr_delta": delta,
                "features": {"commit_gap": 0.2, "counterfactual_risk": 0.1},
            }
        ],
    }


class LLMHypothesisProvenanceTest(unittest.TestCase):
    def test_adjacent_opponent_is_part_of_the_rule_match(self):
        row = _record(10, .5)
        row["candidates"][0]["step_source_relation_id"] = 0
        rule = dict(id="adjacent", signature=row["signature"], prev_rel=7, top1=3, choice=2, opponent=0)
        self.assertEqual(len(list(LLMHypothesisProvenance._iter_matches(row, [rule]))), 1)
        self.assertEqual(list(LLMHypothesisProvenance._iter_matches(row, [dict(rule, opponent=3)])), [])

    def test_zero_relation_id_is_not_missing(self):
        row = _record(10, .5)
        row.update(signature="rel|local_sparse|prev=0", prev_rel=0, base_top=0)
        rule = dict(id="zero", signature="rel|local_sparse|prev=0", prev_rel=0, top1=0, choice=2)
        self.assertEqual(len(list(LLMHypothesisProvenance._iter_matches(row, [rule]))), 1)
        missing = dict(rule, id="missing", signature="rel|local_sparse|prev=none", prev_rel=-1)
        self.assertEqual(list(LLMHypothesisProvenance._iter_matches(row, [missing])), [])

    def test_targets_have_separate_provenance_keys(self):
        from awesome_agent.llm_hypothesis_provenance import _record_key
        row = _record(10, .5)
        self.assertNotEqual(_record_key(dict(row, target_relation_id=0)),
                            _record_key(dict(row, target_relation_id=2)))

    def _build(self, holdout_delta: float) -> LLMHypothesisProvenance:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            trace_path = root / "trace.jsonl"
            hypothesis_path = root / "hypotheses.json"
            records = [_record(10, 0.5), _record(11, 0.5), _record(12, 0.5), _record(13, holdout_delta)]
            trace_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            hypothesis_path.write_text(
                json.dumps(
                    {
                        "hypotheses": [
                            {
                                "id": "llm_rule",
                                "source": "llm_reflection_hypothesis",
                                "signature": "rel|local_sparse|prev=7",
                                "prev_rel": 7,
                                "top1": 3,
                                "choice": 2,
                                "conditions": {
                                    "min_commit_gap": 0.1,
                                    "max_counterfactual_risk": 0.2,
                                },
                            },
                            {
                                "id": "statistical_fallback",
                                "source": "trace_statistical_fallback",
                                "signature": "*",
                                "choice": 2,
                                "conditions": {},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return LLMHypothesisProvenance.from_trace_paths(
                [trace_path],
                hypothesis_path=str(hypothesis_path),
                holdout_fraction=0.25,
                min_support=2,
                min_fixes=2,
                min_precision=0.7,
                min_reward=0.02,
                max_fail_rate=0.3,
                holdout_min_support=1,
                holdout_min_fixes=1,
                holdout_min_precision=0.7,
                holdout_min_reward=0.02,
                holdout_max_fail_rate=0.0,
            )

    def test_accepts_llm_rule_and_excludes_holdout_evidence(self) -> None:
        provenance = self._build(0.5)
        self.assertEqual(provenance.summary.llm_rules, 1)
        self.assertEqual(provenance.summary.accepted_rules, 1)
        self.assertEqual(provenance.summary.qualified_candidates, 3)
        self.assertEqual([rule["id"] for rule in provenance.accepted_rules], ["llm_rule"])
        self.assertIsNone(provenance.evidence_for(_record(13, 0.5), 2))

    def test_rejects_rule_that_fails_temporal_holdout(self) -> None:
        provenance = self._build(-0.5)
        self.assertEqual(provenance.summary.accepted_rules, 0)
        self.assertEqual(provenance.summary.qualified_candidates, 0)
        self.assertEqual(len(provenance.rule_audits), 1)
        self.assertEqual(provenance.rule_audits[0]["holdout"]["fails"], 1)
        self.assertIn("holdout.fail_rate", provenance.rule_audits[0]["rejected_by"])


if __name__ == "__main__":
    unittest.main()
