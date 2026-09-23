from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class LLMTraceDistillProvenanceTeacherTest(unittest.TestCase):
    def test_augment_preserves_base_teacher_and_adds_llm_pairs(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = object
        fake_torch.device = object
        fake_torch.dtype = object
        fake_torch.float32 = "float32"
        records = []
        for time_id in (10, 11, 12, 13):
            records.append(
                {
                    "query": {"s": 1, "o": 2, "t": time_id},
                    "signature": "rel|local_sparse|prev=7",
                    "prev_rel": 7,
                    "base_top": 3,
                    "base_order": [3, 2, 4],
                    "gt": [2],
                    "candidates": [
                        {
                            "rid": 2,
                            "rank": 2,
                            "is_gt": True,
                            "step_mrr_delta": 0.5,
                            "features": {"commit_gap": 0.2, "counterfactual_risk": 0.1},
                        }
                    ],
                }
            )
        hypotheses = {
            "hypotheses": [
                {
                    "id": "llm_rule",
                    "source": "llm_reflection_hypothesis",
                    "signature": "rel|local_sparse|prev=7",
                    "prev_rel": 7,
                    "top1": 3,
                    "choice": 2,
                    "conditions": {"min_commit_gap": 0.1, "max_counterfactual_risk": 0.2},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            trace_path = root / "trace.jsonl"
            hypothesis_path = root / "hypotheses.json"
            trace_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            hypothesis_path.write_text(json.dumps(hypotheses), encoding="utf-8")
            with patch.dict(sys.modules, {"torch": fake_torch}):
                sys.modules.pop("awesome_agent.llm_trace_distill", None)
                module = importlib.import_module("awesome_agent.llm_trace_distill")
                teacher = module.LLMTraceDistillTeacher.from_trace_file(
                    trace_path,
                    num_rels=5,
                    min_delta=0.02,
                    pair_min_support=2,
                    pair_min_precision=0.7,
                    pair_min_reward=0.1,
                    provenance_mode="augment",
                    provenance_hypothesis_path=str(hypothesis_path),
                    provenance_min_support=2,
                    provenance_min_fixes=2,
                    provenance_min_precision=0.7,
                    provenance_min_reward=0.02,
                    provenance_holdout_min_support=1,
                    provenance_holdout_min_fixes=1,
                    provenance_holdout_min_precision=0.7,
                    provenance_holdout_min_reward=0.02,
                    provenance_pair_min_support=2,
                    provenance_pair_min_precision=0.7,
                    provenance_pair_min_reward=0.1,
                )
        self.assertGreater(len(teacher.query_targets), 0)
        self.assertGreater(len(teacher.pair_weights), 0)
        self.assertGreater(len(teacher.provenance_pair_weights), 0)
        self.assertGreater(len(teacher.provenance_context_weights), 0)
        self.assertIn((7, 3, 2), teacher.provenance_context_weights)
        self.assertEqual(teacher.summary.provenance_accepted_rules, 1)
        self.assertGreater(teacher.summary.provenance_context_rules, 0)


if __name__ == "__main__":
    unittest.main()
