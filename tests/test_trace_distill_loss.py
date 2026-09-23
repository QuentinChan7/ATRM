from __future__ import annotations

import unittest

from awesome_agent.trace_distill_loss import (
    relation_history_context_matches,
    resolve_trace_distill_enabled,
    sparse_coverage_normalization,
)


class SparseCoverageNormalizationTest(unittest.TestCase):
    def test_default_preserves_full_batch_normalization(self) -> None:
        result = sparse_coverage_normalization(batch_rows=500, matched_rows=2, target_coverage=1.0)
        self.assertEqual(result.denominator, 500.0)
        self.assertAlmostEqual(result.observed_coverage, 0.004)
        self.assertEqual(result.amplification, 1.0)

    def test_target_coverage_bounds_sparse_amplification(self) -> None:
        result = sparse_coverage_normalization(batch_rows=500, matched_rows=2, target_coverage=0.05)
        self.assertEqual(result.denominator, 25.0)
        self.assertEqual(result.amplification, 20.0)

    def test_dense_matches_are_never_downscaled_below_active_mean(self) -> None:
        result = sparse_coverage_normalization(batch_rows=500, matched_rows=100, target_coverage=0.05)
        self.assertEqual(result.denominator, 100.0)
        self.assertEqual(result.amplification, 5.0)

    def test_prev_none_requires_an_empty_relation_history(self) -> None:
        self.assertTrue(
            relation_history_context_matches(previous_relation=-1, history_total=0.0)
        )
        self.assertFalse(
            relation_history_context_matches(previous_relation=-1, history_total=1.0)
        )

    def test_explicit_previous_relation_must_be_present(self) -> None:
        self.assertTrue(
            relation_history_context_matches(
                previous_relation=7,
                history_total=3.0,
                previous_relation_seen=1.0,
            )
        )
        self.assertFalse(
            relation_history_context_matches(
                previous_relation=7,
                history_total=3.0,
                previous_relation_seen=0.0,
            )
        )


class TraceDistillActivationTest(unittest.TestCase):
    def test_explicit_disable_wins_over_framework_and_environment(self) -> None:
        self.assertFalse(
            resolve_trace_distill_enabled(False, "llm_trace_distill", "1")
        )

    def test_explicit_enable_wins_over_disabled_environment(self) -> None:
        self.assertTrue(
            resolve_trace_distill_enabled(True, "graph_token_listwise", "0")
        )

    def test_framework_is_only_a_final_fallback(self) -> None:
        self.assertTrue(resolve_trace_distill_enabled(None, "llm_trace_distill", None))
        self.assertFalse(resolve_trace_distill_enabled(None, "llm_trace_distill", "0"))


if __name__ == "__main__":
    unittest.main()
