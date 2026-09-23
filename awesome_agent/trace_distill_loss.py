from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_TRACE_DISTILL_FRAMEWORKS = frozenset(
    {
        "llm_trace_distill",
        "llm_training_teacher",
        "llm_trace_training_teacher",
        "llm_teacher_training_loss",
        "llm_query_trace_distill",
        "query_trace_distill",
    }
)


def resolve_trace_distill_enabled(
    explicit: Optional[bool],
    framework: str,
    env_value: Optional[str] = None,
) -> bool:
    """Resolve activation with CLI override, environment, then framework precedence."""
    if explicit is not None:
        return bool(explicit)
    if env_value is not None:
        normalized = str(env_value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return str(framework or "").strip().lower() in _TRACE_DISTILL_FRAMEWORKS


@dataclass(frozen=True)
class SparseCoverageNormalization:
    denominator: float
    observed_coverage: float
    amplification: float


def relation_history_context_matches(
    *,
    previous_relation: int,
    history_total: float,
    previous_relation_seen: float = 0.0,
) -> bool:
    if int(previous_relation) == -1:
        return float(history_total) <= 0.0
    return int(previous_relation) >= 0 and float(previous_relation_seen) > 0.0


def sparse_coverage_normalization(
    *,
    batch_rows: int,
    matched_rows: int,
    target_coverage: float,
) -> SparseCoverageNormalization:
    """Bound sparse-rule amplification without giving it active-row normalization."""
    batch = max(1, int(batch_rows))
    matched = max(0, min(batch, int(matched_rows)))
    coverage = float(matched) / float(batch)
    target = max(0.0, min(1.0, float(target_coverage)))
    denominator = max(1.0, float(matched), float(batch) * target)
    return SparseCoverageNormalization(
        denominator=float(denominator),
        observed_coverage=float(coverage),
        amplification=float(batch) / float(denominator),
    )
