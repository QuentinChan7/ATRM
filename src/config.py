"""Fixed architecture and Memory settings for ATRM."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    history: int
    angle: int
    layers: int = 2
    semantic: bool = False
    static: bool = False
    selection: str = "relation_filter"


DATASETS = {
    "ICEWS14": DatasetConfig(9, 14, semantic=True, static=True),
    "ICEWS18": DatasetConfig(10, 9, semantic=True, static=True),
    "GDELT": DatasetConfig(7, 10),
    "WIKI": DatasetConfig(2, 10),
    "YAGO": DatasetConfig(1, 10, layers=1, selection="relation_raw"),
}


def fusion_config(num_rels, trace_path=None):
    from awesome_agent.fusion import FusionConfig

    return FusionConfig(
        relation_upper_bound=2 * num_rels, base_topn=14, expand_add=12,
        max_delta=1.65, min_delta=-0.25, tool_weight=0.50,
        llm_choice_delta=0.62,
        memory_total_weight=0.22, recall_weight=0.08,
        recall_only_max_delta=0.06, target_min_score=0.25,
        target_min_precision=0.48, target_fail_penalty=1.80,
        target_neutral_penalty=0.025, target_min_valid_fixes=20,
        target_use_rule_calibrator=True, target_rule_min_precision=0.70,
        target_rule_max_neutral_rate=0.78,
        trace_enabled=trace_path is not None,
        llm_reflection_hypothesis_trace_path=str(trace_path or ""),
    )
