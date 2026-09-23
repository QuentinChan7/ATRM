"""Exact target-filter rewards for adjacent swaps in recorded relation rankings."""
from __future__ import annotations

from typing import Mapping


TARGET_SCHEMA = "target_filter_step_v1"


def known_order(record: Mapping) -> dict[int, int]:
    positions = {i: int(rid) for i, rid in enumerate(record["base_order"], 1)}
    for candidate in record["candidates"]:
        rank, rid = int(candidate["base_rank"]), int(candidate["rid"])
        if rank < 1 or (rank in positions and positions[rank] != rid):
            raise ValueError("Inconsistent candidate/base ranking in trace")
        positions[rank] = rid
    if len(set(positions.values())) != len(positions):
        raise ValueError("A relation occupies multiple base ranks")
    return positions


def step_reward(positions: Mapping[int, int], answers: set[int], target: int,
                rank: int) -> tuple[float, int | None, int | None] | None:
    """None means insufficient recorded ranks, never a neutral training label."""
    if target not in answers:
        raise ValueError("Target relation is missing from the valid-answer set")
    if rank == 1:
        return 0.0, None, None
    if rank not in positions or rank - 1 not in positions:
        return None
    candidate, above = positions[rank], positions[rank - 1]
    if target not in (candidate, above):
        return 0.0, None, None
    other = above if candidate == target else candidate
    if other in answers:
        return 0.0, None, None
    target_position = rank if candidate == target else rank - 1
    if any(i not in positions for i in range(1, target_position + 1)):
        return None
    before = 1 + sum(positions[i] not in answers for i in range(1, target_position))
    after = before - 1 if candidate == target else before + 1
    return 1.0 / after - 1.0 / before, before, after


def label_record(record: Mapping, target: int) -> tuple[dict, int]:
    """Return a new trace; retain only actions with exactly recoverable rewards."""
    if record.get("target_relation_id") is not None and int(record["target_relation_id"]) != target:
        raise ValueError("Stored target differs from the source validation triple")
    positions = known_order(record)
    answers = set(map(int, record["gt"]))
    if target not in answers:
        raise ValueError("Recovered target is absent from trace answers")
    candidates, dropped = [], 0
    for raw in record["candidates"]:
        result = step_reward(positions, answers, target, int(raw["base_rank"]))
        if result is None:
            dropped += 1
            continue
        delta, before, after = result
        features = dict(raw.get("features") or {})
        features["step_mrr_delta"] = delta
        candidates.append({
            **raw, "legacy_step_mrr_delta": raw.get("step_mrr_delta"),
            "step_mrr_delta": delta, "features": features,
            "is_target": int(raw["rid"]) == target,
            "step_source_relation_id": positions.get(int(raw["base_rank"]) - 1),
            "target_filter_before_rank": before, "target_filter_after_rank": after,
            "outcome_if_step_lift": "fix" if delta > 1e-12 else "fail" if delta < -1e-12 else "neutral",
        })
    return {
        **record, "reward_schema": TARGET_SCHEMA, "target_relation_id": target,
        "candidates": candidates, "unrecoverable_actions": dropped,
    }, dropped
