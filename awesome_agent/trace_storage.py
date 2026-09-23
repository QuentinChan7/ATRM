"""Bounded-memory, target-filter trace collection for offline-only experiments."""
from __future__ import annotations

import json
from pathlib import Path

from awesome_agent.target_trace import known_order, label_record


MTR_FEATURES = ("commit_gap", "tool_gap", "verifier_gap", "counterfactual_risk", "reflection")


def compact_target_record(raw):
    if raw.get("target_relation_id") is None:
        raise ValueError("Streaming trace requires the evaluated target, including relation ID zero")
    row, dropped = label_record(raw, int(raw["target_relation_id"]))
    result = {key: row[key] for key in (
        "query", "target_relation_id", "reward_schema", "signature", "selected_tools",
        "prev_rel", "base_top", "base_order", "gt", "unrecoverable_actions",
    )}
    result["base_positions"] = sorted(known_order(raw).items())
    result["candidates"] = [
        {**{key: c[key] for key in (
            "rid", "rank", "base_rank", "is_gt", "is_target", "step_mrr_delta",
            "step_source_relation_id", "target_filter_before_rank", "target_filter_after_rank",
        )}, "features": {key: c["features"][key] for key in MTR_FEATURES if key in c["features"]}}
        for c in row["candidates"]
    ]
    return result, dropped


def validate_collector(config):
    if not config.trace_enabled:
        raise ValueError("Trace collection is disabled")


class TargetTraceSink:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8")
        self.count = self.actions = self.dropped = 0

    def append(self, raw):
        row, dropped = compact_target_record(raw)
        self.handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        self.count += 1
        self.actions += len(row["candidates"])
        self.dropped += dropped
        if self.count % 1000 == 0:
            self.handle.flush()

    def close(self):
        self.handle.close()
