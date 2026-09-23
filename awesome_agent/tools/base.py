from __future__ import annotations

from collections import Counter
from typing import Any

from awesome_agent.contracts import EvidenceBundle, RelationQuery, ToolOutput
from awesome_agent.memory.stores import GraphMemory


class MemoryTool:
    """Base class for deterministic tools computed over GraphMemory."""

    name = "base"

    def __init__(self, memory: GraphMemory, mapper: Any):
        self.memory = memory
        self.mapper = mapper

    @staticmethod
    def norm_counter_score(counter: Counter, key: int) -> float:
        total = float(sum(counter.values()))
        return float(counter.get(int(key), 0.0) / (total + 1.0))

    def run_relation(self, query: RelationQuery, evidence: EvidenceBundle) -> ToolOutput:
        raise NotImplementedError
