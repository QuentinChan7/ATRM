"""Incremental history vocabularies for large temporal datasets."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


class TemporalHistoryVocabulary:
    """Build the two binary decoder vocabularies without cumulative NPZ files.

    The state contains facts from timestamps strictly earlier than the snapshot
    being scored. ``update`` accepts original (non-inverse) dataset rows, while
    ``vocabularies`` accepts the forward+inverse rows consumed by the decoder.
    """

    def __init__(self, num_nodes: int, num_rels: int):
        if num_nodes <= 0 or num_rels <= 0:
            raise ValueError("num_nodes and num_rels must be positive")
        self.num_nodes = int(num_nodes)
        self.num_rels = int(num_rels)
        self.relation_width = 2 * self.num_rels
        self._tails = defaultdict(set)
        self._relations = defaultdict(set)
        self.snapshots = 0
        self.facts = 0

    @staticmethod
    def _array(rows) -> np.ndarray:
        try:
            import torch
            if torch.is_tensor(rows):
                rows = rows.detach().cpu().numpy()
        except ImportError:
            pass
        array = np.asarray(rows)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"history rows must have shape [n, >=3], got {array.shape}")
        return array

    def update(self, snapshot) -> None:
        rows = self._array(snapshot)
        if rows.size:
            # The saved sparse matrices are binary, so duplicate facts do not
            # change their contents. Deduplicating each snapshot reduces work.
            triples = np.unique(rows[:, :3].astype(np.int64, copy=False), axis=0)
            for source, relation, target in triples:
                source, relation, target = int(source), int(relation), int(target)
                if not (0 <= source < self.num_nodes and 0 <= target < self.num_nodes):
                    raise ValueError(f"entity id outside [0, {self.num_nodes}): {(source, target)}")
                if not 0 <= relation < self.num_rels:
                    raise ValueError(f"raw relation id outside [0, {self.num_rels}): {relation}")
                inverse = relation + self.num_rels
                self._tails[source * self.relation_width + relation].add(target)
                self._tails[target * self.relation_width + inverse].add(source)
                self._relations[source * self.num_nodes + target].add(relation)
                self._relations[target * self.num_nodes + source].add(inverse)
        self.snapshots += 1
        self.facts += int(len(rows))

    def seed(self, snapshots: Iterable) -> None:
        for snapshot in snapshots:
            self.update(snapshot)

    def vocabularies(self, forward_and_inverse_rows):
        import torch

        rows = self._array(forward_and_inverse_rows).astype(np.int64, copy=False)
        tails = np.zeros((len(rows), self.num_nodes), dtype=np.float32)
        relations = np.zeros((len(rows), self.relation_width), dtype=np.float32)
        for index, row in enumerate(rows):
            source, relation, target = map(int, row[:3])
            if not (0 <= source < self.num_nodes and 0 <= target < self.num_nodes):
                raise ValueError(f"entity id outside [0, {self.num_nodes}): {(source, target)}")
            if not 0 <= relation < self.relation_width:
                raise ValueError(
                    f"relation id outside [0, {self.relation_width}): {relation}"
                )
            known_tails = self._tails.get(source * self.relation_width + relation)
            if known_tails:
                tails[index, list(known_tails)] = 1.0
            known_relations = self._relations.get(source * self.num_nodes + target)
            if known_relations:
                relations[index, list(known_relations)] = 1.0
        return torch.from_numpy(tails), torch.from_numpy(relations)

    def summary(self) -> dict[str, int]:
        return {
            "snapshots": self.snapshots,
            "facts": self.facts,
            "tail_keys": len(self._tails),
            "relation_keys": len(self._relations),
        }
