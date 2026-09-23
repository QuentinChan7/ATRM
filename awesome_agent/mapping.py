from __future__ import annotations

from pathlib import Path
from typing import Dict


class NLMapper:
    """Small id/name mapper used only for evidence rendering and prompts."""

    def __init__(self, ent_path: str = "", rel_path: str = ""):
        self.id2ent: Dict[int, str] = {}
        self.id2rel: Dict[int, str] = {}
        if ent_path:
            self._load(Path(ent_path), self.id2ent)
        if rel_path:
            self._load(Path(rel_path), self.id2rel)

    @staticmethod
    def _load(path: Path, mapping: Dict[int, str]) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = raw_line.strip().split("\t")
                if len(parts) < 2:
                    continue
                try:
                    mapping[int(parts[1])] = parts[0]
                except Exception:
                    continue

    def ent_name(self, eid: int) -> str:
        return self.id2ent.get(int(eid), f"ENT_{int(eid)}")

    def base_relation_count(self) -> int:
        if not self.id2rel:
            return 0
        return max(int(rid) for rid in self.id2rel.keys()) + 1

    def base_rel_id(self, rid: int) -> int:
        rid = int(rid)
        base_count = self.base_relation_count()
        if base_count > 0 and base_count <= rid < 2 * base_count:
            return rid - base_count
        return rid

    def is_inverse_rel(self, rid: int) -> bool:
        rid = int(rid)
        base_count = self.base_relation_count()
        return bool(base_count > 0 and base_count <= rid < 2 * base_count and (rid - base_count) in self.id2rel)

    def rel_name(self, rid: int) -> str:
        rid = int(rid)
        if rid in self.id2rel:
            return self.id2rel[rid]
        base_id = self.base_rel_id(rid)
        if base_id in self.id2rel and self.is_inverse_rel(rid):
            return f"reverse of {self.id2rel[base_id]}"
        return f"REL_{rid}"

    def rel_direction(self, rid: int) -> str:
        return "reverse" if self.is_inverse_rel(int(rid)) else "forward"

    def rel_phrase(self, rid: int, subject: str = "subject", object_: str = "object") -> str:
        rid = int(rid)
        base_id = self.base_rel_id(rid)
        rel = self.id2rel.get(base_id, self.rel_name(rid))
        if self.is_inverse_rel(rid):
            return f"{object_} -[{rel}]-> {subject}"
        return f"{subject} -[{rel}]-> {object_}"
