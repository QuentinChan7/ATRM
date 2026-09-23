from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]

def _iter_ints(values: Iterable[Any]) -> List[int]:
    out: List[int] = []
    for value in values or []:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def _answer_set(ans_dict: Any, s: int, o: int, r_gt: int) -> Set[int]:
    raw = None
    if hasattr(ans_dict, "get"):
        raw = ans_dict.get((int(s), int(o)), None)
        if raw is None:
            by_subject = ans_dict.get(int(s), None)
            if hasattr(by_subject, "get"):
                raw = by_subject.get(int(o), None)
    if raw is None:
        raw = [int(r_gt)]
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    if isinstance(raw, dict):
        merged: List[int] = []
        for value in raw.values():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            merged.extend(_iter_ints(value if isinstance(value, (list, tuple, set)) else [value]))
        raw = merged
    valid = set(_iter_ints(raw))
    valid.add(int(r_gt))
    return valid


def _safe_order(scores: torch.Tensor, upper_bound: int) -> List[int]:
    upper_bound = int(max(0, upper_bound))
    if scores is None or upper_bound <= 0:
        return []
    row = scores[:upper_bound].detach().clone().to(torch.float32)
    finite = torch.isfinite(row)
    if bool(finite.any().item()):
        finite_vals = row[finite]
        floor = torch.min(finite_vals) - torch.abs(torch.max(finite_vals) - torch.min(finite_vals)) - 1.0
        row = torch.where(finite, row, torch.full_like(row, floor))
    else:
        row = torch.zeros_like(row)
    try:
        return [int(x) for x in torch.argsort(row, descending=True, stable=True).detach().cpu().tolist()]
    except TypeError:
        return [int(x) for x in torch.argsort(row, descending=True).detach().cpu().tolist()]


def _order_after_deltas(scores: torch.Tensor, deltas: Dict[int, float], upper_bound: int) -> List[int]:
    if scores is None:
        return []
    row = scores[: int(upper_bound)].detach().clone().to(torch.float32)
    for rid, delta in (deltas or {}).items():
        try:
            rid_int = int(rid)
            if 0 <= rid_int < int(upper_bound):
                row[rid_int] = row[rid_int] + float(delta)
        except Exception:
            continue
    return _safe_order(row, upper_bound)


def _set_global_seed(seed: int = 42) -> None:
    """Best-effort reproducibility for GNN training, warm-up sampling, and agent policy."""
    try:
        seed = int(seed)
    except Exception:
        seed = 42
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _resolve_semantic_feature_path(args, dataset: str) -> Path:
    raw_path = str(getattr(args, "semantic_feature_path", "") or "").strip()
    if raw_path:
        first = Path(raw_path).expanduser()
        candidates = [first]
        if not first.is_absolute():
            candidates.extend([
                _REPO_ROOT / first,
                Path.cwd() / first,
                Path(__file__).resolve().parent / first,
            ])
    else:
        candidates = [
            _REPO_ROOT / "data" / dataset / "sbert_features.pt",
            Path.cwd() / "data" / dataset / "sbert_features.pt",
        ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def _semantic_tensor(payload: Dict[str, Any], names: Tuple[str, ...], label: str) -> torch.Tensor:
    for name in names:
        if name in payload:
            value = payload[name]
            if value is not None:
                return torch.as_tensor(value, dtype=torch.float32)
    raise KeyError(f"SBERT feature file is missing {label}; expected one of {names}")


def _load_semantic_features(args, dataset: str, num_nodes: int, num_rels: int) -> Tuple[torch.Tensor, torch.Tensor, Path]:
    path = _resolve_semantic_feature_path(args, dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Semantic features enabled but file not found: {path}. "
            f"Build it with scripts/build_sbert_features.py or set --semantic-feature-path."
        )
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Semantic feature file must contain a dict, got {type(payload)!r}: {path}")
    ent_features = _semantic_tensor(payload, ("entity_features", "entity", "entities"), "entity features")
    rel_features = _semantic_tensor(payload, ("relation_features", "relation", "relations"), "relation features")
    if ent_features.dim() != 2 or ent_features.size(0) != int(num_nodes):
        raise ValueError(
            f"Entity feature shape mismatch: expected [{num_nodes}, dim], got {tuple(ent_features.shape)} from {path}"
        )
    if rel_features.dim() != 2 or rel_features.size(0) not in {int(num_rels), int(num_rels) * 2}:
        raise ValueError(
            f"Relation feature shape mismatch: expected [{num_rels}, dim] or [{num_rels * 2}, dim], "
            f"got {tuple(rel_features.shape)} from {path}"
        )
    return ent_features.contiguous(), rel_features.contiguous(), path
