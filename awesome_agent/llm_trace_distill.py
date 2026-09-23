from __future__ import annotations

import glob
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch

from awesome_agent.llm_hypothesis_provenance import LLMHypothesisProvenance
from awesome_agent.trace_distill_loss import relation_history_context_matches
from awesome_agent.target_trace import TARGET_SCHEMA


QueryKey = Tuple[int, int, int]
EntityPairKey = Tuple[int, int]
PairKey = Tuple[int, int]
ContextPairKey = Tuple[int, int, int]
MarginKey = Tuple[int, int, int]


def _resolve_trace_path(path: Union[str, Path]) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path.cwd() / raw,
        repo_root / raw,
        repo_root / "src" / raw,
        repo_root / "src" / "logs" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / raw).resolve()


def _resolve_trace_paths(path: Union[str, Path]) -> Tuple[Path, ...]:
    raw_text = str(path).strip()
    if not raw_text:
        return (_resolve_trace_path(path),)
    parts = [part.strip() for part in raw_text.replace(";", ",").split(",") if part.strip()]
    resolved = []
    seen = set()
    repo_root = Path(__file__).resolve().parents[1]
    for part in parts:
        raw = Path(part).expanduser()
        glob_candidates = []
        if any(token in part for token in ("*", "?", "[")):
            patterns = [str(raw)] if raw.is_absolute() else [
                str(Path.cwd() / raw),
                str(repo_root / raw),
                str(repo_root / "src" / raw),
            ]
            for pattern in patterns:
                glob_candidates.extend(Path(item) for item in sorted(glob.glob(pattern)))
        else:
            glob_candidates.append(_resolve_trace_path(raw))
        for candidate in glob_candidates:
            try:
                final = candidate.resolve()
            except Exception:
                final = candidate
            key = str(final)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(final)
    return tuple(resolved)


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _candidate_rank(candidate: Mapping[str, Any]) -> Optional[int]:
    for name in ("rank", "base_rank", "current_rank"):
        value = _as_int(candidate.get(name))
        if value is not None and value > 0:
            return value
    return None


@dataclass(frozen=True)
class LLMTraceDistillSummary:
    path: str
    rows_seen: int
    rows_used: int
    query_targets: int
    entity_pair_targets: int
    pair_rules: int
    pair_candidates: int
    source_field: str
    min_delta: float
    pair_min_support: int
    pair_min_precision: float
    pair_min_reward: float
    pair_max_neutral_rate: float
    confidence_weighting: bool
    margin_targets: int
    margin_pairs: int
    margin_candidates: int
    margin_min_neg_delta: float
    margin_max_negatives_per_target: int
    margin_scope: str
    margin_rule_targets: int
    margin_rule_pairs: int
    margin_rule_min_support: int
    margin_rule_min_weight: float
    margin_rule_min_precision: float
    dedupe_records: bool
    provenance_mode: str
    provenance_hypothesis_path: str
    provenance_llm_rules: int
    provenance_accepted_rules: int
    provenance_qualified_candidates: int
    provenance_pair_rules: int
    provenance_context_rules: int
    provenance_pair_min_support: int
    provenance_pair_min_precision: float
    provenance_pair_min_reward: float
    provenance_pair_max_neutral_rate: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "rows_seen": int(self.rows_seen),
            "rows_used": int(self.rows_used),
            "query_targets": int(self.query_targets),
            "entity_pair_targets": int(self.entity_pair_targets),
            "pair_rules": int(self.pair_rules),
            "pair_candidates": int(self.pair_candidates),
            "source_field": self.source_field,
            "min_delta": float(self.min_delta),
            "pair_min_support": int(self.pair_min_support),
            "pair_min_precision": float(self.pair_min_precision),
            "pair_min_reward": float(self.pair_min_reward),
            "pair_max_neutral_rate": float(self.pair_max_neutral_rate),
            "confidence_weighting": bool(self.confidence_weighting),
            "margin_targets": int(self.margin_targets),
            "margin_pairs": int(self.margin_pairs),
            "margin_candidates": int(self.margin_candidates),
            "margin_min_neg_delta": float(self.margin_min_neg_delta),
            "margin_max_negatives_per_target": int(self.margin_max_negatives_per_target),
            "margin_scope": self.margin_scope,
            "margin_rule_targets": int(self.margin_rule_targets),
            "margin_rule_pairs": int(self.margin_rule_pairs),
            "margin_rule_min_support": int(self.margin_rule_min_support),
            "margin_rule_min_weight": float(self.margin_rule_min_weight),
            "margin_rule_min_precision": float(self.margin_rule_min_precision),
            "dedupe_records": bool(self.dedupe_records),
            "provenance_mode": str(self.provenance_mode),
            "provenance_hypothesis_path": str(self.provenance_hypothesis_path),
            "provenance_llm_rules": int(self.provenance_llm_rules),
            "provenance_accepted_rules": int(self.provenance_accepted_rules),
            "provenance_qualified_candidates": int(self.provenance_qualified_candidates),
            "provenance_pair_rules": int(self.provenance_pair_rules),
            "provenance_context_rules": int(self.provenance_context_rules),
            "provenance_pair_min_support": int(self.provenance_pair_min_support),
            "provenance_pair_min_precision": float(self.provenance_pair_min_precision),
            "provenance_pair_min_reward": float(self.provenance_pair_min_reward),
            "provenance_pair_max_neutral_rate": float(self.provenance_pair_max_neutral_rate),
        }


class LLMTraceDistillTeacher:
    """Offline teacher compiled from memory/tool/reflection trace outcomes."""

    def __init__(
        self,
        *,
        num_rels: int,
        query_targets: Mapping[QueryKey, Mapping[int, float]],
        entity_pair_targets: Mapping[EntityPairKey, Mapping[int, float]],
        pair_weights: Mapping[PairKey, float],
        provenance_pair_weights: Mapping[PairKey, float],
        provenance_context_weights: Mapping[ContextPairKey, float],
        margin_negatives: Mapping[MarginKey, Mapping[int, float]],
        margin_relation_negatives: Mapping[int, Mapping[int, float]],
        summary: LLMTraceDistillSummary,
        confidence_weighting: bool = False,
        margin_scope: str = "relation",
    ) -> None:
        self.num_rels = int(num_rels)
        self.output_rels = int(num_rels) * 2
        self.query_targets: Dict[QueryKey, Dict[int, float]] = {
            key: {int(rid): float(weight) for rid, weight in value.items() if 0 <= int(rid) < self.output_rels}
            for key, value in query_targets.items()
        }
        self.entity_pair_targets: Dict[EntityPairKey, Dict[int, float]] = {
            key: {int(rid): float(weight) for rid, weight in value.items() if 0 <= int(rid) < self.output_rels}
            for key, value in entity_pair_targets.items()
        }
        self.pair_weights: Dict[PairKey, float] = {
            (int(src), int(dst)): float(weight)
            for (src, dst), weight in pair_weights.items()
            if 0 <= int(src) < self.output_rels and 0 <= int(dst) < self.output_rels
        }
        self.provenance_pair_weights: Dict[PairKey, float] = {
            (int(src), int(dst)): float(weight)
            for (src, dst), weight in provenance_pair_weights.items()
            if 0 <= int(src) < self.output_rels and 0 <= int(dst) < self.output_rels
        }
        self.provenance_context_weights: Dict[ContextPairKey, float] = {
            (int(prev_rel), int(src), int(dst)): float(weight)
            for (prev_rel, src, dst), weight in provenance_context_weights.items()
            if int(prev_rel) == -1 or 0 <= int(prev_rel) < self.output_rels
            if 0 <= int(src) < self.output_rels and 0 <= int(dst) < self.output_rels
        }
        self._provenance_context_by_target: DefaultDict[int, list] = defaultdict(list)
        for (prev_rel, src, dst), weight in self.provenance_context_weights.items():
            self._provenance_context_by_target[int(dst)].append((int(prev_rel), int(src), float(weight)))
        self.margin_negatives: Dict[MarginKey, Dict[int, float]] = {
            (int(s), int(o), int(pos_rel)): {
                int(neg_rel): float(weight)
                for neg_rel, weight in value.items()
                if 0 <= int(neg_rel) < self.output_rels and int(neg_rel) != int(pos_rel) and float(weight) > 0
            }
            for (s, o, pos_rel), value in margin_negatives.items()
            if 0 <= int(pos_rel) < self.output_rels
        }
        self.margin_relation_negatives: Dict[int, Dict[int, float]] = {
            int(pos_rel): {
                int(neg_rel): float(weight)
                for neg_rel, weight in value.items()
                if 0 <= int(neg_rel) < self.output_rels and int(neg_rel) != int(pos_rel) and float(weight) > 0
            }
            for pos_rel, value in margin_relation_negatives.items()
            if 0 <= int(pos_rel) < self.output_rels
        }
        self.summary = summary
        self.confidence_weighting = bool(confidence_weighting)
        self.margin_scope = str(margin_scope or "relation").strip().lower()
        self._pair_matrix_cpu: Optional[torch.Tensor] = None
        self._provenance_pair_matrix_cpu: Optional[torch.Tensor] = None

    @classmethod
    def from_trace_file(
        cls,
        path: Union[str, Path],
        *,
        num_rels: int,
        min_delta: float = 0.02,
        max_rows: int = 0,
        max_targets_per_query: int = 3,
        source_field: str = "base_top",
        pair_min_support: int = 4,
        pair_min_precision: float = 0.72,
        pair_min_reward: float = 0.50,
        pair_max_neutral_rate: float = 0.92,
        mirror_inverse: bool = True,
        confidence_weighting: bool = False,
        margin_min_neg_delta: float = 0.02,
        margin_max_negatives_per_target: int = 4,
        margin_rank_above_weight: float = 0.0,
        margin_source_weight: float = 0.30,
        margin_scope: str = "relation",
        margin_rule_min_support: int = 4,
        margin_rule_min_weight: float = 0.18,
        margin_rule_min_precision: float = 0.0,
        dedupe_records: bool = False,
        provenance_mode: str = "off",
        provenance_hypothesis_path: str = "",
        provenance_holdout_fraction: float = 0.25,
        provenance_min_support: int = 6,
        provenance_min_fixes: int = 2,
        provenance_min_precision: float = 0.70,
        provenance_min_reward: float = 0.02,
        provenance_max_fail_rate: float = 0.30,
        provenance_holdout_min_support: int = 2,
        provenance_holdout_min_fixes: int = 1,
        provenance_holdout_min_precision: float = 0.70,
        provenance_holdout_min_reward: float = 0.02,
        provenance_holdout_max_fail_rate: float = 0.0,
        provenance_pair_min_support: int = 2,
        provenance_pair_min_precision: float = 0.80,
        provenance_pair_min_reward: float = 0.10,
        provenance_pair_max_neutral_rate: float = 0.90,
    ) -> "LLMTraceDistillTeacher":
        trace_paths = _resolve_trace_paths(path)
        trace_path = trace_paths[0] if trace_paths else _resolve_trace_path(path)
        output_rels = int(num_rels) * 2
        query_accum: DefaultDict[QueryKey, DefaultDict[int, float]] = defaultdict(lambda: defaultdict(float))
        entity_pair_accum: DefaultDict[EntityPairKey, DefaultDict[int, float]] = defaultdict(lambda: defaultdict(float))
        pair_pos: DefaultDict[PairKey, int] = defaultdict(int)
        pair_neg: DefaultDict[PairKey, int] = defaultdict(int)
        pair_neu: DefaultDict[PairKey, int] = defaultdict(int)
        pair_reward: DefaultDict[PairKey, float] = defaultdict(float)
        provenance_pair_pos: DefaultDict[PairKey, int] = defaultdict(int)
        provenance_pair_neg: DefaultDict[PairKey, int] = defaultdict(int)
        provenance_pair_neu: DefaultDict[PairKey, int] = defaultdict(int)
        provenance_pair_reward: DefaultDict[PairKey, float] = defaultdict(float)
        provenance_context_pos: DefaultDict[ContextPairKey, int] = defaultdict(int)
        provenance_context_neg: DefaultDict[ContextPairKey, int] = defaultdict(int)
        provenance_context_neu: DefaultDict[ContextPairKey, int] = defaultdict(int)
        provenance_context_reward: DefaultDict[ContextPairKey, float] = defaultdict(float)
        margin_accum: DefaultDict[MarginKey, DefaultDict[int, float]] = defaultdict(lambda: defaultdict(float))
        margin_rule_accum: DefaultDict[int, DefaultDict[int, float]] = defaultdict(lambda: defaultdict(float))
        margin_rule_support: DefaultDict[PairKey, int] = defaultdict(int)
        rows_seen = 0
        rows_used = 0
        pair_candidates = 0
        margin_candidates = 0
        seen_records = set()

        normalized_provenance_mode = str(provenance_mode or "off").strip().lower()
        if normalized_provenance_mode not in {"off", "augment", "require"}:
            normalized_provenance_mode = "off"

        existing_trace_paths = tuple(trace_path for trace_path in trace_paths if trace_path.exists())
        if not existing_trace_paths:
            summary = LLMTraceDistillSummary(
                path=str(trace_path),
                rows_seen=0,
                rows_used=0,
                query_targets=0,
                entity_pair_targets=0,
                pair_rules=0,
                pair_candidates=0,
                source_field=str(source_field),
                min_delta=float(min_delta),
                pair_min_support=int(pair_min_support),
                pair_min_precision=float(pair_min_precision),
                pair_min_reward=float(pair_min_reward),
                pair_max_neutral_rate=float(pair_max_neutral_rate),
                confidence_weighting=bool(confidence_weighting),
                margin_targets=0,
                margin_pairs=0,
                margin_candidates=0,
                margin_min_neg_delta=float(margin_min_neg_delta),
                margin_max_negatives_per_target=int(margin_max_negatives_per_target),
                margin_scope=str(margin_scope or "relation"),
                margin_rule_targets=0,
                margin_rule_pairs=0,
                margin_rule_min_support=int(margin_rule_min_support),
                margin_rule_min_weight=float(margin_rule_min_weight),
                margin_rule_min_precision=float(margin_rule_min_precision),
                dedupe_records=bool(dedupe_records),
                provenance_mode=normalized_provenance_mode,
                provenance_hypothesis_path=str(provenance_hypothesis_path or ""),
                provenance_llm_rules=0,
                provenance_accepted_rules=0,
                provenance_qualified_candidates=0,
                provenance_pair_rules=0,
                provenance_context_rules=0,
                provenance_pair_min_support=int(provenance_pair_min_support),
                provenance_pair_min_precision=float(provenance_pair_min_precision),
                provenance_pair_min_reward=float(provenance_pair_min_reward),
                provenance_pair_max_neutral_rate=float(provenance_pair_max_neutral_rate),
            )
            return cls(
                num_rels=num_rels,
                query_targets={},
                entity_pair_targets={},
                pair_weights={},
                provenance_pair_weights={},
                provenance_context_weights={},
                margin_negatives={},
                margin_relation_negatives={},
                summary=summary,
                confidence_weighting=bool(confidence_weighting),
                margin_scope=str(margin_scope or "relation"),
            )

        provenance_index = None
        provenance_rule_prev_rel: Dict[str, int] = {}
        if normalized_provenance_mode != "off" and str(provenance_hypothesis_path or "").strip():
            provenance_index = LLMHypothesisProvenance.from_trace_paths(
                existing_trace_paths,
                hypothesis_path=str(provenance_hypothesis_path),
                max_rows=int(max_rows),
                holdout_fraction=float(provenance_holdout_fraction),
                min_support=int(provenance_min_support),
                min_fixes=int(provenance_min_fixes),
                min_precision=float(provenance_min_precision),
                min_reward=float(provenance_min_reward),
                max_fail_rate=float(provenance_max_fail_rate),
                holdout_min_support=int(provenance_holdout_min_support),
                holdout_min_fixes=int(provenance_holdout_min_fixes),
                holdout_min_precision=float(provenance_holdout_min_precision),
                holdout_min_reward=float(provenance_holdout_min_reward),
                holdout_max_fail_rate=float(provenance_holdout_max_fail_rate),
            )
            print(provenance_index.format_summary())
            provenance_rule_prev_rel = {
                str(rule.get("id", "")): int(_as_int(rule.get("prev_rel"), -1))
                for rule in provenance_index.accepted_rules
                if str(rule.get("id", ""))
            }

        for trace_path in existing_trace_paths:
            with trace_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    rows_seen += 1
                    if max_rows > 0 and rows_used >= int(max_rows):
                        break
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    query = record.get("query") if isinstance(record, dict) else None
                    candidates = record.get("candidates") if isinstance(record, dict) else None
                    if not isinstance(query, dict) or not isinstance(candidates, list):
                        continue
                    s = _as_int(query.get("s"))
                    o = _as_int(query.get("o"))
                    t = _as_int(query.get("t"))
                    if s is None or o is None or t is None:
                        continue
                    if dedupe_records:
                        record_key = (
                            s,
                            o,
                            t,
                            tuple(int(rid) for rid in (record.get("gt") or []) if _as_int(rid) is not None),
                            tuple(int(rid) for rid in (record.get("base_order") or []) if _as_int(rid) is not None),
                            _as_int(record.get("target_relation_id"), -1),
                        )
                        if record_key in seen_records:
                            continue
                        seen_records.add(record_key)
                    source_rel = _as_int(record.get(source_field))
                    if source_rel is None or source_rel < 0 or source_rel >= output_rels:
                        source_rel = _as_int(record.get("base_top"))
                    rows_used += 1
                    parsed_candidates = []
                    for candidate in candidates:
                        if not isinstance(candidate, dict):
                            continue
                        rid = _as_int(candidate.get("rid"))
                        if rid is None or rid < 0 or rid >= output_rels:
                            continue
                        delta = _as_float(candidate.get("step_mrr_delta"), 0.0)
                        candidate_source = source_rel
                        if record.get("reward_schema") == TARGET_SCHEMA:
                            candidate_source = _as_int(candidate.get("step_source_relation_id"))
                            if abs(delta) > 1e-12 and candidate_source is None:
                                raise ValueError("Target-filter action is missing its actual adjacent opponent")
                        is_gt = bool(candidate.get("is_gt", False))
                        provenance = provenance_index.evidence_for(record, int(rid)) if provenance_index is not None else None
                        base_signal_allowed = normalized_provenance_mode != "require" or provenance is not None
                        parsed = {
                            "rid": int(rid),
                            "delta": float(delta),
                            "is_gt": bool(is_gt),
                            "rank": _candidate_rank(candidate),
                            "source_rel": candidate_source,
                            "provenance": provenance,
                            "base_signal_allowed": bool(base_signal_allowed),
                        }
                        parsed_candidates.append(parsed)
                        if base_signal_allowed and is_gt and delta > float(min_delta):
                            query_accum[(s, o, t)][rid] += max(delta, float(min_delta))
                            entity_pair_accum[(s, o)][rid] += max(delta, float(min_delta))
                            if mirror_inverse:
                                mirror = cls._mirror_relation(rid, int(num_rels))
                                if mirror is not None:
                                    query_accum[(o, s, t)][mirror] += max(delta, float(min_delta))
                                    entity_pair_accum[(o, s)][mirror] += max(delta, float(min_delta))
                        if candidate_source is None or candidate_source < 0 or candidate_source >= output_rels:
                            continue
                        if int(candidate_source) == int(rid):
                            continue
                        key = (int(candidate_source), int(rid))
                        if base_signal_allowed:
                            pair_candidates += 1
                            if delta > float(min_delta):
                                pair_pos[key] += 1
                                pair_reward[key] += delta
                            elif delta < -float(min_delta):
                                pair_neg[key] += 1
                                pair_reward[key] += delta
                            else:
                                pair_neu[key] += 1
                        if provenance is not None:
                            if delta > float(min_delta):
                                provenance_pair_pos[key] += 1
                                provenance_pair_reward[key] += delta * max(0.0, float(provenance.confidence))
                            elif delta < -float(min_delta):
                                provenance_pair_neg[key] += 1
                                provenance_pair_reward[key] += delta
                            else:
                                provenance_pair_neu[key] += 1
                            context_prev_rels = {
                                provenance_rule_prev_rel[rule_id]
                                for rule_id in provenance.rule_ids
                                if rule_id in provenance_rule_prev_rel
                            }
                            for context_prev_rel in context_prev_rels:
                                context_key = (int(context_prev_rel), int(candidate_source), int(rid))
                                if delta > float(min_delta):
                                    provenance_context_pos[context_key] += 1
                                    provenance_context_reward[context_key] += delta * max(
                                        0.0,
                                        float(provenance.confidence),
                                    )
                                elif delta < -float(min_delta):
                                    provenance_context_neg[context_key] += 1
                                    provenance_context_reward[context_key] += delta
                                else:
                                    provenance_context_neu[context_key] += 1
                    positive_candidates = [
                        item for item in parsed_candidates
                        if bool(item["base_signal_allowed"])
                        and bool(item["is_gt"])
                        and float(item["delta"]) > float(min_delta)
                    ]
                    if positive_candidates:
                        for positive in positive_candidates:
                            pos_rel = int(positive["rid"])
                            pos_rank = positive.get("rank")
                            for negative in parsed_candidates:
                                neg_rel = int(negative["rid"])
                                if bool(negative["is_gt"]) or neg_rel == pos_rel:
                                    continue
                                neg_delta = float(negative["delta"])
                                weight = 0.0
                                if neg_delta < -float(margin_min_neg_delta):
                                    weight = max(weight, min(1.0, abs(neg_delta)))
                                neg_rank = negative.get("rank")
                                if (
                                    float(margin_rank_above_weight) > 0
                                    and pos_rank is not None
                                    and neg_rank is not None
                                    and int(neg_rank) < int(pos_rank)
                                ):
                                    weight = max(weight, float(margin_rank_above_weight))
                                positive_source = positive["source_rel"]
                                if positive_source is not None and 0 <= int(positive_source) < output_rels and int(positive_source) == neg_rel:
                                    weight = max(weight, float(margin_source_weight))
                                if weight <= 0:
                                    continue
                                margin_candidates += 1
                                margin_accum[(s, o, pos_rel)][neg_rel] += weight
                                margin_rule_accum[pos_rel][neg_rel] += weight
                                margin_rule_support[(pos_rel, neg_rel)] += 1
                                if mirror_inverse:
                                    pos_mirror = cls._mirror_relation(pos_rel, int(num_rels))
                                    neg_mirror = cls._mirror_relation(neg_rel, int(num_rels))
                                    if pos_mirror is not None and neg_mirror is not None and pos_mirror != neg_mirror:
                                        margin_accum[(o, s, pos_mirror)][neg_mirror] += weight
                                        margin_rule_accum[pos_mirror][neg_mirror] += weight
                                        margin_rule_support[(pos_mirror, neg_mirror)] += 1
            if max_rows > 0 and rows_used >= int(max_rows):
                break

        query_targets = cls._trim_query_targets(
            query_accum,
            max_targets_per_query=max_targets_per_query,
            confidence_weighting=bool(confidence_weighting),
        )
        entity_pair_targets = cls._trim_query_targets(
            entity_pair_accum,
            max_targets_per_query=max_targets_per_query,
            confidence_weighting=bool(confidence_weighting),
        )
        pair_weights = cls._accept_pair_weights(
            pair_pos,
            pair_neg,
            pair_neu,
            pair_reward,
            num_rels=int(num_rels),
            min_support=int(pair_min_support),
            min_precision=float(pair_min_precision),
            min_reward=float(pair_min_reward),
            max_neutral_rate=float(pair_max_neutral_rate),
            mirror_inverse=bool(mirror_inverse),
        )
        provenance_pair_weights = cls._accept_pair_weights(
            provenance_pair_pos,
            provenance_pair_neg,
            provenance_pair_neu,
            provenance_pair_reward,
            num_rels=int(num_rels),
            min_support=int(provenance_pair_min_support),
            min_precision=float(provenance_pair_min_precision),
            min_reward=float(provenance_pair_min_reward),
            max_neutral_rate=float(provenance_pair_max_neutral_rate),
            mirror_inverse=bool(mirror_inverse),
        )
        provenance_context_weights = cls._accept_context_pair_weights(
            provenance_context_pos,
            provenance_context_neg,
            provenance_context_neu,
            provenance_context_reward,
            num_rels=int(num_rels),
            min_support=int(provenance_pair_min_support),
            min_precision=float(provenance_pair_min_precision),
            min_reward=float(provenance_pair_min_reward),
            max_neutral_rate=float(provenance_pair_max_neutral_rate),
            mirror_inverse=bool(mirror_inverse),
        )
        margin_negatives = cls._trim_margin_negatives(
            margin_accum,
            max_negatives_per_target=int(margin_max_negatives_per_target),
        )
        margin_pairs = sum(len(value) for value in margin_negatives.values())
        margin_relation_negatives = cls._trim_relation_margin_negatives(
            margin_rule_accum,
            margin_rule_support,
            max_negatives_per_target=int(margin_max_negatives_per_target),
            min_support=int(margin_rule_min_support),
            min_weight=float(margin_rule_min_weight),
            min_precision=float(margin_rule_min_precision),
        )
        margin_rule_pairs = sum(len(value) for value in margin_relation_negatives.values())
        summary = LLMTraceDistillSummary(
            path=",".join(str(trace_path) for trace_path in existing_trace_paths),
            rows_seen=rows_seen,
            rows_used=rows_used,
            query_targets=len(query_targets),
            entity_pair_targets=len(entity_pair_targets),
            pair_rules=len(pair_weights),
            pair_candidates=pair_candidates,
            source_field=str(source_field),
            min_delta=float(min_delta),
            pair_min_support=int(pair_min_support),
            pair_min_precision=float(pair_min_precision),
            pair_min_reward=float(pair_min_reward),
            pair_max_neutral_rate=float(pair_max_neutral_rate),
            confidence_weighting=bool(confidence_weighting),
            margin_targets=len(margin_negatives),
            margin_pairs=margin_pairs,
            margin_candidates=margin_candidates,
            margin_min_neg_delta=float(margin_min_neg_delta),
            margin_max_negatives_per_target=int(margin_max_negatives_per_target),
            margin_scope=str(margin_scope or "relation"),
            margin_rule_targets=len(margin_relation_negatives),
            margin_rule_pairs=margin_rule_pairs,
            margin_rule_min_support=int(margin_rule_min_support),
            margin_rule_min_weight=float(margin_rule_min_weight),
            margin_rule_min_precision=float(margin_rule_min_precision),
            dedupe_records=bool(dedupe_records),
            provenance_mode=normalized_provenance_mode,
            provenance_hypothesis_path=str(provenance_hypothesis_path or ""),
            provenance_llm_rules=int(provenance_index.summary.llm_rules) if provenance_index is not None else 0,
            provenance_accepted_rules=int(provenance_index.summary.accepted_rules) if provenance_index is not None else 0,
            provenance_qualified_candidates=int(provenance_index.summary.qualified_candidates) if provenance_index is not None else 0,
            provenance_pair_rules=len(provenance_pair_weights),
            provenance_context_rules=len(provenance_context_weights),
            provenance_pair_min_support=int(provenance_pair_min_support),
            provenance_pair_min_precision=float(provenance_pair_min_precision),
            provenance_pair_min_reward=float(provenance_pair_min_reward),
            provenance_pair_max_neutral_rate=float(provenance_pair_max_neutral_rate),
        )
        return cls(
            num_rels=num_rels,
            query_targets=query_targets,
            entity_pair_targets=entity_pair_targets,
            pair_weights=pair_weights,
            provenance_pair_weights=provenance_pair_weights,
            provenance_context_weights=provenance_context_weights,
            margin_negatives=margin_negatives,
            margin_relation_negatives=margin_relation_negatives,
            summary=summary,
            confidence_weighting=bool(confidence_weighting),
            margin_scope=str(margin_scope or "relation"),
        )

    @staticmethod
    def _mirror_relation(rid: int, num_rels: int) -> Optional[int]:
        rid = int(rid)
        num_rels = int(num_rels)
        if 0 <= rid < num_rels:
            return rid + num_rels
        if num_rels <= rid < num_rels * 2:
            return rid - num_rels
        return None

    @staticmethod
    def _trim_query_targets(
        query_accum: Mapping[Tuple[int, ...], Mapping[int, float]],
        *,
        max_targets_per_query: int,
        confidence_weighting: bool,
    ) -> Dict[Tuple[int, ...], Dict[int, float]]:
        out: Dict[Tuple[int, ...], Dict[int, float]] = {}
        limit = max(1, int(max_targets_per_query))
        for key, values in query_accum.items():
            ordered = sorted(values.items(), key=lambda item: float(item[1]), reverse=True)[:limit]
            total = sum(max(0.0, float(weight)) for _, weight in ordered)
            if total <= 0:
                continue
            if confidence_weighting:
                out[key] = {int(rid): max(0.0, float(weight)) for rid, weight in ordered}
            else:
                out[key] = {int(rid): max(0.0, float(weight)) / total for rid, weight in ordered}
        return out

    @staticmethod
    def _trim_margin_negatives(
        margin_accum: Mapping[MarginKey, Mapping[int, float]],
        *,
        max_negatives_per_target: int,
    ) -> Dict[MarginKey, Dict[int, float]]:
        out: Dict[MarginKey, Dict[int, float]] = {}
        limit = max(1, int(max_negatives_per_target))
        for key, values in margin_accum.items():
            pos_rel = int(key[2])
            ordered = [
                (int(neg_rel), max(0.0, float(weight)))
                for neg_rel, weight in values.items()
                if int(neg_rel) != pos_rel and float(weight) > 0
            ]
            ordered = sorted(ordered, key=lambda item: float(item[1]), reverse=True)[:limit]
            if not ordered:
                continue
            max_weight = max(float(weight) for _, weight in ordered)
            scale = max(1.0, max_weight)
            out[(int(key[0]), int(key[1]), pos_rel)] = {
                int(neg_rel): min(1.0, float(weight) / scale)
                for neg_rel, weight in ordered
                if float(weight) > 0
            }
        return out

    @staticmethod
    def _trim_relation_margin_negatives(
        margin_rule_accum: Mapping[int, Mapping[int, float]],
        margin_rule_support: Mapping[PairKey, int],
        *,
        max_negatives_per_target: int,
        min_support: int,
        min_weight: float,
        min_precision: float = 0.0,
    ) -> Dict[int, Dict[int, float]]:
        out: Dict[int, Dict[int, float]] = {}
        limit = max(1, int(max_negatives_per_target))
        support_floor = max(1, int(min_support))
        weight_floor = max(0.0, float(min_weight))
        precision_floor = max(0.0, min(1.0, float(min_precision)))
        for pos_rel, values in margin_rule_accum.items():
            ordered = []
            for neg_rel, total_weight in values.items():
                if int(neg_rel) == int(pos_rel):
                    continue
                support = int(margin_rule_support.get((int(pos_rel), int(neg_rel)), 0))
                if support < support_floor:
                    continue
                reverse_support = int(margin_rule_support.get((int(neg_rel), int(pos_rel)), 0))
                precision = float(support) / float(max(1, support + reverse_support))
                if precision_floor > 0 and precision < precision_floor:
                    continue
                avg_weight = float(total_weight) / float(max(1, support))
                if avg_weight < weight_floor:
                    continue
                ordered.append((int(neg_rel), min(1.0, avg_weight)))
            ordered = sorted(ordered, key=lambda item: float(item[1]), reverse=True)[:limit]
            if ordered:
                out[int(pos_rel)] = {int(neg_rel): float(weight) for neg_rel, weight in ordered}
        return out

    @classmethod
    def _accept_context_pair_weights(
        cls,
        context_pos: Mapping[ContextPairKey, int],
        context_neg: Mapping[ContextPairKey, int],
        context_neu: Mapping[ContextPairKey, int],
        context_reward: Mapping[ContextPairKey, float],
        *,
        num_rels: int,
        min_support: int,
        min_precision: float,
        min_reward: float,
        max_neutral_rate: float,
        mirror_inverse: bool,
    ) -> Dict[ContextPairKey, float]:
        out: Dict[ContextPairKey, float] = {}
        prev_rels = {int(key[0]) for key in context_pos}
        for prev_rel in prev_rels:
            positives = {
                (int(src), int(dst)): int(value)
                for (context, src, dst), value in context_pos.items()
                if int(context) == prev_rel
            }
            negatives = {
                (int(src), int(dst)): int(value)
                for (context, src, dst), value in context_neg.items()
                if int(context) == prev_rel
            }
            neutrals = {
                (int(src), int(dst)): int(value)
                for (context, src, dst), value in context_neu.items()
                if int(context) == prev_rel
            }
            rewards = {
                (int(src), int(dst)): float(value)
                for (context, src, dst), value in context_reward.items()
                if int(context) == prev_rel
            }
            accepted = cls._accept_pair_weights(
                positives,
                negatives,
                neutrals,
                rewards,
                num_rels=int(num_rels),
                min_support=int(min_support),
                min_precision=float(min_precision),
                min_reward=float(min_reward),
                max_neutral_rate=float(max_neutral_rate),
                mirror_inverse=False,
            )
            for (src, dst), weight in accepted.items():
                out[(int(prev_rel), int(src), int(dst))] = float(weight)
                if not mirror_inverse:
                    continue
                src_mirror = cls._mirror_relation(int(src), int(num_rels))
                dst_mirror = cls._mirror_relation(int(dst), int(num_rels))
                prev_mirror = -1 if int(prev_rel) == -1 else cls._mirror_relation(int(prev_rel), int(num_rels))
                if src_mirror is None or dst_mirror is None or prev_mirror is None:
                    continue
                out[(int(prev_mirror), int(src_mirror), int(dst_mirror))] = max(
                    out.get((int(prev_mirror), int(src_mirror), int(dst_mirror)), 0.0),
                    float(weight),
                )
        return out

    @classmethod
    def _accept_pair_weights(
        cls,
        pair_pos: Mapping[PairKey, int],
        pair_neg: Mapping[PairKey, int],
        pair_neu: Mapping[PairKey, int],
        pair_reward: Mapping[PairKey, float],
        *,
        num_rels: int,
        min_support: int,
        min_precision: float,
        min_reward: float,
        max_neutral_rate: float,
        mirror_inverse: bool,
    ) -> Dict[PairKey, float]:
        out: Dict[PairKey, float] = {}
        for key, positives in pair_pos.items():
            negatives = int(pair_neg.get(key, 0))
            neutrals = int(pair_neu.get(key, 0))
            total = int(positives) + negatives + neutrals
            if total <= 0:
                continue
            precision_denom = int(positives) + negatives
            precision = float(positives) / float(max(1, precision_denom))
            neutral_rate = float(neutrals) / float(max(1, total))
            reward = float(pair_reward.get(key, 0.0))
            if int(positives) < int(min_support):
                continue
            if precision < float(min_precision):
                continue
            if reward < float(min_reward):
                continue
            if neutral_rate > float(max_neutral_rate):
                continue
            weight = max(0.0, min(1.0, precision * min(1.0, reward / max(1.0, float(positives)))))
            if weight <= 0:
                continue
            out[(int(key[0]), int(key[1]))] = weight
            if mirror_inverse:
                src_mirror = cls._mirror_relation(int(key[0]), int(num_rels))
                dst_mirror = cls._mirror_relation(int(key[1]), int(num_rels))
                if src_mirror is not None and dst_mirror is not None and src_mirror != dst_mirror:
                    out[(src_mirror, dst_mirror)] = max(out.get((src_mirror, dst_mirror), 0.0), weight)
        return out

    def has_signal(self) -> bool:
        return bool(
            self.query_targets
            or self.entity_pair_targets
            or self.pair_weights
            or self.provenance_pair_weights
            or self.provenance_context_weights
            or self.margin_negatives
            or self.margin_relation_negatives
        )

    def format_summary(self) -> str:
        data = self.summary.as_dict()
        return (
            "[LLMTraceDistillTeacher] "
            f"path={data['path']} rows={data['rows_used']}/{data['rows_seen']} "
            f"query_targets={data['query_targets']} entity_pair_targets={data['entity_pair_targets']} "
            f"pair_rules={data['pair_rules']} "
            f"pair_candidates={data['pair_candidates']} source={data['source_field']} "
            f"delta>{data['min_delta']} pair_gate={data['pair_min_support']}/"
            f"{data['pair_min_precision']}/{data['pair_min_reward']} "
            f"neutral<={data['pair_max_neutral_rate']} "
            f"confidence_weighting={int(bool(data['confidence_weighting']))} "
            f"margin_targets={data['margin_targets']} margin_pairs={data['margin_pairs']} "
            f"margin_candidates={data['margin_candidates']} "
            f"margin_neg_delta>{data['margin_min_neg_delta']} "
            f"margin_max_neg={data['margin_max_negatives_per_target']} "
            f"margin_scope={data['margin_scope']} "
            f"margin_rule_targets={data['margin_rule_targets']} "
            f"margin_rule_pairs={data['margin_rule_pairs']} "
            f"margin_rule_gate={data['margin_rule_min_support']}/{data['margin_rule_min_weight']}/"
            f"{data['margin_rule_min_precision']} "
            f"dedupe={int(bool(data['dedupe_records']))} "
            f"provenance={data['provenance_mode']} rules={data['provenance_accepted_rules']}/"
            f"{data['provenance_llm_rules']} qualified={data['provenance_qualified_candidates']} "
            f"provenance_pairs={data['provenance_pair_rules']} "
            f"provenance_contexts={data['provenance_context_rules']} "
            f"gate={data['provenance_pair_min_support']}/"
            f"{data['provenance_pair_min_precision']}/{data['provenance_pair_min_reward']}/"
            f"neutral<={data['provenance_pair_max_neutral_rate']}"
        )

    def pair_weight_matrix(self, *, device: Union[torch.device, str], dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
        if not self.pair_weights:
            return None
        if self._pair_matrix_cpu is None:
            matrix = torch.zeros((self.output_rels, self.output_rels), dtype=torch.float32)
            for (src, dst), weight in self.pair_weights.items():
                if 0 <= src < self.output_rels and 0 <= dst < self.output_rels:
                    matrix[src, dst] = max(float(matrix[src, dst].item()), float(weight))
            self._pair_matrix_cpu = matrix
        return self._pair_matrix_cpu.to(device=device, dtype=dtype)

    def provenance_pair_weight_matrix(
        self,
        *,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        if not self.provenance_pair_weights:
            return None
        if self._provenance_pair_matrix_cpu is None:
            matrix = torch.zeros((self.output_rels, self.output_rels), dtype=torch.float32)
            for (src, dst), weight in self.provenance_pair_weights.items():
                if 0 <= src < self.output_rels and 0 <= dst < self.output_rels:
                    matrix[src, dst] = max(float(matrix[src, dst].item()), float(weight))
            self._provenance_pair_matrix_cpu = matrix
        return self._provenance_pair_matrix_cpu.to(device=device, dtype=dtype)

    def provenance_source_weights_for_triples(
        self,
        triples: torch.Tensor,
        relation_history: torch.Tensor,
        *,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """Compile exact provenance context into per-row allowed source relations."""
        if not self.provenance_context_weights or triples is None or relation_history is None:
            return None
        if triples.numel() == 0 or relation_history.numel() == 0:
            return None
        triples_cpu = triples.detach().cpu()
        history_cpu = relation_history.detach().cpu()
        if history_cpu.dim() != 2 or int(history_cpu.size(0)) != int(triples_cpu.size(0)):
            return None
        source_weights = torch.zeros((int(triples_cpu.size(0)), self.output_rels), dtype=torch.float32)
        matched = 0
        for row_idx, row in enumerate(triples_cpu):
            gold_rel = int(row[1].item())
            contexts = self._provenance_context_by_target.get(gold_rel, ())
            if not contexts:
                continue
            history_row = history_cpu[row_idx]
            history_total = float(history_row.sum().item())
            row_matched = False
            for prev_rel, source_rel, weight in contexts:
                previous_relation_seen = (
                    float(history_row[prev_rel].item())
                    if 0 <= prev_rel < int(history_row.numel())
                    else 0.0
                )
                context_matches = relation_history_context_matches(
                    previous_relation=prev_rel,
                    history_total=history_total,
                    previous_relation_seen=previous_relation_seen,
                )
                if not context_matches:
                    continue
                source_weights[row_idx, source_rel] = max(
                    float(source_weights[row_idx, source_rel].item()),
                    float(weight),
                )
                row_matched = True
            if row_matched:
                matched += 1
        if matched <= 0:
            return None
        return source_weights.to(device=device, dtype=dtype)

    def targets_for_triples(
        self,
        triples: torch.Tensor,
        *,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        if (not self.query_targets and not self.entity_pair_targets) or triples is None or triples.numel() == 0:
            return None
        triples_cpu = triples.detach().cpu()
        target = torch.zeros((int(triples_cpu.size(0)), self.output_rels), dtype=torch.float32)
        matched = 0
        for row_idx, row in enumerate(triples_cpu):
            s = int(row[0].item())
            gold_rel = int(row[1].item())
            o = int(row[2].item())
            t = int(row[3].item())
            key = (s, o, t)
            values = self.query_targets.get(key)
            if not values or gold_rel not in values:
                values = self.entity_pair_targets.get((s, o))
            if not values or gold_rel not in values:
                continue
            confidence = max(0.0, float(values.get(gold_rel, 0.0)))
            target[row_idx, gold_rel] = confidence if self.confidence_weighting else 1.0
            matched += 1
        if matched <= 0:
            return None
        return target.to(device=device, dtype=dtype)

    def margin_negatives_for_triples(
        self,
        triples: torch.Tensor,
        *,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        if (not self.margin_negatives and not self.margin_relation_negatives) or triples is None or triples.numel() == 0:
            return None
        triples_cpu = triples.detach().cpu()
        negatives = torch.zeros((int(triples_cpu.size(0)), self.output_rels), dtype=torch.float32)
        matched = 0
        use_entity = self.margin_scope in {"entity", "entity_pair", "both", "all"}
        use_relation = self.margin_scope in {"relation", "rel", "both", "all"}
        for row_idx, row in enumerate(triples_cpu):
            s = int(row[0].item())
            gold_rel = int(row[1].item())
            o = int(row[2].item())
            if use_entity:
                key = (s, o, gold_rel)
                values = self.margin_negatives.get(key)
                if values:
                    for neg_rel, weight in values.items():
                        if 0 <= int(neg_rel) < self.output_rels and int(neg_rel) != gold_rel:
                            negatives[row_idx, int(neg_rel)] = max(
                                float(negatives[row_idx, int(neg_rel)].item()),
                                float(weight),
                            )
            if use_relation:
                values = self.margin_relation_negatives.get(gold_rel)
                if values:
                    for neg_rel, weight in values.items():
                        if 0 <= int(neg_rel) < self.output_rels and int(neg_rel) != gold_rel:
                            negatives[row_idx, int(neg_rel)] = max(
                                float(negatives[row_idx, int(neg_rel)].item()),
                                float(weight),
                            )
            if float(negatives[row_idx].sum().item()) > 0:
                matched += 1
        if matched <= 0:
            return None
        return negatives.to(device=device, dtype=dtype)


def iter_top_pair_rules(teacher: LLMTraceDistillTeacher, limit: int = 12) -> Iterable[Tuple[PairKey, float]]:
    ordered = sorted(teacher.pair_weights.items(), key=lambda item: float(item[1]), reverse=True)
    return ordered[: max(0, int(limit))]
