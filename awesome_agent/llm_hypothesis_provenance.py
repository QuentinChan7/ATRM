from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RecordKey = Tuple[int, int, int, int, Tuple[int, ...], int]
EvidenceKey = Tuple[RecordKey, int]


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _resolve_path(path: str) -> Path:
    raw = Path(str(path or "")).expanduser()
    if raw.is_absolute():
        return raw
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (Path.cwd() / raw, repo_root / raw, repo_root / "src" / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / raw).resolve()


def _record_key(record: Mapping[str, Any]) -> Optional[RecordKey]:
    query = record.get("query")
    if not isinstance(query, Mapping):
        return None
    s = _as_int(query.get("s"))
    o = _as_int(query.get("o"))
    t = _as_int(query.get("t"))
    base_top = _as_int(record.get("base_top"), -1)
    if s is None or o is None or t is None or base_top is None:
        return None
    base_order = tuple(
        int(value)
        for value in list(record.get("base_order") or [])
        if _as_int(value) is not None
    )
    return int(s), int(o), int(t), int(base_top), base_order, int(_as_int(record.get("target_relation_id"), -1))


@dataclass(frozen=True)
class LLMProvenanceEvidence:
    rule_ids: Tuple[str, ...]
    confidence: float
    fit_precision: float
    fit_reward: float
    holdout_precision: float
    holdout_reward: float


@dataclass(frozen=True)
class LLMProvenanceSummary:
    hypothesis_path: str
    llm_rules: int
    accepted_rules: int
    fit_time_groups: int
    holdout_time_groups: int
    matched_candidates: int
    qualified_candidates: int
    min_support: int
    min_fixes: int
    min_precision: float
    min_reward: float
    max_fail_rate: float
    holdout_min_support: int
    holdout_min_fixes: int
    holdout_min_precision: float
    holdout_min_reward: float
    holdout_max_fail_rate: float


class LLMHypothesisProvenance:
    """Cross-fitted provenance index for candidates selected by LLM hypotheses."""

    def __init__(
        self,
        *,
        evidence: Mapping[EvidenceKey, LLMProvenanceEvidence],
        accepted_rules: Sequence[Mapping[str, Any]],
        summary: LLMProvenanceSummary,
        rule_audits: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self._evidence = dict(evidence)
        self.accepted_rules = tuple(dict(rule) for rule in accepted_rules)
        self.summary = summary
        self.rule_audits = tuple(dict(row) for row in rule_audits)

    @classmethod
    def from_trace_paths(
        cls,
        trace_paths: Sequence[Path],
        *,
        hypothesis_path: str,
        max_rows: int = 0,
        holdout_fraction: float = 0.25,
        min_support: int = 6,
        min_fixes: int = 2,
        min_precision: float = 0.70,
        min_reward: float = 0.02,
        max_fail_rate: float = 0.30,
        holdout_min_support: int = 2,
        holdout_min_fixes: int = 1,
        holdout_min_precision: float = 0.70,
        holdout_min_reward: float = 0.02,
        holdout_max_fail_rate: float = 0.0,
    ) -> "LLMHypothesisProvenance":
        resolved_hypothesis_path = _resolve_path(hypothesis_path)
        rules = cls._load_llm_rules(resolved_hypothesis_path)
        records = cls._load_records(trace_paths, max_rows=max_rows)
        known_rule_ids = {str(rule["id"]) for rule in rules}
        for record in records:
            for candidate in list(record.get("candidates") or []):
                if not isinstance(candidate, Mapping):
                    continue
                provenance = candidate.get("llm_provenance")
                if not isinstance(provenance, Mapping):
                    continue
                if str(provenance.get("source", "") or "") != "llm_reflection_hypothesis":
                    continue
                rule_id = str(provenance.get("rule_id", "") or "").strip()
                if not rule_id or rule_id in known_rule_ids:
                    continue
                known_rule_ids.add(rule_id)
                rules.append(
                    {
                        "id": rule_id,
                        "source": "llm_reflection_hypothesis",
                        "signature": "*",
                        "prev_rel": None,
                        "top1": None,
                        "choice": None,
                        "conditions": {},
                        "explicit_only": True,
                    }
                )
        time_groups = sorted(
            {
                int(key[2])
                for record in records
                for key in [_record_key(record)]
                if key is not None
            }
        )
        fraction = max(0.0, min(0.80, float(holdout_fraction)))
        holdout_count = max(1, int(math.ceil(len(time_groups) * fraction))) if len(time_groups) > 1 and fraction > 0 else 0
        holdout_groups = set(time_groups[-holdout_count:]) if holdout_count > 0 else set()

        stats: DefaultDict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "fit_support": 0,
                "fit_fixes": 0,
                "fit_fails": 0,
                "fit_reward_sum": 0.0,
                "holdout_support": 0,
                "holdout_fixes": 0,
                "holdout_fails": 0,
                "holdout_reward_sum": 0.0,
            }
        )
        matched_candidates = 0
        matches_by_record: DefaultDict[EvidenceKey, List[str]] = defaultdict(list)
        rule_by_id = {str(rule["id"]): dict(rule) for rule in rules}

        for record in records:
            key = _record_key(record)
            if key is None:
                continue
            is_holdout = int(key[2]) in holdout_groups
            for rule, rid, candidate in cls._iter_provenance_matches(record, rules):
                matched_candidates += 1
                rule_id = str(rule["id"])
                delta = _as_float(candidate.get("step_mrr_delta"), 0.0)
                prefix = "holdout" if is_holdout else "fit"
                row_stats = stats[rule_id]
                row_stats[f"{prefix}_support"] += 1
                row_stats[f"{prefix}_reward_sum"] += float(delta)
                if delta > 1e-12:
                    row_stats[f"{prefix}_fixes"] += 1
                elif delta < -1e-12:
                    row_stats[f"{prefix}_fails"] += 1
                matches_by_record[(key, int(rid))].append(rule_id)

        accepted: List[Dict[str, Any]] = []
        rule_audits: List[Dict[str, Any]] = []
        accepted_ids = set()
        for rule_id, rule in rule_by_id.items():
            row_stats = stats[rule_id]
            fit = cls._metrics(row_stats, "fit")
            holdout = cls._metrics(row_stats, "holdout")
            fit_ok = bool(
                fit["support"] >= max(1, int(min_support))
                and fit["fixes"] >= max(1, int(min_fixes))
                and fit["precision"] >= float(min_precision)
                and fit["reward"] >= float(min_reward)
                and fit["fail_rate"] <= float(max_fail_rate)
            )
            holdout_ok = bool(
                holdout["support"] >= max(1, int(holdout_min_support))
                and holdout["fixes"] >= max(1, int(holdout_min_fixes))
                and holdout["precision"] >= float(holdout_min_precision)
                and holdout["reward"] >= float(holdout_min_reward)
                and holdout["fail_rate"] <= float(holdout_max_fail_rate)
            )
            reasons = []
            for split, metrics, limits in (
                ("fit", fit, (min_support, min_fixes, min_precision, min_reward, max_fail_rate)),
                ("holdout", holdout, (holdout_min_support, holdout_min_fixes, holdout_min_precision, holdout_min_reward, holdout_max_fail_rate)),
            ):
                for name, limit in zip(("support", "fixes", "precision", "reward"), limits[:4]):
                    threshold = max(1, int(limit)) if name in {"support", "fixes"} else float(limit)
                    if metrics[name] < threshold:
                        reasons.append(f"{split}.{name}")
                if metrics["fail_rate"] > limits[4]:
                    reasons.append(f"{split}.fail_rate")
            rule_audits.append({"id": rule_id, "fit": fit, "holdout": holdout,
                                "accepted": fit_ok and holdout_ok, "rejected_by": reasons})
            if not (fit_ok and holdout_ok):
                continue
            accepted_rule = dict(rule)
            accepted_rule["provenance_validation"] = {"fit": fit, "holdout": holdout}
            accepted.append(accepted_rule)
            accepted_ids.add(rule_id)

        evidence: Dict[EvidenceKey, LLMProvenanceEvidence] = {}
        for evidence_key, rule_ids in matches_by_record.items():
            record_time = int(evidence_key[0][2])
            if record_time in holdout_groups:
                continue
            qualified_ids = sorted(set(rule_ids).intersection(accepted_ids))
            if not qualified_ids:
                continue
            fit_precision = max(cls._metrics(stats[rule_id], "fit")["precision"] for rule_id in qualified_ids)
            fit_reward = max(cls._metrics(stats[rule_id], "fit")["reward"] for rule_id in qualified_ids)
            holdout_precision = max(cls._metrics(stats[rule_id], "holdout")["precision"] for rule_id in qualified_ids)
            holdout_reward = max(cls._metrics(stats[rule_id], "holdout")["reward"] for rule_id in qualified_ids)
            confidence = max(0.0, min(1.0, min(fit_precision, holdout_precision)))
            evidence[evidence_key] = LLMProvenanceEvidence(
                rule_ids=tuple(qualified_ids),
                confidence=float(confidence),
                fit_precision=float(fit_precision),
                fit_reward=float(fit_reward),
                holdout_precision=float(holdout_precision),
                holdout_reward=float(holdout_reward),
            )

        summary = LLMProvenanceSummary(
            hypothesis_path=str(resolved_hypothesis_path),
            llm_rules=len(rules),
            accepted_rules=len(accepted),
            fit_time_groups=max(0, len(time_groups) - len(holdout_groups)),
            holdout_time_groups=len(holdout_groups),
            matched_candidates=int(matched_candidates),
            qualified_candidates=len(evidence),
            min_support=int(min_support),
            min_fixes=int(min_fixes),
            min_precision=float(min_precision),
            min_reward=float(min_reward),
            max_fail_rate=float(max_fail_rate),
            holdout_min_support=int(holdout_min_support),
            holdout_min_fixes=int(holdout_min_fixes),
            holdout_min_precision=float(holdout_min_precision),
            holdout_min_reward=float(holdout_min_reward),
            holdout_max_fail_rate=float(holdout_max_fail_rate),
        )
        return cls(evidence=evidence, accepted_rules=accepted, summary=summary, rule_audits=rule_audits)

    @staticmethod
    def _load_records(trace_paths: Sequence[Path], *, max_rows: int) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for trace_path in trace_paths:
            if not trace_path.exists():
                continue
            with trace_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if max_rows > 0 and len(records) >= int(max_rows):
                        return records
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(record, dict) and isinstance(record.get("query"), dict) and isinstance(record.get("candidates"), list):
                        records.append(record)
        return records

    @staticmethod
    def _load_llm_rules(path: Path) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        raw_rules = payload.get("hypotheses", payload.get("rules", [])) if isinstance(payload, dict) else payload
        if not isinstance(raw_rules, list):
            return []
        rules = []
        for idx, raw in enumerate(raw_rules):
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source", "llm_reflection_hypothesis") or "llm_reflection_hypothesis")
            if source not in {
                "llm_reflection_hypothesis", "statistical_anchor_selection", "random_anchor_selection"
            }:
                continue
            conditions = raw.get("conditions") if isinstance(raw.get("conditions"), dict) else {}
            rules.append(
                {
                    "id": str(raw.get("id", f"llm_hypothesis_{idx}") or f"llm_hypothesis_{idx}"),
                    "source": source,
                    "signature": str(raw.get("signature", "*") or "*"),
                    "prev_rel": _as_int(raw.get("prev_rel")),
                    "top1": _as_int(raw.get("top1")),
                    "choice": _as_int(raw.get("choice", raw.get("candidate_id"))),
                    "opponent": _as_int(raw.get("opponent")),
                    "conditions": dict(conditions),
                }
            )
        return rules

    @staticmethod
    def _metrics(stats: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
        support = int(stats.get(f"{prefix}_support", 0) or 0)
        fixes = int(stats.get(f"{prefix}_fixes", 0) or 0)
        fails = int(stats.get(f"{prefix}_fails", 0) or 0)
        reward_sum = _as_float(stats.get(f"{prefix}_reward_sum"), 0.0)
        return {
            "support": support,
            "fixes": fixes,
            "fails": fails,
            "neutral": max(0, support - fixes - fails),
            "precision": float(fixes / max(1, fixes + fails)),
            "fail_rate": float(fails / max(1, fixes + fails)),
            "reward": float(reward_sum / max(1, support)),
            "reward_sum": float(reward_sum),
        }

    @classmethod
    def _iter_matches(
        cls,
        record: Mapping[str, Any],
        rules: Sequence[Mapping[str, Any]],
    ) -> Iterable[Tuple[Dict[str, Any], int, Dict[str, Any]]]:
        candidates = [dict(candidate) for candidate in list(record.get("candidates") or []) if isinstance(candidate, dict)]
        candidate_by_id = {
            int(rid): candidate
            for candidate in candidates
            for rid in [_as_int(candidate.get("rid"))]
            if rid is not None
        }
        order = [int(rid) for rid in (_as_int(candidate.get("rid")) for candidate in candidates) if rid is not None]
        top1 = _as_int(record.get("base_top"), -1)
        prev_rel = _as_int(record.get("prev_rel"), -1)
        signature = str(record.get("signature", "") or "")
        for raw_rule in rules:
            rule = dict(raw_rule)
            if bool(rule.get("explicit_only", False)):
                continue
            fixed_choice = _as_int(rule.get("choice"))
            choices = [fixed_choice] if fixed_choice is not None else order
            for rid in choices:
                if rid is None or int(rid) == int(top1) or int(rid) not in candidate_by_id:
                    continue
                candidate = candidate_by_id[int(rid)]
                opponent = _as_int(rule.get("opponent"))
                if opponent is not None and _as_int(candidate.get("step_source_relation_id")) != opponent:
                    continue
                features = dict(candidate.get("features") or {})
                if cls._matches(
                    rule,
                    signature=signature,
                    prev_rel=int(prev_rel),
                    top1=int(top1),
                    choice=int(rid),
                    features=features,
                ):
                    yield rule, int(rid), candidate

    @classmethod
    def _iter_provenance_matches(
        cls,
        record: Mapping[str, Any],
        rules: Sequence[Mapping[str, Any]],
    ) -> Iterable[Tuple[Dict[str, Any], int, Dict[str, Any]]]:
        rule_by_id = {str(rule.get("id", "")): dict(rule) for rule in rules}
        seen = set()
        for candidate in list(record.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            rid = _as_int(candidate.get("rid"))
            provenance = candidate.get("llm_provenance")
            if rid is None or not isinstance(provenance, Mapping):
                continue
            if str(provenance.get("source", "") or "") != "llm_reflection_hypothesis":
                continue
            rule_id = str(provenance.get("rule_id", "") or "").strip()
            rule = rule_by_id.get(rule_id)
            if not rule_id or rule is None:
                continue
            seen.add((rule_id, int(rid)))
            yield dict(rule), int(rid), dict(candidate)
        for rule, rid, candidate in cls._iter_matches(record, rules):
            identity = (str(rule.get("id", "")), int(rid))
            if identity in seen:
                continue
            yield rule, rid, candidate

    @classmethod
    def _matches(
        cls,
        rule: Mapping[str, Any],
        *,
        signature: str,
        prev_rel: int,
        top1: int,
        choice: int,
        features: Mapping[str, Any],
    ) -> bool:
        if not cls._signature_matches(str(rule.get("signature", "*") or "*"), signature):
            return False
        for field, actual in (("prev_rel", prev_rel), ("top1", top1), ("choice", choice)):
            expected = _as_int(rule.get(field))
            if expected is not None and int(expected) != int(actual):
                return False
        for raw_name, raw_value in dict(rule.get("conditions") or {}).items():
            name = str(raw_name)
            if name == "baseline_protected":
                if bool(raw_value) != bool(cls._feature_value(features, "baseline_protected") > 0.5):
                    return False
                continue
            if name.startswith("min_"):
                if cls._feature_value(features, name[4:]) + 1e-12 < _as_float(raw_value):
                    return False
            elif name.startswith("max_"):
                if cls._feature_value(features, name[4:]) - 1e-12 > _as_float(raw_value):
                    return False
            elif name.endswith("_gte"):
                if cls._feature_value(features, name[:-4]) + 1e-12 < _as_float(raw_value):
                    return False
            elif name.endswith("_lte"):
                if cls._feature_value(features, name[:-4]) - 1e-12 > _as_float(raw_value):
                    return False
        return True

    @staticmethod
    def _signature_matches(rule_signature: str, actual_signature: str) -> bool:
        if rule_signature in {"", "*"}:
            return True
        if rule_signature == actual_signature:
            return True
        rule_parts = rule_signature.split("|")
        actual_parts = actual_signature.split("|")
        if len(rule_parts) != len(actual_parts):
            return False
        for expected, actual in zip(rule_parts, actual_parts):
            if expected in {"*", "prev=none", "prev=null"}:
                continue
            if expected != actual:
                return False
        return True

    @staticmethod
    def _feature_value(features: Mapping[str, Any], name: str) -> float:
        if name == "hard_source":
            return max(
                _as_float(features.get("direct")),
                _as_float(features.get("pair_transition")),
                _as_float(features.get("prior")),
            )
        if name == "risk":
            return _as_float(features.get("counterfactual_risk")) + _as_float(
                features.get("negative_penalty", features.get("neg_penalty"))
            )
        return _as_float(features.get(name), 0.0)

    def evidence_for(self, record: Mapping[str, Any], candidate_id: int) -> Optional[LLMProvenanceEvidence]:
        key = _record_key(record)
        if key is None:
            return None
        return self._evidence.get((key, int(candidate_id)))

    def format_summary(self) -> str:
        summary = self.summary
        return (
            "[LLMHypothesisProvenance] "
            f"path={summary.hypothesis_path} llm_rules={summary.llm_rules} "
            f"accepted={summary.accepted_rules} groups={summary.fit_time_groups}+{summary.holdout_time_groups} "
            f"matched={summary.matched_candidates} qualified={summary.qualified_candidates} "
            f"fit_gate={summary.min_support}/{summary.min_fixes}/{summary.min_precision}/{summary.min_reward}/"
            f"fail<={summary.max_fail_rate} holdout_gate={summary.holdout_min_support}/"
            f"{summary.holdout_min_fixes}/{summary.holdout_min_precision}/{summary.holdout_min_reward}/"
            f"fail<={summary.holdout_max_fail_rate}"
        )
