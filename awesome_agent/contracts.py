from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def dedup_ints(values: Iterable[Any], lower: Optional[int] = None, upper: Optional[int] = None) -> List[int]:
    out: List[int] = []
    seen = set()
    if values is None:
        return out
    for value in values:
        try:
            item = int(value)
        except Exception:
            continue
        if lower is not None and item < int(lower):
            continue
        if upper is not None and item >= int(upper):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


@dataclass(frozen=True)
class RelationQuery:
    """Closed-world relation prediction query: predict r in (s, r, o, t)."""

    s: int
    o: int
    t: int

    def as_key(self) -> Tuple[int, int, int]:
        return int(self.s), int(self.o), int(self.t)


@dataclass
class Candidate:
    candidate_id: int
    logit: float = 0.0
    source: str = "gnn"
    rank: int = 0


@dataclass
class CandidateSet:
    """A relation candidate set supplied by a System-1 provider such as a GNN."""

    candidates: List[Candidate] = field(default_factory=list)

    @classmethod
    def from_topk(cls, ids: Sequence[Any], logits: Optional[Sequence[Any]] = None, source: str = "gnn") -> "CandidateSet":
        ids_list = list(ids) if ids is not None else []
        logits_list = list(logits) if logits is not None else []
        items: List[Candidate] = []
        seen = set()
        for rank, cid in enumerate(ids_list):
            item = safe_int(cid, -1)
            if item < 0:
                continue
            if item in seen:
                continue
            seen.add(item)
            logit = safe_float(logits_list[rank], 0.0) if rank < len(logits_list) else 0.0
            items.append(Candidate(candidate_id=item, logit=logit, source=source, rank=rank))
        return cls(candidates=items)

    @property
    def ids(self) -> List[int]:
        return [int(x.candidate_id) for x in self.candidates]

    @property
    def logits(self) -> List[float]:
        return [float(x.logit) for x in self.candidates]

    @property
    def top1(self) -> int:
        return int(self.candidates[0].candidate_id) if self.candidates else -1

    def logit_of(self, candidate_id: int) -> float:
        cid = int(candidate_id)
        for cand in self.candidates:
            if int(cand.candidate_id) == cid:
                return float(cand.logit)
        return 0.0

    def top1_probability(self, cap: int = 6) -> float:
        vals = [float(x) for x in self.logits[: max(1, int(cap))]]
        if not vals:
            return 0.0
        max_v = max(vals)
        try:
            exps = [pow(2.718281828459045, v - max_v) for v in vals]
            denom = sum(exps)
            return float(exps[0] / denom) if denom > 0 else 0.0
        except Exception:
            return 0.0

    def margin(self) -> float:
        vals = self.logits
        if len(vals) < 2:
            return 0.0
        return float(vals[0] - vals[1])


@dataclass
class RelationEvidence:
    candidate_id: int
    work: float = 0.0
    direct: float = 0.0
    reciprocal: float = 0.0
    twohop: float = 0.0
    prior: float = 0.0
    transition: float = 0.0
    pair_transition: float = 0.0
    global_transition: float = 0.0
    cluster: float = 0.0
    event: float = 0.0
    commit: float = 0.0
    recall: float = 0.0
    semantic_fit: float = 0.0
    counterfactual_risk: float = 0.0
    verifier_support: float = 0.0
    verifier_penalty: float = 0.0
    reflection_policy: float = 0.0
    pop_penalty: float = 0.0
    neg_penalty: float = 0.0
    total: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, candidate_id: int, values: Mapping[str, Any]) -> "RelationEvidence":
        return cls(
            candidate_id=int(candidate_id),
            work=safe_float(values.get("work", 0.0)),
            direct=safe_float(values.get("direct", 0.0)),
            reciprocal=safe_float(values.get("reciprocal", 0.0)),
            twohop=safe_float(values.get("twohop", 0.0)),
            prior=safe_float(values.get("prior", 0.0)),
            transition=safe_float(values.get("transition", 0.0)),
            pair_transition=safe_float(values.get("pair_transition", values.get("transition", 0.0))),
            global_transition=safe_float(values.get("global_transition", 0.0)),
            cluster=safe_float(values.get("cluster", 0.0)),
            event=safe_float(values.get("event", 0.0)),
            commit=safe_float(values.get("commit", 0.0)),
            recall=safe_float(values.get("recall", 0.0)),
            semantic_fit=safe_float(values.get("semantic_fit", 0.0)),
            counterfactual_risk=safe_float(values.get("counterfactual_risk", 0.0)),
            verifier_support=safe_float(values.get("verifier_support", 0.0)),
            verifier_penalty=safe_float(values.get("verifier_penalty", 0.0)),
            reflection_policy=safe_float(values.get("reflection_policy", 0.0)),
            pop_penalty=safe_float(values.get("pop_penalty", 0.0)),
            neg_penalty=safe_float(values.get("neg_penalty", 0.0)),
            total=safe_float(values.get("total", 0.0)),
            provenance=dict(values.get("provenance", {}) or {}),
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "work": float(self.work),
            "direct": float(self.direct),
            "reciprocal": float(self.reciprocal),
            "twohop": float(self.twohop),
            "prior": float(self.prior),
            "transition": float(self.transition),
            "pair_transition": float(self.pair_transition),
            "global_transition": float(self.global_transition),
            "cluster": float(self.cluster),
            "event": float(self.event),
            "commit": float(self.commit),
            "recall": float(self.recall),
            "semantic_fit": float(self.semantic_fit),
            "counterfactual_risk": float(self.counterfactual_risk),
            "verifier_support": float(self.verifier_support),
            "verifier_penalty": float(self.verifier_penalty),
            "reflection_policy": float(self.reflection_policy),
            "pop_penalty": float(self.pop_penalty),
            "neg_penalty": float(self.neg_penalty),
            "total": float(self.total),
        }


@dataclass
class EvidenceBundle:
    query: RelationQuery
    candidate_set: CandidateSet
    by_id: Dict[int, RelationEvidence] = field(default_factory=dict)
    prev_rel: int = -1
    prev_t: int = -1

    def evidence(self, candidate_id: int) -> RelationEvidence:
        cid = int(candidate_id)
        if cid not in self.by_id:
            self.by_id[cid] = RelationEvidence(candidate_id=cid)
        return self.by_id[cid]

    def best_by_total(self) -> int:
        ids = self.candidate_set.ids
        if not ids:
            return -1
        return max(ids, key=lambda cid: self.evidence(cid).total)

    def as_support_dict(self) -> Dict[int, Dict[str, float]]:
        return {int(cid): ev.as_dict() for cid, ev in self.by_id.items()}


@dataclass
class ToolOutput:
    name: str
    scores: Dict[int, float] = field(default_factory=dict)
    penalties: Dict[int, float] = field(default_factory=dict)
    suggestions: List[int] = field(default_factory=list)
    summary: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def net(self, candidate_id: int) -> float:
        cid = int(candidate_id)
        return float(self.scores.get(cid, 0.0)) - float(self.penalties.get(cid, 0.0))


@dataclass
class ToolTrace:
    task: str = "rel"
    conflict_type: str = "stable"
    signature: str = "rel|stable"
    selected_tools: List[str] = field(default_factory=list)
    outputs: Dict[str, ToolOutput] = field(default_factory=dict)
    normalized_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)

    @property
    def suggestions(self) -> List[int]:
        return dedup_ints(x for out in self.outputs.values() for x in out.suggestions)

    def tool_net(self, candidate_id: int) -> float:
        return float(sum(out.net(int(candidate_id)) for out in self.outputs.values()))

    def as_feedback_record(self, query: RelationQuery, choice: int, top1: int) -> Dict[str, Any]:
        return {
            "task": "rel",
            "s": int(query.s),
            "o": int(query.o),
            "t": int(query.t),
            "choice": int(choice),
            "top1": int(top1),
            "final_action": "override" if int(choice) != int(top1) else "keep",
            "signature": str(self.signature),
            "selected_tools": list(self.selected_tools),
            "per_tool_scores": {name: dict(out.scores) for name, out in self.outputs.items()},
            "per_tool_penalties": {name: dict(out.penalties) for name, out in self.outputs.items()},
        }

    def as_legacy_dict(self) -> Dict[str, Any]:
        """Dict shape expected by the old 0407 writeback gate.

        The legacy evaluator does isinstance(..., dict) checks, so passing this
        structured dict is not optional when the refactored agent is run through
        that pipeline.
        """

        return {
            "task": str(self.task),
            "conflict_type": str(self.conflict_type),
            "signature": str(self.signature),
            "plan": list(self.selected_tools),
            "selected_tools": list(self.selected_tools),
            "outputs": {
                name: {
                    "scores": dict(out.scores),
                    "penalties": dict(out.penalties),
                    "suggestions": list(out.suggestions),
                    "summary": list(out.summary),
                    "metadata": dict(out.metadata),
                }
                for name, out in self.outputs.items()
            },
            "per_tool_scores": {name: dict(out.scores) for name, out in self.outputs.items()},
            "per_tool_penalties": {name: dict(out.penalties) for name, out in self.outputs.items()},
            "per_tool_claims": {
                name: {"support": dict(out.scores), "oppose": dict(out.penalties)}
                for name, out in self.outputs.items()
            },
            "per_tool_norm_scores": {name: dict(scores) for name, scores in self.normalized_scores.items()},
            "suggestions": self.suggestions,
        }


@dataclass
class LLMDecision:
    used_llm: bool = False
    choice_id: int = -1
    ranked_ids: List[int] = field(default_factory=list)
    action: str = "keep"
    confidence: float = 0.0
    explanation: str = ""
    raw_outputs: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    failure_modes_checked: List[str] = field(default_factory=list)
    evidence_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    critic_supports_choice: bool = False
    critic_attacks_top1: bool = False
    critic_abstain: bool = False
    critic_risk: str = "unknown"
    critic_score: float = 0.0
    target_supported: bool = False
    top1_supported: bool = False
    target_vs_top1_margin: float = 0.0
    evidence_sufficiency: float = 0.0
    should_commit: bool = False
    risk_level: str = "unknown"
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecision:
    query: RelationQuery
    top1: int
    choice_id: int
    ranked_ids: List[int]
    action: str
    used_llm: bool = False
    confidence: float = 0.0
    reason: str = ""
    evidence_bundle: Optional[EvidenceBundle] = None
    tool_trace: Optional[ToolTrace] = None
    llm_decision: Optional[LLMDecision] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def override(self) -> bool:
        return int(self.choice_id) != int(self.top1)
