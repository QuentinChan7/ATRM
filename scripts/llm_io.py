from __future__ import annotations


import argparse


import hashlib


import json


import math


import os


import time


import urllib.error


import urllib.request


from collections import Counter, defaultdict


from pathlib import Path


from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _json_payload(raw: str) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    try:
        payload = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            return None
    return dict(payload) if isinstance(payload, dict) else None


def extract_anchor_selections(raw: str) -> List[Dict[str, str]]:
    payload = _json_payload(raw)
    raw_selections = payload.get("selections", []) if payload is not None else []
    selections: List[Dict[str, str]] = []
    seen = set()
    for item in raw_selections if isinstance(raw_selections, list) else []:
        if isinstance(item, str):
            anchor_id = item.strip()
            rationale = ""
        elif isinstance(item, dict):
            anchor_id = str(item.get("anchor_id", item.get("id", "")) or "").strip()
            rationale = str(item.get("rationale", item.get("reason", "")) or "").strip()
        else:
            continue
        if not anchor_id or anchor_id in seen:
            continue
        seen.add(anchor_id)
        selections.append({"anchor_id": anchor_id, "rationale": rationale[:240]})
    return selections


def extract_hypotheses(raw: str) -> List[Dict[str, Any]]:
    payload = _json_payload(raw)
    raw_rules = payload.get("hypotheses", []) if isinstance(payload, dict) else []
    rules = []
    for idx, rule in enumerate(raw_rules if isinstance(raw_rules, list) else []):
        if not isinstance(rule, dict):
            continue
        normalized = dict(rule)
        normalized["id"] = str(rule.get("id", f"llm_mtr_{idx}") or f"llm_mtr_{idx}")
        normalized["source"] = "llm_reflection_hypothesis"
        normalized["validated"] = False
        normalized["validation"] = {}
        rules.append(normalized)
    return rules[:4]


def _api_root(base_url: str) -> str:
    root = str(base_url or "").rstrip("/")
    if root.endswith("/chat/completions"):
        root = root[: -len("/chat/completions")]
    return root


def discover_models(base_url: str, api_key: str, timeout: float = 15.0) -> List[str]:
    print(f"[LLMRequest] checking model endpoint (timeout={timeout:g}s)", flush=True)
    request = urllib.request.Request(
        _api_root(base_url) + "/models",
        headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"LLM model endpoint is unavailable ({type(exc).__name__}); "
            "check the Qwen/vLLM service before rerunning teacher."
        ) from exc
    models = [
        str(item.get("id", "") or "").strip()
        for item in list(payload.get("data", []) or [])
        if isinstance(item, dict) and str(item.get("id", "") or "").strip()
    ]
    if not models:
        raise RuntimeError("LLM model endpoint returned no served models")
    print(f"[LLMRequest] served_models={models}", flush=True)
    return models


def call_llm(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    allowed_anchor_ids: Optional[Sequence[str]] = None,
    anchor_families: Optional[Mapping[str, str]] = None,
    min_anchor_selections: int = 1,
    max_anchor_selections: Optional[int] = None,
    request_timeout: float = 60.0,
    total_timeout: float = 180.0,
    max_attempts: int = 6,
    served_models: Optional[Sequence[str]] = None,
) -> Tuple[str, str, int]:
    if not (math.isfinite(request_timeout) and request_timeout > 0
            and math.isfinite(total_timeout) and total_timeout > 0 and max_attempts > 0):
        raise ValueError("LLM timeouts and max_attempts must be positive and finite")
    started = time.monotonic()
    deadline = started + total_timeout
    endpoint = _api_root(base_url) + "/chat/completions"
    discovered_models = list(served_models) if served_models is not None else discover_models(
        base_url, api_key, timeout=min(15.0, total_timeout)
    )
    requested_model = str(model or "").strip()
    model_candidates = []
    if requested_model and (not discovered_models or requested_model in discovered_models):
        model_candidates.append(requested_model)
    model_candidates.extend(item for item in discovered_models if item not in model_candidates)
    if requested_model and requested_model not in model_candidates:
        model_candidates.append(requested_model)
    if not model_candidates:
        model_candidates = [requested_model or "qwen3-8b"]

    token_candidates = []
    for value in (int(max_tokens), min(int(max_tokens), 900), min(int(max_tokens), 600), min(int(max_tokens), 320)):
        if value >= 128 and value not in token_candidates:
            token_candidates.append(value)

    errors = []
    attempts = 0
    last_successful_response: Optional[Tuple[str, str, int]] = None
    allowed_ids = {str(value) for value in allowed_anchor_ids} if allowed_anchor_ids is not None else None
    family_by_id = {str(key): str(value) for key, value in dict(anchor_families or {}).items()}
    required_families = set(family_by_id.values())
    required_selections = max(0, int(min_anchor_selections))
    request_variants = (
        (
            "json_no_think",
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object"},
            },
        ),
        ("no_think", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("plain_no_think", {}),
    )
    for model_name in model_candidates:
        for token_budget in token_candidates:
            context_rejected = False
            model_rejected = False
            for variant_name, extra_payload in request_variants:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or attempts >= max_attempts:
                    raise RuntimeError(
                        f"LLM request retry budget exhausted: attempts={attempts}/{max_attempts} "
                        f"elapsed={time.monotonic() - started:.1f}s. " + " | ".join(errors[-3:])
                    )
                attempts += 1
                timeout = min(request_timeout, remaining)
                print(
                    f"[LLMRequest] attempt={attempts}/{max_attempts} model={model_name} "
                    f"max_tokens={token_budget} format={variant_name} timeout={timeout:.1f}s",
                    flush=True,
                )
                request_payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "/no_think\n" + prompt}],
                    "max_tokens": int(token_budget),
                    "temperature": float(temperature),
                    **extra_payload,
                }
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(request_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'EMPTY'}"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    content = str(payload["choices"][0]["message"].get("content", "") or "")
                    if not content.strip():
                        errors.append(
                            f"model={model_name} max_tokens={token_budget} variant={variant_name}: empty content"
                        )
                        print(f"[LLMRequest] rejected: {errors[-1]}", flush=True)
                        continue
                    last_successful_response = content, model_name, int(token_budget)
                    selections = extract_anchor_selections(content)
                    valid_selection_count = sum(
                        1
                        for selection in selections
                        if allowed_ids is None or str(selection["anchor_id"]) in allowed_ids
                    )
                    selected_families = {
                        family_by_id[str(selection["anchor_id"])]
                        for selection in selections
                        if str(selection["anchor_id"]) in family_by_id
                    }
                    family_coverage_ok = not required_families or selected_families == required_families
                    selection_payload = _json_payload(content)
                    explicit_selection_list = isinstance((selection_payload or {}).get("selections"), list)
                    valid_selection_shape = explicit_selection_list and len(selections) == len(selection_payload["selections"])
                    all_ids_valid = allowed_ids is None or all(s["anchor_id"] in allowed_ids for s in selections)
                    count_ok = max_anchor_selections is None or len(selections) <= max_anchor_selections
                    if (valid_selection_shape and all_ids_valid and count_ok
                            and valid_selection_count >= required_selections and family_coverage_ok):
                        print(f"[LLMRequest] accepted after {time.monotonic() - started:.1f}s", flush=True)
                        return last_successful_response
                    if allowed_ids is None and extract_hypotheses(content):
                        return last_successful_response
                    errors.append(
                        f"model={model_name} max_tokens={token_budget} variant={variant_name}: "
                        f"valid_anchor_selections={valid_selection_count}/{required_selections} "
                        f"family_coverage={len(selected_families)}/{len(required_families)} "
                        f"response={content[:240]!r}"
                    )
                    print(f"[LLMRequest] rejected: {errors[-1]}", flush=True)
                except urllib.error.HTTPError as exc:
                    error_body = exc.read().decode("utf-8", errors="replace")
                    errors.append(
                        f"model={model_name} max_tokens={token_budget} variant={variant_name} "
                        f"HTTP {exc.code}: {error_body[:500]}"
                    )
                    print(f"[LLMRequest] rejected: {errors[-1]}", flush=True)
                    if int(exc.code) != 400:
                        raise RuntimeError(errors[-1]) from exc
                    lowered = error_body.lower()
                    if any(token in lowered for token in ("maximum context", "max model len", "max_tokens", "context length")):
                        context_rejected = True
                        break
                    if any(token in lowered for token in ("model does not exist", "model not found", "not served")):
                        model_rejected = True
                        break
                except (TimeoutError, urllib.error.URLError, OSError) as exc:
                    raise RuntimeError(
                        f"LLM request transport failed ({type(exc).__name__}, "
                        f"attempt={attempts}, timeout={timeout:.1f}s). "
                        "Stopped without retrying request formats; check Qwen/vLLM inference health."
                    ) from exc
                except Exception as exc:
                    errors.append(
                        f"model={model_name} max_tokens={token_budget} variant={variant_name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"[LLMRequest] rejected: {errors[-1]}", flush=True)
            if model_rejected:
                break
            if context_rejected:
                continue
    detail = "\n".join(errors[-8:])
    raise RuntimeError(
        "No LLM request produced an acceptable response. "
        f"discovered_models={discovered_models or 'none'} endpoint={endpoint}\n{detail}"
    )


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def load_relation_names(path: Path, num_rels: int) -> Dict[int, str]:
    names: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if not text:
                continue
            if "\t" in text:
                name, raw_id = text.rsplit("\t", 1)
            else:
                parts = text.rsplit(maxsplit=1)
                if len(parts) != 2:
                    continue
                name, raw_id = parts
            rid = _as_int(raw_id)
            if rid < 0:
                continue
            names[rid] = name.strip()
            names[rid + int(num_rels)] = "inverse of " + name.strip()
    return names
