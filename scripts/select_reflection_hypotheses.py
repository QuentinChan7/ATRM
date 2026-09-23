#!/usr/bin/env python3
"""Batched, fit-only MTR evidence selection with explicit LLM abstention."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from awesome_agent.artifacts import sha256, write_json
from awesome_agent.target_trace import TARGET_SCHEMA
from scripts.llm_io import (
    _json_payload, call_llm, discover_models, extract_anchor_selections,
)
from scripts.llm_io import load_relation_names


def records(path):
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("reward_schema") != TARGET_SCHEMA or row.get("target_relation_id") is None:
                raise ValueError("Target teacher requires an explicitly target-filter-labelled trace")
            yield row


def metrics(stats):
    n, f, h = stats["support"], stats["fixes"], stats["fails"]
    return {"n": n, "fix": f, "fail": h, "neutral": n-f-h,
            "precision": round(f / max(1, f+h), 6),
            "mean_rr": round(stats["rr"] / max(1, n), 6)}


def add(stats, delta):
    stats["support"] += 1
    stats["fixes"] += delta > 1e-12
    stats["fails"] += delta < -1e-12
    stats["rr"] += delta


def build_catalog(path, holdout_fraction=.25):
    times = sorted({int(r["query"]["t"]) for r in records(path)})
    if len(times) < 4 or not 0 < holdout_fraction < 1:
        raise ValueError("Need multiple chronological fit and holdout groups")
    count = math.ceil(len(times) * holdout_fraction)
    fit_times, holdout_times = times[:-count], times[-count:]
    fit_set = set(fit_times)
    blocks = {t: min(2, i * 3 // len(fit_times)) for i, t in enumerate(fit_times)}
    clusters = {}
    fit_records = 0
    for row in records(path):
        t = int(row["query"]["t"])
        if t not in fit_set:
            continue
        fit_records += 1
        for candidate in row["candidates"]:
            rid = int(candidate["rid"])
            if rid == int(row["base_top"]):
                continue
            opponent = candidate.get("step_source_relation_id")
            if opponent is None:
                raise ValueError("Non-top candidate has no adjacent opponent")
            key = (row["signature"], int(row["prev_rel"]), int(row["base_top"]), rid, int(opponent))
            cluster = clusters.setdefault(key, {"all": Counter(), "blocks": [Counter() for _ in range(3)],
                                               "days": defaultdict(Counter), "examples": {}})
            delta = float(candidate["step_mrr_delta"])
            add(cluster["all"], delta)
            add(cluster["blocks"][blocks[t]], delta)
            add(cluster["days"][t], delta)
            outcome = "fix" if delta > 1e-12 else "fail" if delta < -1e-12 else "neutral"
            if outcome != "neutral" and outcome not in cluster["examples"]:
                f = candidate.get("features", {})
                cluster["examples"][outcome] = {
                    "query": dict(row["query"]), "target": int(row["target_relation_id"]),
                    "current_rank": candidate["base_rank"], "target_rr_delta": round(delta, 6),
                    "tools": row.get("selected_tools", [])[:3],
                    "mtr": {k: round(float(f.get(k, 0)), 4) for k in (
                        "commit_gap", "tool_gap", "verifier_gap", "counterfactual_risk", "reflection")},
                }
    anchors = []
    for key, c in sorted(clusters.items()):
        s = c["all"]
        n, f, h = s["support"], s["fixes"], s["fails"]
        if n < 6 or f < 2 or f/max(1, f+h) < .7 or s["rr"]/n < .02 or h/max(1, f+h) > .3:
            continue
        anchors.append({
            "id": f"A{len(anchors):04d}", "signature": key[0], "prev_rel": key[1],
            "top1": key[2], "choice": key[3], "family": "|".join(key[0].split("|")[:2]),
            "opponent": key[4],
            "fit": metrics(s), "chronological_fit_blocks": [metrics(b) for b in c["blocks"]],
            "positive_days": sum(v["rr"] > 1e-12 for v in c["days"].values()),
            "negative_days": sum(v["rr"] < -1e-12 for v in c["days"].values()),
            "active_days": len(c["days"]), "examples": c["examples"],
        })
    families = defaultdict(list)
    for anchor in anchors:
        families[anchor["family"]].append(anchor)
    queues = []
    for family in sorted(families):
        ranked = sorted(families[family], key=lambda a: (
            -sum(b["mean_rr"] > 0 for b in a["chronological_fit_blocks"]),
            -a["fit"]["precision"], -a["positive_days"]/a["active_days"], a["id"]))
        queues.append(deque(ranked))
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.popleft())
    return {"reward_schema": TARGET_SCHEMA, "fit_records": fit_records,
            "fit_times": fit_times, "holdout_times": holdout_times, "anchors": ordered}


def prompt_for(anchors, relation_names):
    def name(rid):
        return "no previous relation" if rid == -1 else relation_names.get(rid, str(rid))[:90]

    rendered = []
    for a in anchors:
        row = dict(a)
        row["relations"] = {k: name(a[k]) for k in ("prev_rel", "top1", "choice", "opponent")}
        rendered.append(row)
    header = (
        "You are an offline Memory-Tool-Reflection teacher. All evidence below is from the FIT partition only.\n"
        "top1 is the post-Memory baseline, not the raw GNN prediction. The action lifts choice over opponent by one rank. "
        "Reward is per-target FILTERED reciprocal-rank gain; other valid relations are filtered. "
        "fix/fail count rank improvements/harms, not necessarily top-1 corrections.\n"
        "Compare successful and harmful tool traces, relation meanings, and the three chronological fit blocks. "
        "Large support alone does not imply transfer. The mtr gaps compare a candidate with the baseline; "
        "counterfactual_risk is risk level, not risk reduction. Prefer repeatable patterns with consistent recent evidence.\n"
        "Select at most 2 distinct IDs worth using as training contexts. You may reject all. "
        "Never invent IDs, conditions, or holdout outcomes. Return JSON only: "
        '{"selections":[{"anchor_id":"A0000","rationale":"brief evidence-based reason"}]}. '
        'For abstention return {"selections":[]}.\n'
    )
    return header + json.dumps(rendered, separators=(",", ":"), ensure_ascii=True)


def make_batches(anchors, relation_names, max_chars=3600):
    batches, pending = [], []
    for anchor in anchors:
        if len(prompt_for([anchor], relation_names)) > max_chars:
            raise ValueError(f"One anchor exceeds prompt budget: {anchor['id']}; increase context explicitly")
        if pending and (len(pending) >= 3 or len(prompt_for(pending + [anchor], relation_names)) > max_chars):
            batches.append(pending)
            pending = []
        pending.append(anchor)
    if pending:
        batches.append(pending)
    return batches


def generate(trace, relations, num_rels, work, *, dry_run=False):
    work.mkdir(parents=True, exist_ok=True)
    print("[TargetTeacher] building fit-only catalog from the complete trace", flush=True)
    names = load_relation_names(relations, num_rels)
    catalog = build_catalog(trace)
    batches = make_batches(catalog["anchors"], names)
    write_json(work / "fit_catalog.json", {**catalog, "batch_count": len(batches)})
    print(f"[TargetTeacher] eligible={len(catalog['anchors'])} families={len({a['family'] for a in catalog['anchors']})} "
          f"batches={len(batches)}; all eligible anchors are offered, holdout outcomes hidden", flush=True)
    if not batches:
        raise RuntimeError("No fit-qualified target-filter anchors; no LLM requests made")
    served = None
    hypotheses = []
    model = os.environ.get("AGENT_LLM_MODEL", "qwen3-8b")
    for number, batch in enumerate(batches, 1):
        prompt = prompt_for(batch, names)
        prompt_path = work / f"batch_{number:04d}_prompt.txt"
        response_path = work / f"batch_{number:04d}.json"
        prompt_path.write_text(prompt)
        recipe = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "model": model,
                  "max_tokens": 600, "temperature": 0.0}
        if dry_run:
            continue
        if response_path.exists():
            response = json.loads(response_path.read_text())
            if response["recipe"] != recipe:
                raise RuntimeError(f"Cached teacher batch changed: {response_path}; use a new work directory")
            raw = response["raw"]
            print(f"[TargetTeacher] batch {number}/{len(batches)} reused", flush=True)
        else:
            print(f"[TargetTeacher] batch {number}/{len(batches)} anchors={[a['id'] for a in batch]}", flush=True)
            base_url = os.environ.get("AGENT_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
            api_key = os.environ.get("AGENT_LLM_API_KEY", "EMPTY")
            if served is None:
                served = discover_models(base_url, api_key)
                if model not in served:
                    raise RuntimeError(f"Frozen teacher model {model!r} is not served: {served}")
            raw, resolved_model, tokens = call_llm(
                base_url=base_url, model=model, api_key=api_key, prompt=prompt, max_tokens=600,
                temperature=0.0, allowed_anchor_ids=[a["id"] for a in batch],
                min_anchor_selections=0, max_anchor_selections=2, served_models=[model],
            )
            if resolved_model != model:
                raise RuntimeError("Teacher response used a different model than the frozen recipe")
            write_json(response_path, {"recipe": recipe, "raw": raw,
                                       "resolved_model": resolved_model, "max_tokens": tokens})
        by_id = {a["id"]: a for a in batch}
        payload = _json_payload(raw) or {}
        selections = extract_anchor_selections(raw)
        if (not isinstance(payload.get("selections"), list) or len(selections) != len(payload["selections"])
                or len(selections) > 2 or any(s["anchor_id"] not in by_id for s in selections)):
            raise ValueError("Invalid cached LLM selections")
        for selection in selections:
            anchor = by_id[selection["anchor_id"]]
            hypotheses.append({
                "id": "llm_target_" + anchor["id"], "anchor_id": anchor["id"],
                "source": "llm_reflection_hypothesis", "description": selection["rationale"],
                **{k: anchor[k] for k in ("signature", "prev_rel", "top1", "choice", "opponent")},
                "conditions": {}, "action": "boost_choice", "delta": 0.0,
                "fit_anchor": anchor["fit"],
            })
    if not dry_run:
        write_json(work / "hypotheses.json", {"metadata": {
            "reward_schema": TARGET_SCHEMA, "trace_sha256": sha256(trace), "model": model,
            "offered_anchors": len(catalog["anchors"]), "batches": len(batches),
            "holdout_times": catalog["holdout_times"], "fit_only": True,
        }, "hypotheses": hypotheses})
        print(f"[TargetTeacher] selected={len(hypotheses)}; temporal audit still required", flush=True)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--num-rels", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    generate(args.trace, args.relations, args.num_rels, args.work_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
