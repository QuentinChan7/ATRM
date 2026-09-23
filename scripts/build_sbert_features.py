#!/usr/bin/env python3
"""Build SBERT semantic features for TKG entity/relation IDs.

The training code intentionally does not import sentence-transformers. This
script produces a small .pt artifact that can be enabled explicitly with
AGENT_SEMANTIC_INIT=1 or --semantic-init.
"""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from typing import List, Tuple


def _read_id_texts(path: Path, label: str) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    pairs: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Bad {label} line {line_no}: expected '<text>\\t<id>', got {line!r}")
            text = parts[0].strip()
            try:
                idx = int(parts[1])
            except ValueError as exc:
                raise ValueError(f"Bad {label} id at line {line_no}: {parts[1]!r}") from exc
            pairs.append((idx, text))
    if not pairs:
        raise ValueError(f"No {label} rows found in {path}")
    max_idx = max(idx for idx, _ in pairs)
    texts = [""] * (max_idx + 1)
    seen = set()
    for idx, text in pairs:
        if idx in seen:
            raise ValueError(f"Duplicate {label} id {idx} in {path}")
        seen.add(idx)
        texts[idx] = text
    missing = [idx for idx, text in enumerate(texts) if not text]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(f"{label} ids must be contiguous; missing ids: {preview}")
    return texts


def _encode_sentence_transformers(model, texts: List[str], batch_size: int, normalize: bool):
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    return vectors.detach().cpu().float().contiguous()


def _force_ipv4_only() -> None:
    original_getaddrinfo = socket.getaddrinfo

    def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_getaddrinfo


def _mean_pool(last_hidden_state, attention_mask, torch):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_transformers(
    torch,
    model_name: str,
    texts: List[str],
    batch_size: int,
    normalize: bool,
    device: str,
    max_length: int,
    local_files_only: bool,
    trust_remote_code: bool,
    use_safetensors: bool,
    pooling: str,
):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Neither sentence-transformers nor transformers is available. "
            "Install one of them, or pass a local precomputed feature file."
        ) from exc

    target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    load_kwargs = {
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
    }
    if use_safetensors:
        load_kwargs["use_safetensors"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    model = AutoModel.from_pretrained(model_name, **load_kwargs).to(target_device)
    model.eval()

    chunks = []
    total = len(texts)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            output = model(**encoded)
            if pooling == "cls":
                pooled = output.last_hidden_state[:, 0]
            else:
                pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"], torch)
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.detach().cpu().float())
            end = min(start + batch_size, total)
            print(f"[Transformers] encoded {end}/{total}", flush=True)
    return torch.cat(chunks, dim=0).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SBERT features for ATRM entity/relation IDs.")
    parser.add_argument("--dataset", required=True, help="Dataset folder name under --data-dir, e.g. ICEWS14")
    parser.add_argument("--data-dir", default="data", help="Repository data directory")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sentence-transformers", "transformers"],
        default="auto",
        help="Encoding backend. auto falls back to transformers mean pooling if sentence-transformers is unavailable.",
    )
    parser.add_argument("--output", default="", help="Output .pt path; default data/<dataset>/sbert_features.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="", help="Optional encoder device, e.g. cuda:0 or cpu")
    parser.add_argument("--max-length", type=int, default=64, help="Max token length for transformers fallback")
    parser.add_argument("--local-files-only", action="store_true", help="Never download from HuggingFace; use local model/cache only")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code for local transformer models")
    parser.add_argument("--force-ipv4", action="store_true", help="Force Python downloads to use IPv4 DNS results only")
    parser.add_argument(
        "--pooling",
        choices=["auto", "mean", "cls"],
        default="auto",
        help="Pooling for transformers backend. auto uses cls for BGE models and mean otherwise.",
    )
    parser.add_argument(
        "--allow-bin-weights",
        action="store_true",
        help="Allow loading pytorch_model.bin. Default requires safetensors to avoid torch.load restrictions.",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize SBERT embeddings")
    args = parser.parse_args()

    if args.force_ipv4:
        _force_ipv4_only()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to save the SBERT feature tensor artifact.") from exc

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir
    dataset_dir = data_dir / args.dataset
    output_path = Path(args.output).expanduser() if args.output else dataset_dir / "sbert_features.pt"
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    entity_texts = _read_id_texts(dataset_dir / "entity2id.txt", "entity")
    relation_texts = _read_id_texts(dataset_dir / "relation2id.txt", "relation")

    normalize = not bool(args.no_normalize)
    entity_inputs = [f"entity: {text}" for text in entity_texts]
    relation_inputs = [f"relation: {text}" for text in relation_texts]
    backend_used = args.backend
    pooling = str(args.pooling).strip().lower()
    if pooling == "auto":
        pooling = "cls" if "bge" in str(args.model).lower() else "mean"

    if args.backend in {"auto", "sentence-transformers"}:
        try:
            from sentence_transformers import SentenceTransformer
            model_kwargs = {"device": args.device} if args.device else {}
            model = SentenceTransformer(args.model, **model_kwargs)
            backend_used = "sentence-transformers"
            entity_features = _encode_sentence_transformers(model, entity_inputs, args.batch_size, normalize)
            relation_features = _encode_sentence_transformers(model, relation_inputs, args.batch_size, normalize)
        except Exception as exc:
            if args.backend == "sentence-transformers":
                raise SystemExit(
                    f"sentence-transformers could not load {args.model!r}: {exc}. "
                    "Use a complete local model directory or --backend transformers."
                ) from exc
            print(
                f"[SBERT] sentence-transformers could not load {args.model!r} "
                f"({type(exc).__name__}: {exc}); falling back to transformers.",
                flush=True,
            )
            backend_used = "transformers"
            entity_features = _encode_transformers(
                torch,
                args.model,
                entity_inputs,
                args.batch_size,
                normalize,
                args.device,
                int(args.max_length),
                bool(args.local_files_only),
                bool(args.trust_remote_code),
                not bool(args.allow_bin_weights),
                pooling,
            )
            relation_features = _encode_transformers(
                torch,
                args.model,
                relation_inputs,
                args.batch_size,
                normalize,
                args.device,
                int(args.max_length),
                bool(args.local_files_only),
                bool(args.trust_remote_code),
                not bool(args.allow_bin_weights),
                pooling,
            )
    else:
        backend_used = "transformers"
        entity_features = _encode_transformers(
            torch,
            args.model,
            entity_inputs,
            args.batch_size,
            normalize,
            args.device,
            int(args.max_length),
            bool(args.local_files_only),
            bool(args.trust_remote_code),
            not bool(args.allow_bin_weights),
            pooling,
        )
        relation_features = _encode_transformers(
            torch,
            args.model,
            relation_inputs,
            args.batch_size,
            normalize,
            args.device,
            int(args.max_length),
            bool(args.local_files_only),
            bool(args.trust_remote_code),
            not bool(args.allow_bin_weights),
            pooling,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": args.dataset,
            "model_name": args.model,
            "backend": backend_used,
            "pooling": pooling if backend_used == "transformers" else "sentence-transformers",
            "normalized": normalize,
            "entity_texts": entity_texts,
            "relation_texts": relation_texts,
            "entity_features": entity_features,
            "relation_features": relation_features,
        },
        str(output_path),
    )
    print(
        f"[SBERT] saved {output_path} | "
        f"entities={tuple(entity_features.shape)} relations={tuple(relation_features.shape)} "
        f"model={args.model} backend={backend_used} pooling={pooling if backend_used == 'transformers' else 'sentence-transformers'} "
        f"normalized={int(normalize)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
