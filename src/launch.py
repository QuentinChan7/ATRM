"""ATRM reproduction entry point."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATASETS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'trace', 'select', 'distill', 'evaluate'])
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference-checkpoint', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.stage == 'train' and args.epochs is None:
        parser.error('train requires --epochs')
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.lr is not None and not 0 < args.lr < float('inf'):
        parser.error('--lr must be finite and positive')
    for field in ('data_root', 'work_dir', 'checkpoint', 'reference_checkpoint'):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    trace = work / 'traces.jsonl'
    hypotheses = work / 'selection' / 'hypotheses.json'
    reference = args.reference_checkpoint or work / 'base.pt'
    if args.stage == 'select':
        from scripts.select_reflection_hypotheses import generate
        from src.relation_names import prompt_mapping
        stats = (args.data_root / args.dataset / 'stat.txt').read_text().split()
        mapping = prompt_mapping(args.dataset, args.data_root / args.dataset / 'relation2id.txt', work / 'relation_names.txt')
        generate(trace, mapping, int(stats[1]), work / 'selection')
        from awesome_agent.llm_trace_distill import LLMTraceDistillTeacher
        compiled = LLMTraceDistillTeacher.from_trace_file(
            trace, num_rels=int(stats[1]), provenance_mode='augment',
            provenance_hypothesis_path=str(hypotheses), margin_max_negatives_per_target=2,
            margin_source_weight=0.18, margin_rule_min_precision=0.70,
            source_field='base_top' if args.dataset == 'ICEWS14' else 'step_source_relation_id')
        if not compiled.provenance_context_weights:
            raise RuntimeError('No validated contexts; do not start distillation')
        print(f'[ATRM] validated_contexts={len(compiled.provenance_context_weights)}', flush=True)
        return
    import torch
    from awesome_agent.artifacts import load_weights, write_json
    from src.data import GraphData
    from src.evaluation import evaluate, make_agent
    from src.runtime_helpers import _set_global_seed
    from src.training import compile_supervision, train

    if args.gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError('Training and graph evaluation require a CUDA device')
    torch.cuda.set_device(args.gpu)
    _set_global_seed(args.seed)
    data = GraphData(args.dataset, args.data_root)
    model, static_graph = data.model(args.gpu)
    if args.stage in {'train', 'distill'}:
        distill = args.stage == 'distill'
        compiled = compile_supervision(trace, hypotheses, data.num_rels, args.dataset) if distill else None
        output = work / ('atrm.pt' if distill else 'base.pt')
        source = reference if distill else args.checkpoint
        train(model, data, static_graph, gpu=args.gpu, output=output,
              epochs=args.epochs or 1, lr=args.lr or (0.0001 if distill else 0.001),
              source=source, compiled=compiled)
        return
    if args.stage == 'trace':
        if trace.exists():
            raise FileExistsError(f'Trace already exists: {trace}; use a new work directory')
        os.environ['AGENT_LLM_TRACE_STREAM'] = '1'
        load_weights(model, args.checkpoint or reference)
        agent = make_agent(data, trace)
        evaluate(model, data, static_graph, 'valid', args.gpu, agent, learn=True)
        return
    # Calibrate the shared Memory on the reference; evaluate the selected weights.
    load_weights(model, reference)
    agent = make_agent(data)
    evaluate(model, data, static_graph, 'valid', args.gpu, agent, learn=True)
    load_weights(model, args.checkpoint or work / 'atrm.pt')
    result = evaluate(model, data, static_graph, 'test', args.gpu, agent)
    write_json(work / 'metrics.json', result)


if __name__ == '__main__':
    main()
