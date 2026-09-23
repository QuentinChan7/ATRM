"""Full-snapshot evaluation with validation-only reflection learning."""
import torch
from tqdm import tqdm

from awesome_agent.fusion import RelationScoreFusionAgent
from awesome_agent.mapping import NLMapper
from awesome_agent.memory.stores import GraphMemory
from src.config import DATASETS, fusion_config
from src.runtime_helpers import _answer_set, _safe_order, _order_after_deltas


def make_agent(data, trace_path=None):
    mapper = NLMapper(str(data.directory / 'entity2id.txt'), str(data.directory / 'relation2id.txt'))
    config = fusion_config(data.num_rels, trace_path)
    config.llm_reflection_hypothesis_trace_limit = 2 * sum(map(len, data.snapshots['valid']))
    return RelationScoreFusionAgent(GraphMemory(), mapper, config)


def apply_memory(agent, triples, scores, answers, learn):
    """Score the entire snapshot before feedback can affect the next snapshot."""
    scores = scores.to(torch.float32)
    rel_dim = scores.shape[1]
    values, ids = torch.topk(scores, min(24, rel_dim), dim=1)
    pending = []
    for row, triple in enumerate(triples):
        s, target, o, t = map(int, triple.detach().cpu().tolist())
        valid = _answer_set(answers, s, o, target)
        before = _safe_order(scores[row], rel_dim)
        result = agent.score_relation(s, o, t, ids[row].tolist(), values[row].tolist(),
                                      score_lookup=scores[row], use_llm=False, count_row=True)
        result.debug.update(target_relation_id=target, valid_relation_ids=sorted(valid))
        after = _order_after_deltas(scores[row], result.deltas, rel_dim)
        if learn:
            agent.observe_target_outcome(result, after, valid)
        pending.append((row, result, before, after, valid))
    for row, result, before, after, valid in pending:
        for rid, delta in result.deltas.items():
            if 0 <= int(rid) < rel_dim:
                scores[row, int(rid)] += float(delta)
        if learn:
            agent.observe_outcome(result, before, after, valid)
        else:
            agent.update_report_with_outcome(before, after, valid)
    return scores


def finalize_memory(agent):
    agent.finalize_target_calibrator()
    rule = agent.target_rule_summary
    if rule['rules'] > 0:
        agent.memory.add_negative_constraint(
            "Reject target relation replacements unless the target matches "
            "a valid-calibrated high-precision memory/tool/reflection rule.")
        agent.memory.add_distilled_rule(
            f"Valid-calibrated target rules learned: rules={rule['rules']}, "
            f"best_precision={rule['best_precision']:.4f}, best_reward={rule['best_reward']:.2f}.")
    agent._write_llm_reflection_hypothesis_traces()
    print(agent.format_target_calibration(), flush=True)


@torch.no_grad()
def evaluate(model, data, static_graph, split, gpu, agent=None, learn=False, rank_fn=None):
    if split not in {'valid', 'test'} or learn and split != 'valid':
        raise ValueError("Reflection learning is validation-only")
    if rank_fn is None:
        from rgcn.utils import get_total_rank
        rank_fn = get_total_rank
    model.eval()
    cfg = DATASETS[data.name]
    seed = data.history(split)
    history = data.vocabulary(seed)
    context = list(seed[-cfg.history:])
    device = next(model.parameters()).device
    if agent is not None:
        agent.memory.reset_dynamic_state(clear_reflection=learn)
        agent.reset_report()
        if learn:
            agent.reset_target_calibrator()
        for snapshot in seed:
            agent.memory.update_snapshot(snapshot)
    ranks = {key: [] for key in ('entity_raw', 'entity_filter', 'relation_raw', 'relation_filter', 'gnn_relation_filter')}
    snapshots = data.snapshots[split]
    print(f"[Evaluation] split={split} snapshots={len(snapshots)} facts={sum(map(len, snapshots))} "
          f"history_snapshots={len(seed)} learn={int(learn)}", flush=True)
    for i, snapshot in enumerate(tqdm(snapshots, desc=f'{split} snapshots', unit='snapshot')):
        triples, _, tail, rel = data.inputs(snapshot, history, device)
        triples, entity_scores, relation_scores = model.predict(
            data.graphs(context, gpu), data.num_rels, static_graph, triples, tail, rel, True)
        relation_scores = relation_scores.to(torch.float32)
        gnn_scores = relation_scores.clone()
        entity_answers, relation_answers = (answer[i] for answer in data.answers[split])
        if agent is not None:
            relation_scores = apply_memory(agent, triples, relation_scores, relation_answers, learn)
        for label, scores, answers, is_relation in (
            ('entity', entity_scores, entity_answers, 0),
            ('relation', relation_scores, relation_answers, 1),
            ('gnn_relation', gnn_scores, relation_answers, 1),
        ):
            _, _, raw, filtered = rank_fn(triples, scores.clone(), answers, 1000, rel_predict=is_relation)
            if label + '_raw' in ranks:
                ranks[label + '_raw'].append(raw.cpu())
            ranks[label + '_filter'].append(filtered.cpu())
        # Ground-truth facts enter history only after all queries at this time are scored.
        history.update(snapshot)
        if agent is not None:
            agent.memory.update_snapshot(snapshot)
        context = (context + [snapshot])[-cfg.history:]
    expected = 2 * sum(map(len, snapshots))
    metrics = {}
    for name, values in ranks.items():
        all_ranks = torch.cat(values).float()
        if all_ranks.numel() != expected:
            raise RuntimeError("Incomplete evaluation")
        metrics[name] = {'queries': expected, 'mrr': (1 / all_ranks).mean().item(),
                         **{f'hits{k}': (all_ranks <= k).float().mean().item() for k in (1, 3, 10)}}
        print(f"{name}: {metrics[name]}", flush=True)
    if learn:
        finalize_memory(agent)
        if agent.config.trace_enabled and agent.report['llm_reflection_hypothesis_trace_records'] != expected:
            raise RuntimeError('Trace collection did not cover every validation query')
    return metrics
