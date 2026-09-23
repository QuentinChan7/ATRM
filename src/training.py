"""Joint GNN training and offline, provenance-conditioned rank supervision."""
import torch
from tqdm import tqdm

from awesome_agent.artifacts import load_weights
from awesome_agent.llm_trace_distill import LLMTraceDistillTeacher
from src.config import DATASETS
from src.evaluation import evaluate


def compile_supervision(trace, hypotheses, num_rels, dataset):
    compiled = LLMTraceDistillTeacher.from_trace_file(
        trace, num_rels=num_rels, source_field='base_top' if dataset == 'ICEWS14' else 'step_source_relation_id',
        margin_max_negatives_per_target=2, margin_source_weight=0.18,
        margin_rule_min_precision=0.70, provenance_mode='augment',
        provenance_hypothesis_path=str(hypotheses))
    if not compiled.provenance_context_weights:
        raise RuntimeError("No temporally validated LLM contexts survived; training was not started")
    print(compiled.summary, flush=True)
    return compiled


def supervision_kwargs(compiled, triples, relation_history):
    device = triples.device
    return dict(
        relation_teacher_targets=compiled.targets_for_triples(triples, device=device),
        relation_teacher_pair_weights=compiled.pair_weight_matrix(device=device),
        relation_teacher_weight=0.04, relation_teacher_query_weight=0.020,
        relation_teacher_pair_weight=0.04,
        relation_teacher_provenance_pair_weights=compiled.provenance_pair_weight_matrix(device=device),
        relation_teacher_provenance_source_weights=compiled.provenance_source_weights_for_triples(
            triples, relation_history, device=device),
        relation_teacher_conditioned_margin_weight=0.04,
        relation_teacher_conditioned_margin=0.08,
        relation_teacher_conditioned_margin_target_coverage=0.02,
        relation_teacher_margin_negatives=compiled.margin_negatives_for_triples(triples, device=device),
        relation_teacher_margin_weight=0.006, relation_teacher_margin=0.08,
        relation_teacher_margin_active_only=False, relation_teacher_margin_loss_cap=0.0,
    )


def train(model, data, static_graph, *, gpu, output, epochs, lr, source=None, compiled=None):
    if output.exists() or source is not None and output.resolve() == source.resolve():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {output}')
    source_epoch = -1
    if source is not None:
        source_epoch = int(load_weights(model, source).get('epoch', -1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    cfg = DATASETS[data.name]
    device = next(model.parameters()).device
    snapshots = data.snapshots['train']
    best = float('-inf')
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        history = data.vocabulary(snapshots[:1])
        losses, active_rows = [], 0
        for index in tqdm(range(1, len(snapshots)), desc=f'Epoch {epoch}', unit='snapshot'):
            snapshot = snapshots[index]
            triples, both, tail, rel = data.inputs(snapshot, history, device)
            graphs = data.graphs(snapshots[max(0, index - cfg.history):index], gpu)
            kwargs = supervision_kwargs(compiled, both, rel) if compiled is not None else {}
            entity_loss, relation_loss, static_loss = model.get_loss(
                graphs, triples, static_graph, tail, rel, True, **kwargs)
            error = getattr(model, 'last_relation_distill_conditioned_margin_error', '')
            if compiled is not None and error:
                raise RuntimeError(error)
            loss = 0.7 * entity_loss + 0.3 * relation_loss + static_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}, snapshot {index}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            history.update(snapshot)
            losses.append(loss.item())
            active_rows += int(getattr(model, 'last_relation_distill_conditioned_margin_rows', 0))
        if compiled is not None and active_rows == 0:
            raise RuntimeError('No training rows activated the LLM-conditioned objective')
        metrics = evaluate(model, data, static_graph, 'valid', gpu)
        score = metrics[cfg.selection]['mrr']
        print(f'[Training] epoch={epoch} loss={sum(losses) / len(losses):.6f} '
              f'conditioned_rows={active_rows} valid_mrr={score:.6f}', flush=True)
        if score > best:
            best = score
            temporary = output.with_suffix('.partial')
            torch.save({'state_dict': model.state_dict(), 'epoch': source_epoch + epoch + 1,
                        'dataset': data.name, 'selection_metric': cfg.selection, 'valid_mrr': score}, temporary)
            temporary.replace(output)
    load_weights(model, output)
