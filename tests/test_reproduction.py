import ast
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from awesome_agent.history_vocabulary import TemporalHistoryVocabulary
from awesome_agent.target_trace import TARGET_SCHEMA
from scripts import llm_io
from scripts.select_reflection_hypotheses import generate, build_catalog
from src.config import DATASETS, fusion_config
from src.evaluation import apply_memory, evaluate, make_agent
from src.launch import main, parse_args
from src.training import compile_supervision, supervision_kwargs

ROOT = Path(__file__).resolve().parents[1]


def canonical_ranks():
    tree = ast.parse((ROOT / 'rgcn/utils.py').read_text())
    names = {'sort_and_rank', 'filter_score', 'filter_score_r', 'get_total_rank'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<ranking>', 'exec'), scope)
    return scope['get_total_rank']


class ToyData:
    name, num_nodes, num_rels = 'ICEWS14', 3, 2

    def __init__(self, directory):
        self.directory = directory
        self.snapshots = {split: [np.array([[0, 0, 1, t], [0, 1, 1, t]]) for t in times]
                          for split, times in [('train', [0, 1]), ('valid', [2, 3, 4, 5]), ('test', [6, 7])]}
        self.answers = {}
        for split, snapshots in self.snapshots.items():
            self.answers[split] = ([{0: {0: {1}, 1: {1}}, 1: {2: {0}, 3: {0}}} for _ in snapshots],
                                   [{0: {1: {0, 1}}, 1: {0: {2, 3}}} for _ in snapshots])
        self.checked_times = []

    def history(self, split):
        return self.snapshots['train'] + (self.snapshots['valid'] if split == 'test' else [])

    def vocabulary(self, snapshots):
        history = TemporalHistoryVocabulary(num_nodes=3, num_rels=2)
        history.seed(snapshots)
        return history

    def inputs(self, snapshot, history, device):
        triples = torch.as_tensor(snapshot, device=device)
        inverse = triples[:, [2, 1, 0, 3]].clone()
        inverse[:, 1] += 2
        both = torch.cat([triples, inverse])
        self.checked_times.append((int(snapshot[0, 3]), history.snapshots))
        tail, rel = history.vocabularies(both.numpy())
        return triples, both, tail, rel

    def graphs(self, context, gpu):
        return context


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def predict(self, graphs, num_rels, static, triples, tail, rel, cuda):
        assert max(int(g[0, 3]) for g in graphs) < int(triples[0, 3])
        inverse = triples[:, [2, 1, 0, 3]].clone()
        inverse[:, 1] += num_rels
        both = torch.cat([triples, inverse])
        return both, torch.tensor([[-1., -2., -3.]]).repeat(len(both), 1), torch.tensor(
            [[-1.0, -1.1, -1.2, -1.3]]).repeat(len(both), 1)


class ReproductionTest(unittest.TestCase):
    def test_cli_supported_datasets(self):
        self.assertEqual(set(DATASETS), {'ICEWS14', 'ICEWS18', 'GDELT'})
        for dataset in DATASETS:
            for stage in ('train', 'trace', 'select', 'distill', 'evaluate'):
                with self.subTest(dataset=dataset, stage=stage):
                    args = parse_args([stage, '--dataset', dataset, '--work-dir', 'runs/test', '--epochs', '1'])
                    self.assertEqual(args.dataset, dataset)
        with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['trace', '--dataset', 'unsupported', '--work-dir', 'runs/test'])

    def test_cli_requires_training_budget_and_resolves_paths(self):
        args = parse_args(['train', '--dataset', 'ICEWS14', '--work-dir', 'runs/test', '--epochs', '8'])
        self.assertTrue(args.work_dir.is_absolute())
        with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(['train', '--dataset', 'ICEWS14', '--work-dir', 'runs/test'])

    def test_snapshot_is_scored_before_feedback(self):
        events = []
        class Agent:
            def score_relation(self, *args, **kwargs):
                events.append('score')
                return SimpleNamespace(debug={}, deltas={1: .2})
            def observe_target_outcome(self, *args):
                events.append('label')
            def observe_outcome(self, *args):
                events.append('feedback')
        triples = torch.tensor([[0, 1, 2, 4], [0, 0, 2, 4]])
        scores = torch.tensor([[-1., -1.1], [-1., -1.1]])
        apply_memory(Agent(), triples, scores, {0: {2: {0, 1}}}, True)
        self.assertEqual(events, ['score', 'label', 'score', 'label', 'feedback', 'feedback'])
        torch.testing.assert_close(scores[:, 1], torch.tensor([-.9, -.9]))

    def test_full_trace_and_test_are_causal_and_test_does_not_learn(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = ToyData(root)
            trace = root / 'traces.jsonl'
            agent = make_agent(data, trace)
            with patch.dict('os.environ', {'AGENT_LLM_TRACE_STREAM': '1'}):
                valid = evaluate(ToyModel(), data, None, 'valid', 0, agent, True, canonical_ranks())
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(len(rows), 16)
            self.assertEqual(valid['relation_filter']['queries'], 16)
            self.assertEqual([row['target_relation_id'] for row in rows[:4]], [0, 1, 2, 3])
            self.assertTrue(all(row['reward_schema'] == TARGET_SCHEMA for row in rows))
            feedback = copy.deepcopy(dict(agent.memory.tool_experience))
            self.assertTrue(feedback)
            agent.config.trace_enabled = False
            test = evaluate(ToyModel(), data, None, 'test', 0, agent, False, canonical_ranks())
            self.assertEqual(test['relation_filter']['queries'], 8)
            # Query-time lookups may create zero-valued keys, but must not learn new outcomes.
            self.assertEqual(sum(v['calls'] for v in feedback.values()),
                             sum(v['calls'] for v in agent.memory.tool_experience.values()))
            self.assertEqual(data.checked_times, [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)])

    def test_test_split_rejects_reflection_learning(self):
        with self.assertRaisesRegex(ValueError, 'validation-only'):
            evaluate(ToyModel(), None, None, 'test', 0, learn=True)

    def test_filtered_ranking_keeps_the_evaluated_target(self):
        triples = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
        scores = torch.tensor([[3., 2., 1.], [3., 2., 1.]])
        _, _, raw, filtered = canonical_ranks()(triples, scores, {0: {1: {0, 1}}}, 1000, 1)
        self.assertEqual(raw.tolist(), [1, 2])
        self.assertEqual(filtered.tolist(), [1, 1])


class OfflineSelectionTest(unittest.TestCase):
    def test_selection_uses_dataset_relation_mapping(self):
        from awesome_agent.llm_trace_distill import LLMTraceDistillTeacher

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory).resolve()
            for dataset in DATASETS:
                with self.subTest(dataset=dataset):
                    data = root / 'data' / dataset
                    data.mkdir(parents=True)
                    (data / 'stat.txt').write_text('3 2\n')
                    mapping = data / 'relation2id.txt'
                    mapping.write_text('cooperate\t0\nmeet\t1\n')
                    work = root / 'runs' / dataset
                    compiled = SimpleNamespace(provenance_context_weights={(0, 1): 1.0})
                    with patch('scripts.select_reflection_hypotheses.generate') as generate_mock, \
                         patch.object(LLMTraceDistillTeacher, 'from_trace_file',
                                      return_value=compiled) as compile_mock:
                        main(['select', '--dataset', dataset, '--data-root', str(root / 'data'),
                              '--work-dir', str(work)])
                    generate_mock.assert_called_once_with(work / 'traces.jsonl', mapping, 2, work / 'selection')
                    self.assertEqual(compile_mock.call_args.kwargs['source_field'],
                                     'base_top' if dataset == 'ICEWS14' else 'step_source_relation_id')

    def test_llm_selection_temporal_audit_and_training_tensors(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            trace, mapping = root / 'traces.jsonl', root / 'relations.txt'
            mapping.write_text('cooperate\t0\nmeet\t1\n')
            records = []
            for t in range(4):
                for i in range(2):
                    records.append(dict(query=dict(s=i, o=i+1, t=t), target_relation_id=1,
                        reward_schema=TARGET_SCHEMA, signature='rel|local_sparse', prev_rel=0,
                        base_top=0, base_order=[0, 1, 2, 3], gt=[1], selected_tools=['pair_local'],
                        candidates=[dict(rid=1, rank=2, base_rank=2, is_gt=True, is_target=True,
                                         step_source_relation_id=0, step_mrr_delta=.5, features={})]))
            trace.write_text(''.join(json.dumps(r) + '\n' for r in records))
            catalog = build_catalog(trace)
            self.assertEqual(catalog['fit_records'], 6)
            self.assertEqual(catalog['holdout_times'], [3])
            calls = []
            def request(req, timeout):
                if req.full_url.endswith('/models'):
                    return io.BytesIO(json.dumps({'data': [{'id': 'qwen3-8b'}]}).encode())
                payload = json.loads(req.data)
                calls.append(payload)
                return io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps({
                    'selections': [{'anchor_id': 'A0000', 'rationale': 'Repeated cross-time evidence'}]})}}]}).encode())
            with patch.object(llm_io.urllib.request, 'urlopen', request):
                generate(trace, mapping, 2, root / 'selection')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['temperature'], 0)
            compiled = compile_supervision(trace, root / 'selection/hypotheses.json', 2, 'ICEWS18')
            self.assertTrue(compiled.provenance_context_weights)
            triples = torch.tensor([[0, 1, 1, 0], [1, 3, 0, 0]])
            kwargs = supervision_kwargs(compiled, triples, torch.ones(2, 4))
            self.assertGreater(kwargs['relation_teacher_provenance_source_weights'].sum().item(), 0)
            self.assertEqual(kwargs['relation_teacher_conditioned_margin_weight'], .04)

    def test_nonexistent_trace_does_not_silently_train(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                compile_supervision(Path(directory) / 'missing.jsonl', '', 2, 'ICEWS18')


if __name__ == '__main__':
    unittest.main()
