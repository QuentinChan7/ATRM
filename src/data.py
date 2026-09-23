"""Dataset, causal history vocabulary and checkpoint-compatible backbone."""
from types import SimpleNamespace

import numpy as np
import torch

from awesome_agent.history_vocabulary import TemporalHistoryVocabulary
from rgcn import knowledge_graph, utils
from src.config import DATASETS
from src.rrgcn import RecurrentRGCN
from src.runtime_helpers import _load_semantic_features


class GraphData:
    def __init__(self, dataset, data_root):
        self.name = dataset
        self.directory = data_root / dataset
        raw = knowledge_graph.load_from_local(str(data_root), dataset)
        self.num_nodes, self.num_rels = raw.num_nodes, raw.num_rels
        self.snapshots, self.times, self.answers = {}, {}, {}
        for split in ("train", "valid", "test"):
            triples = getattr(raw, split)
            snapshots, times = utils.split_by_time(triples)
            if not snapshots:
                raise ValueError(f"Empty {split} split")
            self.snapshots[split], self.times[split] = snapshots, times
            self.answers[split] = (
                utils.load_all_answers_for_time_filter(triples, self.num_rels, self.num_nodes, False),
                utils.load_all_answers_for_time_filter(triples, self.num_rels, self.num_nodes, True),
            )
        if self.times['train'][-1] >= self.times['valid'][0] or self.times['valid'][-1] >= self.times['test'][0]:
            raise ValueError("Splits must be strictly ordered in time")
        self.num_times = sum(len(v) for v in self.snapshots.values())
        self.time_interval = self.times['train'][1] - self.times['train'][0]

    def history(self, split):
        return self.snapshots['train'] + (self.snapshots['valid'] if split == 'test' else [])

    def vocabulary(self, snapshots):
        history = TemporalHistoryVocabulary(num_nodes=self.num_nodes, num_rels=self.num_rels)
        history.seed(snapshots)
        return history

    def graphs(self, snapshots, gpu):
        return [utils.build_sub_graph(self.num_nodes, self.num_rels, snap, True, gpu) for snap in snapshots]

    def inputs(self, snapshot, history, device):
        triples = torch.as_tensor(snapshot, dtype=torch.long, device=device)
        inverse = triples[:, [2, 1, 0, 3]].clone()
        inverse[:, 1] += self.num_rels
        both = torch.cat([triples, inverse])
        tail, rel = history.vocabularies(both.detach().cpu().numpy())
        return triples, both, tail.masked_fill(tail != 0, 1).to(device), rel.masked_fill(rel != 0, 1).to(device)

    def model(self, gpu):
        cfg = DATASETS[self.name]
        entity_features = relation_features = static_graph = None
        num_static_rels = num_words = 0
        if cfg.semantic:
            args = SimpleNamespace(semantic_feature_path=str(self.directory / 'sbert_features.pt'))
            entity_features, relation_features, _ = _load_semantic_features(args, self.name, self.num_nodes, self.num_rels)
        if cfg.static:
            triples = np.loadtxt(self.directory / 'e-w-graph.txt', dtype=np.int64, ndmin=2)[:, :3]
            num_static_rels = len(np.unique(triples[:, 1]))
            num_words = len(np.unique(triples[:, 2]))
            triples[:, 2] += self.num_nodes
            static_graph = utils.build_sub_graph(self.num_nodes + num_words, num_static_rels, triples, True, gpu)
        model = RecurrentRGCN(
            'timeconvtranse', 'convgcn', self.num_nodes, self.num_rels,
            num_static_rels, num_words, self.num_times, self.time_interval, 200, 'sub', 0.3,
            sequence_len=cfg.history, num_bases=100, num_basis=100,
            num_hidden_layers=cfg.layers, dropout=0.2, self_loop=True,
            skip_connect=False, layer_norm=True, input_dropout=0.2,
            hidden_dropout=0.2, feat_dropout=0.2, aggregation='none',
            weight=0.5, discount=1, angle=cfg.angle, use_static=cfg.static,
            entity_prediction=True, relation_prediction=True, use_cuda=True, gpu=gpu,
            analysis=False, semantic_entity_features=entity_features,
            semantic_relation_features=relation_features,
            semantic_init_scale=0.1, semantic_trainable=False,
        ).cuda(gpu)
        return model, static_graph
