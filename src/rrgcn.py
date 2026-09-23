import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import *
from awesome_agent.trace_distill_loss import sparse_coverage_normalization


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        # --- 核心修改部分 START ---
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop,
                                  skip_connect=sc, rel_emb=self.rel_emb)
        # 新增 uvrgcn 的支持 (使用 RGCNBlockLayer)
        elif self.encoder_name == "uvrgcn":
            return RGCNBlockLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop,
                                  skip_connect=sc)
        else:
            raise NotImplementedError("Encoder {} not implemented".format(self.encoder_name))
        # --- 核心修改部分 END ---


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times, time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False, semantic_entity_features=None, semantic_relation_features=None,
                 semantic_init_scale=0.1, semantic_trainable=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.semantic_init_scale = float(semantic_init_scale)
        self.semantic_trainable = bool(semantic_trainable)
        self.semantic_enabled = semantic_entity_features is not None or semantic_relation_features is not None
        self.sin = torch.sin
        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()
        self.use_cuda = None

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.semantic_entity_proj = None
        self.semantic_relation_proj = None
        if self.semantic_enabled:
            if semantic_entity_features is None or semantic_relation_features is None:
                raise ValueError("semantic_entity_features and semantic_relation_features must be provided together.")
            ent_features = torch.as_tensor(semantic_entity_features, dtype=torch.float32)
            rel_features = torch.as_tensor(semantic_relation_features, dtype=torch.float32)
            if ent_features.dim() != 2 or ent_features.size(0) != self.num_ents:
                raise ValueError(
                    f"semantic_entity_features shape must be [{self.num_ents}, dim], got {tuple(ent_features.shape)}"
                )
            if rel_features.dim() != 2:
                raise ValueError(f"semantic_relation_features must be 2D, got {tuple(rel_features.shape)}")
            if rel_features.size(0) == self.num_rels:
                rel_features = torch.cat([rel_features, rel_features], dim=0)
            elif rel_features.size(0) != self.num_rels * 2:
                raise ValueError(
                    f"semantic_relation_features rows must be {self.num_rels} or {self.num_rels * 2}, got {rel_features.size(0)}"
                )
            ent_features = F.normalize(ent_features, p=2, dim=1)
            rel_features = F.normalize(rel_features, p=2, dim=1)
            if self.semantic_trainable:
                self.semantic_entity_features = nn.Parameter(ent_features, requires_grad=True)
                self.semantic_relation_features = nn.Parameter(rel_features, requires_grad=True)
            else:
                self.register_buffer("semantic_entity_features", ent_features)
                self.register_buffer("semantic_relation_features", rel_features)
            self.semantic_entity_proj = nn.Linear(ent_features.size(1), self.h_dim, bias=False)
            self.semantic_relation_proj = nn.Linear(rel_features.size(1), self.h_dim, bias=False)
            nn.init.xavier_normal_(self.semantic_entity_proj.weight)
            nn.init.xavier_normal_(self.semantic_relation_proj.weight)
        

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))


        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # add
        self.global_weight = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight , gain=nn.init.calculate_gain('relu'))
        self.global_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "timeconvtranse":
            self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 




    def _entity_base_embedding(self):
        base = self.dynamic_emb
        if self.semantic_enabled:
            base = base + self.semantic_init_scale * self.semantic_entity_proj(self.semantic_entity_features)
        return base

    def _relation_base_embedding(self):
        base = self.emb_rel
        if self.semantic_enabled:
            base = base + self.semantic_init_scale * self.semantic_relation_proj(self.semantic_relation_features)
        return base

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []
        entity_base = self._entity_base_embedding()
        relation_base = self._relation_base_embedding()

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((entity_base, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(entity_base) if self.layer_norm else entity_base[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((relation_base, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, relation_base)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((relation_base, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        self.use_cuda = use_cuda
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            time_embs = self.get_init_time(all_triples)

            score_rel_r = self.rel_raw_mode(embedding, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(embedding, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_r = self.raw_mode(embedding, r_emb, time_embs, all_triples)
            score_h = self.history_mode(embedding, r_emb, time_embs, all_triples, entity_history_vocabulary)

            relation_alpha = getattr(self, "relation_history_rate_override", self.history_rate)
            score_rel = relation_alpha * score_rel_h + (1 - relation_alpha) * score_rel_r
            # Preserve the canonical checkpoint ranking exactly when the adapter
            # is disabled. Clamping here changes ties among underflowed scores.
            score_rel_log = torch.log(score_rel)
            score_rel = score_rel_log
            score = self.history_rate * score_h + (1 - self.history_rate) * score_r
            score = torch.log(score)

            return all_triples, score, score_rel


    def get_loss(
        self,
        glist,
        triples,
        static_graph,
        entity_history_vocabulary,
        rel_history_vocabulary,
        use_cuda,
        relation_teacher_targets=None,
        relation_teacher_pair_weights=None,
        relation_teacher_weight=0.0,
        relation_teacher_query_weight=None,
        relation_teacher_pair_weight=None,
        relation_teacher_provenance_pair_weights=None,
        relation_teacher_provenance_source_weights=None,
        relation_teacher_conditioned_margin_weight=0.0,
        relation_teacher_conditioned_margin=0.08,
        relation_teacher_conditioned_margin_target_coverage=1.0,
        relation_teacher_margin_negatives=None,
        relation_teacher_margin_weight=0.0,
        relation_teacher_margin=0.15,
        relation_teacher_margin_active_only=False,
        relation_teacher_margin_loss_cap=0.0,
    ):
        self.use_cuda = use_cuda
        self.last_relation_distill_loss = 0.0
        self.last_relation_distill_query_loss = 0.0
        self.last_relation_distill_pair_loss = 0.0
        self.last_relation_distill_conditioned_margin_loss = 0.0
        self.last_relation_distill_margin_loss = 0.0
        self.last_relation_distill_margin_raw_loss = 0.0
        self.last_relation_distill_margin_scale = 1.0
        self.last_relation_distill_query_rows = 0
        self.last_relation_distill_pair_rows = 0
        self.last_relation_distill_conditioned_margin_rows = 0
        self.last_relation_distill_conditioned_margin_active_rows = 0
        self.last_relation_distill_conditioned_margin_coverage = 0.0
        self.last_relation_distill_conditioned_margin_amplification = 1.0
        self.last_relation_distill_conditioned_margin_error = ""
        self.last_relation_distill_margin_rows = 0
        self.last_relation_distill_margin_active_rows = 0
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        time_embs = self.get_init_time(all_triples)

        if self.entity_prediction:
            score_r = self.raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_h = self.history_mode(pre_emb, r_emb, time_embs, all_triples, entity_history_vocabulary)
            score_en = self.history_rate * score_h + (1 - self.history_rate) * score_r
            scores_en = torch.log(score_en)
            loss_ent += F.nll_loss(scores_en, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(pre_emb, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_re = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            base_scores_re = torch.log(torch.clamp(score_re, min=1e-12))
            scores_re = base_scores_re
            loss_rel += F.nll_loss(base_scores_re, all_triples[:, 1])
            teacher_weight = float(relation_teacher_weight or 0.0)
            query_weight = teacher_weight if relation_teacher_query_weight is None else float(relation_teacher_query_weight or 0.0)
            pair_weight = teacher_weight if relation_teacher_pair_weight is None else float(relation_teacher_pair_weight or 0.0)
            conditioned_margin_weight = float(relation_teacher_conditioned_margin_weight or 0.0)
            conditioned_margin_value = max(0.0, float(relation_teacher_conditioned_margin or 0.0))
            conditioned_target_coverage = max(
                0.0,
                min(1.0, float(relation_teacher_conditioned_margin_target_coverage or 0.0)),
            )
            margin_weight = float(relation_teacher_margin_weight or 0.0)
            margin_value = float(relation_teacher_margin or 0.0)
            margin_active_only = bool(relation_teacher_margin_active_only)
            margin_loss_cap = max(0.0, float(relation_teacher_margin_loss_cap or 0.0))
            if query_weight > 0 or pair_weight > 0 or margin_weight > 0 or conditioned_margin_weight > 0:
                weighted_distill = torch.zeros(1, device=scores_re.device, dtype=scores_re.dtype)
                weighted_terms = 0.0
                if relation_teacher_targets is not None:
                    try:
                        target = relation_teacher_targets.to(device=scores_re.device, dtype=scores_re.dtype)
                        if query_weight > 0 and target.shape == scores_re.shape:
                            mass = target.sum(dim=1)
                            mask = mass > 0
                            if bool(mask.any().item()):
                                normalized_target = target / torch.clamp(mass.unsqueeze(1), min=1e-12)
                                per_row_loss = -(normalized_target[mask] * scores_re[mask]).sum(dim=1)
                                row_weights = torch.clamp(mass[mask], min=1e-6, max=1.0)
                                row_count = torch.clamp(mask.sum().to(dtype=scores_re.dtype), min=1.0)
                                query_loss = (per_row_loss * row_weights).sum() / row_count
                                loss_rel = loss_rel + query_weight * query_loss
                                weighted_distill = weighted_distill + query_weight * query_loss
                                weighted_terms += query_weight
                                self.last_relation_distill_query_rows = int(mask.sum().item())
                                self.last_relation_distill_query_loss = float(query_loss.detach().item())
                    except Exception:
                        self.last_relation_distill_query_rows = 0
                if relation_teacher_pair_weights is not None:
                    try:
                        pair_weights = relation_teacher_pair_weights.to(device=scores_re.device, dtype=scores_re.dtype)
                        if pair_weight > 0 and pair_weights.dim() == 2 and pair_weights.size(0) == scores_re.size(1) and pair_weights.size(1) == scores_re.size(1):
                            pred_top = torch.argmax(score_re.detach(), dim=1)
                            gold_rel = all_triples[:, 1].long()
                            row_weights = pair_weights[pred_top, gold_rel]
                            mask = row_weights > 0
                            if bool(mask.any().item()):
                                per_row_nll = F.nll_loss(scores_re, gold_rel, reduction='none')
                                weighted = per_row_nll[mask] * row_weights[mask]
                                pair_loss = weighted.sum() / torch.clamp(row_weights[mask].sum(), min=1e-12)
                                loss_rel = loss_rel + pair_weight * pair_loss
                                weighted_distill = weighted_distill + pair_weight * pair_loss
                                weighted_terms += pair_weight
                                self.last_relation_distill_pair_rows = int(mask.sum().item())
                                self.last_relation_distill_pair_loss = float(pair_loss.detach().item())
                    except Exception:
                        self.last_relation_distill_pair_rows = 0
                if (
                    relation_teacher_provenance_source_weights is not None
                    or relation_teacher_provenance_pair_weights is not None
                ):
                    try:
                        if conditioned_margin_weight > 0 and conditioned_margin_value > 0:
                            pred_top = torch.argmax(score_re.detach(), dim=1)
                            gold_rel = all_triples[:, 1].long()
                            row_weights = None
                            if relation_teacher_provenance_source_weights is not None:
                                source_weights = relation_teacher_provenance_source_weights.to(
                                    device=scores_re.device,
                                    dtype=scores_re.dtype,
                                )
                                if source_weights.shape == scores_re.shape:
                                    row_weights = source_weights.gather(1, pred_top.view(-1, 1)).squeeze(1)
                            if row_weights is None and relation_teacher_provenance_pair_weights is not None:
                                provenance_weights = relation_teacher_provenance_pair_weights.to(
                                    device=scores_re.device,
                                    dtype=scores_re.dtype,
                                )
                                if (
                                    provenance_weights.dim() == 2
                                    and provenance_weights.size(0) == scores_re.size(1)
                                    and provenance_weights.size(1) == scores_re.size(1)
                                ):
                                    row_weights = provenance_weights[pred_top, gold_rel]
                            if row_weights is None:
                                raise ValueError("invalid provenance source/pair weight shape")
                            mask = (row_weights > 0) & (pred_top != gold_rel)
                            if bool(mask.any().item()):
                                gold_scores = scores_re.gather(1, gold_rel.view(-1, 1)).squeeze(1)
                                predicted_scores = scores_re.gather(1, pred_top.view(-1, 1)).squeeze(1)
                                violations = F.relu(
                                    conditioned_margin_value - (gold_scores - predicted_scores)
                                ) * row_weights
                                active_mask = mask & (violations > 0)
                                normalization = sparse_coverage_normalization(
                                    batch_rows=int(scores_re.size(0)),
                                    matched_rows=int(mask.sum().item()),
                                    target_coverage=conditioned_target_coverage,
                                )
                                conditioned_loss = violations[mask].sum() / normalization.denominator
                                loss_rel = loss_rel + conditioned_margin_weight * conditioned_loss
                                weighted_distill = weighted_distill + conditioned_margin_weight * conditioned_loss
                                weighted_terms += conditioned_margin_weight
                                self.last_relation_distill_conditioned_margin_loss = float(
                                    conditioned_loss.detach().item()
                                )
                                self.last_relation_distill_conditioned_margin_rows = int(mask.sum().item())
                                self.last_relation_distill_conditioned_margin_active_rows = int(active_mask.sum().item())
                                self.last_relation_distill_conditioned_margin_coverage = float(
                                    normalization.observed_coverage
                                )
                                self.last_relation_distill_conditioned_margin_amplification = float(
                                    normalization.amplification
                                )
                    except Exception as exc:
                        self.last_relation_distill_conditioned_margin_rows = 0
                        self.last_relation_distill_conditioned_margin_active_rows = 0
                        self.last_relation_distill_conditioned_margin_error = type(exc).__name__
                if relation_teacher_margin_negatives is not None:
                    try:
                        neg_weights = relation_teacher_margin_negatives.to(device=scores_re.device, dtype=scores_re.dtype)
                        if margin_weight > 0 and neg_weights.shape == scores_re.shape and margin_value > 0:
                            gold_rel = all_triples[:, 1].long()
                            neg_weights = neg_weights.clone()
                            neg_weights.scatter_(1, gold_rel.view(-1, 1), 0.0)
                            row_mass = neg_weights.sum(dim=1)
                            mask = row_mass > 0
                            if bool(mask.any().item()):
                                pos_scores = scores_re.gather(1, gold_rel.view(-1, 1))
                                raw_margin = margin_value - (pos_scores - scores_re)
                                weighted_margin = F.relu(raw_margin) * neg_weights
                                row_violation = weighted_margin.sum(dim=1)
                                active_mask = mask & (row_violation > 0)
                                per_row_margin = row_violation / torch.clamp(row_mass, min=1e-12)
                                train_mask = active_mask if margin_active_only else mask
                                if bool(train_mask.any().item()):
                                    raw_margin_loss = per_row_margin[train_mask].mean()
                                    margin_scale = torch.ones(1, device=scores_re.device, dtype=scores_re.dtype)
                                    if margin_loss_cap > 0:
                                        cap = torch.as_tensor(margin_loss_cap, device=scores_re.device, dtype=scores_re.dtype)
                                        margin_scale = torch.clamp(
                                            cap / torch.clamp(raw_margin_loss.detach(), min=1e-12),
                                            max=1.0,
                                        )
                                    margin_loss = raw_margin_loss * margin_scale
                                    loss_rel = loss_rel + margin_weight * margin_loss
                                    weighted_distill = weighted_distill + margin_weight * margin_loss
                                    weighted_terms += margin_weight
                                    self.last_relation_distill_margin_raw_loss = float(raw_margin_loss.detach().item())
                                    self.last_relation_distill_margin_loss = float(margin_loss.detach().item())
                                    self.last_relation_distill_margin_scale = float(margin_scale.detach().item())
                                self.last_relation_distill_margin_rows = int(mask.sum().item())
                                self.last_relation_distill_margin_active_rows = int(active_mask.sum().item())
                    except Exception:
                        self.last_relation_distill_margin_rows = 0
                        self.last_relation_distill_margin_active_rows = 0
                if weighted_terms > 0:
                    try:
                        self.last_relation_distill_loss = float((weighted_distill / float(weighted_terms)).detach().item())
                    except Exception:
                        self.last_relation_distill_loss = 0.0


        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_ob = self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents)
        score = F.softmax(scores_ob, dim=1)
        return score

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding = global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2 * self.num_rels)
        score = F.softmax(scores_re, dim=1)
        return score

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h
