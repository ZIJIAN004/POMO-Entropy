"""CVRP Model for POMO — 纯 POMO，无 value head。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import EncoderLayer, _reshape, _mha, get_encoding


class CVRPModel(nn.Module):
    def __init__(self, embedding_dim, encoder_layer_num, head_num, qkv_dim,
                 ff_hidden_dim, logit_clipping, **kw):
        super().__init__()
        self.logit_clipping     = logit_clipping
        self.sqrt_embedding_dim = embedding_dim ** 0.5
        self.head_num           = head_num

        # Encoder
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node  = nn.Linear(3, embedding_dim)
        self.encoder_layers  = nn.ModuleList([
            EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim)
            for _ in range(encoder_layer_num)
        ])

        # Decoder：context = last_node_embedding + load
        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk      = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv      = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        node_xy_demand = torch.cat(
            (reset_state.node_xy, reset_state.node_demand[:, :, None]), dim=2)
        x = torch.cat((self.embedding_depot(reset_state.depot_xy),
                        self.embedding_node(node_xy_demand)), dim=1)
        for layer in self.encoder_layers:
            x = layer(x)
        self.encoded_nodes   = x
        self.k               = _reshape(self.Wk(x), self.head_num)
        self.v               = _reshape(self.Wv(x), self.head_num)
        self.single_head_key = x.transpose(1, 2)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size  = state.BATCH_IDX.size(1)
        dev        = state.BATCH_IDX.device

        if state.selected_count == 0:
            selected = torch.zeros(batch_size, pomo_size, dtype=torch.long, device=dev)
            prob     = torch.ones(batch_size, pomo_size, device=dev)

        elif state.selected_count == 1:
            selected = torch.arange(1, pomo_size + 1, device=dev)[None, :].expand(batch_size, -1)
            prob     = torch.ones(batch_size, pomo_size, device=dev)

        else:
            enc_last = get_encoding(self.encoded_nodes, state.current_node)
            q_input  = torch.cat((enc_last, state.load[:, :, None]), dim=2)
            q        = _reshape(self.Wq_last(q_input), self.head_num)
            mh_out   = self.multi_head_combine(_mha(q, self.k, self.v, mask=state.ninf_mask))
            score    = torch.matmul(mh_out, self.single_head_key) / self.sqrt_embedding_dim
            score    = self.logit_clipping * torch.tanh(score) + state.ninf_mask
            probs    = F.softmax(score, dim=2)

            if self.training:
                while True:
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                                        .squeeze(1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected]
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob     = None

        return selected, prob
