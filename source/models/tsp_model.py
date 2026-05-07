"""TSP Model for POMO — 标准 Attention Model，无 depot 区分。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import EncoderLayer, _reshape, _mha, get_encoding


class TSPModel(nn.Module):
    def __init__(self, embedding_dim, encoder_layer_num, head_num, qkv_dim,
                 ff_hidden_dim, logit_clipping, **kw):
        super().__init__()
        self.logit_clipping     = logit_clipping
        self.sqrt_embedding_dim = embedding_dim ** 0.5
        self.head_num           = head_num

        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim)
            for _ in range(encoder_layer_num)
        ])

        # Decoder：query = first_node + last_node
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last  = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk       = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv       = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.encoded_nodes = None
        self.first_node    = None   # 每个 pomo 的起始节点

    def pre_forward(self, reset_state):
        x = self.embedding(reset_state.node_xy)
        for layer in self.encoder_layers:
            x = layer(x)
        self.encoded_nodes = x
        self.k = _reshape(self.Wk(x), self.head_num)
        self.v = _reshape(self.Wv(x), self.head_num)
        self.single_head_key = x.transpose(1, 2)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size  = state.BATCH_IDX.size(1)
        dev        = state.BATCH_IDX.device

        if state.selected_count == 0:
            # POMO 分叉：每个 pomo 从不同节点出发
            selected = torch.arange(pomo_size, device=dev)[None, :].expand(batch_size, -1)
            prob = torch.ones(batch_size, pomo_size, device=dev)
            entropy = torch.zeros(batch_size, pomo_size, device=dev)
            self.first_node = selected
        else:
            enc_first = get_encoding(self.encoded_nodes, self.first_node)
            enc_last  = get_encoding(self.encoded_nodes, state.current_node)

            q = _reshape(self.Wq_first(enc_first), self.head_num) + \
                _reshape(self.Wq_last(enc_last), self.head_num)

            mh_out = self.multi_head_combine(_mha(q, self.k, self.v, mask=state.ninf_mask))
            score  = torch.matmul(mh_out, self.single_head_key) / self.sqrt_embedding_dim
            score  = self.logit_clipping * torch.tanh(score) + state.ninf_mask
            probs  = F.softmax(score, dim=2)

            entropy = -(probs * probs.clamp(min=1e-20).log()).sum(dim=2)

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
                prob = None

        return selected, prob, entropy
