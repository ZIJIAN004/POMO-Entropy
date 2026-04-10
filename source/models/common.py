"""Encoder / Decoder 通用组件，TSP / CVRP / VRPTW 共用。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, ff_hidden_dim):
        super().__init__()
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.norm1   = nn.InstanceNorm1d(embedding_dim, affine=True)
        self.ff      = FeedForward(embedding_dim, ff_hidden_dim)
        self.norm2   = nn.InstanceNorm1d(embedding_dim, affine=True)
        self.head_num = head_num

    def forward(self, x):
        h = self.head_num
        q = _reshape(self.Wq(x), h)
        k = _reshape(self.Wk(x), h)
        v = _reshape(self.Wv(x), h)
        out = self.multi_head_combine(_mha(q, k, v))
        x = self.norm1((x + out).transpose(1, 2)).transpose(1, 2)
        x = self.norm2((x + self.ff(x)).transpose(1, 2)).transpose(1, 2)
        return x


class FeedForward(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.W1 = nn.Linear(d, h)
        self.W2 = nn.Linear(h, d)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


def _reshape(x, h):
    b, n, _ = x.shape
    return x.reshape(b, n, h, -1).transpose(1, 2)


def _mha(q, k, v, mask=None):
    b, h, n, d = q.shape
    score = torch.matmul(q, k.transpose(2, 3)) / d ** 0.5
    if mask is not None:
        score = score + mask[:, None, :, :].expand(b, h, n, k.size(2))
    w = F.softmax(score, dim=3)
    return torch.matmul(w, v).transpose(1, 2).reshape(b, n, h * d)


def get_encoding(encoded_nodes, node_index):
    b, p    = node_index.shape
    emb_dim = encoded_nodes.size(2)
    idx     = node_index[:, :, None].expand(b, p, emb_dim)
    return encoded_nodes.gather(dim=1, index=idx)
