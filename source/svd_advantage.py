"""
SVD-Reward integration for POMO-Entropy training.

Bridges POMO rollouts (tensor-based env) into SVD-Reward's PyG-based GNN
autoencoder, then mixes the SVD-similarity advantage with the standard
POMO cost-baseline advantage (hybrid, not replacement).

Hot path (per training batch):
    1. _build_pyg_inputs(env.node_xy, env.selected_node_list)
        — pure-tensor GPU construction of (x, edge_index, instance_edge_index,
          batch). No Python loop, no PyG Batch.from_data_list.
    2. ae.encode(data)
        — runs InstanceEncoder + TourEncoder on the batched graph; returns
          z of shape (B·P, D).
    3. compute_hybrid_advantage(z, rewards, alpha, rank, top_k)
        — per-instance SVD on top-k anchors → residual → z-scored, mixed
          with z-scored (−length) cost reward.

Cross-project import:
    POMO-Entropy and SVD-Reward live as sibling directories under the user's
    code root. We resolve SVD-Reward via (in order):
      1. SVD_REWARD_PATH env var
      2. ../SVD-Reward relative to POMO-Entropy root
      3. ~/SVD-Reward
"""

import os
import sys

import torch

# ── Locate SVD-Reward and import its modules ───────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_POMO_ENTROPY_ROOT = os.path.dirname(_THIS_DIR)
_SVD_CANDIDATES = [
    os.environ.get('SVD_REWARD_PATH'),
    os.path.abspath(os.path.join(_POMO_ENTROPY_ROOT, '..', 'SVD-Reward')),
    os.path.expanduser('~/SVD-Reward'),
]
_SVD_PATH = None
for _cand in _SVD_CANDIDATES:
    if _cand and os.path.exists(os.path.join(_cand, 'svd_reward.py')):
        _SVD_PATH = _cand
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

if _SVD_PATH is None:
    raise ImportError(
        "SVD-Reward project not found. Tried: "
        f"{[c for c in _SVD_CANDIDATES if c]}. "
        "Set SVD_REWARD_PATH env var or place SVD-Reward as sibling of POMO-Entropy.")

from svd_reward import per_instance_reward_torch, topk_anchor_idx  # noqa: E402
from model import TourAutoEncoder                                   # noqa: E402
from config import Config as SVDConfig                              # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
#  AE checkpoint loading
# ──────────────────────────────────────────────────────────────────────────

def load_ae(ckpt_path: str, device: torch.device) -> TourAutoEncoder:
    """Load the pretrained TourAutoEncoder. eval mode, no grad."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = SVDConfig(**ckpt["config"])
    ae = TourAutoEncoder(cfg).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


# ──────────────────────────────────────────────────────────────────────────
#  GPU-batched PyG input construction (replaces Python loop + Batch.from_data_list)
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _instance_graph_inputs(node_xy: torch.Tensor, knn_k: int):
    """Per-instance KNN graph for InstanceEncoder. B graphs, N nodes each.

    Avoids the P-fold replication that would be needed if we lumped all
    B·P rollouts into one big graph batch — instance encoding only depends
    on coords, which are shared across the P rollouts of an instance.

    Returns:
        x_inst:       (B·N, 2)
        inst_ei:      (2, 2·B·N·k)        KNN edges (undirected) with per-instance offset
        batch_inst:   (B·N,)              instance index per node
    """
    B, N, _ = node_xy.shape
    device = node_xy.device

    x_inst = node_xy.reshape(-1, 2).contiguous()                           # (B·N, 2)
    batch_inst = torch.arange(B, device=device).repeat_interleave(N)       # (B·N,)

    dists = torch.cdist(node_xy, node_xy)                                  # (B, N, N)
    dists.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    _, knn_idx = dists.topk(knn_k, dim=2, largest=False)                   # (B, N, k)

    src_inst = (torch.arange(N, device=device)
                .view(1, N, 1).expand(B, N, knn_k))                        # (B, N, k)
    inst_offset = (torch.arange(B, device=device) * N).view(B, 1, 1)       # (B, 1, 1)
    src_g = (src_inst + inst_offset).reshape(-1)                           # (B·N·k,)
    dst_g = (knn_idx   + inst_offset).reshape(-1)
    inst_ei = torch.stack([
        torch.cat([src_g, dst_g], dim=0),
        torch.cat([dst_g, src_g], dim=0),
    ], dim=0).contiguous()                                                 # (2, 2·B·N·k)

    return x_inst, inst_ei, batch_inst


@torch.no_grad()
def _tour_graph_inputs(selected_list: torch.Tensor):
    """Tour Hamilton-cycle graph for TourEncoder. B·P graphs, N nodes each.

    Returns:
        tour_ei:    (2, 4·B·P·N)         undirected tour edges with per-rollout offset
        batch_tour: (B·P·N,)             rollout index per node (graph id in 0..B·P-1)
    """
    B, P, N = selected_list.shape
    device = selected_list.device

    batch_tour = torch.arange(B * P, device=device).repeat_interleave(N)
    graph_offset = (torch.arange(B * P, device=device) * N).view(B, P, 1)

    src = selected_list                                                    # (B, P, N)
    dst = torch.roll(selected_list, shifts=-1, dims=-1)
    src_g = (src + graph_offset).reshape(-1)
    dst_g = (dst + graph_offset).reshape(-1)
    tour_ei = torch.stack([
        torch.cat([src_g, dst_g], dim=0),
        torch.cat([dst_g, src_g], dim=0),
    ], dim=0).contiguous()                                                 # (2, 4·B·P·N)

    return tour_ei, batch_tour


@torch.no_grad()
def encode_rollouts_to_z(ae: TourAutoEncoder,
                          node_xy: torch.Tensor,
                          selected_list: torch.Tensor,
                          knn_k: int = 10) -> torch.Tensor:
    """
    Two-stage GPU-batched encode:
      1. InstanceEncoder on B instance graphs → h_inst_per_inst (B·N, hidden)
      2. Replicate h_inst per P rollouts → h_inst (B·P·N, hidden)
      3. TourEncoder on B·P tour graphs → z (B·P, embedding_dim)

    Avoids the P-fold redundancy of running InstanceEncoder once per rollout
    (instance geometry is identical across the P rollouts of an instance).
    For B=64, P=100, N=100, k=10 this saves ~100× on instance-stream FLOPs
    and instance_edge_index memory.

    Returns:
        z: (B, P, D)
    """
    B, P, N = selected_list.shape

    # Stage 1: encode instance once per instance.
    x_inst, inst_ei, _ = _instance_graph_inputs(node_xy, knn_k=knn_k)
    h_inst_per_inst = ae.instance_encoder(x_inst, inst_ei)                 # (B·N, hidden)

    # Stage 2: replicate h_inst across P rollouts of each instance.
    hidden = h_inst_per_inst.shape[-1]
    h_inst = (h_inst_per_inst.view(B, N, hidden)
              .unsqueeze(1).expand(B, P, N, hidden)
              .reshape(-1, hidden).contiguous())                           # (B·P·N, hidden)

    # Stage 3: build tour graphs and run TourEncoder.
    tour_ei, batch_tour = _tour_graph_inputs(selected_list)
    z_flat = ae.tour_encoder(h_inst, tour_ei, batch_tour)                  # (B·P, D)
    return z_flat.view(B, P, -1)


# ──────────────────────────────────────────────────────────────────────────
#  Hybrid advantage (POMO cost-baseline + SVD subspace-similarity, both z-scored)
# ──────────────────────────────────────────────────────────────────────────

def _znorm(r: torch.Tensor) -> torch.Tensor:
    """Per-instance (dim=1) z-score; clamps std to avoid div-by-zero."""
    m = r.mean(dim=1, keepdim=True)
    s = r.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (r - m) / s


@torch.no_grad()
def compute_hybrid_advantage(z: torch.Tensor,
                              rewards: torch.Tensor,
                              alpha: float,
                              rank: int,
                              top_k: int,
                              return_diag: bool = False):
    """
    advantage = α · znorm(−svd_residual) + (1−α) · znorm(rewards)

    Args:
        z:       (B, P, D)   tour embeddings
        rewards: (B, P)      cost reward (= −length); higher = shorter tour
        alpha:   ∈ [0,1]; 0 = baseline POMO, 1 = pure SVD, 0.5 = balanced
        rank:    SVD rank for anchor subspace
        top_k:   number of top-shortest rollouts to use as anchors

    Returns:
        advantage: (B, P)
        diag:      dict (only if return_diag=True), see fields below.
    """
    B, P = rewards.shape

    cost_adv = _znorm(rewards)                                             # (B, P)

    # Anchors = top_k shortest tours per instance (largest reward = shortest).
    anchor_idx = topk_anchor_idx(-rewards, top_k=min(top_k, P // 2))       # (B, top_k)

    if return_diag:
        raw_svd, svd_diag = per_instance_reward_torch(
            z, anchor_idx, rank=rank, return_diag=True)
    else:
        raw_svd = per_instance_reward_torch(z, anchor_idx, rank=rank)
    svd_adv = _znorm(raw_svd)                                              # (B, P)

    advantage = alpha * svd_adv + (1.0 - alpha) * cost_adv

    if not return_diag:
        return advantage

    # Signal-mix diagnostics — answers: are cost and SVD redundant?
    cs_c = cost_adv - cost_adv.mean(dim=1, keepdim=True)
    sv_c = svd_adv  - svd_adv.mean(dim=1, keepdim=True)
    num = (cs_c * sv_c).sum(dim=1)
    den = (cs_c.norm(dim=1) * sv_c.norm(dim=1)).clamp(min=1e-8)
    cost_svd_corr = (num / den).mean()

    diag = {
        **svd_diag,
        'cost_svd_corr':     cost_svd_corr.detach(),
        'cost_adv_abs_mean': cost_adv.abs().mean().detach(),
        'svd_adv_abs_mean':  svd_adv.abs().mean().detach(),
    }
    return advantage, diag
