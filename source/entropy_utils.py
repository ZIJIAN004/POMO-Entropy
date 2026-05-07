import torch


@torch.no_grad()
def compute_entropy_weights(entropy, n_feasible, advantage, gamma):
    """
    Args:
        entropy:    (batch, pomo, num_steps)
        n_feasible: (batch, pomo, num_steps)
        advantage:  (batch, pomo)
        gamma:      float — softmax temperature

    Returns:
        weights: (batch, pomo, num_steps), sum along dim=2 == num_steps
    """
    B, P, T = entropy.shape

    H     = entropy.reshape(B, -1)
    log_F = torch.log(n_feasible.float().reshape(B, -1).clamp(min=1))

    log_F_mean     = log_F.mean(dim=1, keepdim=True)
    H_mean         = H.mean(dim=1, keepdim=True)
    log_F_centered = log_F - log_F_mean
    denom          = log_F_centered.square().sum(dim=1, keepdim=True).clamp(min=1e-8)
    a = (H * log_F_centered).sum(dim=1, keepdim=True) / denom
    b = H_mean - a * log_F_mean

    residual = H - (a * log_F + b)

    nf      = n_feasible.reshape(B, -1).long()
    max_nf  = nf.max() + 1
    bid     = torch.arange(B, device=entropy.device)[:, None].expand_as(nf)
    gid     = (bid * max_nf + nf).reshape(-1)
    n_grp   = B * max_nf

    res_flat = residual.reshape(-1)
    ones     = torch.ones_like(res_flat)

    counts   = torch.zeros(n_grp, device=entropy.device).scatter_add_(0, gid, ones)
    sums     = torch.zeros(n_grp, device=entropy.device).scatter_add_(0, gid, res_flat)
    grp_mean = (sums / counts.clamp(min=1))[gid]

    centered = res_flat - grp_mean
    sq_sums  = torch.zeros(n_grp, device=entropy.device).scatter_add_(0, gid, centered.square())
    grp_std  = ((sq_sums / counts.clamp(min=1)).sqrt().clamp(min=1e-8))[gid]

    delta_H = (centered / grp_std).reshape(B, P, T)

    sign_A = advantage.sign().unsqueeze(2)
    logit  = gamma * sign_A * delta_H
    w      = torch.softmax(logit, dim=2) * T

    return w
