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
    H = entropy.reshape(-1)
    log_F = torch.log(n_feasible.float().reshape(-1).clamp(min=1))

    log_F_mean = log_F.mean()
    H_mean = H.mean()
    log_F_centered = log_F - log_F_mean
    denom = log_F_centered.square().sum().clamp(min=1e-8)
    a = (H * log_F_centered).sum() / denom
    b = H_mean - a * log_F_mean

    residual = H - (a * log_F + b)
    nf_flat  = n_feasible.reshape(-1).long()

    max_nf   = nf_flat.max() + 1
    ones     = torch.ones_like(H)
    counts   = torch.zeros(max_nf, device=H.device).scatter_add_(0, nf_flat, ones)
    sums     = torch.zeros(max_nf, device=H.device).scatter_add_(0, nf_flat, residual)
    grp_mean = (sums / counts.clamp(min=1))[nf_flat]

    centered = residual - grp_mean
    sq_sums  = torch.zeros(max_nf, device=H.device).scatter_add_(0, nf_flat, centered.square())
    grp_std  = ((sq_sums / counts.clamp(min=1)).sqrt().clamp(min=1e-8))[nf_flat]

    delta_H  = (centered / grp_std).reshape_as(entropy)

    sign_A = advantage.sign().unsqueeze(2)
    logit  = gamma * sign_A * delta_H
    T      = entropy.size(2)
    w      = torch.softmax(logit, dim=2) * T

    return w
