"""Smoke test for Mode B baseline — runs on CPU, no real rollout."""
import torch
import sys
sys.path.insert(0, '.')

from source.baseline import (
    EntropyBaselineMLP,
    batched_per_instance_ols,
    compute_mode_b_weights,
)

torch.manual_seed(0)

# fake dims
B, P, T, n_state, n_inst, hidden, h_out = 4, 8, 30, 8, 128, 16, 8

mlp = EntropyBaselineMLP(n_state=n_state, n_inst=n_inst, hidden=hidden, h_out=h_out)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

state_feat = torch.randn(B, P, T, n_state)
inst_emb   = torch.randn(B, n_inst)
inst_step  = inst_emb[:, None, None, :].expand(B, P, T, n_inst)

# fake entropy: roughly correlated with one of the features
H = state_feat[..., 0] * 1.5 + state_feat[..., 1] * 0.5 + 0.1 * torch.randn(B, P, T)

# fake n_feasible
n_feas = torch.randint(1, 50, (B, P, T)).float()

# valid mask: skip first 2 steps
valid = torch.ones(B, P, T, dtype=torch.bool)
valid[:, :, :2] = False

# === forward + OLS ===
features = mlp(state_feat, inst_step)
H_hat, beta = batched_per_instance_ols(features, H, valid, ridge=1e-4)
print("features:", features.shape, "  H_hat:", H_hat.shape, "  beta:", beta.shape)

# train step
loss = torch.nn.functional.mse_loss(H_hat[valid], H[valid].detach())
opt.zero_grad()
loss.backward()
opt.step()
print(f"initial loss: {loss.item():.4f}")

# run a few steps to verify training actually reduces loss
for k in range(50):
    features = mlp(state_feat, inst_step)
    H_hat, _ = batched_per_instance_ols(features, H, valid, ridge=1e-4)
    loss = torch.nn.functional.mse_loss(H_hat[valid], H[valid].detach())
    opt.zero_grad(); loss.backward(); opt.step()
print(f"after 50 steps: {loss.item():.4f}")

# === mode-B weights ===
residual = H - H_hat
advantage = torch.randn(B, P)
weights = compute_mode_b_weights(residual, n_feas, advantage, valid, gamma=1.0)
print(f"weights shape: {weights.shape}")
print(f"weights sum per (B,P): {weights.sum(dim=2)[0, :5].tolist()}  (should ≈ T_valid={valid[0, 0].sum().item()})")
print(f"weights min/max: {weights.min().item():.4f}, {weights.max().item():.4f}")
print(f"weights at invalid positions (should be ~0): {weights[:, :, :2].max().item():.4e}")

print("\n=== ALL SMOKE TESTS PASSED ===")
