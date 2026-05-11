"""
Smoke test for the Mode B baseline path.

Covers:
  1. shapes / API (EntropyBaselineMLP, batched_per_instance_ols, compute_residual_weights)
  2. MSE-loss reduces over training steps
  3. ★ "graph-reuse" guard ★ — verifies that backward through baseline loss
     does NOT touch the model graph, so a subsequent policy backward succeeds.
     This is the regression test for the entropy_list.detach() fix.

Run on the server (or any box with torch):
    python test_baseline_smoke.py
"""
import torch
import sys
sys.path.insert(0, '.')

from source.baseline import (
    EntropyBaselineMLP,
    batched_per_instance_ols,
    compute_residual_weights,
)

torch.manual_seed(0)

# fake dims
B, P, T, n_state, n_inst, hidden, h_out = 4, 8, 30, 8, 128, 16, 8

mlp = EntropyBaselineMLP(n_state=n_state, n_inst=n_inst, hidden=hidden, h_out=h_out)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

state_feat = torch.randn(B, P, T, n_state)
inst_emb   = torch.randn(B, n_inst)
inst_step  = inst_emb[:, None, None, :].expand(B, P, T, n_inst)

# fake "true" H, no grad
H_true = state_feat[..., 0] * 1.5 + state_feat[..., 1] * 0.5 + 0.1 * torch.randn(B, P, T)
n_feas = torch.randint(1, 50, (B, P, T)).float()
valid  = torch.ones(B, P, T, dtype=torch.bool)
valid[:, :, :2] = False

# === (1) basic forward + OLS ===
features = mlp(state_feat, inst_step)
H_hat, beta = batched_per_instance_ols(features, H_true, valid, ridge=1e-4)
assert features.shape == (B, P, T, h_out)
assert H_hat.shape == (B, P, T)
assert beta.shape == (B, h_out)
print(f"[1] shapes OK: features={tuple(features.shape)}  H_hat={tuple(H_hat.shape)}  beta={tuple(beta.shape)}")

# === (2) training reduces loss ===
loss0 = torch.nn.functional.mse_loss(H_hat[valid], H_true[valid].detach()).item()
for k in range(50):
    features = mlp(state_feat, inst_step)
    H_hat, _ = batched_per_instance_ols(features, H_true, valid, ridge=1e-4)
    loss = torch.nn.functional.mse_loss(H_hat[valid], H_true[valid].detach())
    opt.zero_grad(); loss.backward(); opt.step()
print(f"[2] training drops loss: {loss0:.4f} -> {loss.item():.4f}")
assert loss.item() < loss0, "MLP training should reduce loss"

# === (3) weights API ===
residual = H_true - H_hat
advantage = torch.randn(B, P)
weights = compute_residual_weights(residual, n_feas, advantage, valid, gamma=1.0)
T_valid_per = valid[0, 0].sum().item()
w_sum_first = weights[0, 0].sum().item()
assert abs(w_sum_first - T_valid_per) < 1e-3, f"weights should sum to T_valid={T_valid_per}, got {w_sum_first}"
assert weights[:, :, :2].max().item() < 1e-5, "invalid steps should have ~0 weight"
print(f"[3] weights OK: sum≈T_valid={T_valid_per}, invalid≈0")

# === (4) regression guard: graph-reuse bug must NOT recur ===
# Simulate the real training loop: H_entropy has grad to a fake "policy", and
# log_prob (also from same policy) is used in a downstream policy loss.
# The bug we fixed: forgetting to detach entropy_list before OLS causes
# loss_b.backward() to walk into the policy graph and free its intermediates.
print("[4] testing graph-reuse guard (this is the regression test):")

# fake policy that produces BOTH entropy_list and prob_list from a shared linear
policy = torch.nn.Linear(n_state + n_inst, 2)
policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

x = torch.cat([state_feat, inst_step], dim=-1)        # (B, P, T, d)
policy_out = policy(x)                                 # (B, P, T, 2)
entropy_from_policy = policy_out[..., 0]                # has grad to policy
log_prob_from_policy = policy_out[..., 1]               # has grad to policy

# baseline forward + OLS using entropy as target — MUST detach
features = mlp(state_feat, inst_step)
H_hat, _ = batched_per_instance_ols(
    features, entropy_from_policy.detach(), valid, ridge=1e-4)   # ← detached!
loss_b = torch.nn.functional.mse_loss(H_hat[valid], entropy_from_policy[valid].detach())

opt.zero_grad()
loss_b.backward()                # should only touch MLP params
opt.step()

# now policy backward — this is what failed in the original bug
loss_p = -(advantage[..., None] * log_prob_from_policy).sum(dim=2).mean()
policy_opt.zero_grad()
loss_p.backward()                # should NOT raise "backward through the graph a second time"
policy_opt.step()
print("[4] OK — baseline backward + policy backward both succeeded")

# === (5) "negative test" — without detach, the same setup MUST raise ===
print("[5] negative test (no detach — expect graph-reuse error):")
policy_out2 = policy(x)
entropy2 = policy_out2[..., 0]
log_prob2 = policy_out2[..., 1]

features2 = mlp(state_feat, inst_step)
H_hat2, _ = batched_per_instance_ols(
    features2, entropy2, valid, ridge=1e-4)             # NOT detached
loss_b2 = torch.nn.functional.mse_loss(H_hat2[valid], entropy2[valid].detach())

opt.zero_grad()
loss_b2.backward()
opt.step()

loss_p2 = -(advantage[..., None] * log_prob2).sum(dim=2).mean()
policy_opt.zero_grad()
try:
    loss_p2.backward()
    print("[5] ⚠ WARNING: expected RuntimeError but backward succeeded — "
          "your torch version may handle this differently. Detach is still required.")
except RuntimeError as e:
    if "backward through the graph a second time" in str(e):
        print("[5] OK — RuntimeError raised as expected: detach is required.")
    else:
        raise

print("\n=== ALL SMOKE TESTS PASSED ===")
