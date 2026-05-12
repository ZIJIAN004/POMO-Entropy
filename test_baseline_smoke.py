"""
Smoke test for the Mode B (MLP) baseline path — NEW single-input interface.

Covers:
  1. shape / API for EntropyBaselineMLP(n_in, hidden, h_out)
  2. batched_per_instance_ols + compute_residual_weights shapes
  3. MSE-loss reduces over training steps
  4. graph-reuse regression guard (detach() target before backward through baseline)
  5. e2e shape check for the per-step input builder mlp_input_dim(...)
     using a mock model + state + env.

Run on the server:
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
from source.TRAIN_N_EVAL.Train import mlp_input_dim, _build_step_emb
from source.models.common import get_encoding

torch.manual_seed(0)

EMBED_DIM = 128
B, P, T   = 4, 8, 30
H_OUT     = 8
HIDDEN    = 64


# === (1) basic forward + OLS shapes ===
n_in_tsp = mlp_input_dim('tsp', EMBED_DIM)
mlp = EntropyBaselineMLP(n_in=n_in_tsp, hidden=HIDDEN, h_out=H_OUT)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

x = torch.randn(B, P, T, n_in_tsp)
H_true = x[..., 0] * 1.5 + x[..., 1] * 0.5 + 0.1 * torch.randn(B, P, T)
n_feas = torch.randint(1, 50, (B, P, T)).float()
valid  = torch.ones(B, P, T, dtype=torch.bool)
valid[:, :, :2] = False

phi = mlp(x)
H_hat, beta = batched_per_instance_ols(phi, H_true, valid, ridge=1e-4)
assert phi.shape == (B, P, T, H_OUT), phi.shape
assert H_hat.shape == (B, P, T)
assert beta.shape == (B, H_OUT)
print(f"[1] shapes OK: phi={tuple(phi.shape)}  H_hat={tuple(H_hat.shape)}  beta={tuple(beta.shape)}")
print(f"    mlp_input_dim: tsp={n_in_tsp}  cvrp={mlp_input_dim('cvrp', EMBED_DIM)}  vrptw={mlp_input_dim('vrptw', EMBED_DIM)}")


# === (2) training reduces loss ===
loss0 = torch.nn.functional.mse_loss(H_hat[valid], H_true[valid].detach()).item()
for _ in range(50):
    phi = mlp(x)
    H_hat, _ = batched_per_instance_ols(phi, H_true, valid, ridge=1e-4)
    loss = torch.nn.functional.mse_loss(H_hat[valid], H_true[valid].detach())
    opt.zero_grad(); loss.backward(); opt.step()
print(f"[2] training drops loss: {loss0:.4f} -> {loss.item():.4f}")
assert loss.item() < loss0, "MLP training should reduce loss"


# === (3) compute_residual_weights API ===
residual = H_true - H_hat
advantage = torch.randn(B, P)
weights = compute_residual_weights(residual, n_feas, advantage, valid, gamma=1.0)
T_valid_per = valid[0, 0].sum().item()
w_sum_first = weights[0, 0].sum().item()
assert abs(w_sum_first - T_valid_per) < 1e-3, (w_sum_first, T_valid_per)
assert weights[:, :, :2].max().item() < 1e-5, "invalid steps should have ~0 weight"
print(f"[3] weights OK: sum≈T_valid={T_valid_per}, invalid≈0")


# === (4) graph-reuse regression guard ===
print("[4] graph-reuse guard:")
policy = torch.nn.Linear(n_in_tsp, 2)
policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

policy_out = policy(x)
entropy_p = policy_out[..., 0]
logp_p    = policy_out[..., 1]

phi2 = mlp(x)
H_hat2, _ = batched_per_instance_ols(phi2, entropy_p.detach(), valid, ridge=1e-4)
loss_b = torch.nn.functional.mse_loss(H_hat2[valid], entropy_p[valid].detach())
opt.zero_grad(); loss_b.backward(); opt.step()

loss_p = -(advantage[..., None] * logp_p).sum(dim=2).mean()
policy_opt.zero_grad()
loss_p.backward()                # must NOT raise
policy_opt.step()
print("[4] OK — baseline backward + policy backward both succeeded")


# === (5) e2e shape check for _build_step_emb on mock objects ===
print("[5] _build_step_emb shape check:")

class MockTSPModel:
    def __init__(self, B, N, D):
        self.encoded_nodes = torch.randn(B, N, D)
        self.first_node    = torch.randint(0, N, (B, P))

class MockCVRPModel:
    def __init__(self, B, N1, D):                # N1 = N+1 (incl. depot)
        self.encoded_nodes = torch.randn(B, N1, D)

class MockState:
    pass

class MockTSPEnv:
    def __init__(self, B, N):
        self.node_xy = torch.rand(B, N, 2)

class MockCVRPEnv:
    def __init__(self, B, N1):
        self.depot_node_xy = torch.rand(B, N1, 2)

# TSP
N = 20
m_tsp  = MockTSPModel(B, N, EMBED_DIM)
e_tsp  = MockTSPEnv(B, N)
s_tsp  = MockState()
s_tsp.BATCH_IDX      = torch.arange(B)[:, None].expand(B, P)
s_tsp.POMO_IDX       = torch.arange(P)[None, :].expand(B, P)
s_tsp.current_node   = torch.randint(0, N, (B, P))
s_tsp.n_feasible     = torch.randint(1, N+1, (B, P))
s_tsp.selected_count = 5

emb = _build_step_emb(m_tsp, s_tsp, e_tsp, 'tsp', n_in_tsp)
assert emb.shape == (B, P, n_in_tsp), emb.shape
print(f"    tsp: {tuple(emb.shape)} (expected ({B}, {P}, {n_in_tsp}))")

# TSP forced step (current_node is None) → zeros
s_tsp.current_node = None
emb = _build_step_emb(m_tsp, s_tsp, e_tsp, 'tsp', n_in_tsp)
assert emb.shape == (B, P, n_in_tsp)
assert emb.abs().max().item() == 0.0
print(f"    tsp forced step → all zeros ✓")

# CVRP
N1 = 21  # 1 depot + 20 customers
n_in_cvrp = mlp_input_dim('cvrp', EMBED_DIM)
m_cv = MockCVRPModel(B, N1, EMBED_DIM)
e_cv = MockCVRPEnv(B, N1)
s_cv = MockState()
s_cv.BATCH_IDX               = torch.arange(B)[:, None].expand(B, P)
s_cv.POMO_IDX                = torch.arange(P)[None, :].expand(B, P)
s_cv.current_node            = torch.randint(0, N1, (B, P))
s_cv.n_feasible              = torch.randint(1, N1+1, (B, P))
s_cv.selected_count          = 7
s_cv.load                    = torch.rand(B, P)
s_cv.visited_customer_count  = torch.randint(0, N1, (B, P)).float()

emb = _build_step_emb(m_cv, s_cv, e_cv, 'cvrp', n_in_cvrp)
assert emb.shape == (B, P, n_in_cvrp), emb.shape
print(f"    cvrp: {tuple(emb.shape)} (expected ({B}, {P}, {n_in_cvrp}))")

# VRPTW
n_in_tw = mlp_input_dim('vrptw', EMBED_DIM)
m_tw = MockCVRPModel(B, N1, EMBED_DIM)            # same shape
e_tw = MockCVRPEnv(B, N1)
s_tw = MockState()
s_tw.BATCH_IDX               = torch.arange(B)[:, None].expand(B, P)
s_tw.POMO_IDX                = torch.arange(P)[None, :].expand(B, P)
s_tw.current_node            = torch.randint(0, N1, (B, P))
s_tw.n_feasible              = torch.randint(1, N1+1, (B, P))
s_tw.selected_count          = 7
s_tw.current_time            = torch.rand(B, P)
s_tw.visited_customer_count  = torch.randint(0, N1, (B, P)).float()
emb = _build_step_emb(m_tw, s_tw, e_tw, 'vrptw', n_in_tw)
assert emb.shape == (B, P, n_in_tw), emb.shape
print(f"    vrptw: {tuple(emb.shape)} (expected ({B}, {P}, {n_in_tw}))")

# (6) MLP forward on a stacked feat_list mimic
feat_list = []
for _ in range(T):
    feat_list.append(_build_step_emb(m_tsp, s_tsp, e_tsp, 'tsp', n_in_tsp))   # forced - zeros
s_tsp.current_node = torch.randint(0, N, (B, P))
for _ in range(T):
    feat_list.append(_build_step_emb(m_tsp, s_tsp, e_tsp, 'tsp', n_in_tsp))   # real
features = torch.stack(feat_list, dim=2)
mlp_t = EntropyBaselineMLP(n_in=n_in_tsp, hidden=HIDDEN, h_out=H_OUT)
phi_t = mlp_t(features)
assert phi_t.shape == (B, P, 2*T, H_OUT)
print(f"[6] end-to-end stack → MLP: {tuple(phi_t.shape)}")


print("\n=== ALL SMOKE TESTS PASSED ===")
