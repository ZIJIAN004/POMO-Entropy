"""
Diagnose whether the step-entropy structure H_{i,t} on a POMO rollout is
better described as additively or multiplicatively separable in
(trajectory, step).

We posit two competing decompositions per instance (P trajectories × T steps):

    Additive       : H_{i,t}      = α_i + β_t + ε_{i,t}
    Multiplicative : log H_{i,t}  = α_i + β_t + ε_{i,t}   (i.e. H = a_i · b_t · e^ε)

Both are fit by two-way ANOVA (closed-form: row mean + column mean − grand mean).
Residuals are then evaluated *in H-space* so the two models are comparable.

Diagnostics reported:
  • R² (in H-space) — higher = more variance explained
  • Residual variance per Ĥ-quintile + top/bottom ratio — close to 1 means
    homoscedastic (the residuals look like noise of constant scale, the
    assumption the model implicitly makes)

Usage:
    python diagnose_entropy.py                            # random-init, TSP
    python diagnose_entropy.py --problem cvrp
    python diagnose_entropy.py --ckpt result/.../CheckPoint_ep00050
    python diagnose_entropy.py --n_batches 4 --batch_size 64 --size 50
"""
import argparse
import os
import sys
sys.path.insert(0, '.')

import torch

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--problem',    type=str, default='tsp', choices=['tsp', 'cvrp'])
parser.add_argument('--size',       type=int, default=None,
                    help='problem size override (defaults to HYPER_PARAMS.PROBLEM_SIZE)')
parser.add_argument('--n_batches',  type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--ckpt',       type=str, default=None,
                    help='checkpoint directory; omitted → random-init model')
parser.add_argument('--seed',       type=int, default=0)
args = parser.parse_args()

# Override HYPER_PARAMS before the regular import.
import HYPER_PARAMS as _HP
_HP.PROBLEM_TYPE = args.problem
if args.size is not None:
    _HP.PROBLEM_SIZE = args.size
    _HP.POMO_SIZE    = args.size
from HYPER_PARAMS import *

if args.problem == 'tsp':
    from source.models.tsp_model import TSPModel as Model
    from source.envs.tsp_env     import TSPEnv   as Env
else:
    from source.models.cvrp_model import CVRPModel as Model
    from source.envs.cvrp_env     import CVRPEnv   as Env

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model ────────────────────────────────────────────────────────────────────
model = Model(
    embedding_dim     = EMBEDDING_DIM,
    encoder_layer_num = ENCODER_LAYER_NUM,
    head_num          = HEAD_NUM,
    qkv_dim           = QKV_DIM,
    ff_hidden_dim     = FF_HIDDEN_DIM,
    logit_clipping    = LOGIT_CLIPPING,
).to(device)

if args.ckpt:
    state = torch.load(os.path.join(args.ckpt, 'MODEL_state_dic.pt'),
                       map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.ckpt}")
else:
    print("Using random-init model (untrained).")
model.eval()

env = Env(problem_size=PROBLEM_SIZE, pomo_size=POMO_SIZE)
print(f"Problem={args.problem.upper()}  size={PROBLEM_SIZE}  POMO={POMO_SIZE}  "
      f"batches={args.n_batches}×{args.batch_size}  device={device}")


# ── Rollout: collect H and valid_mask ────────────────────────────────────────
all_H, all_valid = [], []
forced_steps = 1 if args.problem == 'tsp' else 2

with torch.no_grad():
    for _ in range(args.n_batches):
        env.load_problems(args.batch_size, device=device)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        H_list   = torch.zeros(args.batch_size, POMO_SIZE, 0, device=device)
        fin_list = (torch.zeros(args.batch_size, POMO_SIZE, 0,
                                dtype=torch.bool, device=device)
                    if args.problem == 'cvrp' else None)

        state, _, done = env.pre_step()
        while not done:
            selected, _, entropy = model(state)
            H_list = torch.cat((H_list, entropy[:, :, None]), dim=2)
            if fin_list is not None:
                fin_list = torch.cat((fin_list, state.finished[:, :, None]), dim=2)
            state, _, done = env.step(selected)

        T_total = H_list.size(2)
        valid   = torch.ones(args.batch_size, POMO_SIZE, T_total,
                              dtype=torch.bool, device=device)
        if T_total > forced_steps:
            valid[:, :, :forced_steps] = False
        if fin_list is not None:
            valid &= ~fin_list

        all_H.append(H_list)
        all_valid.append(valid)

H  = torch.cat(all_H,     dim=0)   # (B_total, P, T)
vm = torch.cat(all_valid, dim=0)
print(f"\nCollected H shape={tuple(H.shape)};  "
      f"valid steps={int(vm.sum())}/{vm.numel()} ({100*vm.float().mean():.1f}%)")
print(f"H stats on valid:  min={H[vm].min():.4f}  max={H[vm].max():.4f}  "
      f"mean={H[vm].mean():.4f}  std={H[vm].std():.4f}")


# ── Two-way ANOVA (per-instance) ─────────────────────────────────────────────
@torch.no_grad()
def two_way_anova(X, vm):
    """Per-instance closed-form two-way ANOVA on the (P, T) slab.
    X, vm: (B, P, T). Returns (pred, resid) in X's space, masked to valid."""
    vmf = vm.float()
    cnt_t = vmf.sum(dim=2, keepdim=True).clamp(min=1.0)
    cnt_s = vmf.sum(dim=1, keepdim=True).clamp(min=1.0)
    cnt_a = vmf.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
    mu_t  = (X * vmf).sum(dim=2, keepdim=True)        / cnt_t   # per-trajectory mean
    mu_s  = (X * vmf).sum(dim=1, keepdim=True)        / cnt_s   # per-step mean
    mu_a  = (X * vmf).sum(dim=(1, 2), keepdim=True)   / cnt_a   # grand mean
    pred  = mu_t + mu_s - mu_a
    resid = (X - pred) * vmf
    return pred, resid


@torch.no_grad()
def diagnostics(name, pred, resid, H_ref, vm):
    """All metrics computed in H-space (resid already in H-space)."""
    H_valid    = H_ref[vm]
    pred_valid = pred[vm]
    res_valid  = resid[vm]

    # R² in H-space, against grand mean of H.
    total_ss = ((H_valid - H_valid.mean()) ** 2).sum()
    resid_ss = (res_valid ** 2).sum()
    r2 = 1.0 - resid_ss / total_ss.clamp(min=1e-12)

    # Heteroscedasticity: residual variance per Ĥ-quintile.
    qs   = torch.tensor([0.2, 0.4, 0.6, 0.8], device=pred_valid.device)
    cuts = torch.quantile(pred_valid, qs)
    bins = torch.bucketize(pred_valid, cuts)
    var_per_bin = []
    for b in range(5):
        sel = (bins == b)
        if int(sel.sum()) < 2:
            var_per_bin.append(float('nan'))
        else:
            var_per_bin.append(float(res_valid[sel].var()))
    finite = [v for v in var_per_bin if v == v]
    ratio  = (var_per_bin[-1] / max(var_per_bin[0], 1e-12)) if len(finite) == 5 else float('nan')

    print(f"\n[{name}]")
    print(f"  R²              = {float(r2):.4f}")
    print(f"  resid var by Ĥ-quintile (low→high):")
    for i, v in enumerate(var_per_bin):
        print(f"    Q{i+1}: {v:.3e}")
    verdict = ("~1 → homoscedastic ✓"
               if (0.5 < ratio < 2.0)
               else f"{ratio:.2f} → heteroscedastic")
    print(f"  top/bottom ratio = {verdict}")
    return float(r2), ratio


# (1) Additive in H-space
pred_add, resid_add = two_way_anova(H, vm)
r2_add, ratio_add = diagnostics('Additive  (H = α + β + ε)', pred_add, resid_add, H, vm)

# (2) Multiplicative: fit in log H, then exponentiate; residuals measured in H-space
eps      = 1e-6
logH     = H.clamp(min=eps).log()
# Note: 用 vm 把强制步排除在均值估计外（强制步 H=0 → logH 极小，会污染 mean）
pred_log, _  = two_way_anova(logH, vm)
pred_mul     = pred_log.exp()
resid_mul    = (H - pred_mul) * vm.float()
r2_mul, ratio_mul = diagnostics('Multiplicative (log H = α + β + ε)',
                                pred_mul, resid_mul, H, vm)


# ── Verdict ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"  Additive       :  R²={r2_add:.4f}   het.ratio={ratio_add:.2f}")
print(f"  Multiplicative :  R²={r2_mul:.4f}   het.ratio={ratio_mul:.2f}")

def closer_to_one(a, b):
    return 'additive' if abs(a - 1) < abs(b - 1) else 'multiplicative'

print(f"\n  R² 谁更高?              {'additive' if r2_add > r2_mul else 'multiplicative'}")
print(f"  Het. ratio 谁更接近 1?  {closer_to_one(ratio_add, ratio_mul)}")
print("\n  解读:")
print("    • R² 显著更高 + het.ratio 接近 1  → 该空间结构更对")
print("    • 两个都低 / 都异方差              → H 含交互项，需用非参数 (rank/quantile)")
