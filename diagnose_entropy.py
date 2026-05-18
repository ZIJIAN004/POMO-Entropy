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

from source.baseline import build_group_id

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
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.ckpt}")
else:
    print("Using random-init model (untrained).")
model.eval()

env = Env(problem_size=PROBLEM_SIZE, pomo_size=POMO_SIZE)
print(f"Problem={args.problem.upper()}  size={PROBLEM_SIZE}  POMO={POMO_SIZE}  "
      f"batches={args.n_batches}×{args.batch_size}  device={device}")


# ── Rollout: collect H, valid_mask, and gid features ────────────────────────
all_H, all_valid, all_nf = [], [], []
all_at_depot, all_load, all_vis_ratio = [], [], []   # CVRP only
forced_steps = 1 if args.problem == 'tsp' else 2

with torch.no_grad():
    for _ in range(args.n_batches):
        env.load_problems(args.batch_size, device=device)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        H_list   = torch.zeros(args.batch_size, POMO_SIZE, 0, device=device)
        nf_list  = torch.zeros(args.batch_size, POMO_SIZE, 0, device=device)
        fin_list = (torch.zeros(args.batch_size, POMO_SIZE, 0,
                                dtype=torch.bool, device=device)
                    if args.problem == 'cvrp' else None)
        ad_list  = torch.zeros(args.batch_size, POMO_SIZE, 0, device=device) if args.problem == 'cvrp' else None
        ld_list  = torch.zeros(args.batch_size, POMO_SIZE, 0, device=device) if args.problem == 'cvrp' else None
        vr_list  = torch.zeros(args.batch_size, POMO_SIZE, 0, device=device) if args.problem == 'cvrp' else None

        state, _, done = env.pre_step()
        while not done:
            selected, _, entropy, *_ = model(state)
            H_list  = torch.cat((H_list,  entropy[:, :, None]),                dim=2)
            nf_list = torch.cat((nf_list, state.n_feasible[:, :, None].float()), dim=2)
            if fin_list is not None:
                fin_list = torch.cat((fin_list, state.finished[:, :, None]),    dim=2)
                ad = (torch.zeros(args.batch_size, POMO_SIZE, device=device)
                      if state.current_node is None
                      else (state.current_node == 0).float())
                ad_list = torch.cat((ad_list, ad[:, :, None]), dim=2)
                ld_list = torch.cat((ld_list, state.load[:, :, None]), dim=2)
                vis_step = state.visited_customer_count.float() / max(PROBLEM_SIZE, 1)
                vr_list = torch.cat((vr_list, vis_step[:, :, None]), dim=2)
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
        all_nf.append(nf_list)
        if args.problem == 'cvrp':
            all_at_depot.append(ad_list)
            all_load.append(ld_list)
            all_vis_ratio.append(vr_list)

H  = torch.cat(all_H,     dim=0)   # (B_total, P, T)
vm = torch.cat(all_valid, dim=0)
nf = torch.cat(all_nf,    dim=0)
if args.problem == 'cvrp':
    at_depot  = torch.cat(all_at_depot,  dim=0)
    load      = torch.cat(all_load,      dim=0)
    vis_ratio = torch.cat(all_vis_ratio, dim=0)
    gid, n_grp_per_inst = build_group_id(
        'cvrp', n_feasible=nf, at_depot=at_depot, load=load, vis_ratio=vis_ratio, n_bins=10)
else:
    gid, n_grp_per_inst = build_group_id('tsp', n_feasible=nf, n_bins=10)
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
    """All metrics computed in H-space (resid already in H-space).

    R² is computed **per-instance** (each instance subtracts its own grand mean
    when forming SS_total), then summarised two ways:
      • r2_mean   : mean across instances (each instance weighted equally)
      • r2_pooled : Σ_b SS_res_b / Σ_b SS_tot_b (weighted by within-instance variance)

    Reasoning: ANOVA is fit per-instance, so its "explainable" denominator
    must also be the within-instance variance. Pooling H across instances
    inflates SS_total with inter-instance differences (different problems
    naturally have different entropy levels), which ANOVA wasn't trying to
    explain — that would systematically deflate R².
    """
    vmf = vm.float()

    # ── Per-instance R² ─────────────────────────────────────────────────
    cnt_b = vmf.sum(dim=(1, 2)).clamp(min=1.0)                          # (B,)
    mu_b  = ((H_ref * vmf).sum(dim=(1, 2)) / cnt_b).view(-1, 1, 1)      # (B,1,1)
    tot_ss_b   = (((H_ref - mu_b) ** 2) * vmf).sum(dim=(1, 2))          # (B,)
    resid_ss_b = (resid ** 2).sum(dim=(1, 2))                            # (B,)
    r2_b       = 1.0 - resid_ss_b / tot_ss_b.clamp(min=1e-12)            # (B,)
    r2_mean    = float(r2_b.mean())
    r2_pooled  = float(1.0 - resid_ss_b.sum() / tot_ss_b.sum().clamp(min=1e-12))

    # ── Heteroscedasticity (pooled by Ĥ-quintile across instances) ──────
    pred_valid = pred[vm]
    res_valid  = resid[vm]
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
    print(f"  R² per-instance:  mean={r2_mean:.4f}   pooled={r2_pooled:.4f}")
    print(f"  R² per-instance:  min={float(r2_b.min()):.3f}  "
          f"med={float(r2_b.median()):.3f}  max={float(r2_b.max()):.3f}")
    print(f"  resid var by Ĥ-quintile (low→high):")
    for i, v in enumerate(var_per_bin):
        print(f"    Q{i+1}: {v:.3e}")
    verdict = ("~1 → homoscedastic ✓"
               if (0.5 < ratio < 2.0)
               else f"{ratio:.2f} → heteroscedastic")
    print(f"  top/bottom ratio = {verdict}")
    return r2_mean, r2_pooled, ratio


# (1) Additive in H-space
pred_add, resid_add = two_way_anova(H, vm)
r2_add_m, r2_add_p, ratio_add = diagnostics(
    'Additive  (H = α + β + ε)', pred_add, resid_add, H, vm)

# (2) Multiplicative: fit in log H, then exponentiate; residuals measured in H-space
eps      = 1e-6
logH     = H.clamp(min=eps).log()
# Note: 用 vm 把强制步排除在均值估计外（强制步 H=0 → logH 极小，会污染 mean）
pred_log, _  = two_way_anova(logH, vm)
pred_mul     = pred_log.exp()
resid_mul    = (H - pred_mul) * vm.float()
r2_mul_m, r2_mul_p, ratio_mul = diagnostics(
    'Multiplicative (log H = α + β + ε)', pred_mul, resid_mul, H, vm)


# ── Verdict ───────────────────────────────────────────────────────────────────
print("\n=== Summary (per-instance R²) ===")
print(f"  Additive       :  R²_mean={r2_add_m:.4f}  R²_pooled={r2_add_p:.4f}  "
      f"het.ratio={ratio_add:.2f}")
print(f"  Multiplicative :  R²_mean={r2_mul_m:.4f}  R²_pooled={r2_mul_p:.4f}  "
      f"het.ratio={ratio_mul:.2f}")

def closer_to_one(a, b):
    return 'additive' if abs(a - 1) < abs(b - 1) else 'multiplicative'

print(f"\n  R²_mean 谁更高?            {'additive' if r2_add_m > r2_mul_m else 'multiplicative'}")
print(f"  Het. ratio 谁更接近 1?     {closer_to_one(ratio_add, ratio_mul)}")
print("\n  解读:")
print("    • R² 显著更高 + het.ratio 接近 1  → 该空间结构更对")
print("    • 两个都低 / 都异方差              → H 含交互项，需用非参数 (rank/quantile)")
print("    • 注意: 上面的 het.ratio 混合了 跨桶 + 桶内 异方差，下面单独拆开测")


# ─────────────────────────────────────────────────────────────────────────────
# Group-aware heteroscedasticity diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# baseline.py 在每个 (instance, gid) 桶内做 z-score 标准化。这意味着：
#   • 跨桶异方差（不同桶 grp_std 不同）→ 已被处理（每桶用自己的 std）
#   • 桶内异方差（同桶内不同样本潜在方差不同）→ z-score 失真，是真正待修复对象
# 下面两段独立测这两件事。

@torch.no_grad()
def cross_group_std_mean_relation(H, vm, gid, n_grp_per_inst):
    """跨桶异方差：聚合所有 (instance, gid) 桶，拟合 log(grp_std) ~ α·log(grp_mean) + b。

      α ≈ 1.0  → 乘性结构 (std ∝ mean)，用 /grp_mean 标准化 (CoV)
      α ≈ 0.5  → Poisson-like (std ∝ √mean)，用 /√grp_mean
      α ≈ 0.0  → 同方差 (std 与 mean 无关)，当前 /grp_std 即可
      R² 高    → std-mean 关系清晰，可用参数化分母
      R² 低    → 关系混乱，单凭 grp_mean 抓不住，需要细化 gid 或非参数
    """
    B, P, T = H.shape
    dev = H.device

    gid_flat = gid.reshape(-1)
    bid = torch.arange(B, device=dev).repeat_interleave(P * T)
    gid_global = bid * n_grp_per_inst + gid_flat
    n_grp_total = B * n_grp_per_inst

    v_flat = vm.reshape(-1).float()
    H_flat = H.reshape(-1)

    counts = torch.zeros(n_grp_total, device=dev).scatter_add_(0, gid_global, v_flat)
    sums   = torch.zeros(n_grp_total, device=dev).scatter_add_(0, gid_global, H_flat * v_flat)
    grp_mean = sums / counts.clamp(min=1)
    centered = (H_flat - grp_mean[gid_global]) * v_flat
    sq_sums  = torch.zeros(n_grp_total, device=dev).scatter_add_(0, gid_global, centered.square())
    grp_var  = sq_sums / counts.clamp(min=1)
    grp_std  = grp_var.sqrt()

    valid_grp = counts >= 4
    gm = grp_mean[valid_grp]
    gs = grp_std[valid_grp]

    # 过滤掉接近 0 的（log 会爆）
    keep = (gm > 1e-3) & (gs > 1e-6)
    log_m = gm[keep].log()
    log_s = gs[keep].log()
    n_kept = log_m.numel()
    if n_kept < 10:
        print("\n[Cross-group std-mean relation]  too few valid groups, skip.")
        return

    lm_mean = log_m.mean()
    ls_mean = log_s.mean()
    cov  = ((log_m - lm_mean) * (log_s - ls_mean)).sum()
    varm = ((log_m - lm_mean) ** 2).sum()
    alpha = cov / varm.clamp(min=1e-12)
    b_int = ls_mean - alpha * lm_mean
    pred  = alpha * log_m + b_int
    r2    = 1 - ((log_s - pred) ** 2).sum() / \
                ((log_s - ls_mean) ** 2).sum().clamp(min=1e-12)

    # 跨桶 std-mean 五分位扫描（看 grp_std 量级是否随 grp_mean 量级单调）
    qs   = torch.tensor([0.2, 0.4, 0.6, 0.8], device=dev)
    cuts = torch.quantile(gm[keep], qs)
    bins = torch.bucketize(gm[keep], cuts)
    std_per_bin = []
    for i in range(5):
        sel = (bins == i)
        std_per_bin.append(float(gs[keep][sel].mean()) if int(sel.sum()) >= 2 else float('nan'))

    print(f"\n[Cross-group std-mean relation  (n_groups={n_kept})]")
    print(f"  log(grp_std) ≈ {float(alpha):.3f} · log(grp_mean) + {float(b_int):.3f}")
    print(f"  R² (log-log fit) = {float(r2):.4f}")
    print(f"  grp_std (avg) by grp_mean-quintile (low→high):")
    for i, v in enumerate(std_per_bin):
        print(f"    Q{i+1}: {v:.3e}")
    finite = [v for v in std_per_bin if v == v]
    cross_ratio = (std_per_bin[-1] / max(std_per_bin[0], 1e-12)) if len(finite) == 5 else float('nan')
    print(f"  跨桶比 (Q5/Q1): {cross_ratio:.2f}  "
          f"(baseline.py 桶内 std 标准化已吸收这一部分)")
    print(f"  ⇒ α 解读: ≈1 用 /grp_mean   ≈0.5 用 /√grp_mean   ≈0 用 /grp_std")
    return float(alpha), float(r2), cross_ratio


@torch.no_grad()
def within_group_heteroscedasticity(H, vm, gid, n_grp_per_inst):
    """桶内异方差：同一 (instance, gid) 桶内，残差是否随该样本所在轨迹的 μ_t 变化？

    检验逻辑：
      • 每个 (B, P, T) 位置都属于某个桶
      • 该位置的"残差"= H - grp_mean
      • 该位置所在轨迹的 μ_t = 该轨迹在所有 valid 步的 H 均值
      • 按 μ_t 五分位分组（across all valid positions），看残差平方均值是否单调

    比值 >>1 ⇒ 桶内不同轨迹的方差量级不一致 ⇒ baseline.py 的 grp_std 是混合估计，
                z-score 在不同轨迹上系统性失真。这是当前实现真正未处理的部分。
    """
    B, P, T = H.shape
    dev = H.device
    vmf = vm.float()

    # 轨迹均值 μ_t，broadcast 到每个位置
    mu_t = (H * vmf).sum(dim=2, keepdim=True) / \
            vmf.sum(dim=2, keepdim=True).clamp(min=1)         # (B, P, 1)
    mu_t_pos = mu_t.expand_as(H).reshape(-1)                   # (B·P·T,)

    # 桶内残差 H - grp_mean
    gid_flat = gid.reshape(-1)
    bid = torch.arange(B, device=dev).repeat_interleave(P * T)
    gid_global = bid * n_grp_per_inst + gid_flat
    n_grp_total = B * n_grp_per_inst
    v_flat = vmf.reshape(-1)
    H_flat = H.reshape(-1)
    counts = torch.zeros(n_grp_total, device=dev).scatter_add_(0, gid_global, v_flat)
    sums   = torch.zeros(n_grp_total, device=dev).scatter_add_(0, gid_global, H_flat * v_flat)
    grp_mean_pos = (sums / counts.clamp(min=1))[gid_global]
    centered_pos = (H_flat - grp_mean_pos) * v_flat

    valid_idx = v_flat.bool()
    mu_t_v   = mu_t_pos[valid_idx]
    sq_v     = centered_pos[valid_idx].square()

    qs   = torch.tensor([0.2, 0.4, 0.6, 0.8], device=dev)
    cuts = torch.quantile(mu_t_v, qs)
    bins = torch.bucketize(mu_t_v, cuts)
    sq_per_bin = []
    for i in range(5):
        sel = (bins == i)
        sq_per_bin.append(float(sq_v[sel].mean()) if int(sel.sum()) >= 2 else float('nan'))
    finite = [v for v in sq_per_bin if v == v]
    inner_ratio = (sq_per_bin[-1] / max(sq_per_bin[0], 1e-12)) if len(finite) == 5 else float('nan')

    print(f"\n[Within-group heteroscedasticity  (by trajectory μ_t quintile)]")
    print(f"  桶内残差方差 by μ_t-quintile (low→high):")
    for i, v in enumerate(sq_per_bin):
        print(f"    Q{i+1}: {v:.3e}")
    print(f"  桶内比 (Q5/Q1): {inner_ratio:.2f}")
    if 0.5 < inner_ratio < 2.0:
        print(f"    ~1 → 桶内同方差 ✓  baseline.py 的 grp_std 标准化够用")
    else:
        print(f"    >>1 → 桶内异方差 ✗  baseline.py 的 grp_std 是混合估计，z-score 失真")
        print(f"           → 需要细化 gid (解药 1) 或非参数 rank (解药 3)")
    return inner_ratio


# ── 跑两个新诊断 ──────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("Group-aware diagnostics  (baseline.py 视角)")
print("=" * 72)
cross_group_std_mean_relation(H, vm, gid, n_grp_per_inst)
within_group_heteroscedasticity(H, vm, gid, n_grp_per_inst)
