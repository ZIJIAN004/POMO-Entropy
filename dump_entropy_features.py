"""
dump_entropy_features.py — per-step (entropy, conditioning, candidate-feature)
                            tuples for residual heteroscedasticity diagnostics.

Use case
========
Discover which environment variables are *still leaking* into ΔH after the
current bucket method removes E[H | state]. Diagnostic principle: the model's
pure confidence component of H should have constant variance; any feature `z`
for which std(residual | z) varies systematically is an unmodelled environment
variable that should be added to the conditioning set.

Run
===
    python dump_entropy_features.py --problem tsp  --ckpt result/.../MODEL_FINAL.pt
    python dump_entropy_features.py --problem cvrp --ckpt result/.../MODEL_FINAL.pt

    # Without --ckpt the model is random-init; entropy is meaningless — only
    # use that to smoke-test the pipeline.

Output
======
    dump_{problem}.npz  — one record per (batch, instance, traj, step) cell.
                          Filter rows by `valid == True` before analysis
                          (forced steps and CVRP-finished padding are kept
                          but marked invalid; their feature values are NaN
                          where not well-defined).

    Common fields (TSP + CVRP):
        batch_id, instance_id, traj_id, step_id, step_norm
        H, n_feasible, valid, n_unvisited
        dist_nearest_unvisited, mean_dist_unvisited, std_dist_unvisited

    CVRP-only extra fields:
        at_depot, load, vis_ratio             — already in current bucket
        dist_to_depot                         — distance from current to depot
        n_demand_feasible                     — # customers with demand ≤ load
        max_feasible_demand                   — biggest demand still affordable
        total_remaining_demand                — Σ demand of unvisited customers
        prev_was_depot                        — did last step return to depot
"""

import argparse
import numpy as np
import torch

from HYPER_PARAMS import (EMBEDDING_DIM, ENCODER_LAYER_NUM, HEAD_NUM, QKV_DIM,
                          FF_HIDDEN_DIM, LOGIT_CLIPPING)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', required=True, choices=['tsp', 'cvrp'])
    parser.add_argument('--size', type=int, default=100,
                        help='problem size; sets pomo_size = size too')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to MODEL_state_dic.pt; without this the model '
                             'is random-init and entropy stats are meaningless')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-batches', type=int, default=4)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.problem == 'tsp':
        from source.models.tsp_model import TSPModel as Model
        from source.envs.tsp_env     import TSPEnv   as Env
    else:
        from source.models.cvrp_model import CVRPModel as Model
        from source.envs.cvrp_env     import CVRPEnv   as Env

    model = Model(embedding_dim=EMBEDDING_DIM, encoder_layer_num=ENCODER_LAYER_NUM,
                  head_num=HEAD_NUM, qkv_dim=QKV_DIM, ff_hidden_dim=FF_HIDDEN_DIM,
                  logit_clipping=LOGIT_CLIPPING).to(device)

    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"[ckpt] loaded {args.ckpt}")
    else:
        print("[WARN] no --ckpt: random-init model. Use only for smoke-testing.")

    model.train()    # multinomial sampling — entropy reflects training-time behavior
    env = Env(problem_size=args.size, pomo_size=args.size)

    records = {k: [] for k in _record_keys(args.problem)}
    for bi in range(args.n_batches):
        _rollout_one_batch(args, env, model, device, bi, records)
        n_rows = sum(a.size for a in records['H'])
        print(f"[batch {bi+1}/{args.n_batches}] rows so far = {n_rows}")

    out = {k: np.concatenate(v) for k, v in records.items()}
    out['step_norm'] = _compute_step_norm(out)

    out_path = args.out or f"dump_{args.problem}.npz"
    np.savez_compressed(out_path, **out)
    print(f"[done] saved {out_path}: total={out['H'].size}, "
          f"valid={int(out['valid'].sum())}, "
          f"valid_frac={out['valid'].mean():.3f}")


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

def _record_keys(problem):
    keys = ['batch_id', 'instance_id', 'traj_id', 'step_id',
            'H', 'n_feasible', 'valid', 'n_unvisited',
            'dist_nearest_unvisited', 'mean_dist_unvisited', 'std_dist_unvisited']
    if problem == 'cvrp':
        keys += ['at_depot', 'load', 'vis_ratio',
                 'dist_to_depot', 'n_demand_feasible', 'max_feasible_demand',
                 'total_remaining_demand', 'prev_was_depot']
    return keys


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_one_batch(args, env, model, device, bi, records):
    B, P = args.batch_size, args.size

    env.load_problems(B, device=device)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    if args.problem == 'tsp':
        ctx = {'node_xy': env.node_xy}
    else:
        ctx = {
            'depot_xy':    env.depot_node_xy[:, 0:1, :],   # (B, 1, 2)
            'customer_xy': env.depot_node_xy[:, 1:,  :],   # (B, N, 2)
            'cust_dem':    env.depot_node_demand[:, 1:],   # (B, N)
            'prev_depot':  torch.zeros(B, P, dtype=torch.bool, device=device),
        }

    state, _, done = env.pre_step()
    step_id = 0

    while not done:
        feats = (_features_tsp(state, B, P, device, ctx) if args.problem == 'tsp'
                 else _features_cvrp(state, env, B, P, device, ctx))

        selected, _, entropy, *_ = model(state)

        forced = 1 if args.problem == 'tsp' else 2
        valid = torch.ones(B, P, dtype=torch.bool, device=device)
        if step_id < forced:
            valid[:] = False
        if args.problem == 'cvrp':
            valid = valid & ~state.finished

        _append(records, bi, step_id, B, P, entropy, state.n_feasible, valid, feats)

        state, _, done = env.step(selected)
        if args.problem == 'cvrp':
            ctx['prev_depot'] = (selected == 0)
        step_id += 1


def _append(records, bi, step_id, B, P, entropy, n_feasible, valid, feats):
    records['batch_id'].append(np.full(B * P, bi, dtype=np.int32))
    records['instance_id'].append(np.repeat(np.arange(B, dtype=np.int32), P))
    records['traj_id'].append(np.tile(np.arange(P, dtype=np.int32), B))
    records['step_id'].append(np.full(B * P, step_id, dtype=np.int32))
    records['H'].append(entropy.detach().cpu().numpy().reshape(-1).astype(np.float32))
    records['n_feasible'].append(
        n_feasible.detach().cpu().numpy().reshape(-1).astype(np.float32))
    records['valid'].append(valid.detach().cpu().numpy().reshape(-1))

    for k, v in feats.items():
        arr = v.detach().cpu().numpy().reshape(-1)
        dt = np.float32 if v.dtype.is_floating_point else np.int8
        records[k].append(arr.astype(dt))


# ---------------------------------------------------------------------------
# feature computation
# ---------------------------------------------------------------------------

def _features_tsp(state, B, P, device, ctx):
    cn = state.current_node                                  # (B, P) or None
    if cn is None:
        return _nan_dict(B, P, device,
                         keys=['dist_nearest_unvisited', 'mean_dist_unvisited',
                               'std_dist_unvisited', 'n_unvisited'])
    node_xy = ctx['node_xy']                                 # (B, N, 2)
    cur_xy = _gather_xy(node_xy, cn, device)                 # (B, P, 2)
    unvisited = (state.ninf_mask == 0)                       # (B, P, N)
    return _geom_feats(cur_xy, node_xy, unvisited)


def _features_cvrp(state, env, B, P, device, ctx):
    cn = state.current_node                                  # (B, P) or None
    extra_keys = ['dist_to_depot', 'n_demand_feasible', 'max_feasible_demand',
                  'total_remaining_demand']

    if cn is None:
        feats = _nan_dict(B, P, device,
                          keys=['dist_nearest_unvisited', 'mean_dist_unvisited',
                                'std_dist_unvisited', 'n_unvisited'] + extra_keys)
        feats['at_depot']       = torch.zeros(B, P, dtype=torch.int8, device=device)
        feats['load']           = state.load.detach()
        feats['vis_ratio']      = state.visited_customer_count.detach() / max(P, 1)
        feats['prev_was_depot'] = ctx['prev_depot'].to(torch.int8)
        return feats

    customer_xy = ctx['customer_xy']                         # (B, N, 2)
    depot_xy    = ctx['depot_xy']                            # (B, 1, 2)
    cust_dem    = ctx['cust_dem']                            # (B, N)

    cur_xy = _gather_xy(env.depot_node_xy, cn, device)       # (B, P, 2)
    unvisited_customer = (env.visited_ninf_flag[:, :, 1:] == 0)   # (B, P, N)

    feats = _geom_feats(cur_xy, customer_xy, unvisited_customer)

    # depot distance
    feats['dist_to_depot'] = (
        ((cur_xy - depot_xy.expand(-1, P, -1)) ** 2).sum(-1).clamp(min=1e-20).sqrt())

    # demand-feasibility
    cust_dem_b = cust_dem.unsqueeze(1).expand(-1, P, -1)     # (B, P, N)
    load = state.load                                         # (B, P)
    demand_feas = unvisited_customer & (cust_dem_b <= load.unsqueeze(-1) + 1e-5)
    feats['n_demand_feasible']      = demand_feas.float().sum(dim=2)
    feats['max_feasible_demand']    = torch.where(
        demand_feas, cust_dem_b, torch.zeros_like(cust_dem_b)).max(dim=2).values
    feats['total_remaining_demand'] = (cust_dem_b * unvisited_customer.float()).sum(dim=2)

    feats['at_depot']       = (cn == 0).to(torch.int8)
    feats['load']           = load
    feats['vis_ratio']      = state.visited_customer_count / max(P, 1)
    feats['prev_was_depot'] = ctx['prev_depot'].to(torch.int8)
    return feats


def _geom_feats(cur_xy, node_xy, unvisited):
    """Distances from cur_xy to all nodes in node_xy, masked to unvisited."""
    # node_xy: (B, N, 2); cur_xy: (B, P, 2); unvisited: (B, P, N)
    diffs = node_xy.unsqueeze(1) - cur_xy.unsqueeze(2)        # (B, P, N, 2)
    dists = (diffs ** 2).sum(-1).clamp(min=1e-20).sqrt()       # (B, P, N)
    uvf = unvisited.float()
    n_unvisited = uvf.sum(dim=2)                               # (B, P) — true count
    safe_cnt    = n_unvisited.clamp(min=1.0)                    # divisor only

    big   = torch.full_like(dists, float('inf'))
    d_min = torch.where(unvisited, dists, big).min(dim=2).values

    d_mean = (dists * uvf).sum(dim=2) / safe_cnt
    d_sq   = (((dists - d_mean.unsqueeze(2)) ** 2) * uvf).sum(dim=2)
    d_std  = (d_sq / safe_cnt).sqrt()

    # rows with zero unvisited (e.g., last step) → d_min = inf; set to NaN.
    has_any = n_unvisited > 0
    d_min  = torch.where(has_any, d_min,  torch.full_like(d_min, float('nan')))
    d_mean = torch.where(has_any, d_mean, torch.full_like(d_mean, float('nan')))
    d_std  = torch.where(has_any, d_std,  torch.full_like(d_std, float('nan')))

    return {
        'dist_nearest_unvisited': d_min,
        'mean_dist_unvisited':    d_mean,
        'std_dist_unvisited':     d_std,
        'n_unvisited':            n_unvisited,
    }


def _gather_xy(node_xy, cn, device):
    """node_xy (B, N, 2), cn (B, P) long → cur_xy (B, P, 2)."""
    B = node_xy.size(0)
    return node_xy[torch.arange(B, device=device).unsqueeze(1), cn.long()]


def _nan_dict(B, P, device, keys):
    return {k: torch.full((B, P), float('nan'), device=device) for k in keys}


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------

def _compute_step_norm(out):
    """step_norm = step_id / (max step_id in same trajectory among valid rows + 1)."""
    key = ((out['batch_id'].astype(np.int64) << 32)
           | (out['instance_id'].astype(np.int64) << 16)
           | out['traj_id'].astype(np.int64))
    uniq, inv = np.unique(key, return_inverse=True)

    T_per_key = np.zeros(uniq.size, dtype=np.int64)
    v = out['valid']
    np.maximum.at(T_per_key, inv[v], out['step_id'][v] + 1)
    T_per_step = T_per_key[inv].clip(min=1)
    return (out['step_id'] / T_per_step).astype(np.float32)


if __name__ == '__main__':
    main()
