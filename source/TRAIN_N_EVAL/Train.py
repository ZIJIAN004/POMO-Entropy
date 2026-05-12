"""
Train.py — POMO REINFORCE with optional entropy schemes.

Supported modes (mutually exclusive priority: mlp > hand > none):
  • baseline POMO (no reweighting)
  • USE_HAND_FEATURES : per-instance multivariate OLS on hand-crafted features
                        (TSP: [log F, 1]; CVRP/VRPTW: [log F, load|time,
                         at_depot, visited_customer_ratio, 1])
  • USE_MLP_FEATURES  : truly end-to-end MLP baseline.
                        Input = [inst_summary, enc_last, (enc_first), raw_scalar]
                        — all encoder-side or raw env, no decoder-side tensors
                        (no q / mh_out), so the OLS residual still carries the
                        policy selection-confidence signal we want to reweight.
                        The MLP itself learns log / ratio / interaction terms.
  • USE_ENTROPY_BONUS : A2C-style entropy bonus (can stack on top of either)

Both the hand- and the MLP-feature paths exclude:
  - forced steps (selected_count 0/1 → entropy = 0)
  - finished padding (CVRP/VRPTW)
from the OLS fit and from the softmax.

advantage_i = r_i - mean_j(r_j)
"""

import time
import torch
import torch.nn.functional as F

from HYPER_PARAMS import *
from source.utilities import Average_Meter
from source.models.common import get_encoding

# ── hand-feature imports (only if MLP mode is OFF) ───────────────────────────
_USE_HAND = USE_HAND_FEATURES and not USE_MLP_FEATURES
if _USE_HAND:
    from source.entropy_utils import compute_entropy_weights
    from source.baseline import (
        extract_hand_features_tsp,
        extract_hand_features_cvrp,
        extract_hand_features_vrptw,
    )
    _HAND_EXTRACTOR = {
        'tsp':   extract_hand_features_tsp,
        'cvrp':  extract_hand_features_cvrp,
        'vrptw': extract_hand_features_vrptw,
    }[PROBLEM_TYPE]

# ── MLP-feature imports ──────────────────────────────────────────────────────
if USE_MLP_FEATURES:
    from source.baseline import (
        batched_per_instance_ols,
        compute_residual_weights,
    )


_COLLECT_ENTROPY  = USE_HAND_FEATURES or USE_ENTROPY_BONUS or USE_MLP_FEATURES
_COLLECT_FEATURES = _USE_HAND or USE_MLP_FEATURES
_COLLECT_FINISHED = _COLLECT_FEATURES and PROBLEM_TYPE in ('cvrp', 'vrptw')


# ─── MLP-mode step-embedding builders ────────────────────────────────────────
# Pure policy-encoder outputs (instance representation) + raw env scalars.
# NO decoder-side tensors. NO hand-engineered nonlinearities. MLP learns those.

def mlp_input_dim(problem_type, embed_dim):
    """Total dimension of the per-step MLP input vector."""
    if problem_type == 'tsp':
        # inst_summary + enc_last + enc_first + [F, sc, current_xy_x, current_xy_y]
        return 3 * embed_dim + 4
    elif problem_type == 'cvrp':
        # inst_summary + enc_last + [F, sc, load, current_xy_x, current_xy_y, vis_count]
        return 2 * embed_dim + 6
    elif problem_type == 'vrptw':
        # inst_summary + enc_last + [F, sc, current_time, current_xy_x, current_xy_y, vis_count]
        return 2 * embed_dim + 6
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


def _build_step_emb(model, state, env, problem_type, n_in):
    """
    Returns (B, P, n_in). Forced steps (state.current_node is None) get zeros,
    which is fine because valid_mask excludes them from OLS and softmax.

    All embeddings are detached: baseline never backprops into the policy.
    """
    B = state.BATCH_IDX.size(0)
    P = state.BATCH_IDX.size(1)
    enc_nodes = model.encoded_nodes
    D = enc_nodes.size(-1)
    device = enc_nodes.device

    if state.current_node is None:
        # forced step (CVRP/VRPTW selected_count==0, TSP selected_count==0)
        return torch.zeros(B, P, n_in, device=device)

    # instance summary: step-independent → in per-instance OLS it acts as bias,
    # but it also lets the MLP switch ReLU paths per-instance (instance-conditional
    # nonlinearity), so we keep it.
    inst_summary = enc_nodes.mean(dim=1).detach()                           # (B, D)
    inst_summary = inst_summary[:, None, :].expand(B, P, D)                 # (B, P, D)

    enc_last = get_encoding(enc_nodes, state.current_node).detach()         # (B, P, D)

    # ── per-problem raw env scalars (NO log/ratio/square — MLP learns) ──
    F_t  = state.n_feasible.float()                                          # (B, P)
    sc_t = torch.full_like(F_t, float(state.selected_count))                 # (B, P)

    cur_idx = state.current_node                                             # (B, P)

    if problem_type == 'tsp':
        enc_first = get_encoding(enc_nodes, model.first_node).detach()       # (B, P, D)
        node_xy_exp = env.node_xy[:, None, :, :].expand(B, P, -1, 2)         # (B, P, N, 2)
        current_xy = node_xy_exp.gather(
            2, cur_idx[..., None, None].expand(B, P, 1, 2)).squeeze(2)       # (B, P, 2)
        raw = torch.cat([F_t[..., None], sc_t[..., None], current_xy], dim=-1)   # (B, P, 4)
        return torch.cat([inst_summary, enc_last, enc_first, raw], dim=-1)       # (B, P, 3D+4)

    elif problem_type == 'cvrp':
        dnxy_exp = env.depot_node_xy[:, None, :, :].expand(B, P, -1, 2)      # (B, P, N+1, 2)
        current_xy = dnxy_exp.gather(
            2, cur_idx[..., None, None].expand(B, P, 1, 2)).squeeze(2)       # (B, P, 2)
        load_t    = state.load                                                # (B, P)
        visited_t = state.visited_customer_count.float()                      # (B, P)
        raw = torch.cat([
            F_t[..., None], sc_t[..., None], load_t[..., None],
            current_xy, visited_t[..., None],
        ], dim=-1)                                                            # (B, P, 6)
        return torch.cat([inst_summary, enc_last, raw], dim=-1)               # (B, P, 2D+6)

    elif problem_type == 'vrptw':
        dnxy_exp = env.depot_node_xy[:, None, :, :].expand(B, P, -1, 2)      # (B, P, N+1, 2)
        current_xy = dnxy_exp.gather(
            2, cur_idx[..., None, None].expand(B, P, 1, 2)).squeeze(2)       # (B, P, 2)
        ct_t      = state.current_time                                        # (B, P)
        visited_t = state.visited_customer_count.float()                      # (B, P)
        raw = torch.cat([
            F_t[..., None], sc_t[..., None], ct_t[..., None],
            current_xy, visited_t[..., None],
        ], dim=-1)                                                            # (B, P, 6)
        return torch.cat([inst_summary, enc_last, raw], dim=-1)               # (B, P, 2D+6)

    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


def _make_valid_mask(T_total, batch_size, device, finished_list=None):
    """forced-steps + finished padding → invalid; rest valid."""
    valid = torch.ones(batch_size, POMO_SIZE, T_total, dtype=torch.bool, device=device)
    forced_steps = 1 if PROBLEM_TYPE == 'tsp' else 2
    if T_total > forced_steps:
        valid[:, :, :forced_steps] = False
    else:
        valid[:] = False
    if finished_list is not None:
        valid &= ~finished_list
    return valid


def TRAIN(model, env, optimizer, lr_scheduler, epoch, timer_start, logger,
          baseline_module=None, baseline_optim=None):
    model.train()

    score_AM    = Average_Meter()
    loss_AM     = Average_Meter()
    baseline_AM = Average_Meter() if baseline_module is not None else None
    # diagnostic: MLP_MSE / Var(H[valid])  → indicates baseline strength
    ratio_AM    = Average_Meter() if baseline_module is not None else None

    logger_start = time.time()
    episode      = 0
    device       = next(model.parameters()).device

    n_in = mlp_input_dim(PROBLEM_TYPE, EMBEDDING_DIM) if USE_MLP_FEATURES else None

    # whether to use MLP-feature residuals for reweighting (post-warmup)
    use_mlp_weights = (USE_MLP_FEATURES and
                       baseline_module is not None and
                       epoch > MLP_WARMUP_EPOCHS)

    while episode < TRAIN_EPISODES:
        batch_size = min(TRAIN_BATCH_SIZE, TRAIN_EPISODES - episode)
        episode   += batch_size

        # ── Rollout ──────────────────────────────────────────────────────
        env.load_problems(batch_size, device=device)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        prob_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if _COLLECT_ENTROPY:
            entropy_list    = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
            n_feasible_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if _COLLECT_FEATURES:
            feat_list = []
            finished_list = (torch.zeros(batch_size, POMO_SIZE, 0,
                                          dtype=torch.bool, device=device)
                             if _COLLECT_FINISHED else None)

        state, reward, done = env.pre_step()

        while not done:
            selected, prob, entropy = model(state)
            if _COLLECT_ENTROPY:
                entropy_list = torch.cat(
                    (entropy_list, entropy[:, :, None]), dim=2)
                n_feasible_list = torch.cat(
                    (n_feasible_list, state.n_feasible[:, :, None].float()), dim=2)
            if _COLLECT_FEATURES:
                if USE_MLP_FEATURES:
                    feat_list.append(_build_step_emb(model, state, env, PROBLEM_TYPE, n_in))
                else:
                    feat_list.append(_HAND_EXTRACTOR(state, PROBLEM_SIZE))   # (B, P, d)
                if finished_list is not None:
                    finished_list = torch.cat(
                        (finished_list, state.finished[:, :, None]), dim=2)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # ── REINFORCE advantage ──────────────────────────────────────────
        reward_f  = reward.float()
        advantage = reward_f - reward_f.mean(dim=1, keepdim=True)

        # ── Compute weighted log_prob (dispatch by mode) ─────────────────
        if USE_MLP_FEATURES and baseline_module is not None:
            features = torch.stack(feat_list, dim=2)            # (B, P, T, n_in)
            T_total  = features.size(2)
            valid    = _make_valid_mask(T_total, batch_size, device, finished_list)

            mlp_feats = baseline_module(features)               # (B, P, T, h_out)
            # entropy_list is the OLS target (y); detach so gradient flows ONLY
            # through mlp_feats → MLP params, not back into the policy encoder/decoder.
            H_hat, _  = batched_per_instance_ols(
                mlp_feats, entropy_list.detach(), valid, ridge=MLP_RIDGE)

            # train MLP (every batch, including during warmup)
            if valid.any():
                H_target = entropy_list[valid].detach()
                H_pred   = H_hat[valid]
                loss_b   = F.mse_loss(H_pred, H_target)
                baseline_optim.zero_grad()
                loss_b.backward()
                baseline_optim.step()
                if baseline_AM is not None:
                    baseline_AM.push(loss_b.detach().unsqueeze(0))
                # diagnostic: MSE / Var(H[valid])
                if ratio_AM is not None:
                    var_H = H_target.var(unbiased=False).clamp(min=1e-12)
                    ratio = (loss_b.detach() / var_H)
                    ratio_AM.push(ratio.unsqueeze(0))

            # post-warmup: recompute residual under frozen MLP for weight calc
            if use_mlp_weights:
                with torch.no_grad():
                    mlp_feats_f = baseline_module(features)
                    H_hat_f, _  = batched_per_instance_ols(
                        mlp_feats_f, entropy_list, valid, ridge=MLP_RIDGE)
                residual = entropy_list - H_hat_f
                weights = compute_residual_weights(
                    residual, n_feasible_list, advantage, valid, gamma=ENTROPY_GAMMA)
                log_prob = (prob_list.log() * weights).sum(dim=2)
            else:
                log_prob = prob_list.log().sum(dim=2)

        elif _USE_HAND:
            features = torch.stack(feat_list, dim=2)            # (B, P, T, d)
            T_total  = features.size(2)
            valid    = _make_valid_mask(T_total, batch_size, device, finished_list)

            weights = compute_entropy_weights(
                entropy_list, features, n_feasible_list, advantage,
                valid, gamma=ENTROPY_GAMMA)
            log_prob = (prob_list.log() * weights).sum(dim=2)

        else:
            log_prob = prob_list.log().sum(dim=2)

        loss = -(advantage * log_prob).mean()

        if USE_ENTROPY_BONUS:
            loss = loss - ENTROPY_BONUS_BETA * entropy_list.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────
        max_reward, _ = reward.max(dim=1)
        score_AM.push(-max_reward.float())
        loss_AM.push(loss.detach().unsqueeze(0))

        if time.time() - logger_start > LOG_PERIOD_SEC or episode >= TRAIN_EPISODES:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - timer_start))
            extra = ""
            if baseline_AM is not None and baseline_AM.count > 0:
                tag = "MLP(use)" if use_mlp_weights else "MLP(warmup)"
                extra = "  {}_MSE:{:.4f}".format(tag, baseline_AM.result())
                if ratio_AM is not None and ratio_AM.count > 0:
                    extra += "  MSE/Var:{:.3f}".format(ratio_AM.result())
            logger.info('Ep:{:03d}-{:07d}({:5.1f}%)  T:{}  Loss:{:+.4f}  Avg.best_dist:{:.4f}{}'.format(
                epoch, episode, 100. * episode / TRAIN_EPISODES,
                elapsed, loss_AM.result(), score_AM.result(), extra))
            logger_start = time.time()

    lr_scheduler.step()
