#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def softmax_columns(X):
    """X shape (A, O); normalize each column."""
    X = X - X.max(axis=0, keepdims=True)
    E = np.exp(X)
    return E / E.sum(axis=0, keepdims=True)


def dirichlet_sample(alpha, rng):
    y = rng.gamma(alpha, 1.0)
    return y / y.sum()


def policy_spread_mean(pi_list):
    if len(pi_list) < 2:
        return 0.0
    dists = []
    n = pi_list[0].size
    for i in range(len(pi_list)):
        for j in range(i + 1, len(pi_list)):
            dists.append(np.linalg.norm(pi_list[i] - pi_list[j]) / np.sqrt(n))
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# POMDP generation
# ---------------------------------------------------------------------------

def random_pomdp(S, O, A, gamma_range, reward_scale, seed, stationary_bonus):
    rng = np.random.default_rng(seed)
    good = np.arange(S // 2)
    bad = np.arange(S // 2, S)

    T = np.zeros((S, S, A))
    for a in range(A):
        for s in range(S):
            alpha = 0.05 + rng.random(S)
            if s in good:
                alpha[good] += 5.0
            else:
                alpha[bad] += 5.0
            if a == 0:
                alpha[good] += 4.0
            elif a == 1:
                alpha[bad] += 4.0
            alpha[s] += stationary_bonus * S
            T[s, :, a] = dirichlet_sample(alpha, rng)

    if S == 4 and O == 2:
        Z = np.array([[0.85, 0.15, 0.85, 0.15],
                      [0.15, 0.85, 0.15, 0.85]])
    else:
        Z = np.zeros((O, S))
        for s in range(S):
            alpha = 0.05 + rng.random(O)
            if s % 2 == 0:
                alpha[0] += 5.0
            else:
                alpha[min(O - 1, 1)] += 5.0
            Z[:, s] = dirichlet_sample(alpha, rng)

    R = np.zeros((S, A))
    for s in range(S):
        if s in good:
            R[s, 0] = 3.0 + 0.5 * rng.standard_normal()
            R[s, 1] = 0.0 + 0.5 * rng.standard_normal()
        else:
            R[s, 0] = -2.0 + 0.5 * rng.standard_normal()
            R[s, 1] = 2.0 + 0.5 * rng.standard_normal()
    if A > 2:
        R[:, 2:] = 0.3 * rng.standard_normal((S, A - 2))

    mu = dirichlet_sample(0.7 + 2.0 * rng.random(S), rng)
    gamma = gamma_range[0] + (gamma_range[1] - gamma_range[0]) * rng.random()

    return {'S': S, 'O': O, 'A': A, 'T': T, 'Z': Z, 'R': R, 'mu': mu, 'gamma': gamma}


def convert_to_fully_observable_baseline(partial):
    full = dict(partial)
    S = partial['S']
    full['O'] = S
    full['Z'] = np.eye(S)
    return full


# ---------------------------------------------------------------------------
# Value function and gradient
# ---------------------------------------------------------------------------

def induced_markov_chain_and_reward(T, Z, R, pi):
    S = T.shape[0]
    A = T.shape[2]
    q = np.einsum('os,ao->sa', Z, pi)          # (S, A)
    Ppi = np.einsum('sa,sla->sl', q, T)        # (S, S)
    rpi = np.einsum('sa,sa->s', q, R)          # (S,)
    return Ppi, rpi


def objective_and_grad(pomdp, theta):
    S = pomdp['S']
    A = pomdp['A']
    O = pomdp['O']
    T = pomdp['T']
    Z = pomdp['Z']
    R = pomdp['R']
    mu = pomdp['mu']
    gamma = pomdp['gamma']

    pi = softmax_columns(theta)                 # (A, O)
    Ppi, rpi = induced_markov_chain_and_reward(T, Z, R, pi)

    M = np.eye(S) - gamma * Ppi
    V = np.linalg.solve(M, rpi)
    J = float(mu @ V)
    w = np.linalg.solve(M.T, mu)

    grad = np.zeros((A, O))
    for a0 in range(A):
        for o in range(O):
            pio = pi[:, o]                      # (A,)
            e = np.zeros(A)
            e[a0] = 1.0
            dq_base = pio * (e - pio[a0])       # (A,) softmax Jacobian col a0
            dr = Z[o, :] * (R @ dq_base)        # (S,)
            dP = Z[o, :, np.newaxis] * np.einsum('a,sla->sl', dq_base, T)  # (S, S)
            grad[a0, o] = w @ (dr + gamma * dP @ V)

    return J, grad, V, pi


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def policy_gradient_ascent(pomdp, theta0, max_iters, step_size, grad_clip):
    theta = theta0.copy()
    J_history = []
    gnorm_history = []

    for _ in range(max_iters):
        J, grad, V, pi = objective_and_grad(pomdp, theta)
        gnorm = np.linalg.norm(grad)
        J_history.append(J)
        gnorm_history.append(gnorm)

        if gnorm < 1e-7:
            break
        if gnorm > grad_clip:
            grad = grad * (grad_clip / gnorm)

        theta = theta + step_size * grad

    return theta, np.array(J_history), np.array(gnorm_history)


def run_many_restarts(pomdp, num_restarts, max_iters, step_size, grad_clip, rng):
    A = pomdp['A']
    O = pomdp['O']
    final_J = np.zeros(num_restarts)
    final_pi = []

    for r in range(num_restarts):
        theta0 = 0.25 * rng.standard_normal((A, O))
        theta, J_hist, _ = policy_gradient_ascent(pomdp, theta0, max_iters, step_size, grad_clip)
        J_f, _, _, pi_f = objective_and_grad(pomdp, theta)
        final_J[r] = J_f
        final_pi.append(pi_f)

    return {'final_J': final_J, 'final_pi': final_pi}


# ---------------------------------------------------------------------------
# Batch experiment
# ---------------------------------------------------------------------------

def num_instances_per_combo(cfg):
    S_list = cfg['S_list']
    A_list = cfg['A_list']
    O_list = cfg['O_partial_list']
    combos = sum(1 for S in S_list for A in A_list for O in O_list if O <= S)
    if combos == 0:
        return 1
    return max(1, round(cfg['num_instances'] / combos))


def run_batch_experiment(cfg, rng):
    S_list = cfg['S_list']
    A_list = cfg['A_list']
    O_list = cfg['O_partial_list']
    num_restarts = cfg['num_restarts']
    max_iters = cfg['max_iters']
    step_size = cfg['step_size']
    grad_clip = cfg['grad_clip']
    gamma_range = cfg['gamma_range']
    reward_scale = cfg['reward_scale']
    stationary_bonus = cfg['stationary_bonus']
    subopt_threshold = cfg['subopt_threshold']
    verbose = cfg.get('verbose', False)
    n_per_combo = num_instances_per_combo(cfg)

    meta = []
    partial_J_spread = []
    partial_subopt_frac = []
    partial_policy_spread = []
    full_J_spread = []
    full_subopt_frac = []
    full_policy_spread = []
    partial_best_J = []
    full_best_J = []
    gap_arr = []
    S_arr = []
    A_arr = []
    O_arr = []

    inst = 0
    for S in S_list:
        for A in A_list:
            for O in O_list:
                if O > S:
                    continue
                for rep in range(n_per_combo):
                    seed_here = 1000 + 37 * (inst + 1) + 11 * S + 7 * A + 5 * O
                    pomdp = random_pomdp(S, O, A, gamma_range, reward_scale,
                                         seed_here, stationary_bonus)
                    full = convert_to_fully_observable_baseline(pomdp)

                    res_p = run_many_restarts(pomdp, num_restarts, max_iters,
                                             step_size, grad_clip, rng)
                    res_f = run_many_restarts(full, num_restarts, max_iters,
                                             step_size, grad_clip, rng)

                    Jp = res_p['final_J']
                    Jf = res_f['final_J']
                    best_p = float(np.max(Jp))
                    best_f = float(np.max(Jf))

                    spread_p = float(np.max(Jp) - np.min(Jp))
                    spread_f = float(np.max(Jf) - np.min(Jf))
                    sfrac_p = float(np.mean(Jp < best_p - subopt_threshold * max(1e-12, abs(best_p))))
                    sfrac_f = float(np.mean(Jf < best_f - subopt_threshold * max(1e-12, abs(best_f))))
                    pspread_p = policy_spread_mean(res_p['final_pi'])
                    pspread_f = policy_spread_mean(res_f['final_pi'])
                    gap = max(0.0, (best_f - best_p) / max(1e-12, abs(best_f)))

                    meta.append({'S': S, 'A': A, 'O': O, 'rep': rep, 'seed': seed_here})
                    partial_J_spread.append(spread_p)
                    partial_subopt_frac.append(sfrac_p)
                    partial_policy_spread.append(pspread_p)
                    full_J_spread.append(spread_f)
                    full_subopt_frac.append(sfrac_f)
                    full_policy_spread.append(pspread_f)
                    partial_best_J.append(best_p)
                    full_best_J.append(best_f)
                    gap_arr.append(gap)
                    S_arr.append(S)
                    A_arr.append(A)
                    O_arr.append(O)

                    inst += 1
                    if verbose:
                        print(f'  S={S} A={A} O={O} rep={rep}: '
                              f'best_p={best_p:.4f} best_f={best_f:.4f} gap={gap:.4f}')

    return {
        'meta': meta,
        'partial': {
            'J_spread': np.array(partial_J_spread),
            'subopt_frac': np.array(partial_subopt_frac),
            'policy_spread': np.array(partial_policy_spread),
            'best_J': np.array(partial_best_J),
        },
        'full': {
            'J_spread': np.array(full_J_spread),
            'subopt_frac': np.array(full_subopt_frac),
            'policy_spread': np.array(full_policy_spread),
            'best_J': np.array(full_best_J),
        },
        'gap': np.array(gap_arr),
        'S': np.array(S_arr),
        'A': np.array(A_arr),
        'O': np.array(O_arr),
    }


def summarize_by_configuration(batch_results):
    S_arr = batch_results['S']
    A_arr = batch_results['A']
    O_arr = batch_results['O']
    combos = []
    seen = set()
    for S, A, O in zip(S_arr, A_arr, O_arr):
        key = (int(S), int(A), int(O))
        if key not in seen:
            seen.add(key)
            combos.append(key)

    rows = []
    for (S, A, O) in combos:
        mask = (S_arr == S) & (A_arr == A) & (O_arr == O)
        row = {'S': S, 'A': A, 'O': O, 'n': int(mask.sum())}
        for key, arr in batch_results['partial'].items():
            row[f'partial_{key}_mean'] = float(np.mean(arr[mask]))
            row[f'partial_{key}_std'] = float(np.std(arr[mask]))
        for key, arr in batch_results['full'].items():
            row[f'full_{key}_mean'] = float(np.mean(arr[mask]))
            row[f'full_{key}_std'] = float(np.std(arr[mask]))
        row['gap_mean'] = float(np.mean(batch_results['gap'][mask]))
        row['gap_std'] = float(np.std(batch_results['gap'][mask]))
        rows.append(row)

    return pd.DataFrame(rows)


def export_summary_table_markdown(df, path):
    cols = list(df.columns)
    lines = []
    lines.append('| ' + ' | '.join(str(c) for c in cols) + ' |')
    lines.append('| ' + ' | '.join('---' for _ in cols) + ' |')
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f'{v:.4f}')
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Landscape experiment
# ---------------------------------------------------------------------------

def theta_history_to_policy_path(pomdp, theta0, max_iters, step_size, grad_clip):
    theta = theta0.copy()
    p_path = []
    q_path = []

    for _ in range(max_iters):
        _, grad, _, pi = objective_and_grad(pomdp, theta)
        # Record before stepping so the path includes the initial point and every
        # intermediate policy; after the loop we append the final converged policy.
        p_path.append(float(pi[0, 0]))
        q_path.append(float(pi[0, 1]))

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-7:
            break
        if gnorm > grad_clip:
            grad = grad * (grad_clip / gnorm)
        theta = theta + step_size * grad
    else:
        # Loop ran to completion — append the final stepped policy.
        _, _, _, pi_final = objective_and_grad(pomdp, theta)
        p_path.append(float(pi_final[0, 0]))
        q_path.append(float(pi_final[0, 1]))
        return np.array(p_path), np.array(q_path)

    # Early-stop path: append the final converged policy after the breaking step.
    _, _, _, pi_final = objective_and_grad(pomdp, theta)
    p_path.append(float(pi_final[0, 0]))
    q_path.append(float(pi_final[0, 1]))

    return np.array(p_path), np.array(q_path)


def run_landscape_experiment(cfg, rng):
    S = cfg['S']
    A = cfg['A']
    O = cfg['O']
    gamma = cfg['gamma']
    num_restarts = cfg['num_restarts']
    max_iters = cfg['max_iters']
    step_size = cfg['step_size']
    grad_clip = cfg['grad_clip']
    grid_n = cfg['grid_n']
    seed = cfg['seed']

    pomdp = random_pomdp(S, O, A, [gamma, gamma], 1.0, seed, 0.2)

    eps = 1e-3
    p_vals = np.linspace(eps, 1.0 - eps, grid_n)
    q_vals = np.linspace(eps, 1.0 - eps, grid_n)
    J_grid = np.zeros((grid_n, grid_n))

    for j, q in enumerate(q_vals):
        for i, p in enumerate(p_vals):
            pi = np.array([[p, q], [1.0 - p, 1.0 - q]])
            theta = np.log(np.maximum(pi, 1e-12))
            J, _, _, _ = objective_and_grad(pomdp, theta)
            J_grid[j, i] = J

    trajs = []
    final_J = np.zeros(num_restarts)
    final_p = np.zeros(num_restarts)
    final_q = np.zeros(num_restarts)

    for r in range(num_restarts):
        theta0 = 0.5 * rng.standard_normal((A, O))
        p_path, q_path = theta_history_to_policy_path(pomdp, theta0, max_iters,
                                                       step_size, grad_clip)
        trajs.append((p_path, q_path))
        theta_f, _, _ = policy_gradient_ascent(pomdp, theta0, max_iters, step_size, grad_clip)
        J_f, _, _, pi_f = objective_and_grad(pomdp, theta_f)
        final_J[r] = J_f
        final_p[r] = float(pi_f[0, 0])
        final_q[r] = float(pi_f[0, 1])

    return {
        'p_grid': p_vals,
        'q_grid': q_vals,
        'J_grid': J_grid,
        'trajs': trajs,
        'final_J': final_J,
        'final_p': final_p,
        'final_q': final_q,
        'pomdp': pomdp,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_batch_summary(batch_results, df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    S_arr = batch_results['S']
    labels = [f"S={int(S)}" for S in sorted(set(S_arr))]
    S_vals = sorted(set(S_arr))

    ax = axes[0]
    for pkey, fkey, label in [('partial', 'full', '')]:
        for S in S_vals:
            mask = S_arr == S
            ax.scatter(batch_results['partial']['J_spread'][mask],
                       batch_results['full']['J_spread'][mask],
                       s=18, alpha=0.5, label=f'S={S}')
    ax.set_xlabel('Partial-obs J spread')
    ax.set_ylabel('Full-obs J spread')
    ax.set_title('J spread: partial vs full')
    ax.legend(fontsize=7)

    ax = axes[1]
    for S in S_vals:
        mask = S_arr == S
        ax.scatter(batch_results['partial']['subopt_frac'][mask],
                   batch_results['full']['subopt_frac'][mask],
                   s=18, alpha=0.5, label=f'S={S}')
    ax.set_xlabel('Partial subopt fraction')
    ax.set_ylabel('Full subopt fraction')
    ax.set_title('Suboptimal fraction')
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.hist(batch_results['gap'], bins=30, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.set_xlabel('Normalized optimality gap (partial vs full)')
    ax.set_ylabel('Count')
    ax.set_title('Optimality gap distribution')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_landscape_summary(land_res, out_path):
    p_grid = land_res['p_grid']
    q_grid = land_res['q_grid']
    J_grid = land_res['J_grid']
    trajs = land_res['trajs']
    final_J = land_res['final_J']
    final_p = land_res['final_p']
    final_q = land_res['final_q']

    fig = plt.figure(figsize=(13, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    im = ax0.pcolormesh(p_grid, q_grid, J_grid, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax0, label='J(π)')
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(trajs)))
    for idx, (pp, pq) in enumerate(trajs):
        ax0.plot(pp, pq, color=colors[idx], lw=0.7, alpha=0.6)
        if len(pp) > 0:
            ax0.plot(pp[-1], pq[-1], 'o', color=colors[idx], ms=3)
    ax0.set_xlabel('π(a=0 | o=0)')
    ax0.set_ylabel('π(a=0 | o=1)')
    ax0.set_title('J landscape + gradient trajectories')

    ax1 = fig.add_subplot(gs[1])
    ax1.hist(final_J, bins=15, color='steelblue', edgecolor='white', linewidth=0.4,
             orientation='horizontal')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Final J')
    ax1.set_title('Converged J values')

    ax2 = fig.add_subplot(gs[2])
    sc = ax2.scatter(final_p, final_q, c=final_J, cmap='plasma', s=40, zorder=3)
    plt.colorbar(sc, ax=ax2, label='J')
    ax2.set_xlabel('π(a=0 | o=0)')
    ax2.set_ylabel('π(a=0 | o=1)')
    ax2.set_title('Converged policy points')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, '..', 'output', 'localopt')
    os.makedirs(out_dir, exist_ok=True)

    batch_cfg = {
        'S_list': [4, 8, 12],
        'A_list': [2, 3, 4],
        'O_partial_list': [2, 3],
        'num_instances': 100,
        'num_restarts': 50,
        'max_iters': 350,
        'step_size': 0.05,
        'grad_clip': 10.0,
        'gamma_range': [0.95, 0.98],
        'reward_scale': 1.0,
        'verbose': True,
        'stationary_bonus': 0.15,
        'subopt_threshold': 0.01,
    }

    land_cfg = {
        'S': 4, 'A': 2, 'O': 2, 'gamma': 0.95,
        'num_restarts': 20, 'max_iters': 300,
        'step_size': 0.05, 'grad_clip': 10.0,
        'grid_n': 151, 'seed': 19,
    }

    rng = np.random.default_rng(42)

    print('=== Batch experiment ===')
    t0 = time.time()
    batch_results = run_batch_experiment(batch_cfg, rng)
    print(f'Batch done in {time.time() - t0:.1f}s')

    df = summarize_by_configuration(batch_results)
    df_path = os.path.join(out_dir, 'batch_summary.csv')
    df.to_csv(df_path, index=False)
    md_path = os.path.join(out_dir, 'batch_summary.md')
    export_summary_table_markdown(df, md_path)
    print(df.to_string())

    batch_fig = os.path.join(out_dir, 'batch_summary.png')
    plot_batch_summary(batch_results, df, batch_fig)
    print(f'Saved {batch_fig}')

    print('\n=== Landscape experiment ===')
    t0 = time.time()
    land_res = run_landscape_experiment(land_cfg, rng)
    print(f'Landscape done in {time.time() - t0:.1f}s')
    print(f'Final J values: min={land_res["final_J"].min():.4f} '
          f'max={land_res["final_J"].max():.4f}')

    land_fig = os.path.join(out_dir, 'landscape_summary.png')
    plot_landscape_summary(land_res, land_fig)
    print(f'Saved {land_fig}')


if __name__ == '__main__':
    main()
