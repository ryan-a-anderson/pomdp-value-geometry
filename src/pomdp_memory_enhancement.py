#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


# ---------------------------------------------------------------------------
# Core utilities (shared with localopt)
# ---------------------------------------------------------------------------

def softmax_columns(X):
    """X shape (A, Y); normalize each column."""
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
# POMDP generation (random reward — differs from localopt)
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

    R = reward_scale * (2.0 * rng.random((S, A)) - 1.0)

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
# Memory model construction
# ---------------------------------------------------------------------------

def all_tuples(base, length):
    n = base ** length
    result = np.zeros((n, length), dtype=int)
    for i in range(n):
        z = i
        for j in range(length - 1, -1, -1):
            result[i, j] = z % base
            z //= base
    return result


def tuple_to_index(v, base):
    idx = 0
    for x in v:
        idx = idx * base + int(x)
    return idx


def build_observation_memory_model(pomdp, k):
    S = pomdp['S']
    O = pomdp['O']
    A = pomdp['A']
    T = pomdp['T']
    Z = pomdp['Z']
    R = pomdp['R']
    mu = pomdp['mu']
    gamma = pomdp['gamma']

    if k == 0:
        H = 1
        mem_table = np.zeros((1, 0), dtype=int)
    else:
        H = O ** k
        mem_table = all_tuples(O, k)       # (H, k)

    Y = O * H
    y_of_oh = np.zeros((O, H), dtype=int)
    for o in range(O):
        for h in range(H):
            y_of_oh[o, h] = o + h * O

    next_mem_lookup = np.zeros((O, H), dtype=int)
    for o in range(O):
        for h in range(H):
            if k == 0:
                next_mem_lookup[o, h] = 0
            else:
                new_mem = [o] + list(mem_table[h, :k - 1])
                next_mem_lookup[o, h] = tuple_to_index(new_mem, O)

    X = S * H
    x_of_s_and_mem = np.zeros((S, H), dtype=int)
    for s in range(S):
        for h in range(H):
            x_of_s_and_mem[s, h] = h * S + s

    state_s_of_x = np.zeros(X, dtype=int)
    mem_of_x = np.zeros(X, dtype=int)
    for x in range(X):
        state_s_of_x[x] = x % S
        mem_of_x[x] = x // S

    x_list_by_mem = []
    for h in range(H):
        x_list_by_mem.append(np.arange(h * S, (h + 1) * S))

    if k == 0:
        mu_aug = np.zeros(X)
        for s in range(S):
            mu_aug[x_of_s_and_mem[s, 0]] = mu[s]
    else:
        p_obs = Z @ mu
        p_obs_sum = p_obs.sum()
        if p_obs_sum > 0:
            p_obs = p_obs / p_obs_sum

        mem_prob = np.zeros(H)
        for h in range(H):
            prob = 1.0
            for t in range(k):
                prob *= p_obs[mem_table[h, t]]
            mem_prob[h] = prob
        mem_prob_sum = mem_prob.sum()
        if mem_prob_sum > 0:
            mem_prob = mem_prob / mem_prob_sum

        mu_aug = np.zeros(X)
        for h in range(H):
            for s in range(S):
                mu_aug[h * S + s] = mu[s] * mem_prob[h]

    if k == 0:
        obs_of_y = np.arange(Y, dtype=int)
        mem_of_y = np.zeros(Y, dtype=int)
    else:
        obs_of_y = np.zeros(Y, dtype=int)
        mem_of_y_arr = np.zeros(Y, dtype=int)
        for y in range(Y):
            obs_of_y[y] = y % O
            mem_of_y_arr[y] = y // O
        mem_of_y = mem_of_y_arr

    return {
        'S': S, 'O': O, 'A': A, 'k': k,
        'T': T, 'Z': Z, 'R': R, 'gamma': gamma,
        'H': H, 'X': X, 'Y': Y,
        'mem_table': mem_table,
        'y_of_oh': y_of_oh,
        'next_mem_of_oh': next_mem_lookup,
        'x_of_s_and_mem': x_of_s_and_mem,
        'state_s_of_x': state_s_of_x,
        'mem_of_x': mem_of_x,
        'x_list_by_mem': x_list_by_mem,
        'mu_aug': mu_aug,
        'obs_of_y': obs_of_y,
        'mem_of_y': mem_of_y,
    }


# ---------------------------------------------------------------------------
# Generic objective and gradient (supports k-step memory)
# ---------------------------------------------------------------------------

def objective_and_grad_generic(model, theta):
    S = model['S']
    O = model['O']
    A = model['A']
    T = model['T']
    Z = model['Z']
    R = model['R']
    gamma = model['gamma']
    X = model['X']
    Y = model['Y']
    state_s_of_x = model['state_s_of_x']
    mem_of_x = model['mem_of_x']
    y_of_oh = model['y_of_oh']
    next_mem_of_oh = model['next_mem_of_oh']
    x_of_s_and_mem = model['x_of_s_and_mem']
    x_list_by_mem = model['x_list_by_mem']
    mu_aug = model['mu_aug']
    obs_of_y = model['obs_of_y']
    mem_of_y = model['mem_of_y']

    pi = softmax_columns(theta)             # (A, Y)

    Ppi = np.zeros((X, X))
    rpi = np.zeros(X)

    for x in range(X):
        s = state_s_of_x[x]
        h = mem_of_x[x]
        for o in range(O):
            z = Z[o, s]
            if z == 0.0:
                continue
            y = y_of_oh[o, h]
            piy = pi[:, y]                  # (A,)
            rpi[x] += z * (piy @ R[s, :])
            next_h = next_mem_of_oh[o, h]
            cols = x_of_s_and_mem[:, next_h]    # (S,)
            Ppi[x, cols] += z * np.einsum('a,sa->s', piy, T[s, :, :])

    M = np.eye(X) - gamma * Ppi
    V = np.linalg.solve(M, rpi)
    J = float(mu_aug @ V)
    w = np.linalg.solve(M.T, mu_aug)

    grad = np.zeros((A, Y))
    for y0 in range(Y):
        h0 = int(mem_of_y[y0])
        o0 = int(obs_of_y[y0])
        piy = pi[:, y0]                     # (A,)
        x_list = x_list_by_mem[h0]
        for a0 in range(A):
            e = np.zeros(A)
            e[a0] = 1.0
            dpi = piy * (e - piy[a0])       # (A,)
            dr = np.zeros(X)
            dP = np.zeros((X, X))
            for x in x_list:
                s = state_s_of_x[x]
                z = Z[o0, s]
                if z == 0.0:
                    continue
                dr[x] += z * (dpi @ R[s, :])
                next_h = next_mem_of_oh[o0, h0]
                cols = x_of_s_and_mem[:, next_h]
                dP[x, cols] += z * np.einsum('a,sa->s', dpi, T[s, :, :])
            grad[a0, y0] = w @ (dr + gamma * dP @ V)

    return J, grad, V, pi


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def policy_gradient_ascent_generic(model, theta0, max_iters, step_size, grad_clip):
    theta = theta0.copy()
    J_history = []
    gnorm_history = []

    for _ in range(max_iters):
        J, grad, V, pi = objective_and_grad_generic(model, theta)
        gnorm = np.linalg.norm(grad)
        J_history.append(J)
        gnorm_history.append(gnorm)

        if gnorm < 1e-7:
            break
        if gnorm > grad_clip:
            grad = grad * (grad_clip / gnorm)

        theta = theta + step_size * grad

    return theta, np.array(J_history), np.array(gnorm_history)


def run_many_restarts_generic(model, num_restarts, max_iters, step_size, grad_clip, rng):
    A = model['A']
    Y = model['Y']
    final_J = np.zeros(num_restarts)
    final_pi = []

    for r in range(num_restarts):
        theta0 = 0.25 * rng.standard_normal((A, Y))
        theta, _, _ = policy_gradient_ascent_generic(model, theta0, max_iters,
                                                      step_size, grad_clip)
        J_f, _, _, pi_f = objective_and_grad_generic(model, theta)
        final_J[r] = J_f
        final_pi.append(pi_f)

    return {'final_J': final_J, 'final_pi': final_pi}


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def regime_name(k):
    return f'partial_k{k}'


# ---------------------------------------------------------------------------
# Batch experiment
# ---------------------------------------------------------------------------

def run_batch_experiment_memory(cfg, rng):
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
    memory_list = cfg['memory_list']
    n_per_combo = cfg['num_instances_per_combo']
    verbose = cfg.get('verbose', False)

    regimes = [regime_name(k) for k in memory_list] + ['full']
    metrics = {reg: {'final_J': [], 'J_spread': [], 'subopt_frac': [],
                     'policy_spread': []} for reg in regimes}
    meta = []

    inst = 0
    for S in S_list:
        for A in A_list:
            for O in O_list:
                if O > S:
                    continue
                for rep in range(n_per_combo):
                    seed_here = 5000 + 97 * (inst + 1) + 11 * S + 7 * A + 5 * O + (rep + 1)
                    pomdp = random_pomdp(S, O, A, gamma_range, reward_scale,
                                         seed_here, stationary_bonus)
                    full_pomdp = convert_to_fully_observable_baseline(pomdp)

                    meta.append({'S': S, 'A': A, 'O': O, 'rep': rep, 'seed': seed_here})

                    for k in memory_list:
                        reg = regime_name(k)
                        model = build_observation_memory_model(pomdp, k)
                        iters_k = max_iters * (1 + k)
                        res = run_many_restarts_generic(model, num_restarts, iters_k,
                                                        step_size, grad_clip, rng)
                        Jk = res['final_J']
                        best = float(np.max(Jk))
                        spread = float(np.max(Jk) - np.min(Jk))
                        sfrac = float(np.mean(
                            Jk < best - subopt_threshold * max(1e-12, abs(best))))
                        pspread = policy_spread_mean(res['final_pi'])
                        metrics[reg]['final_J'].append(Jk)
                        metrics[reg]['J_spread'].append(spread)
                        metrics[reg]['subopt_frac'].append(sfrac)
                        metrics[reg]['policy_spread'].append(pspread)

                        if verbose:
                            print(f'  S={S} A={A} O={O} rep={rep} k={k}: '
                                  f'best={best:.4f} spread={spread:.4f}')

                    # Full observable baseline (k=0 model on full POMDP)
                    full_model = build_observation_memory_model(full_pomdp, 0)
                    res_f = run_many_restarts_generic(full_model, num_restarts, max_iters,
                                                      step_size, grad_clip, rng)
                    Jf = res_f['final_J']
                    best_f = float(np.max(Jf))
                    spread_f = float(np.max(Jf) - np.min(Jf))
                    sfrac_f = float(np.mean(
                        Jf < best_f - subopt_threshold * max(1e-12, abs(best_f))))
                    pspread_f = policy_spread_mean(res_f['final_pi'])
                    metrics['full']['final_J'].append(Jf)
                    metrics['full']['J_spread'].append(spread_f)
                    metrics['full']['subopt_frac'].append(sfrac_f)
                    metrics['full']['policy_spread'].append(pspread_f)

                    inst += 1

    # Convert lists to arrays
    for reg in regimes:
        metrics[reg]['J_spread'] = np.array(metrics[reg]['J_spread'])
        metrics[reg]['subopt_frac'] = np.array(metrics[reg]['subopt_frac'])
        metrics[reg]['policy_spread'] = np.array(metrics[reg]['policy_spread'])

    return {'metrics': metrics, 'regimes': regimes, 'meta': meta}


def summarize_memory_experiment_by_configuration(batch_results):
    meta = batch_results['meta']
    regimes = batch_results['regimes']
    metrics = batch_results['metrics']

    combos = []
    seen = set()
    for m in meta:
        key = (m['S'], m['A'], m['O'])
        if key not in seen:
            seen.add(key)
            combos.append(key)

    rows = []
    for (S, A, O) in combos:
        idxs = [i for i, m in enumerate(meta) if m['S'] == S and m['A'] == A and m['O'] == O]
        row = {'S': S, 'A': A, 'O': O, 'n': len(idxs)}
        for reg in regimes:
            for mkey in ['J_spread', 'subopt_frac', 'policy_spread']:
                arr = metrics[reg][mkey]
                vals = arr[idxs]
                row[f'{reg}_{mkey}_mean'] = float(np.mean(vals))
                row[f'{reg}_{mkey}_std'] = float(np.std(vals))
        rows.append(row)

    return pd.DataFrame(rows)


def export_summary_table_markdown_memory(df, path):
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
# Plotting
# ---------------------------------------------------------------------------

def plot_memory_summary(batch_results, out_path):
    regimes = batch_results['regimes']
    metrics = batch_results['metrics']

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(regimes)))

    metric_keys = ['J_spread', 'subopt_frac', 'policy_spread']
    metric_labels = ['J value spread', 'Subopt fraction', 'Policy spread']

    for ax_idx, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[ax_idx]
        means = [np.mean(metrics[reg][mkey]) for reg in regimes]
        stds = [np.std(metrics[reg][mkey]) for reg in regimes]
        x = np.arange(len(regimes))
        ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=4,
                    color='steelblue', ecolor='gray', lw=1.5, ms=6)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)

    ax = axes[3]
    data = [metrics[reg]['J_spread'] for reg in regimes]
    bp = ax.boxplot(data, labels=regimes, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(regimes, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('J spread')
    ax.set_title('Final J spread (boxplot)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, '..', 'output', 'memory_enhancement')
    os.makedirs(out_dir, exist_ok=True)

    batch_cfg = {
        'S_list': [4, 8],
        'A_list': [2, 3],
        'O_partial_list': [2],
        'num_instances_per_combo': 5,
        'num_restarts': 40,
        'max_iters': 350,
        'step_size': 0.05,
        'grad_clip': 10.0,
        'gamma_range': [0.95, 0.98],
        'reward_scale': 1.0,
        'verbose': True,
        'stationary_bonus': 0.15,
        'memory_list': [0, 1, 2],
        'subopt_threshold': 0.01,
    }

    rng = np.random.default_rng(7)

    print('=== Memory enhancement batch experiment ===')
    t0 = time.time()
    batch_results = run_batch_experiment_memory(batch_cfg, rng)
    print(f'Batch done in {time.time() - t0:.1f}s')

    df = summarize_memory_experiment_by_configuration(batch_results)
    df_path = os.path.join(out_dir, 'memory_enhancement_summary.csv')
    df.to_csv(df_path, index=False)
    md_path = os.path.join(out_dir, 'memory_enhancement_summary.md')
    export_summary_table_markdown_memory(df, md_path)
    print(df.to_string())

    fig_path = os.path.join(out_dir, 'memory_enhancement_summary.png')
    plot_memory_summary(batch_results, fig_path)
    print(f'Saved {fig_path}')


if __name__ == '__main__':
    main()
