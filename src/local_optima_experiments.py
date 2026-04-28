#!/usr/bin/env python3
"""
Rebuttal Experiment: Local Optima in Large-Scale POMDPs

For random POMDP instances, run vanilla gradient ascent on J(π) = ρ^T V^π
from many random initializations. Show that different initializations
converge to different local optima, many of which are genuinely suboptimal.

This demonstrates the non-convex semi-algebraic structure of the feasible
value function set predicted by the theory.
"""

import numpy as np
import time
import os
import json


# ---------------------------------------------------------------------------
# POMDP generation
# ---------------------------------------------------------------------------

def generate_random_pomdp(n_states, n_actions, n_obs, gamma=0.9, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    alpha = {a: rng.dirichlet(np.ones(n_states), size=n_states)
             for a in range(n_actions)}
    beta = rng.dirichlet(np.ones(n_obs), size=n_states)
    rewards = {a: rng.uniform(0, 10, size=n_states) for a in range(n_actions)}
    return alpha, beta, rewards, gamma


# ---------------------------------------------------------------------------
# Bellman solve + gradient (softmax parametrization)
# ---------------------------------------------------------------------------

def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def solve_and_grad(alpha, beta, rewards, gamma, logits, rho):
    """J = rho^T V^pi, return (J, dJ/dlogits, V, pi)."""
    S = beta.shape[0]
    A = len(alpha)
    pi = softmax(logits)
    tau = beta @ pi

    P_pi = sum(tau[:, a:a+1] * alpha[a] for a in range(A))
    r_pi = sum(tau[:, a] * rewards[a] for a in range(A))

    M = np.eye(S) - gamma * P_pi
    V = np.linalg.solve(M, r_pi)
    J = float(rho @ V)

    lam = np.linalg.solve(M.T, rho)
    Q = np.column_stack([rewards[a] + gamma * (alpha[a] @ V) for a in range(A)])
    dJ_dtau = lam[:, None] * Q
    dJ_dpi = beta.T @ dJ_dtau
    dJ_dlogits = pi * (dJ_dpi - (dJ_dpi * pi).sum(axis=1, keepdims=True))

    return J, dJ_dlogits, V, pi


# ---------------------------------------------------------------------------
# Vanilla gradient ascent
# ---------------------------------------------------------------------------

def optimize(alpha, beta, rewards, gamma, rho, logits_init,
             steps=3000, lr=0.005):
    """Fixed-lr gradient ascent. No momentum, no adaptive rates."""
    logits = logits_init.copy()
    for _ in range(steps):
        J, grad, V, pi = solve_and_grad(alpha, beta, rewards, gamma, logits, rho)
        logits = logits + lr * grad
    J_f, _, V_f, pi_f = solve_and_grad(alpha, beta, rewards, gamma, logits, rho)
    return {"J_final": J_f, "V_final": V_f, "pi_final": pi_f}


# ---------------------------------------------------------------------------
# Clustering converged runs
# ---------------------------------------------------------------------------

def cluster_runs(results, j_tol=0.1):
    """Cluster by J value. Returns list of clusters sorted by J descending."""
    J = np.array([r["J_final"] for r in results])
    idx = np.argsort(J)[::-1]
    clusters = []
    for i in idx:
        placed = False
        for c in clusters:
            if abs(J[i] - c["J_mean"]) < j_tol:
                c["members"].append(i)
                c["J_mean"] = np.mean(J[c["members"]])
                placed = True
                break
        if not placed:
            clusters.append({"J_mean": J[i], "members": [i]})
    clusters.sort(key=lambda c: c["J_mean"], reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Single instance analysis
# ---------------------------------------------------------------------------

def analyze_instance(alpha, beta, rewards, gamma, rho,
                     n_starts=50, steps=3000, lr=0.005, rng=None):
    """Run n_starts optimizations, cluster results, return detailed stats."""
    if rng is None:
        rng = np.random.default_rng()

    O = beta.shape[1]
    A = len(alpha)
    results = []
    for _ in range(n_starts):
        logits_init = rng.standard_normal((O, A))
        res = optimize(alpha, beta, rewards, gamma, rho, logits_init,
                       steps=steps, lr=lr)
        results.append(res)

    clusters = cluster_runs(results)
    J_all = np.array([r["J_final"] for r in results])
    V_all = np.array([r["V_final"] for r in results])

    n_optima = len(clusters)
    cluster_sizes = [len(c["members"]) for c in clusters]
    cluster_J = [c["J_mean"] for c in clusters]

    # Suboptimal fraction: runs NOT in the best cluster
    best_count = cluster_sizes[0]
    subopt_frac = 1.0 - best_count / len(results)

    # Gap between best and worst cluster
    J_gap = cluster_J[0] - cluster_J[-1] if n_optima > 1 else 0.0

    # V distance between best and worst cluster (representative members)
    if n_optima > 1:
        best_idx = clusters[0]["members"][0]
        worst_idx = clusters[-1]["members"][0]
        v_dist = float(np.linalg.norm(V_all[best_idx] - V_all[worst_idx]))
    else:
        v_dist = 0.0

    return {
        "n_optima": n_optima,
        "cluster_sizes": cluster_sizes,
        "cluster_J": cluster_J,
        "subopt_frac": subopt_frac,
        "J_best": cluster_J[0],
        "J_gap": J_gap,
        "V_dist": v_dist,
    }


# ---------------------------------------------------------------------------
# Sweep over rho values for one POMDP instance
# ---------------------------------------------------------------------------

def analyze_instance_rho_sweep(alpha, beta, rewards, gamma,
                               n_rhos=10, n_starts=50, steps=3000,
                               lr=0.005, rng=None):
    """Try multiple rho values per instance, return best (most optima)."""
    if rng is None:
        rng = np.random.default_rng()
    S = beta.shape[0]

    all_stats = []
    for _ in range(n_rhos):
        rho = rng.dirichlet(np.ones(S))
        stats = analyze_instance(alpha, beta, rewards, gamma, rho,
                                 n_starts=n_starts, steps=steps, lr=lr,
                                 rng=rng)
        all_stats.append(stats)

    # Return stats for all rho values
    return all_stats


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiments(configs, n_instances=20, n_rhos=10, n_starts=50,
                    steps=3000, lr=0.005, gamma=0.9, seed=42):
    rng = np.random.default_rng(seed)
    all_results = []

    for S, A, O in configs:
        print(f"\n{'='*60}")
        print(f"Config (S={S}, A={A}, O={O})")
        print(f"{'='*60}")

        config_stats = []

        for inst in range(n_instances):
            t0 = time.time()
            alpha, beta, rewards, gamma_val = generate_random_pomdp(
                S, A, O, gamma=gamma, rng=rng
            )
            rho_stats = analyze_instance_rho_sweep(
                alpha, beta, rewards, gamma_val,
                n_rhos=n_rhos, n_starts=n_starts, steps=steps, lr=lr,
                rng=rng
            )
            config_stats.append(rho_stats)
            elapsed = time.time() - t0

            # Summary for this instance: take the rho with most optima
            max_optima = max(s["n_optima"] for s in rho_stats)
            mean_optima = np.mean([s["n_optima"] for s in rho_stats])
            if (inst + 1) % 5 == 0 or inst == 0:
                print(f"  Instance {inst+1}/{n_instances}  "
                      f"max_optima={max_optima}  mean_optima={mean_optima:.1f}  "
                      f"({elapsed:.1f}s)")

        # Aggregate across all instances and rhos
        all_n_optima = [s["n_optima"]
                        for inst_stats in config_stats for s in inst_stats]
        all_subopt = [s["subopt_frac"]
                      for inst_stats in config_stats for s in inst_stats]
        all_gaps = [s["J_gap"]
                    for inst_stats in config_stats for s in inst_stats
                    if s["n_optima"] > 1]
        all_vdist = [s["V_dist"]
                     for inst_stats in config_stats for s in inst_stats
                     if s["n_optima"] > 1]

        n_arr = np.array(all_n_optima)
        row = {
            "config": f"({S},{A},{O})",
            "S": S, "A": A, "O": O,
            "mean_n_optima": float(n_arr.mean()),
            "max_n_optima": int(n_arr.max()),
            "frac_multi": float(np.mean(n_arr > 1)),
            "optima_dist": dict(zip(
                [int(k) for k in np.unique(n_arr)],
                [int(v) for v in np.unique(n_arr, return_counts=True)[1]]
            )),
            "mean_subopt_frac": float(np.mean(all_subopt)),
            "mean_J_gap": float(np.mean(all_gaps)) if all_gaps else 0.0,
            "max_J_gap": float(np.max(all_gaps)) if all_gaps else 0.0,
            "mean_V_dist": float(np.mean(all_vdist)) if all_vdist else 0.0,
            "max_V_dist": float(np.max(all_vdist)) if all_vdist else 0.0,
        }
        all_results.append(row)

        print(f"\n  Summary:")
        print(f"    Mean # optima: {row['mean_n_optima']:.2f}  "
              f"Max: {row['max_n_optima']}  "
              f"Frac with >1: {row['frac_multi']:.0%}")
        print(f"    Distribution: {row['optima_dist']}")
        print(f"    Mean subopt fraction: {row['mean_subopt_frac']:.3f}")
        if all_gaps:
            print(f"    J gap (best-worst):  mean={row['mean_J_gap']:.3f}  "
                  f"max={row['max_J_gap']:.3f}")
            print(f"    V distance:          mean={row['mean_V_dist']:.2f}  "
                  f"max={row['max_V_dist']:.2f}")

    return all_results


def format_results_table(results):
    header = ("| Config | Mean # Optima | Max # Optima | % Multi-Optima "
              "| Mean Subopt Frac | Mean J Gap | Max J Gap "
              "| Mean V Dist | Max V Dist |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]

    for r in results:
        line = (f"| {r['config']} "
                f"| {r['mean_n_optima']:.2f} "
                f"| {r['max_n_optima']} "
                f"| {r['frac_multi']:.0%} "
                f"| {r['mean_subopt_frac']:.3f} "
                f"| {r['mean_J_gap']:.3f} "
                f"| {r['max_J_gap']:.3f} "
                f"| {r['mean_V_dist']:.2f} "
                f"| {r['max_V_dist']:.2f} |")
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    configs = [
        (8,  2, 2), (8,  3, 2), (8,  3, 3), (8,  4, 3),
        (12, 2, 2), (12, 3, 2), (12, 3, 3), (12, 4, 3),
        (16, 2, 2), (16, 3, 2), (16, 3, 3), (16, 4, 3),
        (20, 2, 2), (20, 3, 2), (20, 3, 3), (20, 4, 3),
        (24, 2, 2), (24, 3, 2), (24, 3, 3), (24, 4, 3),
    ]

    print("=" * 60)
    print("Local Optima in Large-Scale POMDPs")
    print("Vanilla gradient ascent, 50 starts × 10 rho × 20 instances")
    print("=" * 60)

    results = run_experiments(
        configs,
        n_instances=20,
        n_rhos=10,
        n_starts=50,
        steps=3000,
        lr=0.005,
        gamma=0.9,
        seed=42,
    )

    outdir = os.path.dirname(os.path.abspath(__file__))

    table = format_results_table(results)
    with open(os.path.join(outdir, "local_optima_summary.md"), "w") as f:
        f.write("# Local Optima in POMDPs\n\n")
        f.write("Vanilla gradient ascent (lr=0.005, 3000 steps) from 50 random "
                "initializations, 10 random ρ per instance, 20 instances per config.\n\n")
        f.write(table + "\n")

    with open(os.path.join(outdir, "local_optima_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(table)
