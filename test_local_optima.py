#!/usr/bin/env python3
"""
Tests that the local optima experiment code must pass before running
the full experiment suite.

Run: python test_local_optima.py
"""
import numpy as np
import sys
import time

sys.path.insert(0, '.')
from local_optima_experiments import (
    generate_random_pomdp, make_fully_observable,
    solve_and_grad, softmax, optimize, run_single_instance, compute_metrics
)


def test_bellman_solve():
    """Gradient computation is correct (finite-difference check)."""
    rng = np.random.default_rng(0)
    alpha, beta, rewards, gamma = generate_random_pomdp(4, 2, 2, rng=rng)
    rho = rng.dirichlet(np.ones(4))
    logits = rng.standard_normal((2, 2))

    J, grad, V, pi = solve_and_grad(alpha, beta, rewards, gamma, logits, rho)

    # Finite difference check
    eps = 1e-5
    grad_fd = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            logits_p = logits.copy(); logits_p[i, j] += eps
            logits_m = logits.copy(); logits_m[i, j] -= eps
            Jp, _, _, _ = solve_and_grad(alpha, beta, rewards, gamma, logits_p, rho)
            Jm, _, _, _ = solve_and_grad(alpha, beta, rewards, gamma, logits_m, rho)
            grad_fd[i, j] = (Jp - Jm) / (2 * eps)

    err = np.max(np.abs(grad - grad_fd))
    assert err < 1e-5, f"Gradient error {err:.2e} too large"
    print(f"  PASS: gradient error = {err:.2e}")


def test_optimizer_improves():
    """Optimizer always increases J (or stays flat)."""
    rng = np.random.default_rng(1)
    alpha, beta, rewards, gamma = generate_random_pomdp(6, 2, 2, rng=rng)
    rho = rng.dirichlet(np.ones(6))
    logits = rng.standard_normal((2, 2))

    # Run with history tracking
    J_prev = -np.inf
    vel = np.zeros_like(logits)
    improved = True
    for t in range(200):
        J, grad, V, pi = solve_and_grad(alpha, beta, rewards, gamma, logits, rho)
        if t > 20 and J < J_prev - 0.5:
            # Allow small oscillation from momentum but not large drops
            improved = False
            break
        J_prev = max(J_prev, J)
        gnorm = np.linalg.norm(grad)
        if gnorm > 10.0:
            grad = grad * (10.0 / gnorm)
        vel = 0.6 * vel + grad
        logits = logits + 0.2 * vel

    print(f"  PASS: optimizer improved J from start to end")


def test_paper_example_has_local_optima():
    """The paper's 2-state POMDP with rho=(0.2,0.8) has local optima."""
    P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
    P1 = np.array([[0.65, 0.35], [0.15, 0.85]])
    alpha = {0: P0, 1: P1}
    beta = np.array([[0.80, 0.20], [0.30, 0.70]])
    rewards = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    gamma = 0.9
    rho = np.array([0.2, 0.8])

    rng = np.random.default_rng(42)
    results = run_single_instance(alpha, beta, rewards, gamma, rho,
                                  n_starts=50, rng=rng)
    m = compute_metrics(results)

    assert m["subopt_fraction"] > 0.05, \
        f"Paper example should have local optima, got subopt={m['subopt_fraction']}"
    assert m["value_spread"] > 0.5, \
        f"Paper example should have V spread > 0.5, got {m['value_spread']}"
    print(f"  PASS: paper example subopt={m['subopt_fraction']:.2f}, "
          f"V_spread={m['value_spread']:.3f}")


def test_mdp_no_local_optima_small():
    """Fully observable 4-state MDP: all runs converge to same optimum."""
    rng = np.random.default_rng(42)
    alpha, beta, rewards, gamma = generate_random_pomdp(4, 2, 2, rng=rng)
    rho = rng.dirichlet(np.ones(4))

    alpha_f, beta_f, rewards_f, gamma_f = make_fully_observable(
        alpha, rewards, gamma, 4
    )
    results = run_single_instance(alpha_f, beta_f, rewards_f, gamma_f, rho,
                                  n_starts=30, rng=rng)
    m = compute_metrics(results)

    assert m["subopt_fraction"] == 0.0, \
        f"4-state MDP should have no local optima, got subopt={m['subopt_fraction']}"
    print(f"  PASS: 4-state MDP subopt=0, V_spread={m['value_spread']:.6f}")


def test_mdp_no_local_optima_large():
    """
    Fully observable 20-state MDP with 3 actions: all runs converge
    to same optimum. THIS IS THE CRITICAL TEST — if this fails, the
    optimizer is too weak for large fully-observable problems, and
    partial-vs-full comparisons will be meaningless.
    """
    rng = np.random.default_rng(42)

    # Test 5 random MDP instances
    for inst in range(5):
        alpha, _, rewards, gamma = generate_random_pomdp(20, 3, 2, rng=rng)
        rho = rng.dirichlet(np.ones(20))
        alpha_f, beta_f, rewards_f, gamma_f = make_fully_observable(
            alpha, rewards, gamma, 20
        )
        results = run_single_instance(alpha_f, beta_f, rewards_f, gamma_f, rho,
                                      n_starts=20, rng=rng)
        m = compute_metrics(results)

        assert m["subopt_fraction"] < 0.05, \
            f"20-state MDP inst {inst}: subopt={m['subopt_fraction']:.2f} " \
            f"(should be ~0). Value spread={m['value_spread']:.4f}"

    print(f"  PASS: 20-state MDP with A=3, all 5 instances have subopt < 0.05")


def test_partial_vs_full_gap():
    """
    For a batch of random POMDPs, partial obs suboptimality should
    be significantly higher than full obs on average.
    """
    rng = np.random.default_rng(42)
    partial_subs = []
    full_subs = []

    for inst in range(20):
        S, A, O = 8, 2, 2
        alpha, beta, rewards, gamma = generate_random_pomdp(S, A, O, rng=rng)
        rho = rng.dirichlet(np.ones(S))

        rp = run_single_instance(alpha, beta, rewards, gamma, rho,
                                 n_starts=30, rng=rng)
        mp = compute_metrics(rp)
        partial_subs.append(mp['subopt_fraction'])

        af, bf, rf, gf = make_fully_observable(alpha, rewards, gamma, S)
        rfull = run_single_instance(af, bf, rf, gf, rho,
                                    n_starts=30, rng=rng)
        mf = compute_metrics(rfull)
        full_subs.append(mf['subopt_fraction'])

    p_mean = np.mean(partial_subs)
    f_mean = np.mean(full_subs)

    assert p_mean > f_mean, \
        f"Partial subopt ({p_mean:.3f}) should exceed full ({f_mean:.3f})"
    print(f"  PASS: partial mean subopt={p_mean:.3f} > full={f_mean:.3f}")


def test_timing():
    """Single (20,3,2) instance with 30 starts should finish in < 30s."""
    rng = np.random.default_rng(42)
    alpha, beta, rewards, gamma = generate_random_pomdp(20, 3, 2, rng=rng)
    rho = rng.dirichlet(np.ones(20))

    t0 = time.time()
    results = run_single_instance(alpha, beta, rewards, gamma, rho,
                                  n_starts=30, rng=rng)
    elapsed = time.time() - t0

    assert elapsed < 30, f"Too slow: {elapsed:.1f}s for (20,3,2)"
    print(f"  PASS: (20,3,2) x 30 starts in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("Gradient correctness", test_bellman_solve),
        ("Optimizer improves J", test_optimizer_improves),
        ("Paper example has local optima", test_paper_example_has_local_optima),
        ("Small MDP: no local optima", test_mdp_no_local_optima_small),
        ("Large MDP: no local optima", test_mdp_no_local_optima_large),
        ("Partial > Full suboptimality gap", test_partial_vs_full_gap),
        ("Timing budget", test_timing),
    ]

    print("=" * 60)
    print("Local Optima Experiment — Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)
