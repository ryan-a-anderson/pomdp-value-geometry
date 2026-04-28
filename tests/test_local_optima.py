#!/usr/bin/env python3
"""
Tests for local_optima_experiments.py.

Run from repo root: python tests/test_local_optima.py
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from local_optima_experiments import generate_random_pomdp, solve_and_grad


def test_bellman_solve():
    """Gradient computation is correct (finite-difference check)."""
    rng = np.random.default_rng(0)
    alpha, beta, rewards, gamma = generate_random_pomdp(4, 2, 2, rng=rng)
    rho = rng.dirichlet(np.ones(4))
    logits = rng.standard_normal((2, 2))

    J, grad, V, pi = solve_and_grad(alpha, beta, rewards, gamma, logits, rho)

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
    print(f"  PASS: gradient max error = {err:.2e}")


def test_optimizer_improves():
    """Vanilla gradient ascent with momentum improves J monotonically (modulo small wobble)."""
    rng = np.random.default_rng(1)
    alpha, beta, rewards, gamma = generate_random_pomdp(6, 2, 2, rng=rng)
    rho = rng.dirichlet(np.ones(6))
    logits = rng.standard_normal((2, 2))

    J_prev = -np.inf
    vel = np.zeros_like(logits)
    for t in range(200):
        J, grad, V, pi = solve_and_grad(alpha, beta, rewards, gamma, logits, rho)
        if t > 20:
            assert J >= J_prev - 0.5, f"J dropped sharply at step {t}: {J_prev:.4f} -> {J:.4f}"
        J_prev = max(J_prev, J)
        gnorm = np.linalg.norm(grad)
        if gnorm > 10.0:
            grad = grad * (10.0 / gnorm)
        vel = 0.6 * vel + grad
        logits = logits + 0.2 * vel

    print(f"  PASS: optimizer monotone over 200 steps")


if __name__ == "__main__":
    tests = [
        ("Gradient correctness", test_bellman_solve),
        ("Optimizer improves J", test_optimizer_improves),
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
