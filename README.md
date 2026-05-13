# The Value Function Semi-Algebraic Set in POMDPs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code for the paper **"The Value Function Semi-Algebraic Set in POMDPs"** (Anderson & Montúfar, 2026).

## Overview

This repository contains the experiments and figures from the paper. The central result is that the set of feasible value functions for a POMDP under memoryless stochastic policies admits an explicit semi-algebraic description — a finite system of polynomial equations and inequalities — extending classical polyhedral results for fully observable MDPs to the partially observable setting.

The experiments illustrate two consequences of this geometric structure:
1. The feasible value set has curved (nonlinear) boundary components, captured exactly by finitely many polynomial constraints.
2. Partial observability induces genuinely non-convex optimization landscapes with multiple isolated local optima under policy gradient, unlike the fully observable case.

## Two computational algorithms

The codebase uses two distinct algorithms. They study the same mathematical object from different angles and are deliberately not interchangeable.

---

### Algorithm A — Affine Bellman solve with projected gradient

**Used by:** `pomdp_linear_nonlinear_inequalities.py`, `pomdp_optim_dynamics.py`, `pomdp_optim_dynamics_very_noisy.py`, `initial_distribution_analysis.py`, `initial_distribution_analysis_multi.py`, `pomdp_structural_ablations.py`

**Purpose:** Geometric analysis of the feasible value function set — visualizing its boundary, feasible region, and how the optimal policy changes with the initial state distribution. This is the algorithm that connects directly to the paper's theory.

**How it works:**

Restrict to **binary actions** (|A| = 2). Parametrize the memoryless stochastic policy by a single scalar per observation:

```
p_k = π(a=1 | o=k) ∈ [0, 1],  k = 0, …, n_obs−1
```

The policy-conditioned Bellman system is then **affine** in p:

```
A(p) v = b(p)

A(p) = I − γ P₀ − γ Σ_k p_k · diag(β[:,k]) · ΔP
b(p) = r₀   +      Σ_k p_k · diag(β[:,k]) · Δr
```

where `P₀` is the action-0 transition matrix, `ΔP = P₁ − P₀`, `r₀` is the action-0 reward, `Δr = r₁ − r₀`, and `β[:,k]` is the column of the observation kernel for observation k (i.e., `P(o=k | s)` for each state s).

**Value function:** `v(p) = A(p)⁻¹ b(p)` — one linear solve per policy.

**Gradient:**

Differentiate `A(p) v = b(p)` implicitly with respect to `p_k`:

```
A(p) · dv/dp_k = b_k − A_k · v
dJ/dp_k = ρᵀ · dv/dp_k
```

where `A_k = −γ · diag(β[:,k]) · ΔP` and `b_k = diag(β[:,k]) · Δr`. Each `dv/dp_k` requires one additional linear solve with the same matrix `A(p)`.

**Optimization:** Projected gradient ascent — update p, then clip to `[0,1]^{n_obs}`. Optional momentum.

**Why affine structure matters:** Because `v(p)` is the solution of a linear system whose coefficients are affine in p, the map `p ↦ v(p)` is a rational function of p. The image of `[0,1]^{n_obs}` under this map — the feasible value function set — is therefore a semi-algebraic set. The `_get_q_of_x` function in `POMDPAnalyzer` inverts this map: given a candidate value vector x, it recovers the policy coordinates q such that `A(q) x = b(q)`, and checks `|q_k| ≤ 1` for feasibility. This is the boundary test underlying all geometric visualizations (Figs 2–4).

**Data conventions (Algorithm A):**
- Transition: `P[a]` — shape `(n_actions, n_states, n_states)`, `P[a][s, s']`
- Observation kernel: `Beta[s, o]` — shape `(n_states, n_obs)`, `Beta[s, o] = P(o | s)`
- Reward: `R[a]` — shape `(n_actions, n_states)`, `R[a][s]`
- Policy: `p` — shape `(n_obs,)`, one probability scalar per observation

---

### Algorithm B — Softmax policy gradient with adjoint

**Used by:** `local_optima_experiments.py`, `pomdp_localopt.py`, `pomdp_memory_enhancement.py`, MATLAB scripts in `matlab/`

**Purpose:** Large-scale batch experiments over many random POMDP instances and many random restarts — producing the tables in Appendix B.2. Handles arbitrary numbers of actions.

**How it works:**

Parametrize the memoryless stochastic policy by a matrix of logits:

```
θ ∈ ℝ^{A × O},   π[a, o] = softmax(θ[:, o])[a]
```

The marginal action probability in state s is then `q[s, a] = Σ_o Z[o, s] · π[a, o]`, and the induced Bellman system is:

```
(I − γ P_π) V = r_π

P_π[s, s'] = Σ_a q[s, a] · T[s, s', a]
r_π[s]     = Σ_a q[s, a] · R[s, a]
```

This system is **nonlinear** in θ (through the softmax), so the affine decomposition of Algorithm A does not apply. However, at any fixed θ it remains a linear system and is solved directly.

**Value function:** `V = (I − γ P_π)⁻¹ r_π` — one linear solve per policy evaluation.

**Gradient:**

Use the adjoint (one solve, then chain rule through softmax):

```
w = (I − γ P_π)⁻ᵀ μ                         # adjoint solve

dJ/dθ[a₀, o] = wᵀ (dr + γ · dP · V)         # for each (a₀, o) pair
```

where `dr` and `dP` are the derivatives of `r_π` and `P_π` with respect to `θ[a₀, o]`, computed via the softmax Jacobian `dπ[a, o]/dθ[a₀, o] = π[a, o] · (𝟙[a=a₀] − π[a₀, o])`.

**Optimization:** Gradient ascent on unconstrained logits θ — no projection needed because softmax keeps π in the simplex automatically. Early stopping when `‖∇J‖ < 10⁻⁷`.

**Why softmax parameterization here:** The batch experiments run 50–100 random POMDP instances with 40–50 restarts each, across grid sizes up to (S=12, A=4, O=3). The softmax parameterization avoids numerical issues near the boundary of the policy simplex (where projected gradient stalls), scales naturally to |A| > 2, and is a standard choice in the policy gradient literature that the paper's results directly apply to.

**Relationship to Algorithm A for binary actions:** When |A| = 2, both algorithms optimize the same objective. The policies are related by `p_k = π(a=1 | o=k) = σ(θ[1,k] − θ[0,k])` (sigmoid). The gradients are related by `dJ/dθ = dJ/dp · p(1−p)` — same direction, different scaling. They converge to the same stationary points but trace different paths.

**Data conventions (Algorithm B):**
- Transition: `T[s, s', a]` — shape `(S, S, A)`, `T[s, s', a] = P(s' | s, a)`
- Observation kernel: `Z[o, s]` — shape `(O, S)`, `Z[o, s] = P(o | s)` (**transposed** relative to Algorithm A)
- Reward: `R[s, a]` — shape `(S, A)`
- Policy: `theta` — shape `(A, O)`, logits; `pi[:, o] = softmax(theta[:, o])`

---

## Script-to-algorithm map

| Script | Algorithm | Role |
|--------|-----------|------|
| `pomdp_linear_nonlinear_inequalities.py` | **A** | Core `POMDPAnalyzer` class; affine Bellman solve, boundary visualization (Figs 2, 3) |
| `pomdp_optim_dynamics.py` | **A** | Same `POMDPAnalyzer` extended with `grad_objective`, projected gradient ascent, multi-start trajectory plots |
| `pomdp_optim_dynamics_very_noisy.py` | **A** | Driver script: very-noisy three-region POMDP instance |
| `initial_distribution_analysis.py` | **A** | Sweeps `ρ = (α, 1−α)`; finds extreme points of value set, identifies optimal-policy regions (Fig 4) |
| `initial_distribution_analysis_multi.py` | **A** | Same analysis across multiple POMDP instances |
| `pomdp_structural_ablations.py` | **A** | Compares baseline / 3-action / 3-observation configurations |
| `local_optima_experiments.py` | **B** | Python reference port of Experiments A & B; multi-restart gradient ascent |
| `pomdp_localopt.py` | **B** | Python translation of `matlab/pomdp_localopt/`; batch experiment + landscape visualization |
| `pomdp_memory_enhancement.py` | **B** | Python translation of `matlab/pomdp_memory_enhancement/`; k-step observation memory |
| `matlab/pomdp_localopt/run_pomdp_localopt_experiment.m` | **B** | Canonical Experiments A & B (Tables 1, 2) |
| `matlab/pomdp_memory_enhancement/run_pomdp_memory_enhancement_experiment.m` | **B** | Canonical Experiment C (Tables 3a, 3b) |

**Note on `POMDPAnalyzer`:** The base class is defined in both `pomdp_linear_nonlinear_inequalities.py` and `pomdp_optim_dynamics.py` with shared methods (`_setup_affine_system`, `_compute_A`, `_compute_b`, `solve_v`, `_get_q_of_x`). The second file extends the first with optimization and ablation methods. `initial_distribution_analysis.py` imports from the first; `pomdp_structural_ablations.py` imports from the second.

---

## Components by paper section

### Figures

| Figure | Paper reference | Script | Algorithm |
|--------|----------------|--------|-----------|
| Fig. 1 | Section 2 | Schematic — no script | — |
| Fig. 2 | Theorem 3.1 (linear inequalities) | `pomdp_linear_nonlinear_inequalities.py` | A |
| Fig. 3 | Theorem 3.3 (semi-algebraic boundary) | `pomdp_linear_nonlinear_inequalities.py` | A |
| Fig. 4 | ρ-dependence of optimal policy | `initial_distribution_analysis.py` | A |

### Experiments (Appendix B.2)

The MATLAB scripts in `matlab/` are the canonical implementations that produced the paper's tables. The Python files `pomdp_localopt.py` and `pomdp_memory_enhancement.py` are faithful translations added after submission; `local_optima_experiments.py` is an earlier independent Python reference.

| Experiment | Paper reference | Canonical script | Python equivalent |
|------------|----------------|------------------|-------------------|
| **A** — Value spread, partial vs. full observability | Table 1 | `matlab/pomdp_localopt/run_pomdp_localopt_experiment.m` | `pomdp_localopt.py` |
| **B** — Fraction of suboptimal restarts | Table 2 | `matlab/pomdp_localopt/run_pomdp_localopt_experiment.m` | `pomdp_localopt.py`, `local_optima_experiments.py` |
| **C** — Finite-memory policies (k ∈ {0,1,2}) | Tables 3a/3b | `matlab/pomdp_memory_enhancement/run_pomdp_memory_enhancement_experiment.m` | `pomdp_memory_enhancement.py` |

The MATLAB outputs are checked in:
- `matlab/pomdp_localopt/summary_table.md` — matches Table 1 exactly.
- `matlab/pomdp_memory_enhancement/summary_table.md` — matches Tables 3a/3b exactly.

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   │
│   │   ── Algorithm A (affine Bellman / projected gradient) ──────────────────
│   ├── pomdp_linear_nonlinear_inequalities.py  # POMDPAnalyzer base; Figs 2, 3
│   ├── pomdp_optim_dynamics.py                 # POMDPAnalyzer + optimization
│   ├── pomdp_optim_dynamics_very_noisy.py      # three-region example driver
│   ├── initial_distribution_analysis.py        # Fig 4; ρ-sweep, extreme points
│   ├── initial_distribution_analysis_multi.py  # Fig 4 variants
│   ├── pomdp_structural_ablations.py           # structural configuration sweep
│   │
│   │   ── Algorithm B (softmax policy gradient / adjoint) ──────────────────
│   ├── local_optima_experiments.py             # reference port of Exps A & B
│   ├── pomdp_localopt.py                       # Python translation of matlab/pomdp_localopt/
│   └── pomdp_memory_enhancement.py             # Python translation of matlab/pomdp_memory_enhancement/
│
├── tests/
│   └── test_local_optima.py
│
├── matlab/
│   ├── pomdp_localopt/                         # Canonical Experiments A & B
│   │   ├── run_pomdp_localopt_experiment.m
│   │   ├── export_summary_table_markdown.m
│   │   ├── summary_table.md
│   │   └── README.txt
│   └── pomdp_memory_enhancement/               # Canonical Experiment C
│       ├── run_pomdp_memory_enhancement_experiment.m
│       ├── summary_table.md
│       └── README.txt
│
└── figures/                                    # Generated output figures
```

---

## Installation

### Python

```bash
git clone https://github.com/ryan-a-anderson/pomdp-value-geometry.git
cd pomdp-value-geometry
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `scipy`, `pandas`.

### MATLAB

The MATLAB scripts require only base MATLAB plus the Statistics Toolbox (for `gamrnd`). Tested on MATLAB R2022b+.

---

## Reproducing paper results

### Figures (Python — Algorithm A)

```bash
# Figs 2 and 3 (linear and semi-algebraic boundaries)
python src/pomdp_linear_nonlinear_inequalities.py

# Fig 4 (initial state distribution dependence)
python src/initial_distribution_analysis.py
```

### Experiments A & B (MATLAB — Algorithm B)

```matlab
cd matlab/pomdp_localopt
results = run_pomdp_localopt_experiment;
```

Outputs: `output/batch_summary.png`, `output/landscape_summary.png`, `output/summary_table.csv`, `output/results.mat`.

### Experiment C (MATLAB — Algorithm B)

```matlab
cd matlab/pomdp_memory_enhancement
results = run_pomdp_memory_enhancement_experiment;
```

Outputs: `output/memory_enhancement_summary.png`, `output/summary_table.csv`, `output/results.mat`.

### Python equivalents for Experiments A–C

```bash
python src/pomdp_localopt.py           # Experiments A & B
python src/pomdp_memory_enhancement.py # Experiment C
```

### Tests

```bash
python tests/test_local_optima.py
```

---

## POMDP instance for Figs 2 & 3

States S = {0, 1}, actions A = {0, 1}, observations O = {0, 1, 2}, discount γ = 0.9.

```
P⁰ = [[0.85, 0.15],    P¹ = [[0.65, 0.35],
       [0.25, 0.75]]          [0.15, 0.85]]

R   = [[1, 0],          β = [[0.80, 0.10, 0.10],
        [0, 1]]               [0.30, 0.65, 0.05]]
```

---

## Experiment setup (Appendix B.2)

Random POMDP instances (Algorithm B experiments) are drawn from a structured distribution designed to produce non-trivial optimization landscapes:

- **Transition kernels**: hidden-basin structure — states are split into "good" and "bad" halves; action 0 pushes toward the good basin, action 1 toward the bad basin, with a self-loop bias (stationary bonus).
- **Observation kernel**: for S=4, O=2 a fixed ambiguous matrix is used; otherwise each column is a Dirichlet sample biased by state parity.
- **Rewards** (Experiments A & B): structured — action 0 is high-reward in good-basin states, action 1 is high-reward in bad-basin states.
- **Rewards** (Experiment C): uniform random in [−1, 1].
- **Discount**: γ ~ Uniform[0.95, 0.98].
- **Initial distribution**: μ ~ Dirichlet.

Optimization uses vanilla gradient ascent on softmax logits (step size η = 0.05, gradient clipping at norm 10, up to 350 iterations, early stop at ‖∇J‖ < 10⁻⁷).

---

## Citation

```bibtex
@article{anderson2026value,
  title={The Value Function Semi-Algebraic Set in POMDPs},
  author={Anderson, Ryan A. and Montúfar, Guido},
  year={2026}
}
```

## Contact

Ryan Anderson — raanderson@g.ucla.edu

## License

MIT — see [LICENSE](LICENSE).
