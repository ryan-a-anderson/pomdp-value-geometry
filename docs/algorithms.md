# Computational Algorithms

The codebase uses two distinct algorithms. They study the same mathematical object from different angles and are deliberately not interchangeable.

---

## Algorithm A — Affine Bellman solve with projected gradient

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

## Algorithm B — Softmax policy gradient with adjoint

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
