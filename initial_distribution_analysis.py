#!/usr/bin/env python3
"""
Analyze which extreme points of the POMDP value function boundary are optimal
for different initial state distributions.

For initial distribution α = [α, 1-α]:
    V(π, α) = α·v(s=0) + (1-α)·v(s=1)

We identify:
1. Extreme points (vertices) of the feasible value function set
2. For each α ∈ [0,1], which extreme point policy is optimal
3. Explicit boundary equations separating regions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pomdp_linear_nonlinear_inequalities import POMDPAnalyzer

class InitialDistributionAnalyzer:
    def __init__(self, pomdp_analyzer):
        self.pomdp = pomdp_analyzer

    def find_extreme_points(self, n_samples=10000, n_corner_samples=50):
        """
        Find extreme points (vertices) of the feasible value function boundary.

        Strategy:
        1. Sample corner policies (p_o ∈ {0,1})
        2. Sample boundary policies (where q_i = ±1)
        3. Compute convex hull to identify vertices
        """
        value_points = []
        policy_params = []

        # 1. Corner policies: all combinations of p ∈ {0,1}^n_obs
        for i in range(2**self.pomdp.n_obs):
            p = np.array([(i >> j) & 1 for j in range(self.pomdp.n_obs)], dtype=float)
            v = self.pomdp.solve_v(p)
            value_points.append(v)
            policy_params.append(p)

        # 2. Random sampling on boundary (where at least one q_i ≈ ±1)
        for _ in range(n_samples):
            # Strategy: fix one q_i = ±1, solve for p
            # For simplicity, sample random p and check if feasible
            p = np.random.rand(self.pomdp.n_obs)
            try:
                v = self.pomdp.solve_v(p)
                x = v
                q = self.pomdp._get_q_of_x(x)
                if q is not None and np.all(np.abs(q) <= 1.01):
                    value_points.append(v)
                    policy_params.append(p)
            except:
                pass

        # 3. Sample along edges where q_i = ±1
        for k in range(self.pomdp.n_obs):
            for q_val in [-1, 1]:
                # Sample policies where we try to enforce q_k ≈ q_val
                for _ in range(n_corner_samples):
                    p = np.random.rand(self.pomdp.n_obs)
                    try:
                        v = self.pomdp.solve_v(p)
                        value_points.append(v)
                        policy_params.append(p)
                    except:
                        pass

        value_points = np.array(value_points)
        policy_params = np.array(policy_params)

        # Compute convex hull (use first 2 dimensions for 2-state case)
        hull = ConvexHull(value_points[:, :2])
        vertices_idx = hull.vertices

        extreme_values = value_points[vertices_idx]
        extreme_policies = policy_params[vertices_idx]

        return extreme_values, extreme_policies, value_points

    def compute_value_for_alpha(self, v, alpha):
        """
        Compute value of policy with value function v for initial distribution [α, 1-α].

        V(π, α) = α·v(s=0) + (1-α)·v(s=1)
        """
        return alpha * v[0] + (1 - alpha) * v[1]

    def find_optimal_regions(self, extreme_values, n_alpha=1000):
        """
        For each α ∈ [0,1], determine which extreme point is optimal.
        Returns boundary points where optimal policy changes.
        """
        alphas = np.linspace(0, 1, n_alpha)
        optimal_idx = np.zeros(n_alpha, dtype=int)
        values = np.zeros((len(extreme_values), n_alpha))

        for i, v in enumerate(extreme_values):
            values[i, :] = self.compute_value_for_alpha(v, alphas)

        optimal_idx = np.argmax(values, axis=0)

        # Find boundary points where optimal policy changes
        boundaries = []
        for i in range(len(alphas) - 1):
            if optimal_idx[i] != optimal_idx[i+1]:
                # Boundary between alphas[i] and alphas[i+1]
                # Find exact crossing point
                idx1, idx2 = optimal_idx[i], optimal_idx[i+1]
                v1, v2 = extreme_values[idx1], extreme_values[idx2]

                # Solve: α·v1[0] + (1-α)·v1[1] = α·v2[0] + (1-α)·v2[1]
                # α(v1[0] - v1[1] - v2[0] + v2[1]) = v2[1] - v1[1]
                denom = (v1[0] - v1[1]) - (v2[0] - v2[1])
                if abs(denom) > 1e-10:
                    alpha_boundary = (v2[1] - v1[1]) / denom
                    boundaries.append({
                        'alpha': alpha_boundary,
                        'policy1_idx': idx1,
                        'policy2_idx': idx2,
                        'v1': v1,
                        'v2': v2
                    })

        return alphas, optimal_idx, values, boundaries

    def plot_analysis(self, extreme_values, extreme_policies, all_values,
                      alphas, optimal_idx, values_vs_alpha, boundaries):
        """
        Create comprehensive visualization:
        1. Value function boundary with extreme points
        2. Regions of initial distribution simplex
        3. Heatmap of optimal policy
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # --- Plot 1: Value function boundary with extreme points ---
        ax = axes[0]

        # All sampled feasible points
        ax.scatter(all_values[:, 0], all_values[:, 1], s=1, alpha=0.3, c='lightgray', label='Feasible values')

        # Extreme points (vertices)
        colors = plt.cm.tab10(np.arange(len(extreme_values)))
        for i, (v, p) in enumerate(zip(extreme_values, extreme_policies)):
            ax.scatter(v[0], v[1], s=200, c=[colors[i]], marker='*',
                      edgecolors='black', linewidths=2,
                      label=f'Policy {i}: p={p[:2].round(2)}', zorder=10)

        # Convex hull
        hull = ConvexHull(extreme_values[:, :2])
        for simplex in hull.simplices:
            ax.plot(extreme_values[simplex, 0], extreme_values[simplex, 1],
                   'k-', linewidth=2, alpha=0.5)

        ax.set_xlabel('v(s=0)', fontsize=12)
        ax.set_ylabel('v(s=1)', fontsize=12)
        ax.set_title('Extreme Points of Value Function Boundary', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Optimal policy regions vs α ---
        ax = axes[1]

        # Plot value as function of α for each extreme point
        for i, v in enumerate(extreme_values):
            ax.plot(alphas, values_vs_alpha[i], linewidth=2.5, c=colors[i],
                   label=f'Policy {i}', alpha=0.8)

        # Highlight optimal value
        optimal_values = np.max(values_vs_alpha, axis=0)
        ax.plot(alphas, optimal_values, 'k--', linewidth=3, label='Optimal value', alpha=0.7)

        # Mark boundaries
        for b in boundaries:
            ax.axvline(b['alpha'], color='red', linestyle=':', linewidth=2, alpha=0.6)
            ax.text(b['alpha'], ax.get_ylim()[0], f"α={b['alpha']:.3f}",
                   rotation=90, va='bottom', fontsize=9, color='red')

        ax.set_xlabel('Initial distribution α (for [α, 1-α])', fontsize=12)
        ax.set_ylabel('Value V(π, α)', fontsize=12)
        ax.set_title('Value vs Initial Distribution', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Plot 3: Heatmap showing optimal policy index ---
        ax = axes[2]

        # Create finer heatmap
        alpha_fine = np.linspace(0, 1, 500)
        optimal_idx_fine = np.zeros(len(alpha_fine), dtype=int)

        for j, alpha in enumerate(alpha_fine):
            vals = [self.compute_value_for_alpha(v, alpha) for v in extreme_values]
            optimal_idx_fine[j] = np.argmax(vals)

        # Plot as colored regions
        for i in range(len(extreme_values)):
            mask = optimal_idx_fine == i
            if np.any(mask):
                alpha_region = alpha_fine[mask]
                ax.fill_between(alpha_region, 0, 1, color=colors[i], alpha=0.6, label=f'Policy {i}')

        # Add boundary lines
        for b in boundaries:
            ax.axvline(b['alpha'], color='black', linestyle='--', linewidth=2)
            ax.text(b['alpha'], 0.5, f"{b['alpha']:.4f}",
                   rotation=90, va='center', ha='right', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Initial distribution α (for [α, 1-α])', fontsize=12)
        ax.set_ylabel('Optimal policy region', fontsize=12)
        ax.set_title('Optimal Policy Regions', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('initial_distribution_analysis.png', dpi=150, bbox_inches='tight')
        # plt.show()  # Commented out for non-interactive use

    def print_boundary_equations(self, boundaries, extreme_values):
        """Print explicit boundary equations."""
        print("\n" + "="*80)
        print("EXPLICIT BOUNDARY EQUATIONS")
        print("="*80)
        print("\nFor initial distribution [α, 1-α], policies switch optimality at:\n")

        for i, b in enumerate(boundaries):
            idx1, idx2 = b['policy1_idx'], b['policy2_idx']
            v1, v2 = b['v1'], b['v2']
            alpha_b = b['alpha']

            print(f"Boundary {i+1}:")
            print(f"  Between Policy {idx1} and Policy {idx2}")
            print(f"  α* = {alpha_b:.6f}")
            print(f"  ")
            print(f"  Derivation:")
            print(f"    Policy {idx1}: v = {v1[:2]}")
            print(f"    Policy {idx2}: v = {v2[:2]}")
            print(f"    ")
            print(f"    Setting equal:")
            print(f"    α·{v1[0]:.4f} + (1-α)·{v1[1]:.4f} = α·{v2[0]:.4f} + (1-α)·{v2[1]:.4f}")
            print(f"    ")

            # Simplified form
            slope_diff = (v1[0] - v1[1]) - (v2[0] - v2[1])
            intercept_diff = v2[1] - v1[1]
            print(f"    α·({slope_diff:.4f}) = {intercept_diff:.4f}")
            print(f"    α = {alpha_b:.6f}")
            print(f"    ")
            print(f"  For α < {alpha_b:.6f}: Policy {idx1} optimal")
            print(f"  For α > {alpha_b:.6f}: Policy {idx2} optimal")
            print()


def main():
    # Use example from script
    P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
    PDelta = np.array([[-0.20,  0.20],
                       [-0.10,  0.10]])  # rank 1 (rows proportional)
    P1 = P0 + PDelta
    P = np.array([P0, P1])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    Beta = np.eye(2)

    print("Setting up POMDP...")
    pomdp = POMDPAnalyzer(P, R, Beta)
    analyzer = InitialDistributionAnalyzer(pomdp)

    print("Finding extreme points of value function boundary...")
    extreme_values, extreme_policies, all_values = analyzer.find_extreme_points(n_samples=5000)

    print(f"\nFound {len(extreme_values)} extreme points:")
    for i, (v, p) in enumerate(zip(extreme_values, extreme_policies)):
        print(f"  Policy {i}: p = {p}, v = {v[:2]}")

    print("\nComputing optimal regions across initial distributions...")
    alphas, optimal_idx, values, boundaries = analyzer.find_optimal_regions(extreme_values)

    print(f"\nFound {len(boundaries)} boundary points where optimal policy changes.")

    analyzer.print_boundary_equations(boundaries, extreme_values)

    print("\nGenerating visualizations...")
    analyzer.plot_analysis(extreme_values, extreme_policies, all_values,
                          alphas, optimal_idx, values, boundaries)

    print("\nDone! Plot saved to 'initial_distribution_analysis.png'")


if __name__ == "__main__":
    main()
