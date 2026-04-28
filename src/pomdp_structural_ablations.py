"""
POMDP Structural Ablation Experiments

This script compares optimization landscapes across different POMDP structural configurations:
1. Baseline: 2 states, 2 actions, 2 observations
2. More actions: 2 states, 3 actions, 2 observations
3. More observations: 2 states, 2 actions, 3 observations
4. Asymmetric observations with various structures

For each configuration, we analyze:
- Number of distinct local maxima
- Feasible value function region geometry
- Basin of attraction structure
"""

import numpy as np
import matplotlib.pyplot as plt
from pomdp_optim_dynamics import POMDPAnalyzer
from scipy.spatial import ConvexHull
import json


class StructuralAblationSuite:
    """Suite for comparing POMDP configurations with different structural properties."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.results = {}

    def create_baseline_config(self, use_very_noisy_params=True):
        """
        Baseline: 2 states, 2 actions, 2 observations

        Args:
            use_very_noisy_params: If True, uses P and R from very noisy config.
                                   If False, creates new parameters.
        """
        if use_very_noisy_params:
            # From very noisy configuration
            P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
            PDelta = np.array([[-0.20, 0.20], [-0.10, 0.10]])
            P1 = P0 + PDelta
            P = np.array([P0, P1])
            R = np.array([[1.0, 0.0], [0.0, 1.0]])
            Beta = np.array([[0.65, 0.35], [0.35, 0.65]])
        else:
            # Alternative parameters
            P0 = np.array([[0.8, 0.2], [0.3, 0.7]])
            P1 = np.array([[0.6, 0.4], [0.2, 0.8]])
            P = np.array([P0, P1])
            R = np.array([[1.0, 0.0], [0.0, 1.0]])
            Beta = np.array([[0.7, 0.3], [0.4, 0.6]])

        return P, R, Beta

    def create_3action_config(self, use_very_noisy_params=True):
        """
        2 states, 3 actions, 2 observations

        Adds a third action with its own transition dynamics.
        """
        if use_very_noisy_params:
            P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
            PDelta1 = np.array([[-0.20, 0.20], [-0.10, 0.10]])
            P1 = P0 + PDelta1
            # Third action: intermediate dynamics
            PDelta2 = np.array([[-0.10, 0.10], [-0.05, 0.05]])
            P2 = P0 + PDelta2
            P = np.array([P0, P1, P2])
            R = np.array([[1.0, 0.0], [0.0, 1.0]])
            Beta = np.array([[0.65, 0.35], [0.35, 0.65]])
        else:
            P0 = np.array([[0.8, 0.2], [0.3, 0.7]])
            P1 = np.array([[0.6, 0.4], [0.2, 0.8]])
            P2 = np.array([[0.7, 0.3], [0.25, 0.75]])
            P = np.array([P0, P1, P2])
            R = np.array([[1.0, 0.0], [0.0, 1.0]])
            Beta = np.array([[0.7, 0.3], [0.4, 0.6]])

        return P, R, Beta

    def create_3obs_config(self, use_very_noisy_params=True):
        """
        2 states, 2 actions, 3 observations

        Adds a third observation that both states can emit.
        Beta shape: (n_states, n_obs) where Beta[s, o] = P(o|s)
        """
        if use_very_noisy_params:
            P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
            PDelta = np.array([[-0.20, 0.20], [-0.10, 0.10]])
            P1 = P0 + PDelta
            P = np.array([P0, P1])
            R = np.array([[1.0, 0.0], [0.0, 1.0]])
            # Beta: rows are states, columns are observations
            # Each row must sum to 1 (probability distribution over observations)
            # Observation 0: more likely from state 0
            # Observation 1: more likely from state 1
            # Observation 2: equally likely from both (neutral)
            Beta = np.array([
                [0.5, 0.2, 0.3],   # s0: P(o0|s0)=0.5, P(o1|s0)=0.2, P(o2|s0)=0.3
                [0.2, 0.5, 0.3]    # s1: P(o0|s1)=0.2, P(o1|s1)=0.5, P(o2|s1)=0.3
            ])
        else:
            P0 = np.array([[0.8, 0.2], [0.3, 0.7]])
            P1 = np.array([[0.6, 0.4], [0.2, 0.8]])
            P = np.array([P0, P1])
            R = np.array([[1.0, 0.0], [0.0, 1.0]])
            Beta = np.array([
                [0.55, 0.25, 0.2],
                [0.25, 0.55, 0.2]
            ])

        return P, R, Beta

    def create_asymmetric_3obs_configs(self, use_very_noisy_params=True):
        """
        Multiple asymmetric observation structures with 3 observations.
        Beta shape: (n_states, n_obs) where Beta[s, o] = P(o|s)
        Each row must sum to 1.

        Returns list of (name, P, R, Beta) tuples.
        """
        configs = []

        if use_very_noisy_params:
            P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
            PDelta = np.array([[-0.20, 0.20], [-0.10, 0.10]])
            P1 = P0 + PDelta
            P = np.array([P0, P1])
            R = np.array([[1.0, 0.0], [0.0, 1.0]])
        else:
            P0 = np.array([[0.8, 0.2], [0.3, 0.7]])
            P1 = np.array([[0.6, 0.4], [0.2, 0.8]])
            P = np.array([P0, P1])
            R = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Config 1: P(o0|s0) = P(o1|s1) = 0 (cross-diagonal zeros)
        # s0 can only emit o1 or o2, s1 can only emit o0 or o2
        Beta1 = np.array([
            [0.0, 0.6, 0.4],   # s0: P(o0|s0)=0, P(o1|s0)=0.6, P(o2|s0)=0.4
            [0.6, 0.0, 0.4]    # s1: P(o0|s1)=0.6, P(o1|s1)=0, P(o2|s1)=0.4
        ])
        configs.append(("asym_cross_diagonal", P, R, Beta1))

        # Config 2: s0 fully observable via o0, s1 has two possible observations
        Beta2 = np.array([
            [1.0, 0.0, 0.0],   # s0: always emits o0 (diagnostic)
            [0.0, 0.6, 0.4]    # s1: emits o1 or o2
        ])
        configs.append(("asym_s0_diagnostic", P, R, Beta2))

        # Config 3: s1 fully observable via o1, s0 has two possible observations
        Beta3 = np.array([
            [0.6, 0.0, 0.4],   # s0: emits o0 or o2
            [0.0, 1.0, 0.0]    # s1: always emits o1 (diagnostic)
        ])
        configs.append(("asym_s1_diagnostic", P, R, Beta3))

        # Config 4: One observation is completely uninformative (same prob from both states)
        Beta4 = np.array([
            [0.4, 0.2, 0.4],   # s0: P(o0|s0)=0.4, P(o1|s0)=0.2, P(o2|s0)=0.4
            [0.2, 0.4, 0.4]    # s1: P(o0|s1)=0.2, P(o1|s1)=0.4, P(o2|s1)=0.4
        ])
        configs.append(("asym_o2_uninformative", P, R, Beta4))

        return configs

    def compute_exact_boundaries(self, pomdp, v_samples, grid_res=200):
        """
        Compute exact boundary contours using q = D^-1 R^c method.

        Args:
            pomdp: POMDPAnalyzer instance
            v_samples: Array of feasible value samples (N, 2) for determining grid bounds
            grid_res: Grid resolution for contour computation

        Returns:
            dict with:
                - XX, YY: Meshgrid arrays (grid_res, grid_res)
                - Q_arrays: List of arrays, one per observation (grid_res, grid_res)
                - FEAS: Boolean array marking feasible points (grid_res, grid_res)
        """
        print(f"  Computing exact boundaries (grid_res={grid_res})...")

        # 1. Determine grid bounds with padding
        v0_min, v0_max = v_samples[:, 0].min(), v_samples[:, 0].max()
        v1_min, v1_max = v_samples[:, 1].min(), v_samples[:, 1].max()

        pad0 = 0.3 * (v0_max - v0_min)
        pad1 = 0.3 * (v1_max - v1_min)

        v0_grid = np.linspace(v0_min - pad0, v0_max + pad0, grid_res)
        v1_grid = np.linspace(v1_min - pad1, v1_max + pad1, grid_res)
        XX, YY = np.meshgrid(v0_grid, v1_grid)

        # 2. Initialize output arrays
        n_obs = pomdp.n_obs
        Q_arrays = [np.full(XX.shape, np.nan) for _ in range(n_obs)]
        FEAS = np.zeros_like(XX, dtype=float)

        # 3. Compute q for each grid point
        for i in range(grid_res):
            for j in range(grid_res):
                x = np.array([XX[i, j], YY[i, j]])
                q = pomdp._get_q_of_x(x)

                if q is not None:
                    # Store each q component
                    for k in range(min(len(q), n_obs)):
                        Q_arrays[k][i, j] = q[k]

                    # Mark as feasible if all |q_i| <= 1
                    FEAS[i, j] = 1.0 if np.all(np.abs(q) <= 1.0 + 1e-9) else 0.0

        print(f"  Boundary computation complete")
        return {
            "XX": XX,
            "YY": YY,
            "Q_arrays": Q_arrays,
            "FEAS": FEAS
        }

    def plot_configuration_detailed(self, config_key, config_result, param_set_name):
        """
        Generate detailed boundary plot for a single configuration.

        Similar to reference image showing:
        - Exact boundary contours (q_i = ±1)
        - Feasible region shading
        - Local maxima as red stars
        - Sampled feasible points
        """
        # Extract data
        feasible_points = np.array(config_result["feasible_sample_points"])

        if "boundary_data" not in config_result:
            print(f"  Warning: No boundary data for {config_key}, skipping detailed plot")
            return

        bd = config_result["boundary_data"]
        XX, YY = np.array(bd["XX"]), np.array(bd["YY"])
        Q_arrays = [np.array(Q) for Q in bd["Q_arrays"]]
        FEAS = np.array(bd["FEAS"])

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # 1. Shade feasible region
        ax.contourf(XX, YY, FEAS, levels=[-0.1, 0.5, 1.1],
                    colors=['white', 'lightgreen'], alpha=0.25)

        # 2. Plot exact boundaries (q_i = ±1)
        colors = ['red', 'green', 'orange', 'purple']
        labels = ['q₀=±1', 'q₁=±1', 'q₂=±1', 'q₃=±1']

        for i, Q in enumerate(Q_arrays):
            ax.contour(XX, YY, Q, levels=[-1, 1],
                      colors=colors[i % len(colors)],
                      linewidths=2.5, linestyles='--',
                      label=labels[i])

        # 3. Scatter sampled feasible points
        ax.scatter(feasible_points[:, 0], feasible_points[:, 1],
                  alpha=0.3, s=10, c='blue', label='Sampled feasible', zorder=2)

        # 4. Mark local maxima
        for maxima in config_result["local_maxima"]:
            v = maxima["value"]
            ax.scatter(v[0], v[1], c='red', s=200, marker='*',
                      edgecolors='black', linewidths=1.5, zorder=5)

        ax.plot([], [], 'r*', markersize=15, label='Local maxima')

        # Formatting
        ax.set_xlabel('v(s₀)', fontsize=12)
        ax.set_ylabel('v(s₁)', fontsize=12)
        ax.set_title(f"{config_result['name']}", fontsize=14, weight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save
        filename = f"{config_key}_{param_set_name}_boundary_detailed.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

    def analyze_configuration(self, name, P, R, Beta, gamma=0.9,
                             n_alpha_samples=20, n_starts=15, steps=300):
        """
        Analyze a single POMDP configuration.

        Returns dict with:
        - num_local_maxima: Number of distinct optimal policies found
        - local_maxima: List of (policy, value) tuples for each optimum
        - feasible_region_area: Approximate area of feasible value region
        - basin_info: Dict mapping alpha -> which optimum was reached
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print(f"{'='*60}")
        print(f"States: {P.shape[1]}, Actions: {P.shape[0]}, Observations: {Beta.shape[0]}")
        print(f"Transition matrices: {P.shape}")
        print(f"Beta (obs kernel): {Beta.shape}")

        pomdp = POMDPAnalyzer(P, R, Beta, gamma=gamma)

        # Sample across initial state distributions
        alphas = np.linspace(0.05, 0.95, n_alpha_samples)

        all_endpoints_p = []  # policy parameters
        all_endpoint_values = []
        basin_info = {}
        rng = np.random.default_rng()

        for alpha in alphas:
            rho = np.array([alpha, 1-alpha], dtype=float)

            # Run multiple optimization trajectories
            histories = []
            for _ in range(n_starts):
                p_init = rng.random(Beta.shape[1])  # random policy in [0,1]^n_obs
                hist = pomdp.optimize_projected_gradient(
                    rho=rho,
                    p_init=p_init,
                    steps=steps,
                    lr=0.2,
                    momentum=0.6,
                    clip_grad=10.0,
                    seed=None
                )
                histories.append(hist)

            # Collect endpoints
            for hist in histories:
                policy_p = hist["p"][-1]  # final policy parameter
                value = hist["v"][-1]     # final value function
                all_endpoints_p.append(policy_p)
                all_endpoint_values.append(value)

            basin_info[alpha] = [hist["v"][-1] for hist in histories]

        # Cluster endpoints to find distinct local maxima
        all_endpoints_p = np.array(all_endpoints_p)
        all_endpoint_values = np.array(all_endpoint_values)

        # Use value space clustering (more meaningful for comparing optima)
        cluster_tol_v = 0.05
        clusters = []

        for i, v_end in enumerate(all_endpoint_values):
            placed = False
            for c in clusters:
                if np.linalg.norm(v_end - c["center"]) <= cluster_tol_v:
                    c["members"].append(i)
                    # update center as mean of members
                    mem_vals = all_endpoint_values[c["members"]]
                    c["center"] = mem_vals.mean(axis=0)
                    placed = True
                    break
            if not placed:
                clusters.append({"center": v_end.copy(), "members": [i]})

        n_clusters = len(clusters)
        print(f"Found {n_clusters} distinct local maxima")

        # Extract representative for each cluster
        local_maxima = []
        for c_idx, c in enumerate(clusters):
            mem = c["members"]
            cluster_values = all_endpoint_values[mem]
            cluster_policies = all_endpoints_p[mem]

            # Find best value in cluster
            best_idx = np.argmax(cluster_values[:, 0] + cluster_values[:, 1])
            best_policy = cluster_policies[best_idx]
            best_value = cluster_values[best_idx]

            local_maxima.append({
                "policy": best_policy.tolist(),
                "value": best_value.tolist(),
                "cluster_size": len(mem)
            })

            print(f"  Maxima {c_idx}: policy={best_policy}, value={best_value}, "
                  f"basin={100*len(mem)/len(all_endpoints_p):.1f}%")

        # Compute feasible region properties
        # Sample feasible value functions
        n_samples = 500
        feasible_values = []
        rng_sample = np.random.default_rng(42)

        for _ in range(n_samples):
            # Random policy parameter p in [0,1]^n_obs
            p = rng_sample.random(Beta.shape[1])

            try:
                v = pomdp.solve_v(p)
                if v is not None and not np.any(np.isnan(v)) and not np.any(np.isinf(v)):
                    feasible_values.append(v)
            except:
                pass

        feasible_values = np.array(feasible_values)

        # Compute convex hull area (approximate)
        if len(feasible_values) >= 3:
            try:
                hull = ConvexHull(feasible_values)
                feasible_area = hull.volume  # In 2D, volume = area
            except:
                feasible_area = 0.0
        else:
            feasible_area = 0.0

        print(f"Feasible region approximate area: {feasible_area:.3f}")
        print(f"Sampled {len(feasible_values)} feasible value functions")

        # Compute exact boundaries using q = D^-1 R^c method
        boundary_data = self.compute_exact_boundaries(pomdp, feasible_values, grid_res=200)

        return {
            "name": name,
            "num_local_maxima": n_clusters,
            "local_maxima": local_maxima,
            "feasible_region_area": float(feasible_area),
            "basin_info": {str(k): [v.tolist() for v in vs]
                          for k, vs in basin_info.items()},
            "n_states": int(P.shape[1]),
            "n_actions": int(P.shape[0]),
            "n_observations": int(Beta.shape[0]),
            "feasible_sample_points": feasible_values.tolist(),
            "boundary_data": {
                "XX": boundary_data["XX"].tolist(),
                "YY": boundary_data["YY"].tolist(),
                "Q_arrays": [Q.tolist() for Q in boundary_data["Q_arrays"]],
                "FEAS": boundary_data["FEAS"].tolist()
            }
        }

    def run_all_configurations(self, use_very_noisy_params=True):
        """Run analysis on all configurations WITH boundary computation."""
        results = {}
        param_set_name = "very_noisy" if use_very_noisy_params else "alternative"

        # Baseline
        print("\n" + "="*70)
        print("BASELINE CONFIGURATION")
        print("="*70)
        P, R, Beta = self.create_baseline_config(use_very_noisy_params)
        results["baseline"] = self.analyze_configuration("Baseline (2s, 2a, 2o)",
                                                         P, R, Beta)
        self.plot_configuration_detailed("baseline", results["baseline"], param_set_name)

        # Note: 3 actions configuration skipped because POMDPAnalyzer only supports
        # binary actions (memoryless policies represented as p in [0,1]^n_obs)
        # To support 3+ actions would require simplex parameterization

        # 3 observations
        print("\n" + "="*70)
        print("3 OBSERVATIONS CONFIGURATION")
        print("="*70)
        P, R, Beta = self.create_3obs_config(use_very_noisy_params)
        results["3obs"] = self.analyze_configuration("3 Observations (2s, 2a, 3o)",
                                                     P, R, Beta)
        self.plot_configuration_detailed("3obs", results["3obs"], param_set_name)

        # Asymmetric configurations
        asymmetric_configs = self.create_asymmetric_3obs_configs(use_very_noisy_params)
        for config_name, P, R, Beta in asymmetric_configs:
            print("\n" + "="*70)
            print(f"ASYMMETRIC CONFIGURATION: {config_name}")
            print("="*70)
            results[config_name] = self.analyze_configuration(
                f"Asymmetric {config_name} (2s, 2a, 3o)",
                P, R, Beta
            )
            self.plot_configuration_detailed(config_name, results[config_name], param_set_name)

        self.results = results
        return results

    def plot_comparison(self, param_set_name="very_noisy"):
        """Create comparison visualizations."""
        if not self.results:
            print("No results to plot. Run analysis first.")
            return

        fig = plt.figure(figsize=(16, 10))

        # Extract data
        config_names = []
        num_maxima = []
        feasible_areas = []

        for key, result in self.results.items():
            config_names.append(result["name"])
            num_maxima.append(result["num_local_maxima"])
            feasible_areas.append(result["feasible_region_area"])

        # Plot 1: Number of local maxima
        ax1 = plt.subplot(2, 3, 1)
        bars = ax1.bar(range(len(config_names)), num_maxima,
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                             '#9467bd', '#8c564b', '#e377c2'][:len(config_names)])
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Number of Local Maxima')
        ax1.set_title('Optimization Landscape Complexity')
        ax1.set_xticks(range(len(config_names)))
        ax1.set_xticklabels([name.split('(')[0].strip() for name in config_names],
                           rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, num_maxima)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontsize=9)

        # Plot 2: Feasible region area
        ax2 = plt.subplot(2, 3, 2)
        bars = ax2.bar(range(len(config_names)), feasible_areas,
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                             '#9467bd', '#8c564b', '#e377c2'][:len(config_names)])
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Approximate Area')
        ax2.set_title('Feasible Value Function Region Size')
        ax2.set_xticks(range(len(config_names)))
        ax2.set_xticklabels([name.split('(')[0].strip() for name in config_names],
                           rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)

        # Plot 3: Structure summary
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')

        summary_text = "Configuration Details\n" + "="*30 + "\n\n"
        for result in self.results.values():
            summary_text += f"{result['name']}\n"
            summary_text += f"  States: {result['n_states']}, "
            summary_text += f"Actions: {result['n_actions']}, "
            summary_text += f"Obs: {result['n_observations']}\n"
            summary_text += f"  Local maxima: {result['num_local_maxima']}\n"
            summary_text += f"  Feasible area: {result['feasible_region_area']:.3f}\n\n"

        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', family='monospace')

        # Plot 4-6: Feasible region scatter plots with boundaries for first 3 configs
        for idx, (key, result) in enumerate(list(self.results.items())[:3]):
            ax = plt.subplot(2, 3, 4 + idx)

            # NEW: Plot exact boundaries first (background layer)
            if "boundary_data" in result:
                bd = result["boundary_data"]
                XX = np.array(bd["XX"])
                YY = np.array(bd["YY"])
                Q_arrays = [np.array(Q) for Q in bd["Q_arrays"]]
                FEAS = np.array(bd["FEAS"])

                # Shade feasible region
                ax.contourf(XX, YY, FEAS, levels=[-0.1, 0.5, 1.1],
                           colors=['white', 'lightgreen'], alpha=0.15)

                # Plot q_i = ±1 contours (exact boundaries)
                colors = ['red', 'green', 'orange', 'purple']
                for i, Q in enumerate(Q_arrays):
                    ax.contour(XX, YY, Q, levels=[-1, 1],
                              colors=colors[i % len(colors)],
                              linewidths=1.5, linestyles='--',
                              alpha=0.6)

            # Existing scatter plot
            feasible_points = np.array(result["feasible_sample_points"])
            if len(feasible_points) > 0:
                ax.scatter(feasible_points[:, 0], feasible_points[:, 1],
                          alpha=0.3, s=5, c='blue')

                # Plot local maxima
                for maxima in result["local_maxima"]:
                    v = maxima["value"]
                    ax.scatter(v[0], v[1], c='red', s=100, marker='*',
                             edgecolors='black', linewidths=1.5, zorder=5)

            ax.set_xlabel('v(s₀)')
            ax.set_ylabel('v(s₁)')
            ax.set_title(result['name'].split('(')[0].strip())
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        plt.tight_layout()
        filename = f"structural_ablations_comparison_{param_set_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison figure: {filename}")
        plt.close()

    def save_results(self, filename="structural_ablations_results.json"):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved results to {filename}")


if __name__ == "__main__":
    print("="*70)
    print("POMDP Structural Ablation Study")
    print("="*70)

    # Run with very noisy parameters
    print("\n\n" + "="*70)
    print("EXPERIMENT SET 1: Very Noisy Parameters")
    print("="*70)
    suite1 = StructuralAblationSuite(seed=42)
    results1 = suite1.run_all_configurations(use_very_noisy_params=True)
    suite1.plot_comparison(param_set_name="very_noisy")
    suite1.save_results("structural_ablations_very_noisy.json")

    # Run with alternative parameters
    print("\n\n" + "="*70)
    print("EXPERIMENT SET 2: Alternative Parameters")
    print("="*70)
    suite2 = StructuralAblationSuite(seed=42)
    results2 = suite2.run_all_configurations(use_very_noisy_params=False)
    suite2.plot_comparison(param_set_name="alternative")
    suite2.save_results("structural_ablations_alternative.json")

    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  Summary comparison figures:")
    print("    - structural_ablations_comparison_very_noisy.png")
    print("    - structural_ablations_comparison_alternative.png")
    print("  Detailed boundary figures (12 total):")
    print("    - baseline_{very_noisy,alternative}_boundary_detailed.png")
    print("    - 3obs_{very_noisy,alternative}_boundary_detailed.png")
    print("    - asym_*_{very_noisy,alternative}_boundary_detailed.png (4 configs)")
    print("  Data files:")
    print("    - structural_ablations_very_noisy.json")
    print("    - structural_ablations_alternative.json")
