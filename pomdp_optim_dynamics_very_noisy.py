#!/usr/bin/env python3
"""
Test optimization dynamics with the "Very Noisy" POMDP configuration
that exhibits 3 distinct optimal regions.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

# Import the POMDPAnalyzer class from the main file
from pomdp_optim_dynamics import POMDPAnalyzer

if __name__ == "__main__":
    # Very Noisy Configuration from README
    # This exhibits 3 distinct optimal regions based on initial distribution
    P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
    PDelta = np.array([[-0.20, 0.20], [-0.10, 0.10]])
    P1 = P0 + PDelta
    P = np.array([P0, P1])

    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    Beta = np.array([[0.65, 0.35], [0.35, 0.65]])  # Very noisy observations

    pomdp = POMDPAnalyzer(P, R, Beta, gamma=0.9)

    print("=" * 70)
    print("VERY NOISY POMDP: Optimization Dynamics Analysis")
    print("=" * 70)
    print("\nConfiguration:")
    print("  Beta = [[0.65, 0.35], [0.35, 0.65]]  (Very noisy)")
    print("  This configuration exhibits 3 distinct optimal regions:")
    print("    - Region 1 (α < 0.575): Policy [1,1] optimal")
    print("    - Region 2 (0.575 < α < 0.740): Policy [0,1] optimal")
    print("    - Region 3 (α > 0.740): Policy [0,0] optimal")
    print()

    # Test different initial state distributions across the three regions
    rhos_to_test = [
        (0.2, 0.8),   # Region 1: Should converge to policy [1,1]
        (0.5, 0.5),   # Region 1/2 boundary area
        (0.65, 0.35), # Region 2: Should converge to policy [0,1]
        (0.75, 0.25), # Region 2/3 boundary area
        (0.85, 0.15), # Region 3: Should converge to policy [0,0]
    ]

    print("1. Creating multi-start comparison across different initial distributions...")
    print("   Testing α values: 0.2, 0.5, 0.65, 0.75, 0.85")
    print()

    fig = pomdp.plot_multistart_comparison(
        rhos=rhos_to_test,
        n_starts=8,  # More starts to better capture basin structure
        steps=250,
        lr=0.2,
        momentum=0.65,
        clip_grad=10.0,
        num_y_dirs=20,
        grid_res=200,
        num_iso_levels=14,
        seed=42,
    )
    fig.savefig('optimization_dynamics_very_noisy_multistart.png', dpi=300, bbox_inches='tight')
    print("   Saved: optimization_dynamics_very_noisy_multistart.png")

    # Detailed analysis for a single rho in the middle region
    print("\n2. Detailed analysis for α=0.65 (middle region)...")

    rho = np.array([0.65, 0.35])

    runs = []
    n_inits = 10
    np.random.seed(123)
    for i in range(n_inits):
        p_init = np.random.rand(2)
        hist = pomdp.optimize_projected_gradient(
            rho=rho,
            p_init=p_init,
            steps=300,
            lr=0.2,
            momentum=0.65,
            clip_grad=10.0,
        )
        runs.append(hist)

    pomdp.plot_optimization_dynamics(
        rho=rho,
        histories=runs,
        num_y_dirs=30,
        grid_res=250,
        show_value_space=True,
        show_J_curve=True,
        show_policy_space=True,
        show_iso_lines=True,
        num_iso_levels=18,
    )

    # Fine-grained sweep across the three regions
    print("\n3. Fine-grained sweep across all three optimal regions...")

    # Focus on the regions with transitions
    alphas = np.concatenate([
        np.linspace(0.0, 0.5, 5),      # Region 1
        np.linspace(0.52, 0.62, 8),    # Near boundary 1
        np.linspace(0.64, 0.72, 8),    # Region 2
        np.linspace(0.73, 0.80, 8),    # Near boundary 2
        np.linspace(0.82, 1.0, 5),     # Region 3
    ])

    results = pomdp.ablate_rhos(
        alphas=alphas,
        n_starts=15,
        steps=300,
        lr=0.2,
        momentum=0.65,
        clip_grad=10.0,
        seed=42,
        cluster_tol_v=0.01,
    )

    pomdp.plot_rho_ablation_endpoints(results)
    plt.savefig('very_noisy_rho_endpoints.png', dpi=300, bbox_inches='tight')
    print("   Saved: very_noisy_rho_endpoints.png")
    plt.close()

    pomdp.plot_rho_ablation_trajectories(results, thin=5)
    plt.savefig('very_noisy_rho_trajectories.png', dpi=300, bbox_inches='tight')
    print("   Saved: very_noisy_rho_trajectories.png")
    plt.close()

    pomdp.plot_rho_basin_shares(results)
    plt.savefig('very_noisy_basin_shares.png', dpi=300, bbox_inches='tight')
    print("   Saved: very_noisy_basin_shares.png")
    plt.close()

    # Analyze the clusters to identify the three regions
    print("\n4. Cluster analysis across α values:")
    print("-" * 70)
    print(f"{'α':>6} | {'#Clusters':>10} | Cluster info")
    print("-" * 70)

    for r in results:
        alpha = r["alpha"]
        n_clusters = len(r["clusters"])
        cluster_info = []
        for i, c in enumerate(r["clusters"]):
            v_str = f"v=[{c['center_v'][0]:.2f},{c['center_v'][1]:.2f}]"
            p_str = f"p=[{c['mean_p'][0]:.2f},{c['mean_p'][1]:.2f}]"
            cluster_info.append(f"C{i}({c['count']}/{len(r['finals_v'])}): {v_str}, {p_str}")

        if n_clusters <= 2:
            info_str = "; ".join(cluster_info)
        else:
            info_str = cluster_info[0] + "..."

        print(f"{alpha:>6.3f} | {n_clusters:>10} | {info_str}")

    print("-" * 70)
    print("\n" + "=" * 70)
    print("Very Noisy POMDP Analysis Complete!")
    print("=" * 70)
    print("\nKey findings:")
    print("  - Check the multi-panel plot to see how different α values")
    print("    lead to convergence to different local optima")
    print("  - The basin shares plot shows the transition between regions")
    print("  - Multiple local maxima exist due to the high observation noise")
