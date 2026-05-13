#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

class POMDPAnalyzer:
    def __init__(self, P, R, O, gamma=0.9):
        self.P = P
        self.R = R
        self.beta = O
        self.gamma = gamma
        
        self.n_states = P.shape[1]
        self.n_obs = O.shape[1]
        
        # Verify shapes
        assert P.shape[0] == 2, "Solver handles binary actions (P0, P1)."
        assert O.shape[0] == self.n_states
        
        self._setup_affine_system()
        
    def _setup_affine_system(self):
        """Precompute matrices for A(p)x = b(p)."""
        I = np.eye(self.n_states)
        P0, r0 = self.P[0], self.R[0]
        DeltaP = self.P[1] - self.P[0]
        dr = self.R[1] - self.R[0]
        
        self.A_base = I - self.gamma * P0
        self.b_base = r0
        
        self.A_ks = []
        self.b_ks = []
        
        for k in range(self.n_obs):
            beta_k = self.beta[:, k]
            self.A_ks.append(-self.gamma * np.diag(beta_k) @ DeltaP)
            self.b_ks.append(np.diag(beta_k) @ dr)

        # Centering and Scaling for [-1, 1] parameterization
        self.pc = np.full(self.n_obs, 0.5)
        self.pD = np.full(self.n_obs, 0.5)
        
        self.Ac = self._compute_A(self.pc)
        self.bc = self._compute_b(self.pc)

    def _compute_A(self, p):
        res = self.A_base.copy()
        for k, val in enumerate(p):
            res += val * self.A_ks[k]
        return res

    def _compute_b(self, p):
        res = self.b_base.copy()
        for k, val in enumerate(p):
            res += val * self.b_ks[k]
        return res

    def solve_v(self, p):
        return np.linalg.solve(self._compute_A(p), self._compute_b(p))

    def _get_q_of_x(self, x):
        """Exact test: q = D(x)^-1 * R^c(x)."""
        Rc = self.Ac @ x - self.bc
        cols = [self.pD[k] * (self.A_ks[k] @ x - self.b_ks[k]) for k in range(self.n_obs)]
        D = np.column_stack(cols)
        
        if D.shape[0] != D.shape[1]: 
            return np.linalg.pinv(D) @ Rc # Pseudo-inverse for non-square
        if abs(np.linalg.det(D)) < 1e-10: 
            return None
        return np.linalg.solve(D, Rc)

    def plot_with_inequalities(self, num_y_dirs=12, grid_res=300):
        """
        Plots the solution set with:
        1. Blue lines: Boundaries derived from sampling discrete directions y (Support inequalities).
        2. Red/Green dashed: Exact boundaries derived from the q-inverse test.
        """
        print(f"Generating inequalities for {num_y_dirs} directions...")
        
        # 1. Estimate Bounds via random sampling
        p_samples = np.random.rand(100, self.n_obs)
        vals = np.array([self.solve_v(p) for p in p_samples])
        v0, v1 = vals[:, 0], vals[:, 1]
        
        pad = 0.8 * (v0.max() - v0.min())
        x_min, x_max = v0.min() - pad, v0.max() + pad
        y_min, y_max = v1.min() - pad, v1.max() + pad
        
        X = np.linspace(x_min, x_max, grid_res)
        Y = np.linspace(y_min, y_max, grid_res)
        XX, YY = np.meshgrid(X, Y)
        
        # Flatten grid for vectorized checks
        # shape (2, N*N)
        X_flat = np.vstack([XX.ravel(), YY.ravel()])
        if self.n_states > 2:
            # Pad with zeros if more than 2 states (just for visualization slice)
            X_flat = np.vstack([X_flat, np.zeros((self.n_states - 2, X_flat.shape[1]))])

        # --- EXACT BOUNDARY (Nonlinear/q-test) ---
        # Calculate q for every point
        Q0 = np.full(XX.shape, np.nan)
        Q1 = np.full(XX.shape, np.nan)
        FEAS = np.zeros_like(XX, dtype=float)
        
        # We iterate for exact calculation (harder to vectorise generic solve)
        for i in range(grid_res):
            for j in range(grid_res):
                x_vec = X_flat[:, i*grid_res + j]
                q = self._get_q_of_x(x_vec)
                if q is not None:
                    if len(q) >= 1: Q0[i, j] = q[0]
                    if len(q) >= 2: Q1[i, j] = q[1]
                    FEAS[i,j] = 1.0 if np.all(np.abs(q) <= 1 + 1e-10) else 0.0

        # --- LINEAR INEQUALITIES (Support Function Test) ---
        # Generate y directions (unit circle)
        thetas = np.linspace(0, 2*np.pi, num_y_dirs, endpoint=False)
        Y_dirs = np.vstack([np.cos(thetas), np.sin(thetas)])
        if self.n_states > 2:
             Y_dirs = np.vstack([Y_dirs, np.zeros((self.n_states - 2, num_y_dirs))])
        
        # Precompute terms for vectorized inequality check
        # Rc(x) = Ac*x - bc
        # u_k(x) = A_k*x - b_k
        
        # Ac_X_bc: (2, N_points)
        Ac_X_bc = self.Ac @ X_flat - self.bc[:, None]
        
        # Uk_X_bk: List of (2, N_points)
        Uk_X_bk = [(self.A_ks[k] @ X_flat - self.b_ks[k][:, None]) for k in range(self.n_obs)]
        
        plt.figure(figsize=(10, 8))
        
        # Plot each Y direction inequality
        # Condition: y.T @ Rc <= sum( pD * |y.T @ Uk| )
        # Margin = RHS - LHS >= 0
        
        for d in range(num_y_dirs):
            y = Y_dirs[:, d] # (2,)
            
            # LHS: y^T (Ac x - bc) -> (N_points,)
            lhs = y @ Ac_X_bc
            
            # RHS: sum p_delta * |y^T (Ak x - bk)|
            rhs = np.zeros_like(lhs)
            for k in range(self.n_obs):
                term = y @ Uk_X_bk[k]
                rhs += self.pD[k] * np.abs(term)
            
            margin = rhs - lhs
            Margin_Grid = margin.reshape(grid_res, grid_res)
            
            # Plot the boundary (Margin = 0) for this direction
            plt.contour(XX, YY, Margin_Grid, levels=[0], colors='cornflowerblue', linewidths=1, alpha=0.6)

        # Plot Exact Boundaries
        plt.contour(XX, YY, Q0, levels=[-1, 1], colors='red', linewidths=2.5, linestyles='--')
        plt.contour(XX, YY, Q1, levels=[-1, 1], colors='green', linewidths=2.5, linestyles='--')
        # feasible region shading (via inequality test)
        plt.contourf(XX, YY, FEAS, levels=[-0.1, 0.5, 1.1], alpha=0.18)
        
        # Formatting
        plt.plot([], [], color='cornflowerblue', linewidth=1, label=f'Linear Inequalities')
        plt.plot([], [], color='red', linestyle='--', linewidth=2, label='Exact Boundary q0=±1')
        plt.plot([], [], color='green', linestyle='--', linewidth=2, label='Exact Boundary q1=±1')
        plt.xlabel("v(s=0)")
        plt.ylabel("v(s=1)")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    # ---------------------------
    # Objective + gradient
    # ---------------------------
    def objective(self, p, rho):
        """J(p)= <rho, v(p)>"""
        v = self.solve_v(p)
        return float(np.dot(rho, v))

    def grad_objective(self, p, rho):
        """
        Exact gradient of J(p)=rho^T v(p) where A(p)v=b(p).
        dv/dp_k = A(p)^{-1} (b_k - A_k v)
        dJ/dp_k = rho^T dv/dp_k
        """
        p = np.asarray(p, dtype=float)
        A = self._compute_A(p)
        v = np.linalg.solve(A, self._compute_b(p))

        grad = np.zeros(self.n_obs, dtype=float)
        # Solve A w = (b_k - A_k v) for each k
        for k in range(self.n_obs):
            rhs = self.b_ks[k] - (self.A_ks[k] @ v)
            w = np.linalg.solve(A, rhs)
            grad[k] = float(np.dot(rho, w))
        return grad, v

    # ---------------------------
    # Optimization dynamics
    # ---------------------------
    def optimize_projected_gradient(
        self,
        rho,
        p_init,
        steps=200,
        lr=0.2,
        momentum=0.0,
        clip_grad=None,
        seed=None,
    ):
        """
        Projected gradient ascent on p in [0,1]^n_obs.
        Returns history dict with p, v, J, grad.
        """
        rng = np.random.default_rng(seed)
        p = np.clip(np.array(p_init, dtype=float), 0.0, 1.0)

        v_mom = np.zeros_like(p)
        hist = {"p": [], "v": [], "J": [], "grad": []}

        for t in range(steps):
            g, v = self.grad_objective(p, rho)

            # Record consistent (p, v, J) at current p before updating.
            hist["p"].append(p.copy())
            hist["v"].append(v.copy())
            hist["J"].append(float(np.dot(rho, v)))
            hist["grad"].append(g.copy())

            if clip_grad is not None:
                gn = np.linalg.norm(g)
                if gn > clip_grad:
                    g = g * (clip_grad / (gn + 1e-12))

            # momentum update (optional)
            v_mom = momentum * v_mom + g
            p = p + lr * v_mom

            # project back to box constraints
            p = np.clip(p, 0.0, 1.0)

        # convert to arrays
        hist["p"] = np.array(hist["p"])
        hist["v"] = np.array(hist["v"])
        hist["J"] = np.array(hist["J"])
        hist["grad"] = np.array(hist["grad"])
        return hist

    # ---------------------------
    # Visualization helpers
    # ---------------------------
    def plot_optimization_dynamics(
        self,
        rho,
        histories,
        num_y_dirs=25,
        grid_res=250,
        show_value_space=True,
        show_J_curve=True,
        show_policy_space=True,
        show_iso_lines=True,
        num_iso_levels=15,
    ):
        """
        histories: list of hist dicts returned by optimize_projected_gradient
        Produces up to 3 figures:
          1) value-space feasible region + trajectories (v0,v1) + iso-objective lines
          2) J(t) curves
          3) policy-space J(p0,p1) contour + trajectories (only if n_obs==2)
        """
        # 1) VALUE SPACE: reuse your feasible/boundary plot background,
        #    then overlay trajectories.
        if show_value_space:
            # --- build background (adapted from your plot_with_inequalities) ---
            p_samples = np.random.rand(200, self.n_obs)
            vals = np.array([self.solve_v(p) for p in p_samples])
            v0, v1 = vals[:, 0], vals[:, 1]
            pad = 0.8 * (v0.max() - v0.min())
            x_min, x_max = v0.min() - pad, v0.max() + pad
            y_min, y_max = v1.min() - pad, v1.max() + pad

            X = np.linspace(x_min, x_max, grid_res)
            Y = np.linspace(y_min, y_max, grid_res)
            XX, YY = np.meshgrid(X, Y)

            X_flat = np.vstack([XX.ravel(), YY.ravel()])
            if self.n_states > 2:
                X_flat = np.vstack([X_flat, np.zeros((self.n_states - 2, X_flat.shape[1]))])

            Q0 = np.full(XX.shape, np.nan)
            Q1 = np.full(XX.shape, np.nan)
            FEAS = np.zeros_like(XX, dtype=float)

            for i in range(grid_res):
                for j in range(grid_res):
                    x_vec = X_flat[:, i*grid_res + j]
                    q = self._get_q_of_x(x_vec)
                    if q is not None:
                        if len(q) >= 1: Q0[i, j] = q[0]
                        if len(q) >= 2: Q1[i, j] = q[1]
                        FEAS[i, j] = 1.0 if np.all(np.abs(q) <= 1 + 1e-10) else 0.0

            thetas = np.linspace(0, 2*np.pi, num_y_dirs, endpoint=False)
            Y_dirs = np.vstack([np.cos(thetas), np.sin(thetas)])
            if self.n_states > 2:
                Y_dirs = np.vstack([Y_dirs, np.zeros((self.n_states - 2, num_y_dirs))])

            Ac_X_bc = self.Ac @ X_flat - self.bc[:, None]
            Uk_X_bk = [(self.A_ks[k] @ X_flat - self.b_ks[k][:, None]) for k in range(self.n_obs)]

            plt.figure(figsize=(10, 8))
            for d in range(num_y_dirs):
                y = Y_dirs[:, d]
                lhs = y @ Ac_X_bc
                rhs = np.zeros_like(lhs)
                for k in range(self.n_obs):
                    term = y @ Uk_X_bk[k]
                    rhs += self.pD[k] * np.abs(term)
                margin = rhs - lhs
                Margin_Grid = margin.reshape(grid_res, grid_res)
                plt.contour(XX, YY, Margin_Grid, levels=[0], linewidths=1, alpha=0.35)

            plt.contour(XX, YY, Q0, levels=[-1, 1], linewidths=2.0, linestyles='--', colors='red', alpha=0.6)
            plt.contour(XX, YY, Q1, levels=[-1, 1], linewidths=2.0, linestyles='--', colors='green', alpha=0.6)
            plt.contourf(XX, YY, FEAS, levels=[-0.1, 0.5, 1.1], alpha=0.15, colors=['white', 'lightblue'])

            # --- ISO-OBJECTIVE LINES: J(v) = rho^T v ---
            if show_iso_lines:
                # Compute objective value at each grid point
                J_vals = XX * rho[0] + YY * rho[1]  # rho^T v for v=(XX, YY)

                # Determine iso-level values based on trajectory data
                all_J_vals = []
                for hist in histories:
                    all_J_vals.extend(hist["J"])
                J_min, J_max = min(all_J_vals), max(all_J_vals)
                J_range = J_max - J_min
                iso_levels = np.linspace(J_min - 0.1*J_range, J_max + 0.1*J_range, num_iso_levels)

                # Plot iso-objective contours
                cs = plt.contour(XX, YY, J_vals, levels=iso_levels, colors='purple',
                                linewidths=1.2, alpha=0.4, linestyles='-')
                plt.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

            # overlay trajectories
            for idx, hist in enumerate(histories):
                V = hist["v"]
                plt.plot(V[:, 0], V[:, 1], linewidth=2, alpha=0.9, label=f"run {idx}")
                plt.scatter([V[0, 0]], [V[0, 1]], marker='o', s=50)
                plt.scatter([V[-1, 0]], [V[-1, 1]], marker='X', s=70)

            plt.xlabel("v(s=0)")
            plt.ylabel("v(s=1)")
            plt.title(r"Optimization trajectories in value space for $J(p)=\langle \rho, v(p)\rangle$")
            plt.grid(True, alpha=0.2)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig('optimization_value_space.png', dpi=300, bbox_inches='tight')
            print("   Saved: optimization_value_space.png")
            plt.show()

        # 2) J(t)
        if show_J_curve:
            plt.figure(figsize=(9, 4))
            for idx, hist in enumerate(histories):
                plt.plot(hist["J"], linewidth=2, label=f"run {idx}")
            plt.xlabel("iteration")
            plt.ylabel(r"$J=\langle \rho, v\rangle$")
            plt.title("Objective improvement over iterations")
            plt.grid(True, alpha=0.2)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig('optimization_J_curve.png', dpi=300, bbox_inches='tight')
            print("   Saved: optimization_J_curve.png")
            plt.show()

        # 3) Policy space contour (only for n_obs==2)
        if show_policy_space and self.n_obs == 2:
            n = 151
            p0 = np.linspace(0, 1, n)
            p1 = np.linspace(0, 1, n)
            PP0, PP1 = np.meshgrid(p0, p1)
            JJ = np.zeros_like(PP0)

            for i in range(n):
                for j in range(n):
                    p = np.array([PP0[i, j], PP1[i, j]])
                    JJ[i, j] = self.objective(p, rho)

            plt.figure(figsize=(7.5, 6.2))
            cs = plt.contourf(PP0, PP1, JJ, levels=30, alpha=0.9)
            plt.colorbar(cs, label=r"$J(p)$")
            for idx, hist in enumerate(histories):
                P = hist["p"]
                plt.plot(P[:, 0], P[:, 1], linewidth=2, alpha=0.9, label=f"run {idx}")
                plt.scatter([P[0, 0]], [P[0, 1]], marker='o', s=50)
                plt.scatter([P[-1, 0]], [P[-1, 1]], marker='X', s=70)

            plt.xlabel(r"$p_0=\pi(a{=}1\mid o{=}0)$")
            plt.ylabel(r"$p_1=\pi(a{=}1\mid o{=}1)$")
            plt.title("Policy-parameter space with objective contours")
            plt.grid(True, alpha=0.2)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig('optimization_policy_space.png', dpi=300, bbox_inches='tight')
            print("   Saved: optimization_policy_space.png")
            plt.show()
        
    def ablate_rhos(
        self,
        alphas,
        n_starts=8,
        steps=250,
        lr=0.25,
        momentum=0.6,
        clip_grad=10.0,
        seed=0,
        cluster_tol_v=1e-3,
    ):
        """
        Sweep rho=(alpha,1-alpha) across alphas.
        For each alpha, run n_starts random initial p and store histories.
        Also cluster final endpoints in value-space (roughly identifies distinct maxima).
        """
        rng = np.random.default_rng(seed)
        results = []  # one entry per alpha

        for alpha in alphas:
            rho = np.array([alpha, 1.0 - alpha], dtype=float)
            histories = []
            finals_v = []
            finals_p = []
            finals_J = []

            for _ in range(n_starts):
                p_init = rng.random(self.n_obs)  # uniform in [0,1]^n_obs
                hist = self.optimize_projected_gradient(
                    rho=rho,
                    p_init=p_init,
                    steps=steps,
                    lr=lr,
                    momentum=momentum,
                    clip_grad=clip_grad,
                )
                histories.append(hist)
                finals_v.append(hist["v"][-1].copy())
                finals_p.append(hist["p"][-1].copy())
                finals_J.append(hist["J"][-1])

            finals_v = np.array(finals_v)
            finals_p = np.array(finals_p)
            finals_J = np.array(finals_J)

            # --- cluster final endpoints by value-space proximity ---
            # (works well for 2 states; extend to more dims if you like)
            clusters = []  # list of dicts: {"center":..., "members":[idx,...]}
            for i, v_end in enumerate(finals_v):
                placed = False
                for c in clusters:
                    if np.linalg.norm(v_end[:2] - c["center"][:2]) <= cluster_tol_v:
                        c["members"].append(i)
                        # update center as mean of members
                        mem = np.array([finals_v[j] for j in c["members"]])
                        c["center"] = mem.mean(axis=0)
                        placed = True
                        break
                if not placed:
                    clusters.append({"center": v_end.copy(), "members": [i]})

            # add cluster stats
            cluster_stats = []
            for c in clusters:
                mem = c["members"]
                cluster_stats.append({
                    "center_v": c["center"],
                    "count": len(mem),
                    "mean_p": finals_p[mem].mean(axis=0),
                    "mean_J": finals_J[mem].mean(),
                })

            results.append({
                "alpha": float(alpha),
                "rho": rho,
                "histories": histories,
                "finals_v": finals_v,
                "finals_p": finals_p,
                "finals_J": finals_J,
                "clusters": cluster_stats,
            })

        return results
    
    def plot_rho_ablation_endpoints(self, results):
        # Gather endpoints
        alphas = np.array([r["alpha"] for r in results])
        all_v = np.vstack([r["finals_v"] for r in results])
        all_p = np.vstack([r["finals_p"] for r in results])
        all_alpha_rep = np.concatenate([
            np.full(len(r["finals_v"]), r["alpha"]) for r in results
        ])

        # 1) endpoints in value space (v0,v1) colored by alpha
        plt.figure(figsize=(7.5, 6.5))
        sc = plt.scatter(all_v[:, 0], all_v[:, 1], c=all_alpha_rep, s=35, alpha=0.85)
        plt.colorbar(sc, label=r"$\alpha$ in $\rho=(\alpha,1-\alpha)$")
        plt.xlabel("v(s=0)")
        plt.ylabel("v(s=1)")
        plt.title("Final endpoints in value space across rho sweep")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

        # 2) endpoints in policy space vs alpha
        plt.figure(figsize=(8.5, 4.5))
        # scatter each endpoint; x=alpha
        plt.scatter(all_alpha_rep, all_p[:, 0], s=25, alpha=0.7, label="p0")
        if self.n_obs > 1:
            plt.scatter(all_alpha_rep, all_p[:, 1], s=25, alpha=0.7, label="p1")
        plt.xlabel(r"$\alpha$ (rho=(alpha,1-alpha))")
        plt.ylabel("final p_k")
        plt.title("Final policy parameters vs rho")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_rho_ablation_trajectories(self, results, thin=3):
        """
        Overlay value-space trajectories for each alpha.
        'thin' plots every thin-th iterate to reduce clutter.
        """
        plt.figure(figsize=(9, 7))
        for r in results:
            alpha = r["alpha"]
            for hist in r["histories"]:
                V = hist["v"][::thin]
                plt.plot(V[:, 0], V[:, 1], alpha=0.25)
                plt.scatter([V[-1, 0]], [V[-1, 1]], s=20, alpha=0.6)

        plt.xlabel("v(s=0)")
        plt.ylabel("v(s=1)")
        plt.title("Value-space trajectories across rho sweep (endpoints emphasized)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    def plot_rho_basin_shares(self, results):
        """
        For each alpha, plot the fraction of starts that fall into each detected cluster.
        Note: cluster labels are local to each alpha; this is most useful as 'how many basins exist'
        + rough stability. For globally consistent labeling, we can match clusters across alphas.
        """
        alphas = [r["alpha"] for r in results]
        maxK = max(len(r["clusters"]) for r in results)

        shares = np.zeros((len(results), maxK))
        for i, r in enumerate(results):
            total = len(r["finals_v"])
            for k, c in enumerate(r["clusters"]):
                shares[i, k] = c["count"] / total

        plt.figure(figsize=(8.5, 4.5))
        for k in range(maxK):
            plt.plot(alphas, shares[:, k], marker="o", linewidth=2, label=f"cluster {k}")
        plt.xlabel(r"$\alpha$")
        plt.ylabel("share of random starts")
        plt.title("Basin shares across rho sweep (per-alpha clustering)")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_multistart_comparison(
        self,
        rhos,
        n_starts=5,
        steps=200,
        lr=0.2,
        momentum=0.6,
        clip_grad=10.0,
        num_y_dirs=20,
        grid_res=200,
        num_iso_levels=12,
        seed=None,
    ):
        """
        Creates a multi-panel figure showing how different initial distributions (rho values)
        lead to different optimization trajectories and local maxima.

        Each panel shows:
        - Feasible value function region (shaded)
        - Iso-objective lines for J(v) = rho^T v
        - Multiple trajectories from random initializations
        - Final convergence points

        Args:
            rhos: list of initial state distributions, e.g., [(0.1,0.9), (0.5,0.5), (0.9,0.1)]
            n_starts: number of random initializations per rho
            Other args: optimization and plotting parameters
        """
        rng = np.random.default_rng(seed)
        n_rhos = len(rhos)

        # Compute feasible region once
        p_samples = np.random.rand(200, self.n_obs)
        vals = np.array([self.solve_v(p) for p in p_samples])
        v0, v1 = vals[:, 0], vals[:, 1]
        pad = 0
        x_min, x_max = 3, 8
        y_min, y_max = 3, 8

        X = np.linspace(x_min, x_max, grid_res)
        Y = np.linspace(y_min, y_max, grid_res)
        XX, YY = np.meshgrid(X, Y)

        X_flat = np.vstack([XX.ravel(), YY.ravel()])
        if self.n_states > 2:
            X_flat = np.vstack([X_flat, np.zeros((self.n_states - 2, X_flat.shape[1]))])

        # Compute feasibility mask
        FEAS = np.zeros_like(XX, dtype=float)
        for i in range(grid_res):
            for j in range(grid_res):
                x_vec = X_flat[:, i*grid_res + j]
                q = self._get_q_of_x(x_vec)
                if q is not None:
                    FEAS[i, j] = 1.0 if np.all(np.abs(q) <= 1 + 1e-10) else 0.0

        # Create subplots
        ncols = min(3, n_rhos)
        nrows = (n_rhos + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6*nrows))
        if n_rhos == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, n_starts))

        for idx, rho in enumerate(rhos):
            ax = axes[idx]
            rho_arr = np.array(rho, dtype=float)

            # Plot feasible region
            ax.contourf(XX, YY, FEAS, levels=[-0.1, 0.5, 1.1], alpha=0.2, colors=['white', 'lightblue'])
            ax.contour(XX, YY, FEAS, levels=[0.5], colors='black', linewidths=1.5, alpha=0.4)

            # Plot iso-objective lines: J(v) = rho^T v = rho[0]*v0 + rho[1]*v1
            J_vals = XX * rho_arr[0] + YY * rho_arr[1]
            iso_levels = np.linspace(J_vals.min(), J_vals.max(), num_iso_levels)
            cs = ax.contour(XX, YY, J_vals, levels=iso_levels, colors='gray',
                           linewidths=0.8, alpha=0.5, linestyles='-')
            ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')

            # Run optimization from multiple random starts
            for start_idx in range(n_starts):
                p_init = rng.random(self.n_obs)
                hist = self.optimize_projected_gradient(
                    rho=rho_arr,
                    p_init=p_init,
                    steps=steps,
                    lr=lr,
                    momentum=momentum,
                    clip_grad=clip_grad,
                )

                V = hist["v"]
                ax.plot(V[:, 0], V[:, 1], linewidth=2, alpha=0.7, color=colors[start_idx],
                       label=f"start {start_idx+1}" if idx == 0 else "")
                ax.scatter([V[0, 0]], [V[0, 1]], marker='o', s=60, color=colors[start_idx],
                          edgecolors='black', linewidths=1, zorder=5)
                ax.scatter([V[-1, 0]], [V[-1, 1]], marker='*', s=150, color=colors[start_idx],
                          edgecolors='black', linewidths=1, zorder=5)

            # Arrow showing gradient direction (rho direction)
            # The gradient of J(v) = rho^T v is just rho
            v_center = np.array([(x_min + x_max)/2, (y_min + y_max)/2])
            arrow_scale = 0.15 * (x_max - x_min)
            ax.arrow(v_center[0], v_center[1],
                    arrow_scale * rho_arr[0], arrow_scale * rho_arr[1],
                    head_width=0.3, head_length=0.2, fc='red', ec='red',
                    alpha=0.6, linewidth=2, zorder=10)
            ax.text(v_center[0] + arrow_scale * rho_arr[0] * 1.3,
                   v_center[1] + arrow_scale * rho_arr[1] * 1.3,
                   r'$\nabla J = \rho$', fontsize=11, color='red', fontweight='bold')

            ax.set_xlabel(r"$v(s=0)$", fontsize=12)
            ax.set_ylabel(r"$v(s=1)$", fontsize=12)
            ax.set_title(r"$\rho = ({:.2f}, {:.2f})$".format(rho_arr[0], rho_arr[1]), fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

        # Hide extra subplots
        for idx in range(n_rhos, len(axes)):
            axes[idx].axis('off')

        # Add legend to first subplot
        #if n_rhos > 0:
        #    axes[0].legend(loc='best', fontsize=9)

        #plt.suptitle("Optimization Dynamics: How Initial Distribution Affects Convergence",fontsize=15, fontweight='bold', y=1.0)
        plt.tight_layout()

        return fig


# --- Run ---
if __name__ == "__main__":
    P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
    PDelta = np.array([[-0.20,  0.20],
                       [-0.10,  0.10]])
    P1 = P0 + PDelta
    P = np.array([P0, P1])

    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    Beta = np.array([[0.80, 0.20], [0.30, 0.70]])

    pomdp = POMDPAnalyzer(P, R, Beta, gamma=0.9)

    print("=" * 60)
    print("POMDP Optimization Dynamics Visualization")
    print("=" * 60)

    # ===== MAIN EXPERIMENT: Multi-panel comparison across different rho values =====
    print("\n1. Creating multi-start comparison across different initial distributions...")

    # Test different initial state distributions
    rhos_to_test = [
        (0.1, 0.9),   # Heavily favor state 1
        (0.3, 0.7),   # Moderately favor state 1
        (0.5, 0.5),   # Balanced
        (0.7, 0.3),   # Moderately favor state 0
        (0.9, 0.1),   # Heavily favor state 0
    ]

    fig = pomdp.plot_multistart_comparison(
        rhos=rhos_to_test,
        n_starts=6,
        steps=200,
        lr=0.25,
        momentum=0.6,
        clip_grad=10.0,
        num_y_dirs=20,
        grid_res=180,
        num_iso_levels=12,
        seed=42,
    )
    fig.savefig('optimization_dynamics_multistart.png', dpi=300, bbox_inches='tight')
    print("   Saved: optimization_dynamics_multistart.png")

    # ===== ALTERNATIVE: Single rho with detailed analysis =====
    print("\n2. Detailed single-rho analysis with iso-lines...")

    # Choose a rho (initial belief over states)
    rho = np.array([0.3, 0.7])  # try changing this; it shifts the "preferred" direction in value space

    # Multiple random starts to show different basins / local maxima behavior
    runs = []
    inits = [
        np.array([0.05, 0.95]),
        np.array([0.95, 0.05]),
        np.array([0.5, 0.5]),
        np.random.rand(2),
        np.random.rand(2),
    ]
    for p_init in inits:
        hist = pomdp.optimize_projected_gradient(
            rho=rho,
            p_init=p_init,
            steps=200,
            lr=0.25,
            momentum=0.6,
            clip_grad=10.0,
        )
        runs.append(hist)

    pomdp.plot_optimization_dynamics(
        rho=rho,
        histories=runs,
        num_y_dirs=25,
        grid_res=220,
        show_value_space=True,
        show_J_curve=True,
        show_policy_space=True,
        show_iso_lines=True,
        num_iso_levels=15,
    )

    # ===== RHO SWEEP ANALYSIS =====
    print("\n3. Running rho sweep to analyze basin structure...")

    alphas = np.linspace(0.0, 1.0, 11)   # rho sweep
    results = pomdp.ablate_rhos(
        alphas=alphas,
        n_starts=12,
        steps=250,
        lr=0.25,
        momentum=0.6,
        clip_grad=10.0,
        seed=1,
        cluster_tol_v=5e-3,
    )

    pomdp.plot_rho_ablation_endpoints(results)
    pomdp.plot_rho_ablation_trajectories(results, thin=4)
    pomdp.plot_rho_basin_shares(results)

    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)