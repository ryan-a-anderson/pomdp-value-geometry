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

# --- Run ---
if __name__ == "__main__":
    # Standard Example
    P0 = np.array([[0.85, 0.15], [0.25, 0.75]])
    PDelta = np.array([[-0.20,  0.20],
                       [-0.10,  0.10]])  # rank 1 (rows proportional)
    P1 = P0 + PDelta
    P = np.array([P0, P1])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    #Beta = np.array([[0.80, 0.20], [0.30, 0.70]])
    Beta = np.eye(2)

    pomdp = POMDPAnalyzer(P, R, Beta)
    
    # Try a good number of directions to see the "cuts" clearly
    pomdp.plot_with_inequalities(num_y_dirs=25)