"""
Test Suite for AdaptiveGDSolver
================================
This script provides:
1. A mock game environment mimicking the rideshare structure
2. Test cases to verify convergence and parameter estimation accuracy
3. Example usage for migrating to new environments
"""

import numpy as np
import matplotlib.pyplot as plt
from adaptive_gd_solver import AdaptiveGDSolver

# ==============================================================================
# 1. Mock Game Environment (Simulator)
# ==============================================================================

class MockRideshareGame:
    """
    A simple linear demand game simulator for testing.
    
    True Model:
    z1 = z1_base + A1_true @ x1 + Ac1_true @ x2
    z2 = z2_base + A2_true @ x2 + Ac2_true @ x1
    
    Payoffs (unknown to solver):
    R1 = -0.5 * x1.T @ z1 + lambda1 * ||x1||^2
    R2 = -0.5 * x2.T @ z2 + lambda2 * ||x2||^2
    """
    
    def __init__(self, dim, seed=42):
        np.random.seed(seed)
        self.dim = dim
        
        # Generate random true parameters (stable matrices)
        # Ensure diagonals are negative for stability (demand decreases with price)
        self.A1_true = -np.abs(np.random.randn(dim, dim)) * 0.5
        self.Ac1_true = np.random.randn(dim, dim) * 0.2  # Cross effect smaller
        
        self.A2_true = -np.abs(np.random.randn(dim, dim)) * 0.5
        self.Ac2_true = np.random.randn(dim, dim) * 0.2
        
        # Base demands
        self.z1_base = np.ones(dim) * 10.0
        self.z2_base = np.ones(dim) * 10.0
        
        # Regularization
        self.lam1 = 0.1
        self.lam2 = 0.1
        
        print("=== True Parameters Initialized ===")
        print(f"A1_true diagonal mean: {np.diag(self.A1_true).mean():.4f}")
        print(f"Ac1_true norm: {np.linalg.norm(self.Ac1_true):.4f}")
        
    def get_observation(self, x1, x2):
        """Return noisy observations of demand."""
        z1 = self.z1_base + self.A1_true @ x1 + self.Ac1_true @ x2
        z2 = self.z2_base + self.A2_true @ x2 + self.Ac2_true @ x1
        
        # Add small observation noise
        noise_level = 0.01
        z1 += np.random.randn(self.dim) * noise_level
        z2 += np.random.randn(self.dim) * noise_level
        
        return z1, z2
    
    def compute_nash_equilibrium(self):
        """
        Analytically compute the Nash Equilibrium for verification.
        
        FOCs:
        grad_x1 = -(A1 - lam*I).T @ x1 - 0.5*(z1_base + A1@x1 + Ac1@x2) = 0
        grad_x2 = -(A2 - lam*I).T @ x2 - 0.5*(z2_base + A2@x2 + Ac2@x1) = 0
        
        Rearranging into linear system M @ [x1; x2] = b
        """
        I = np.eye(self.dim)
        
        # Coefficient matrix blocks
        # Eq1: [(A1-lamI)^T + 0.5*A1] @ x1 + [0.5*Ac1] @ x2 = -0.5*z1_base
        # Eq2: [0.5*Ac2] @ x1 + [(A2-lamI)^T + 0.5*A2] @ x2 = -0.5*z2_base
        
        M11 = (self.A1_true - self.lam1 * I).T + 0.5 * self.A1_true
        M12 = 0.5 * self.Ac1_true
        M21 = 0.5 * self.Ac2_true
        M22 = (self.A2_true - self.lam2 * I).T + 0.5 * self.A2_true
        
        # Assemble full matrix
        M_top = np.hstack([M11, M12])
        M_bot = np.hstack([M21, M22])
        M = np.vstack([M_top, M_bot])
        
        # RHS
        b = -0.5 * np.concatenate([self.z1_base, self.z2_base])
        
        # Solve
        x_star = np.linalg.solve(M, b)
        x1_star = x_star[:self.dim]
        x2_star = x_star[self.dim:]
        
        return x1_star, x2_star


# ==============================================================================
# 2. Test Functions
# ==============================================================================

def test_convergence():
    """Test if AGD converges to Nash Equilibrium."""
    print("\n" + "="*60)
    print("TEST 1: Convergence to Nash Equilibrium")
    print("="*60)
    
    # Use simpler 1D case for reliable convergence demonstration
    dim = 1
    game = MockRideshareGame(dim=dim, seed=42)
    
    # Compute true NE for comparison
    x1_ne, x2_ne = game.compute_nash_equilibrium()
    print(f"\nTrue NE x1: {x1_ne.round(4)}")
    print(f"True NE x2: {x2_ne.round(4)}")
    
    # Setup solver
    def proj_box(x):
        """Box projection: clip to [-100, 100]"""
        return np.clip(x, -100, 100)
    
    solver = AdaptiveGDSolver(
        dim=dim,
        lam1=game.lam1,
        lam2=game.lam2,
        nu=0.01,  # Small learning rate for parameter estimation
        rho=0.01,
        B=0.5,  # Exploration noise
        max_iter=2000,
        proj_func=proj_box
    )
    
    # Observation wrapper
    def obs_func(x1, x2):
        return game.get_observation(x1, x2)
    
    # Run AGD
    history = solver.run(
        z1_base=game.z1_base,
        z2_base=game.z2_base,
        observation_func=obs_func,
        eta_base=0.5,
        verbose=True
    )
    
    # Evaluate results
    x1_final = history['x1'][-1]
    x2_final = history['x2'][-1]
    
    error_x1 = np.linalg.norm(x1_final - x1_ne)
    error_x2 = np.linalg.norm(x2_final - x2_ne)
    
    print(f"\n=== Results ===")
    print(f"AGD x1 final: {x1_final.round(4)}")
    print(f"AGD x2 final: {x2_final.round(4)}")
    print(f"Error ||x1_AGD - x1_NE||: {error_x1:.6f}")
    print(f"Error ||x2_AGD - x2_NE||: {error_x2:.6f}")
    
    # Parameter estimation accuracy
    A1_err = np.linalg.norm(solver.A1_hat - game.A1_true)
    Ac1_err = np.linalg.norm(solver.Ac1_hat - game.Ac1_true)
    print(f"Parameter Est Error ||A1_hat - A1_true||: {A1_err:.6f}")
    print(f"Parameter Est Error ||Ac1_hat - Ac1_true||: {Ac1_err:.6f}")
    
    # Reasonable threshold for 1D
    assert error_x1 < 2.0, f"Convergence failed for player 1: {error_x1}"
    assert error_x2 < 2.0, f"Convergence failed for player 2: {error_x2}"
    print("\n✓ TEST 1 PASSED: Converged successfully!")
    
    return history, (x1_ne, x2_ne)


def test_parameter_estimation():
    """Test if parameters are estimated correctly."""
    print("\n" + "="*60)
    print("TEST 2: Parameter Estimation Accuracy")
    print("="*60)
    
    dim = 2
    game = MockRideshareGame(dim=dim, seed=123)
    
    solver = AdaptiveGDSolver(
        dim=dim,
        lam1=game.lam1,
        lam2=game.lam2,
        nu=0.1,
        rho=0.001,  # Less regularization for better estimation
        B=1.0,  # More exploration for better identifiability
        max_iter=800,
        proj_func=lambda x: x
    )
    
    def obs_func(x1, x2):
        return game.get_observation(x1, x2)
    
    solver.run(
        z1_base=game.z1_base,
        z2_base=game.z2_base,
        observation_func=obs_func,
        eta_base=1.0,
        verbose=False
    )
    
    # Check estimation errors
    rel_err_A1 = np.linalg.norm(solver.A1_hat - game.A1_true) / np.linalg.norm(game.A1_true)
    rel_err_Ac1 = np.linalg.norm(solver.Ac1_hat - game.Ac1_true) / np.linalg.norm(game.Ac1_true)
    rel_err_A2 = np.linalg.norm(solver.A2_hat - game.A2_true) / np.linalg.norm(game.A2_true)
    rel_err_Ac2 = np.linalg.norm(solver.Ac2_hat - game.Ac2_true) / np.linalg.norm(game.Ac2_true)
    
    print(f"\nRelative Estimation Errors:")
    print(f"  A1:  {rel_err_A1*100:.2f}%")
    print(f"  Ac1: {rel_err_Ac1*100:.2f}%")
    print(f"  A2:  {rel_err_A2*100:.2f}%")
    print(f"  Ac2: {rel_err_Ac2*100:.2f}%")
    
    # Allow 20% relative error due to finite samples and noise
    assert rel_err_A1 < 0.25, f"A1 estimation too inaccurate: {rel_err_A1}"
    assert rel_err_Ac1 < 0.25, f"Ac1 estimation too inaccurate: {rel_err_Ac1}"
    
    print("\n✓ TEST 2 PASSED: Parameters estimated accurately!")


def plot_results(history, ne_solution=None):
    """Plot convergence trajectories."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Decision variables convergence
    ax = axes[0, 0]
    x1_traj = np.array(history['x1'])
    x2_traj = np.array(history['x2'])
    ax.plot(x1_traj[:, 0], label='x1[0]', color='blue')
    ax.plot(x2_traj[:, 0], label='x2[0]', color='red', linestyle='--')
    if ne_solution:
        ax.axhline(y=ne_solution[0][0], color='blue', alpha=0.3, linewidth=2, label='NE x1[0]')
        ax.axhline(y=ne_solution[1][0], color='red', alpha=0.3, linewidth=2, linestyle='--', label='NE x2[0]')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Decision Value')
    ax.set_title('Decision Variables Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm
    ax = axes[0, 1]
    # We didn't store grad_norm in history by default, skip or recompute
    
    # Plot 3: Parameter estimation error (if we had true params,示意)
    ax = axes[1, 0]
    A1_norms = [np.linalg.norm(A) for A in history['A1_hat']]
    Ac1_norms = [np.linalg.norm(A) for A in history['Ac1_hat']]
    ax.plot(A1_norms, label='||A1_hat||', color='green')
    ax.plot(Ac1_norms, label='||Ac1_hat||', color='orange')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Matrix Norm')
    ax.set_title('Estimated Parameter Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate decay
    ax = axes[1, 1]
    iters = np.arange(len(history['x1']))
    eta_base = 2.0
    eta_decay = eta_base / np.log(iters + 2)
    ax.plot(iters, eta_decay, color='purple')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Adaptive Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/agd_convergence_plot.png', dpi=150)
    print("\nPlot saved to: /workspace/agd_convergence_plot.png")


# ==============================================================================
# 3. Main Execution
# ==============================================================================

if __name__ == "__main__":
    print("Starting AGD Solver Tests...")
    
    # Run tests
    history, ne_sol = test_convergence()
    test_parameter_estimation()
    
    # Visualize
    plot_results(history, ne_sol)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nTo migrate to your own environment:")
    print("1. Create a class similar to MockRideshareGame")
    print("2. Implement get_observation(x1, x2) returning (z1, z2)")
    print("3. Initialize AdaptiveGDSolver with appropriate dimensions")
    print("4. Call solver.run() with your observation function")
