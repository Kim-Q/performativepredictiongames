"""
Adaptive Gradient Descent (AGD) Solver for Performative Prediction Games
=========================================================================
This module extracts the core logic of runAGD() from utilsrm.py and encapsulates it
into a reusable class suitable for migration to other game environments.

Key Features:
- Online parameter estimation via Ridge Regression
- Exploratory noise injection for persistent excitation
- Adaptive learning rate schedule
- Projection onto feasible sets
"""

import numpy as np
from typing import Tuple, Optional, Callable, List

class AdaptiveGDSolver:
    """
    Generic Adaptive Gradient Descent solver for two-player games with unknown parameters.
    
    The game structure assumes:
    - Player 1 payoff depends on x1, x2, and unknown params (A1, Ac1)
    - Player 2 payoff depends on x1, x2, and unknown params (A2, Ac2)
    - Demand/Observation model: z = z_base + A*x_self + Ac*x_other
    
    Parameters
    ----------
    dim : int
        Dimension of decision variables x1, x2.
    lam1, lam2 : float
        Regularization coefficients for players 1 and 2.
    nu : float
        Learning rate for parameter estimation (ridge regression update scaling).
    rho : float
        Ridge regression regularization parameter.
    B : float
        Magnitude of exploratory noise.
    max_iter : int
        Maximum number of iterations.
    proj_func : Callable, optional
        Projection function onto the feasible set. Defaults to identity.
    """
    
    def __init__(self, 
                 dim: int, 
                 lam1: float = 0.1, 
                 lam2: float = 0.1,
                 nu: float = 0.1,
                 rho: float = 0.01,
                 B: float = 0.1,
                 max_iter: int = 1000,
                 proj_func: Optional[Callable] = None):
        
        self.dim = dim
        self.lam1 = lam1
        self.lam2 = lam2
        self.nu = nu
        self.rho = rho
        self.B = B
        self.max_iter = max_iter
        
        # Default projection: identity (no constraints)
        self.proj = proj_func if proj_func else lambda x: x
        
        # Initialize decision variables
        self.x1 = np.zeros(dim)
        self.x2 = np.zeros(dim)
        
        # Initialize parameter estimates (zeros)
        self.A1_hat = np.zeros((dim, dim))
        self.Ac1_hat = np.zeros((dim, dim))
        self.A2_hat = np.zeros((dim, dim))
        self.Ac2_hat = np.zeros((dim, dim))
        
        # History storage for estimation
        self._reset_history()
        
        # Trajectory storage
        self.history = {
            'x1': [], 'x2': [],
            'A1_hat': [], 'Ac1_hat': [],
            'A2_hat': [], 'Ac2_hat': []
        }

    def _reset_history(self):
        """Reset trajectory history."""
        pass  # No internal buffers needed for SGD-style updates

    def get_gradient(self, x1: np.ndarray, x2: np.ndarray, 
                     z1_pred: np.ndarray, z2_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Nash Equilibrium gradients using estimated parameters.
        
        Gradient derivation:
        Payoff_1 = -0.5 * x1.T @ z1 + lam1 * ||x1||^2 (simplified L2 reg for smoothness)
                 = -0.5 * x1.T @ (z_base + A1*x1 + Ac1*x2) + ...
        grad_x1 = -(A1.T @ x1) - 0.5 * (z1 + Ac1 @ x2) + lam1 * x1
                = -(A1 - lam1*I).T @ x1 - 0.5 * (z1 + Ac1 @ x2)
        
        Returns
        -------
        grad1, grad2 : np.ndarray
            Gradients for player 1 and 2.
        """
        I = np.eye(self.dim)
        
        # Player 1 Gradient
        term1 = -(self.A1_hat - self.lam1 * I).T @ x1
        term2 = -0.5 * (z1_pred + self.Ac1_hat @ x2)
        grad1 = term1 + term2
        
        # Player 2 Gradient
        term1 = -(self.A2_hat - self.lam2 * I).T @ x2
        term2 = -0.5 * (z2_pred + self.Ac2_hat @ x1)
        grad2 = term1 + term2
        
        return grad1, grad2

    def update_parameters(self, x1: np.ndarray, x2: np.ndarray, 
                          z1_obs: np.ndarray, z2_obs: np.ndarray,
                          z1_base: np.ndarray, z2_base: np.ndarray):
        """
        Update parameter estimates using Stochastic Gradient Descent for Least Squares.
        
        This follows the original implementation in utilsrm.py (update_estimate method).
        
        Model: z_obs = z_base + A*x_self + Ac*x_other + noise
        
        SGD Update Rule:
        theta_new = theta_old + nu * (residual) * (feature)^T
        
        where residual = z_obs - z_base - theta_old @ [x_self; x_other]
              feature = [x_self; x_other]
        
        This is an online/recursive least squares update, not batch ridge regression.
        """
        # --- Player 1 Estimation ---
        # Concatenate features: [x1; x2]
        phi1 = np.concatenate([x1, x2])  # Shape (2*dim,)
        
        # Current prediction using old estimates
        z1_pred_full = z1_base + self.A1_hat @ x1 + self.Ac1_hat @ x2
        
        # Residual
        residual1 = z1_obs - z1_pred_full  # Shape (dim,)
        
        # SGD update for barA1 = [A1, Ac1]
        # Update: barA1_new = barA1_old + nu * residual @ phi^T
        barA1_hat = np.hstack([self.A1_hat, self.Ac1_hat])  # (dim, 2*dim)
        update1 = self.nu * np.outer(residual1, phi1)
        # Clip updates to prevent explosion
        update1 = np.clip(update1, -10, 10)
        barA1_hat = barA1_hat + update1
        
        # Split back
        self.A1_hat = barA1_hat[:, :self.dim]
        self.Ac1_hat = barA1_hat[:, self.dim:]
        
        # --- Player 2 Estimation ---
        # Concatenate features: [x2; x1] (self first, then other)
        phi2 = np.concatenate([x2, x1])  # Shape (2*dim,)
        
        z2_pred_full = z2_base + self.A2_hat @ x2 + self.Ac2_hat @ x1
        residual2 = z2_obs - z2_pred_full
        
        barA2_hat = np.hstack([self.A2_hat, self.Ac2_hat])
        update2 = self.nu * np.outer(residual2, phi2)
        update2 = np.clip(update2, -10, 10)
        barA2_hat = barA2_hat + update2
        
        self.A2_hat = barA2_hat[:, :self.dim]
        self.Ac2_hat = barA2_hat[:, self.dim:]
        
        # Additional stability: clip parameter matrices
        self.A1_hat = np.clip(self.A1_hat, -100, 100)
        self.Ac1_hat = np.clip(self.Ac1_hat, -100, 100)
        self.A2_hat = np.clip(self.A2_hat, -100, 100)
        self.Ac2_hat = np.clip(self.Ac2_hat, -100, 100)

    def step(self, t: int, 
             z1_base: np.ndarray, z2_base: np.ndarray,
             observation_func: Callable,
             eta_base: float = 1.0) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Perform one iteration of Adaptive Gradient Descent.
        
        Parameters
        ----------
        t : int
            Current iteration index.
        z1_base, z2_base : np.ndarray
            Base demand vectors (intercepts).
        observation_func : Callable
            Function signature: func(x1, x2, A1_true, Ac1_true, A2_true, Ac2_true) -> (z1, z2)
            Returns the TRUE observed demands given current actions.
            In a real environment, this interacts with the simulator.
        eta_base : float
            Base learning rate. Actual rate decays as eta_base / log(t+2).
            
        Returns
        -------
        x1, x2 : np.ndarray
            Updated decisions.
        info : dict
            Dictionary containing current estimates and gradients.
        """
        # 1. Add Exploration Noise (Persistent Excitation)
        # Only add noise in early/mid stages or continuously depending on theory
        # Here we follow the original code: add noise to current iterate before querying
        u1 = self.B * np.random.randn(self.dim)
        u2 = self.B * np.random.randn(self.dim)
        
        x1_explore = self.x1 + u1
        x2_explore = self.x2 + u2
        
        # 2. Get True Observations from Environment
        # In migration, this function calls the new game engine's step/observe method
        z1_obs, z2_obs = observation_func(x1_explore, x2_explore)
        
        # 3. Update Parameter Estimates
        self.update_parameters(x1_explore, x2_explore, z1_obs, z2_obs, z1_base, z2_base)
        
        # 4. Construct Predicted Demands for Gradient Step
        # Use the NEWEST estimates to predict what demand WOULD BE at the clean point (x1, x2)
        z1_pred = z1_base + self.A1_hat @ self.x1 + self.Ac1_hat @ self.x2
        z2_pred = z2_base + self.A2_hat @ self.x2 + self.Ac2_hat @ self.x1
        
        # 5. Compute Gradients
        grad1, grad2 = self.get_gradient(self.x1, self.x2, z1_pred, z2_pred)
        
        # 6. Adaptive Learning Rate
        eta_t = eta_base / np.log(t + 2)
        
        # 7. Gradient Descent Step + Projection
        self.x1 = self.proj(self.x1 - eta_t * grad1)
        self.x2 = self.proj(self.x2 - eta_t * grad2)
        
        # Record History
        self.history['x1'].append(self.x1.copy())
        self.history['x2'].append(self.x2.copy())
        self.history['A1_hat'].append(self.A1_hat.copy())
        self.history['Ac1_hat'].append(self.Ac1_hat.copy())
        self.history['A2_hat'].append(self.A2_hat.copy())
        self.history['Ac2_hat'].append(self.Ac2_hat.copy())
        
        info = {
            'grad_norm': np.linalg.norm(np.concatenate([grad1, grad2])),
            'est_error_A1': None, # To be filled by caller if true params known
            'eta': eta_t
        }
        
        return self.x1, self.x2, info

    def run(self, z1_base: np.ndarray, z2_base: np.ndarray,
            observation_func: Callable, 
            eta_base: float = 1.0,
            verbose: bool = True) -> dict:
        """
        Run the full AGD optimization loop.
        """
        self._reset_history()
        self.x1 = np.zeros(self.dim)
        self.x2 = np.zeros(self.dim)
        
        if verbose:
            print(f"Starting AGD for {self.max_iter} iterations...")
            
        for t in range(self.max_iter):
            x1, x2, info = self.step(t, z1_base, z2_base, observation_func, eta_base)
            
            if verbose and (t % (self.max_iter // 10) == 0 or t == self.max_iter - 1):
                print(f"Iter {t}: Grad Norm={info['grad_norm']:.4e}, Eta={info['eta']:.4e}")
                
        return self.history
