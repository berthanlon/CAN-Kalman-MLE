# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:54:20 2024

@author: betti
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

class UnscentedKalmanFilter:
    def __init__(self, x0, P0, Q, R, F, h, alpha=0.1, beta=3, kappa=0):
        """
        Initialize the Unscented Kalman Filter.

        Parameters:
        - x0: Initial state estimate (n-dimensional NumPy array)
        - P0: Initial covariance estimate (n x n NumPy array)
        - Q: Process noise covariance (n x n NumPy array)
        - R: Measurement noise covariance (m x m NumPy array or PyTorch tensor)
        - F: State transition function (callable) or matrix (NumPy array)
        - h: Measurement function (PyTorch model or wrapper)
        - alpha, beta, kappa: Parameters for the sigma point distribution
        """
        # Ensure inputs are floating-point. If not, convert them.
        self.x = x0.astype(np.float64) if not np.issubdtype(x0.dtype, np.floating) else x0
        self.P = P0.astype(np.float64) if not np.issubdtype(P0.dtype, np.floating) else P0
        self.Q = Q.astype(np.float64) if not np.issubdtype(Q.dtype, np.floating) else Q
        
        # For R, if it is not a NumPy array (e.g. a PyTorch tensor), convert it.
        if isinstance(R, np.ndarray):
            self.R = R.astype(np.float64) if not np.issubdtype(R.dtype, np.floating) else R
        else:
            R_np = R.cpu().numpy()
            self.R = R_np.astype(np.float64) if not np.issubdtype(R_np.dtype, np.floating) else R_np

        self.F = F  # State transition function or matrix
        self.h = h  # Measurement function (PyTorch model or wrapper)
        self.n = x0.shape[0]
        self.m = self.R.shape[0]

        # Calculate lambda parameter
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = alpha**2 * (self.n + kappa) - self.n

        # Compute weights (ensure they are floats)
        self.gamma = np.sqrt(self.n + self.lmbda)
        self.Wm = np.zeros(2 * self.n + 1, dtype=np.float64)
        self.Wc = np.zeros(2 * self.n + 1, dtype=np.float64)
        self.Wm[0] = self.lmbda / (self.n + self.lmbda)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)
        self.Wm[1:] = 1 / (2 * (self.n + self.lmbda))
        self.Wc[1:] = self.Wm[1:]

    def generate_sigma_points(self, x, P):
        """
        Generate sigma points.

        Parameters:
        - x: Mean state vector
        - P: Covariance matrix

        Returns:
        - sigma_points: Array of sigma points (2n+1 x n)
        """
        sigma_points = np.zeros((2 * self.n + 1, self.n), dtype=np.float64)
        sigma_points[0] = x
        U = np.linalg.cholesky((self.n + self.lmbda) * P)
        for k in range(self.n):
            sigma_points[k + 1] = x + U[:, k]
            sigma_points[self.n + k + 1] = x - U[:, k]
        return sigma_points

    def predict(self):
        """
        Predict the next state and covariance.
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)

        # Predict sigma points through the process model
        sigma_points_pred = np.zeros_like(sigma_points, dtype=np.float64)
        for i, sp in enumerate(sigma_points):
            if callable(self.F):
                if isinstance(self.F, torch.nn.Module):
                    # F is a PyTorch model
                    sp_tensor = torch.from_numpy(sp).float()
                    sp_tensor = sp_tensor.unsqueeze(0)  # Add batch dimension
                    device = next(self.F.parameters()).device
                    sp_tensor = sp_tensor.to(device)
                    with torch.no_grad():
                        delta_pred_tensor = self.F(sp_tensor)
                    delta_pred_tensor = delta_pred_tensor.squeeze(0)
                    sigma_points_pred[i] = sp + delta_pred_tensor.cpu().numpy()
                else:
                    # F is a regular function
                    delta = self.F(sp)
                    sigma_points_pred[i] = sp + delta
            else:
                raise ValueError("State transition model F must be callable or a PyTorch model.")

        # Compute predicted mean
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)

        # Compute predicted covariance, ensuring Q is copied as float
        self.P = self.Q.copy()
        for i in range(2 * self.n + 1):
            y = sigma_points_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(y, y)

        # Store predicted sigma points for update step
        self.sigma_points_pred = sigma_points_pred

    def update(self, z):
        """
        Update the state estimate with a new measurement.

        Parameters:
        - z: Measurement vector (NumPy array)
        """
        # Predict measurement sigma points
        Z_sigma = np.zeros((2 * self.n + 1, self.m), dtype=np.float64)
        for i, sp in enumerate(self.sigma_points_pred):
            sp_tensor = torch.from_numpy(sp).float()
            sp_tensor = sp_tensor.unsqueeze(0)  # Add batch dimension

            # Determine the device of the model from either a wrapper or directly
            if hasattr(self.h, 'model'):
                # `self.h` is a wrapper containing a PyTorch model
                device = next(self.h.model.parameters()).device
                with torch.no_grad():
                    z_pred_tensor = self.h.model(sp_tensor.to(device))
            else:
                # `self.h` is directly a PyTorch model
                device = next(self.h.parameters()).device
                with torch.no_grad():
                    z_pred_tensor = self.h(sp_tensor.to(device))

            z_pred_tensor = z_pred_tensor.squeeze(0)
            Z_sigma[i] = z_pred_tensor.cpu().numpy()

        # Compute predicted measurement mean
        z_pred = np.sum(self.Wm[:, np.newaxis] * Z_sigma, axis=0)

        # Compute innovation covariance S (ensure S is float)
        S = self.R.copy()
        for i in range(2 * self.n + 1):
            y = Z_sigma[i] - z_pred
            S += self.Wc[i] * np.outer(y, y)

        # Compute cross covariance T
        T = np.zeros((self.n, self.m), dtype=np.float64)
        for i in range(2 * self.n + 1):
            x_diff = self.sigma_points_pred[i] - self.x
            z_diff = Z_sigma[i] - z_pred
            T += self.Wc[i] * np.outer(x_diff, z_diff)

        # Kalman gain
        K = T @ np.linalg.inv(S)

        # Update state
        y = z - z_pred
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T

    def get_state(self):
        """
        Get the current state estimate.

        Returns:
        - x: Current state estimate
        - P: Current covariance estimate
        """
        return self.x, self.P

    def plot_state_estimate(self, state_history, gt_history, state_labels=None, case=""):
        """
        Plot the state estimates and ground truths for dimensions 1 and 3 (indices 0 and 2).
        """
        if len(state_history.shape) != 2 or len(gt_history.shape) != 2:
            raise ValueError("Expected state_history and gt_history to be 2D arrays with shape (n_steps, n_state_dims).")

        dims_to_plot = [0, 2]
        if state_labels is None:
            state_labels = [f'State {i+1}' for i in range(state_history.shape[1])]

        plt.figure(figsize=(10, 6))
        for i in dims_to_plot:
            plt.plot(state_history[:, i], label=f'Estimate {state_labels[i]}')
            plt.plot(gt_history[:, i], label=f'Ground Truth {state_labels[i]}', linestyle='--')

        plt.xlabel('Time Step')
        plt.ylabel('State Value')
        plt.title(case)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_positions(self, state_history, gt_history, case=""):
        """
        Plot the x and y positions from the state estimates and the ground truth.
        """
        x_estimated = state_history[:, 0]
        y_estimated = state_history[:, 2]
        x_gt = gt_history[:, 0]
        y_gt = gt_history[:, 2]

        plt.figure(figsize=(10, 6))
        plt.plot(x_gt, y_gt, 'b-x', label='Ground Truth Position')
        plt.plot(x_estimated, y_estimated, 'ro-', label='Estimated Position')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(case)
        plt.legend()
        plt.grid(True)
        plt.show()

    def computeMSEsForSequences(self, X_True: np.array, X_gen: np.array, case: str) -> np.array:
        """
        Computes RMSE values for each timestep across all test sequences.
        """
        assert X_True.shape == X_gen.shape, f"Shape mismatch: X_True {X_True.shape}, X_gen {X_gen.shape}"

        n_sequences, n_timesteps, _ = X_True.shape
        mse_T = np.zeros(n_timesteps)

        for t in range(n_timesteps):
            residuals = X_gen[:, t, :] - X_True[:, t, :]
            squared_errors = np.sum(residuals ** 2, axis=1)
            mse_T[t] = np.sqrt(np.mean(squared_errors))

        overall_rmse = np.sqrt(np.mean((X_gen - X_True) ** 2))
        print(f"Case: {case} - Overall RMSE = {overall_rmse:.4f}")

        return mse_T


########################################
# Helper function to run a UKF instance
########################################
def run_ukf(ukf, test_measurements, test_gt, mean_ini, P_ini, device):
    """
    Runs the given UKF on all sequences in test_measurements.

    :param ukf: An instance of the UnscentedKalmanFilter (UKF2 or UKF3)
    :param test_measurements: Tensor of shape (n_sequences, measurement_dim, n_steps)
    :param test_gt: Tensor of shape (n_sequences, state_dim, n_steps)
    :param mean_ini: Initial state mean (numpy array)
    :param P_ini: Initial state covariance (numpy array)
    :param device: torch device (cpu or cuda)
    :return: state_histories of shape (n_sequences, n_steps, state_dim)
    """

    # Convert ground truth to numpy if needed (shape becomes (n_sequences, n_steps, state_dim))
    test_gt_np = test_gt.cpu().numpy().transpose(0, 2, 1)
    n_sequences, measurement_dim, n_steps = test_measurements.shape
    state_dim = ukf.get_state()[0].shape[0]

    # Allocate space for results (as float64)
    state_histories = np.zeros((n_sequences, n_steps, state_dim), dtype=np.float64)

    for s in range(n_sequences):
        # Reset filter for each sequence
        ukf.x = mean_ini.copy()
        ukf.P = P_ini.copy()

        for t in range(n_steps):
            ukf.predict()
            z = test_measurements[s, :, t].cpu().numpy()  # measurement
            ukf.update(z)
            state_histories[s, t, :] = ukf.get_state()[0]

    return state_histories

########################################
# MSE / RMSE Helper
########################################
def compute_rmse(ground_truth, estimates, label=""):
    """
    Computes RMSE over all sequences and time steps.
    
    :param ground_truth: (n_sequences, n_steps, state_dim)
    :param estimates: (n_sequences, n_steps, state_dim)
    :param label: Optional label for printing
    :return: 1D numpy array of length n_steps (RMSE at each time)
    """
    # Ensure same shape
    assert ground_truth.shape == estimates.shape, f"Shape mismatch: {ground_truth.shape} vs {estimates.shape}"
    
    # Mean squared error across sequences, then average across state dims
    mse = np.mean((ground_truth - estimates)**2, axis=(0, 2))  # shape: (n_steps,)
    rmse = np.sqrt(mse)  # shape: (n_steps,)

    if label:
        print(f"{label} - Final RMSE (averaged over all timesteps): {rmse.mean():.4f}")
    return rmse
