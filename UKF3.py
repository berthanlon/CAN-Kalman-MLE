import numpy as np
import torch
import matplotlib.pyplot as plt

class UnscentedKalmanFilter:
    def __init__(self, x0, P0, Q, R, F, h, alpha=0.01, beta=3, kappa=0):
        self.x = x0.astype(np.float64)           # Ensure x0 is float
        self.P = P0.astype(np.float64)           # Ensure P0 is float
        self.Q = Q.astype(np.float64)            # Ensure Q is float
        self.R = R.astype(np.float64)            # Ensure R is float
        self.F = F  # State transition function or matrix
        self.h = h  # Measurement function
        self.n = x0.shape[0]
        self.m = R.shape[0]

        # Calculate lambda parameter
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = alpha**2 * (self.n + kappa) - self.n

        # Compute weights (these will be float by default)
        self.gamma = np.sqrt(self.n + self.lmbda)
        self.Wm = np.zeros(2 * self.n + 1, dtype=np.float64)
        self.Wc = np.zeros(2 * self.n + 1, dtype=np.float64)
        self.Wm[0] = self.lmbda / (self.n + self.lmbda)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)
        self.Wm[1:] = 1 / (2 * (self.n + self.lmbda))
        self.Wc[1:] = self.Wm[1:]

    def generate_sigma_points(self, x, P):
        sigma_points = np.zeros((2 * self.n + 1, self.n), dtype=np.float64)
        sigma_points[0] = x
        U = np.linalg.cholesky((self.n + self.lmbda) * P)
        for k in range(self.n):
            sigma_points[k + 1] = x + U[:, k]
            sigma_points[self.n + k + 1] = x - U[:, k]
        return sigma_points

    def predict(self):
        sigma_points = self.generate_sigma_points(self.x, self.P)
        sigma_points_pred = np.zeros_like(sigma_points, dtype=np.float64)
        for i, sp in enumerate(sigma_points):
            if callable(self.F):
                if isinstance(self.F, torch.nn.Module):
                    sp_tensor = torch.from_numpy(sp).float()
                    sp_tensor = sp_tensor.unsqueeze(0)
                    device = next(self.F.parameters()).device
                    sp_tensor = sp_tensor.to(device)
                    with torch.no_grad():
                        sp_pred_tensor = self.F(sp_tensor)
                    sp_pred_tensor = sp_pred_tensor.squeeze(0)
                    sigma_points_pred[i] = sp_pred_tensor.cpu().numpy()
                else:
                    sigma_points_pred[i] = self.F(sp)
            else:
                sigma_points_pred[i] = self.F @ sp

        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        self.P = self.Q.copy().astype(np.float64)
        for i in range(2 * self.n + 1):
            y = sigma_points_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(y, y)
        self.sigma_points_pred = sigma_points_pred

    def update(self, z):
        Z_sigma = np.zeros((2 * self.n + 1, self.m), dtype=np.float64)
        for i, sp in enumerate(self.sigma_points_pred):
            if callable(self.h):
                if isinstance(self.h, torch.nn.Module):
                    sp_tensor = torch.from_numpy(sp).float()
                    sp_tensor = sp_tensor.unsqueeze(0)
                    device = next(self.h.parameters()).device
                    sp_tensor = sp_tensor.to(device)
                    with torch.no_grad():
                        z_pred_tensor = self.h(sp_tensor)
                    z_pred_tensor = z_pred_tensor.squeeze(0)
                    Z_sigma[i] = z_pred_tensor.cpu().numpy()
                else:
                    Z_sigma[i] = self.h(sp)
            else:
                raise ValueError("Measurement function h must be callable")

        z_pred = np.sum(self.Wm[:, np.newaxis] * Z_sigma, axis=0)
        S = self.R.copy().astype(np.float64)
        for i in range(2 * self.n + 1):
            y = Z_sigma[i] - z_pred
            S += self.Wc[i] * np.outer(y, y)

        T = np.zeros((self.n, self.m), dtype=np.float64)
        for i in range(2 * self.n + 1):
            x_diff = self.sigma_points_pred[i] - self.x
            z_diff = Z_sigma[i] - z_pred
            T += self.Wc[i] * np.outer(x_diff, z_diff)

        K = T @ np.linalg.inv(S)
        y = z - z_pred
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T

    def get_state(self):
        return self.x, self.P

    # [Plotting methods remain unchanged]



    def plot_state_estimate(self, state_history, gt_history, state_labels=None):
        """
        Plot the state estimates and ground truths for dimensions 1 and 3 (i.e., indices 0 and 2).
        
        Parameters:
        - state_history: Array of state estimates at each time step.
                         Expected shape is (n_steps, n_state_dims).
        - gt_history: Array of ground truth values at each time step.
                      Expected shape is (n_steps, n_state_dims).
        - state_labels: List of labels for each state dimension (optional).
        """
        if len(state_history.shape) != 2 or len(gt_history.shape) != 2:
            raise ValueError("Expected state_history and gt_history to be 2D arrays with shape (n_steps, n_state_dims).")
        
        # Define which dimensions to plot (1 and 3 in 1-based indexing correspond to 0 and 2 in 0-based indexing)
        dims_to_plot = [0, 2]

        # If no labels are provided, create default labels
        if state_labels is None:
            state_labels = [f'State {i+1}' for i in range(state_history.shape[1])]

        # Create a plot for each selected dimension (0 and 2)
        plt.figure(figsize=(10, 6))
        
        for i in dims_to_plot:
            plt.plot(state_history[:, i], label=f'Estimate {state_labels[i]}')
            plt.plot(gt_history[:, i], label=f'Ground Truth {state_labels[i]}', linestyle='--')

        plt.xlabel('Time Step')
        plt.ylabel('State Value')
        plt.title('State Estimates vs Ground Truth (Dimensions 1 and 3)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_positions(self, state_history, gt_history):
        """
        Plot the x and y positions from the state estimates and the ground truth.

        Parameters:
        - state_history: Array of state estimates over time. Expected shape is (n_steps, n_state_dims).
        - gt_history: Ground truth positions over time. Expected shape is (n_steps, n_state_dims).
        """
        # Extract x and y positions from state history and ground truth
        x_estimated = state_history[:, 0]  # Assuming x position is at index 0
        y_estimated = state_history[:, 2]  # Assuming y position is at index 2
        x_gt = gt_history[:, 0]  # Ground truth x position
        y_gt = gt_history[:, 2]  # Ground truth y position
        
        # Plot the ground truth positions
        plt.figure(figsize=(10, 6))
        plt.plot(x_gt, y_gt, 'b-x', label='Ground Truth Position')  # Plot ground truth positions
        
        # Plot the estimated positions from the state history
        plt.plot(x_estimated, y_estimated, 'ro-', label='Estimated Position')  # Plot estimated positions
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Estimated Position vs Ground Truth Position')
        plt.legend()
        plt.grid(True)
        plt.show()