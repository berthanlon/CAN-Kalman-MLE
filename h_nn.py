# -*- coding: utf-8 -*- 
"""
Created on Wed Dec 18 21:23:05 2024
@author: betti
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Running on CUDA')
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


def preprocess_unknown(training_gt, training_measurements) -> torch.tensor:
    """
    Reshapes the ground truth and measurements into a 2D format:
      - ground_truth shape: (N_sequences, 4, T) -> (N_sequences*T, 4)
      - measurements shape: (N_sequences, 2, T) -> (N_sequences*T, 2)
    """
    gt_reshaped = training_gt.permute(0, 2, 1).reshape(-1, 4)
    meas_reshaped = training_measurements.permute(0, 2, 1).reshape(-1, 2)
    return gt_reshaped, meas_reshaped


#######################################
## Mahalanobis Log-Det Loss Function ##
#######################################
class MahalanobisLoss(nn.Module):
    """
    Custom loss using R^{-1} directly.
    """
    def __init__(self, R):
        super(MahalanobisLoss, self).__init__()
        self.R_inv = torch.inverse(R)

    def forward(self, outputs, targets):
        """
        outputs: Model predictions (batch_size, d)
        targets: Ground-truth measurements (batch_size, d)
        
        returns: scalar loss = mean over batch of (z - h(x))^T R^{-1} (z - h(x))
        """
        residuals = outputs - targets  # Shape: (batch_size, d)
        losses = torch.sum((residuals @ self.R_inv) * residuals, dim=1)  # Mahalanobis distance
        return torch.mean(losses)  # Mean over batch


######################################################
## Neural Network with BatchNorm + 2 Hidden Layers ###
######################################################
class FunctionApproximator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(FunctionApproximator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


######################################
## Model HU (Hidden Unknowns) Class ##
######################################

class ModelHU:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, R_initial):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize R
        self.R = R_initial.clone().to(self.device).to(torch.float32)
        
        # Build the MLP model
        self.model = FunctionApproximator(
            input_size, 
            hidden_size, 
            output_size, 
            dropout_rate=0
        ).to(self.device)
        
        self.criterion = MahalanobisLoss(self.R)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, state, observation, num_epochs, batch_size, base_dir, save_model=True, update_interval=10):
        """
        Trains for `num_epochs`, updating R every `update_interval` epochs.
        """
        training_input = state.to(self.device)
        training_target = observation.to(self.device)
        
        dataset = TensorDataset(training_input, training_target)
        generator = torch.Generator(device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        R_history = [] 
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_inputs, batch_targets in dataloader:
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)  
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_epoch_loss:.4f}")
            
            # Update R every `update_interval` epochs
            if (epoch + 1) % update_interval == 0:
                residuals = self.compute_residuals(training_input, training_target, batch_size)
                self.update_R(residuals)

                current_R = self.R.detach().cpu().numpy()
                R_history.append(current_R)
                print('updated R', current_R)
                #epoch_path = os.path.join(base_dir, f"R_Epoch_{epoch+1}.npy")
                #np.save(epoch_path, current_R)
        
        if save_model:
            model_path = os.path.join(base_dir, "model_h_unknown_R_best.pth")
            torch.save({'model_state_dict': self.model.state_dict(), 'R': self.R.cpu()}, model_path)
            print(f"Model and R saved to: {model_path}")

    def compute_residuals(self, state_tensor, observation_tensor, batch_size):
        """
        Compute residuals r_k^i = z_k^i - h_{\theta_l}(x_k^i) over entire dataset.
        """
        self.model.eval()
        dataset = TensorDataset(state_tensor, observation_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        residuals = []
        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                outputs = self.model(batch_inputs)
                residuals.append((batch_targets - outputs).cpu().numpy())
        return np.concatenate(residuals, axis=0)

    def update_R(self, residuals):
        """
        Update the covariance matrix R based on the residuals.
        """
        R_sum = np.zeros((2, 2))
        for pair in range(residuals.shape[0]):
            z_diff = residuals[pair]
            R_sum += np.outer(z_diff, z_diff)

        R_est = (1 / residuals.shape[0]) * R_sum
        
        # Regularization for numerical stability
        epsilon = 1e-6
        R_est += epsilon * np.eye(R_est.shape[0])

        self.R = torch.tensor(R_est, dtype=torch.float32, device=self.device)
        self.criterion = MahalanobisLoss(self.R) 

    def load_best_model(self, file_path):
        """
        Loads the best trained model and covariance matrix R.
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
        # Load R if present
        self.R = checkpoint.get('R', torch.eye(2, device=self.device))
        self.criterion = MahalanobisLoss(self.R)
    
        self.model.eval()
        print(f"Model and R matrix loaded successfully from: {file_path}")
        return {'model': self.model, 'R': self.R}
