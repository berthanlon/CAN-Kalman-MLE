"""
Created on Thu Sep 19 20:19:29 2024

@author: betti
modified to learn delta k
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from parameters import R, chol_R, A  # Assuming A is the Cholesky decomposition of Q^-1
import os

def dynamic_preprocess(ground_truth):
    """
    Prepare the sequences of x_{k-1} and (x_k - x_{k-1}) for training.

    Args:
        ground_truth (torch.Tensor): Tensor of shape [batch_size, state_dims, time_steps].
    
    Returns:
        x_k_m1 (torch.Tensor): States at time steps 1, 2, ..., T-1, reshaped for training.
        delta_x (torch.Tensor): Differences (x_k - x_{k-1}), reshaped for training.
    """
    # Ensure the input is a torch tensor
    if not isinstance(ground_truth, torch.Tensor):
        raise ValueError("Input ground_truth must be a torch.Tensor")
    
    # Extract sequences of consecutive time steps
    # x_{k-1}: States at time steps 1, 2, ..., T-1
    x_k_m1 = ground_truth[:, :, :-1]
    
    # Compute differences: x_k - x_{k-1}
    delta_x = ground_truth[:, :, 1:] - ground_truth[:, :, :-1]
    
    # Reshape the data for training
    # From [batch_size, state_dims, time_steps] -> [batch_size * (time_steps - 1), state_dims]
    x_k_m1 = x_k_m1.permute(0, 2, 1).reshape(-1, ground_truth.shape[1])
    delta_x = delta_x.permute(0, 2, 1).reshape(-1, ground_truth.shape[1])
    
    return x_k_m1, delta_x


# Custom loss function
class CustomLoss(nn.Module):
    def __init__(self, Q):
        super(CustomLoss, self).__init__()
        self.Q_inv = torch.inverse(Q)
    
    def forward(self, outputs, targets):
        residuals = outputs - targets  # Shape: (batch_size, d)
        losses = torch.sum((residuals @ self.Q_inv) * residuals, dim=1)  # Mahalanobis distance
        return torch.mean(losses)  # Mean over batch


# Neural network class to learn the dynamics (state transition)
class DynamicModelLearner(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(DynamicModelLearner, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# DynamicModel Class
class DynamicModel:
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 learning_rate, 
                 Q):  # Pass matrix A (Cholesky of Q^-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Q = Q.to(self.device)  # Cholesky matrix A of Q^-1
        self.model = DynamicModelLearner(input_size, hidden_size, output_size, dropout_rate=0.5).to(self.device)
        self.criterion = CustomLoss(self.Q)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train(self,
              state_k_minus_1,  # x_{k-1}
              delta_k,          # x_k - x_{k-1}
              num_epochs, 
              batch_size,
              base_dir):     
        training_input = state_k_minus_1.to(self.device)
        training_target = delta_k.to(self.device)
        dataset = TensorDataset(training_input, training_target)
        
        # Create generator explicitly with the correct device (ensure it's on 'cuda' if using GPU)
        generator = torch.Generator(device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            all_outputs = []
            all_targets = []

            for batch_inputs, batch_targets in dataloader:
                outputs = self.model(batch_inputs)
                
                # Compute the loss using the custom loss function
                loss = self.criterion(outputs, batch_targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Store outputs and targets for plotting
                all_outputs.append(outputs.detach().cpu().numpy())
                all_targets.append(batch_targets.detach().cpu().numpy())
            
            if (epoch + 1) % 10 == 0:
                avg_epoch_loss = epoch_loss / len(dataloader)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
                
                # Concatenate all batches
                all_outputs = np.concatenate(all_outputs, axis=0)
                all_targets = np.concatenate(all_targets, axis=0)

                # Reconstruct absolute states for evaluation or plotting
                reconstructed_outputs = training_input.cpu().numpy() + all_outputs
                reconstructed_targets = training_input.cpu().numpy() + all_targets
                
                # Extract features for plotting
                x_targets = reconstructed_targets[:, 0]
                y_targets = reconstructed_targets[:, 2]

                x_outputs = reconstructed_outputs[:, 0]
                y_outputs = reconstructed_outputs[:, 2]
                
                # Plot y vs x
                plt.figure(figsize=(10, 6))
                plt.scatter(x_targets, y_targets, label='Targets', color='blue', alpha=0.6, edgecolor='k')
                plt.scatter(x_outputs, y_outputs, label='Outputs', color='red', alpha=0.6, edgecolor='k')
                plt.legend()
                plt.title(f'Reconstructed Outputs vs Targets (y vs x) at Epoch {epoch + 1}')
                plt.xlabel('x (First Feature)')
                plt.ylabel('y (Third Feature)')
                plt.grid(True)
                plt.show()
                        
        print("Training complete.")
        
        # Save the trained model
        model_filename = "dynamic_model_F.pth"
        model_filepath = os.path.join(base_dir, model_filename)
        self.save_model(model_filepath)
        
    def save_model(self, file_path):
        """Saves the model and optimizer state to the specified file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, file_path)
        print(f"Model saved to {file_path}")
        
    def load_model(self, file_path):
        """Loads the model and optimizer state from the specified file."""
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()
            print(f"Model loaded from {file_path}")
            return self.model  # Return the loaded model
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except KeyError as e:
            print(f"Missing key in checkpoint file: {e}")
        except Exception as e:
            print(f"Error loading model from {file_path}: {e}")
        return None  # Return None if there was an issue

