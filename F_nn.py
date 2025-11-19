"""
NN to learn F and Q
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
##initisalisng space on device (CPU)
if torch.cuda.is_available():
   dev = torch.device("cuda:0")
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print('Running on CUDA')
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

def dynamic_preprocess_unknown(ground_truth):
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
    x_k_m1 = ground_truth[:, :, :-1]
    delta_x = ground_truth[:, :, 1:] - ground_truth[:, :, :-1]
    
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
    

# Neural network class to learn the dynamics
class DynamicModelLearner(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(DynamicModelLearner, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
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

# DynamicModel Class with Q saving
class DynamicModelUnknown:
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
        #self.Q = torch.eye(A.size(0)).to(self.device)  # Initialize Q as identity matrix

    def update_Q_and_A(self, residuals):
        """
        Update Q and compute A (Cholesky decomposition of Q^-1).
        """
        # Combine all residuals
        residuals_all = torch.cat(residuals, dim=0)

        # Compute the empirical covariance matrix Q
        Q_new = torch.cov(residuals_all.T)
        Q_new += 1e-6 * torch.eye(Q_new.size(0)).to(self.device)  # Add small epsilon for numerical stability

        # Compute Q^-1 and Cholesky decomposition
        Q_inv = torch.linalg.inv(Q_new)
        A_new = torch.linalg.cholesky(Q_inv)

        return Q_new, A_new

    def train(self,
              state_k_minus_1,  # x_{k-1}
              delta_k,          # x_k - x_{k-1}
              num_epochs, 
              batch_size,
              base_dir,
              update_interval=10):     
        training_input = state_k_minus_1.to(self.device)
        training_target = delta_k.to(self.device)
        dataset = TensorDataset(training_input, training_target)

        # Create generator explicitly with the correct device (ensure it's on 'cuda' if using GPU)
        generator = torch.Generator(device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        prev_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            residuals_all = []  # Store residuals for Q update

            for batch_inputs, batch_targets in dataloader:
                outputs = self.model(batch_inputs)

                # Compute the loss using the custom loss function
                loss = self.criterion(outputs, batch_targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Store residuals
                residuals = batch_targets - outputs
                residuals_all.append(residuals.detach())

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

            # Update Q and A every `update_interval` epochs
            if (epoch + 1) % update_interval == 0:
                self.Q, self.A = self.update_Q_and_A(residuals_all)
                self.criterion = CustomLoss(self.Q)  # Update the loss function with new A
                print(f"Epoch [{epoch + 1}]: Updated Q and A.")

        print("Training complete.")
        
        # Save the trained model and Q
        model_filename = "dynamic_model_F_unknown.pth"
        model_filepath = os.path.join(base_dir, model_filename)
        self.save_model(model_filepath)

    def save_model(self, file_path):
        """
        Saves the model, optimizer state, and learned Q matrix to the specified file.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'Q': self.Q.cpu().numpy()  # Save Q as a NumPy array
        }
        torch.save(checkpoint, file_path)
        print(f"Model and learned Q saved to {file_path}")

    def load_model(self, file_path):
        """
        Loads the model, optimizer state, and learned Q matrix from the specified file.
        """
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.Q = torch.tensor(checkpoint['Q']).to(self.device)  # Load Q and move to the correct device
            self.A = torch.linalg.cholesky(torch.linalg.inv(self.Q))  # Recompute A from Q
            self.criterion = CustomLoss(self.Q)  # Update the loss function with the loaded A
            self.model.eval()
            print(f"Model and learned Q loaded from {file_path}")
            return self.model
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except KeyError as e:
            print(f"Missing key in checkpoint file: {e}")
        except Exception as e:
            print(f"Error loading model from {file_path}: {e}")
        return None  # Return None if there was an issue

