# -*- coding: utf-8 -*-
"""
Neural network class 
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from parameters import R, chol_R, A
import os
import numpy as np
import matplotlib.pyplot as plt

#######################################
## Mahalanobis Log-Det Loss Function ##
#######################################
class CustomLoss(nn.Module):
    """
    Custom loss using R^{-1} directly.
    """
    def __init__(self, R):
        super(CustomLoss, self).__init__()
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

# Neural network class
class FunctionApproximator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, B):
        super(FunctionApproximator, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.B = B #torch.from_numpy(chol_R).float().to(self.device)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout(x)  
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
def preprocess(training_gt,                 #gt
               training_measurements        #meas
               ) -> torch.tensor:
    
    gt_reshaped = training_gt.permute(0, 2, 1).reshape(-1, 4)
    meas_reshaped = training_measurements.permute(0, 2, 1).reshape(-1, 2)
    
    return gt_reshaped, meas_reshaped

# Model class for training and evaluation
class ModelH:
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 learning_rate):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.R = torch.tensor(R).float()
        print('self.R init', self.R)
        self.B = torch.from_numpy(chol_R).float().to(self.device)
        self.model = FunctionApproximator(input_size, hidden_size, output_size, dropout_rate=0.5, B=self.B).to(self.device)
        self.criterion = CustomLoss(self.R)  # Use custom loss function with B
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)


    def train(self,
              state,
              observation, 
              num_epochs, 
              batch_size,
              base_dir):     
        training_input = state
        training_target = observation
        num_samples = training_input.size(0)
        
        training_input, training_target = training_input.to(self.device), training_target.to(self.device)
        dataset = TensorDataset(training_input, training_target)
        generator = torch.Generator(device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            all_outputs = []
            all_targets = []

            for batch_inputs, batch_targets in dataloader:
                outputs = self.model(batch_inputs)
                
                # Compute the loss using the custom loss function (with B transformation)
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
                
                print(f'all_targets shape: {all_targets.shape}')
                print(f'all_outputs shape: {all_outputs.shape}')

                # Extract the first and second features (x and y)
                x_targets = all_targets[:, 0]
                y_targets = all_targets[:, 1]

                x_outputs = all_outputs[:, 0]
                y_outputs = all_outputs[:, 1]

                # Plot y vs x
                plt.figure(figsize=(10, 6))
                plt.scatter(x_targets, y_targets, label='Targets', color='blue', alpha=0.6, edgecolor='k')
                plt.scatter(x_outputs, y_outputs, label='Outputs', color='red', alpha=0.6, edgecolor='k')
                plt.legend()
                plt.title(f'Outputs vs Targets (y vs x) at Epoch {epoch + 1}')
                plt.xlabel('x (First Feature)')
                plt.ylabel('y (Second Feature)')
                plt.grid(True)
                plt.show()
                    
        print("Training complete.")
        
        # Construct the full file path and save the model
        model_filename = "model_checkpoint.pth"
        model_filepath = os.path.join(base_dir, model_filename)
        self.save_model(model_filepath)
    
    def evaluate(self, eval_input, eval_target, batch_size):
        num_samples = eval_input.size(0)
        
        eval_input, eval_target = eval_input.to(self.device), eval_target.to(self.device)
        dataset = TensorDataset(eval_input, eval_target)
        generator = torch.Generator(device=self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=generator)
        
        self.model.eval()
        eval_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                predictions = self.model(batch_inputs)
                
                # Apply transformation using the methods in the model
                transformed_predictions = self.model.transform(predictions)
                transformed_targets = self.model.transform(batch_targets)
                
                # Store the predictions and targets for plotting
                all_predictions.append(transformed_predictions.cpu().numpy())
                all_targets.append(transformed_targets.cpu().numpy())
                
                # Compute the loss between the transformed predictions and targets
                loss = self.criterion(transformed_predictions, transformed_targets)
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / len(dataloader)
        print(f'Evaluation Loss: {avg_eval_loss:.4f}')
        
        # Concatenate all the predictions and targets from each batch
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Plot the predictions against the ground truth
        plt.figure(figsize=(10, 5))
        plt.plot(all_targets[:, 0], all_targets[:, 1], label='Ground Truth', color='blue', linestyle='--', marker='o')
        plt.plot(all_predictions[:, 0], all_predictions[:, 1], label='Predictions', color='red', linestyle='-', marker='x')
        plt.legend()
        plt.title('Model Predictions vs Ground Truth')
        plt.xlabel('Feature 1 (e.g., X)')
        plt.ylabel('Feature 2 (e.g., Y)')
        plt.grid(True)
        plt.show()
        
        return avg_eval_loss

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
