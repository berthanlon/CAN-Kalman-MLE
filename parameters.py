# -*- coding: utf-8 -*-
"""
Parameters 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#######################
### Size of DataSet ###
#######################

# Number of Training Examples
N_E = 1600

# Number of Cross Validation Examples
N_CV = 20

N_T = 200

# Sequence Length for Linear Case
n_steps = 50

#########################################
##### Neural network parameters #########
#########################################

mean_ini = np.array([100, 1, 0, 2], dtype = np.float32)

P_ini = np.diag([1, 0.1, 1, 0.1]).astype(np.float32)
chol_ini = np.linalg.cholesky(P_ini)

# Nearly constant velocity model
T = 0.5 # sampling time
F = np.array([[1, T, 0, 0], 
              [0, 1, 0, 0],
              [0, 0, 1, T],
              [0, 0, 0, 1]], dtype = np.float32) # state transition matrix

sigma_u = 0.1 # standard deviation of the acceleration    ####
Q = np.array([[T**3/3, T**2/2, 0, 0], 
              [T**2/2, T, 0, 0],
              [0, 0, T**3/3, T**2/2],
              [0, 0, T**2/2, T]], dtype = np.float32) * sigma_u**2 # covariance of the process noise
chol_Q = np.linalg.cholesky(Q) # Cholesky decomposition of Q
Q_inv = np.linalg.inv(Q)
chol_Q_inv = np.linalg.cholesky(Q) #_inv)
A = torch.from_numpy(chol_Q_inv)

sigma_r = 1
R = np.diag([2,2]) # covariance of the measurement noise
R_inv = np.linalg.inv(R)
chol_R = np.linalg.cholesky(R) #_inv) # Cholesky decomposition of R

B = chol_R # torch.from_numpy(chol_R).float().to(self.device)

m = 4
n = 2

def reshape_data(data):
    # If the input is a PyTorch tensor, move to CPU if necessary and convert to NumPy
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Perform the transpose operation
    data_transposed = data.transpose(0, 2, 1)  # Swap the last two dimensions
    
    # Convert back to a PyTorch tensor
    return torch.tensor(data_transposed, dtype=torch.float32)