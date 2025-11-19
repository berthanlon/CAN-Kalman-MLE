# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:44:23 2024

@author: betti
"""

import numpy as np

def generate_estimates_max_lk(
        training_input, 
        training_target, 
        F, h
        ) -> tuple:
    """
    Function to estimate the matrices R and Q for maximum likelihood estimation.

    Parameters:
    - training_input: Input data, expected to be in tensor form.
    - training_target: Target data (ground truth), expected to be in tensor form.
    - F: State transition matrix.
    - h_func: Measurement function that maps state to measurement space.

    Returns:
    - R_est: Estimated measurement noise covariance matrix.
    - Q_est: Estimated process noise covariance matrix.
    """

    print(f"Loading data and generating R and Q estimate")

    R_sum = np.array(np.zeros((2,2)))
    X_diff_Q_sum = np.array(np.zeros((4,4)))
    
    n_residuals = 0.0 
            
    for s in range(0, training_target.shape[0]):
        
        X_true_gen = training_target[s,:,:].cpu().numpy()
        measurements_gen = training_input[s,:,:].cpu().numpy()
        
        
        for k in range(1, training_target.shape[2]): # starts from 1 because measurements start from 1
        
            X_prev = X_true_gen[:, k-1] # for first itereation, state at 0
                 
            Xik = X_true_gen[:, k] # prediction at 1
    
            X_diff = Xik - F @ X_prev # State Error 1-0
            
            X_diff_mat = np.asmatrix(X_diff)
                
            X_diff_Q_sum += X_diff_mat.T @ X_diff_mat # for the Covariance of the vectors
            
            z_diff = (measurements_gen[:,k])-(h(Xik)) # Prediction error
            
            z_diff_mat = np.asmatrix(z_diff)
            
            R_sum += (z_diff_mat.T @ z_diff_mat) # for the Covariance of Prediction error
    
            n_residuals += 1
            
    #print(f"n_residuals = {n_residuals}, for R_est and Q_est")
    
    R_est = (1.0/ n_residuals) *  R_sum # cov prediction error
    Q_est = (1.0/ n_residuals) *  X_diff_Q_sum #cov state error
    print(f"R_est = {R_est}, Q_est = {Q_est}")

    return R_est, Q_est
