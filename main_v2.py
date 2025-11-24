# -*- coding: utf-8 -*-
"""
mainfor two range scenario
"""
######## RUN ON CPU VERSION 
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.nn as nn
import torch
import torch.optim as optim
import new_sim 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from parameters import n_steps, F, Q, R, mean_ini, B, P_ini, A, reshape_data, sigma_u, sigma_r
from h_known_nn import ModelH, preprocess
from h_nn import ModelHU, preprocess_unknown
from F_nn_knownQR import dynamic_preprocess, DynamicModel
from F_nn import dynamic_preprocess_unknown, DynamicModelUnknown
from UKF2 import run_ukf, compute_rmse
import UKF2
import UKF3
from QR_estimator import generate_estimates_max_lk

torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float32)

##initisalisng space on device (CPU)
if torch.cuda.is_available():
   dev = torch.device("cuda:0")
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print('Running on CUDA')
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

print('n_steps,', n_steps)

directory_name = f"KNetFiles_{n_steps}"
current_working_directory = os.getcwd()
directory_path = os.path.join(current_working_directory, directory_name)
os.makedirs(directory_path, exist_ok=True)

subdirectory_name = f"q{sigma_u}_r{sigma_r}_T{n_steps}"
subdirectory_path = os.path.join(directory_path, subdirectory_name)
os.makedirs(subdirectory_path, exist_ok=True)
startTS = pd.Timestamp.utcnow()

startTS = pd.Timestamp.utcnow()

base_dir = f'C:/Users/betti/Desktop/two_radar_scenario/KNetFiles_{n_steps}/q{sigma_u}_r{sigma_r}_T{n_steps}/'
fname_base = 'MCSim_test'
fname_data = fname_base + f'_data_{n_steps}'
fname_mse_KNet = base_dir + fname_base + f'_KNet_{n_steps}'
fname_mse_MLE = base_dir + fname_base + f'_MLE_{n_steps}'
fname_plot = base_dir + fname_base + f'_plot_{n_steps}'

init_mcs = new_sim.MonteCarloSimulation(n_steps, base_dir, fname_data, 311)
init_mcs.DataGen(base_dir+fname_data)

[training_measurements, training_gt, cv_measurements, cv_gt, test_measurements, test_gt]= init_mcs.DataLoader(base_dir+fname_data)

#######################################################
################# MODEL h known R #####################
#######################################################
hidden_size_h = 256
learning_rate_h = 1e-3
num_epochs_h = 600
batch_size_h = 160
input_size_h = training_gt.shape[1]
output_size_h = training_measurements.shape[1]

(training_input_h, training_target_h), (cv_input_h, cv_target_h), (test_input_h, test_target_h) = map(preprocess, [training_gt, cv_gt, test_gt], [training_measurements, cv_measurements, test_measurements])

model = ModelH(
    input_size_h, 
    hidden_size_h, 
    output_size_h, 
    learning_rate_h
    ) #, torch.tensor(R))

#uncomment line below to train
#model.train(training_input_h, training_target_h, num_epochs_h, batch_size_h, base_dir)

# Load pretrained model back in
modelH = model.load_model(base_dir + 'model_checkpoint.pth')
print('modelH loaded successfully:', modelH)

#########################################################
################# MODEL h UNKNOWN R #####################
#########################################################

hidden_size_hu = 256
learning_rate_hu = 1e-3
num_epochs_hu = 800
batch_size_hu = 160
input_size_hu = training_gt.shape[1]
output_size_hu = training_measurements.shape[1]

# Preprocess data
(training_input_hu, training_target_hu), (cv_input_hu, cv_target_hu), (test_input_hu, test_target_hu) = map(
    preprocess_unknown, 
    [training_gt, cv_gt, test_gt], 
    [training_measurements, cv_measurements, test_measurements]
)

# Initialize the model
state_dim = training_gt.shape[1]
measurement_dim = training_measurements.shape[1]
R_0 = torch.diag(torch.tensor([1, 1]))  #torch.eye(measurement_dim).to(dev)

modelHU = ModelHU(
    input_size=input_size_hu,
    hidden_size=hidden_size_hu,
    output_size=output_size_hu,
    learning_rate=learning_rate_hu,
    R_initial=R_0
)

#uncomment line below to train
#modelHU.train(training_input_hu, training_target_hu, num_epochs=num_epochs_hu, batch_size=batch_size_hu, base_dir=base_dir)

# Load the best saved model
checkpoint_path_h = os.path.join(base_dir, 'model_h_unknown_R_best.pth')
best_model_data = modelHU.load_best_model(checkpoint_path_h)

# Assign best parameters
modelHU.model = best_model_data['model']
R_learned = best_model_data['R']

print(f"Successfully loaded best learned R:\n{R_learned.cpu().numpy()}")

########################################################
################# MODEL F KNOWN QR #####################
########################################################
hidden_size_F = 600
learning_rate_F = 1e-4
num_epochs_F = 150
batch_size_F = 32
input_size_F = training_gt.shape[1]
output_size_F = training_gt.shape[1]

(training_input_F, training_target_F), (cv_input_F, cv_target_F), (test_input_F, test_target_F) = map(dynamic_preprocess, [training_gt, cv_gt, test_gt])

modelF = DynamicModel(input_size_F, hidden_size_F, output_size_F, learning_rate_F, torch.tensor(Q))
#modelF.train(training_input_F, training_target_F,  num_epochs_F, batch_size_F, base_dir) #, 10)

modelF = modelF.load_model(base_dir + 'dynamic_model_F.pth')

########################################################
################# MODEL F UNKNOWN QR #####################
########################################################
hidden_size_Fu = 600
learning_rate_Fu = 1e-4
num_epochs_Fu = 150
batch_size_Fu = 32
input_size_Fu = training_gt.shape[1]
output_size_Fu = training_gt.shape[1]

state_dim = 4  # (un hard code this soon) 
Q_0 = torch.eye(state_dim).to(dev)  # Q_0 starts as identity matrix
A_0 = torch.linalg.cholesky(torch.inverse(Q_0))  # Cholesky decomposition of Q_0^{-1}

(training_input_Fu, training_target_Fu), (cv_input_Fu, cv_target_Fu), (test_input_Fu, test_target_Fu) = map(dynamic_preprocess_unknown, [training_gt, cv_gt, test_gt])

modelFU = DynamicModelUnknown(input_size_Fu, hidden_size_Fu, output_size_Fu, learning_rate_Fu, Q_0)
#modelFU.train(training_input_Fu, training_target_Fu,  num_epochs_Fu, batch_size_Fu, base_dir, 10)

# Load model and Q from the saved file
checkpoint_path = base_dir + 'dynamic_model_F_unknown.pth'
checkpoint = torch.load(checkpoint_path, map_location=dev)  # Load the checkpoint dictionary

# Load the model
modelFU.model.load_state_dict(checkpoint['model_state_dict'])
modelFU.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
modelFU.model.eval()
print(f"Model loaded from {checkpoint_path}")

# Load the learned Q matrix
Q_learned = checkpoint.get('Q', None)  
if Q_learned is None:
    raise ValueError("Q matrix not found in the saved model file.")
Q_learned = Q_learned #.cpu().numpy()  # Convert to numpy for use in UKF
print(f"Learned Q matrix loaded: \n{Q_learned}")

###################################################
################ Q and R estimator ################
###################################################
R_est, Q_est = generate_estimates_max_lk(training_measurements, training_gt, F, init_mcs.h)
#print(f"R_est = {R_est}, Q_est = {Q_est}")

################################################
#################### U K F #####################
################################################

########################################
# LOAD / PREPARE DATA
########################################

# Assume you already have:
# test_measurements: (n_sequences, measurement_dim=2, n_steps)
# test_gt: (n_sequences, state_dim=4, n_steps)
# mean_ini, P_ini, Q, R, Q_learned, R_learned, Q_est, R_est, etc.

# Convert ground truth to a NumPy array for plotting and MSE:
ground_truth_np = test_gt.cpu().numpy().transpose(0, 2, 1)  # (n_sequences, n_steps, state_dim=4)

##############################################################
# (Unknown f,h,Q,R)
##############################################################
print("=== Running UKF2 with learned models (f,h) and unknown Q,R ===")

ukf2_unknown_fhQR = UKF2.UnscentedKalmanFilter(
    mean_ini,
    P_ini,
    Q_learned,
    R_learned,
    modelFU.model,  # learned f
    modelHU         # learned h wrapper (if that's how your code sets it up)
)

state_histories_ukf2_unknown_fhQR = run_ukf(
    ukf2_unknown_fhQR,
    test_measurements,
    test_gt,
    mean_ini, P_ini,
    dev
)

# Save results
ukf2_unknown_path = os.path.join(base_dir, "unknown_fhQR_state_estimates.npy")
np.save(ukf2_unknown_path, state_histories_ukf2_unknown_fhQR)
print(f"Saved UKF2 with unknown fhQR estimates to: {ukf2_unknown_path}")

##############################################################
# known f,h,Q,R 
##############################################################
print("=== Running UKF3 with the true measurement function (known f,h,Q,R) ===")

ukf3_known_fhQR = UKF3.UnscentedKalmanFilter(
    mean_ini, P_ini,
    Q, R,     # True Q, R
    F,        # True dynamic function
    init_mcs.h  # True measurement function
)

state_histories_ukf3_known_fhQR = run_ukf(
    ukf3_known_fhQR,
    test_measurements,
    test_gt,
    mean_ini, P_ini,
    dev
)

# Save results
ukf3_known_path = os.path.join(base_dir, "known_fhQR_state_estimates.npy")
np.save(ukf3_known_path, state_histories_ukf3_known_fhQR)
print(f"Saved UKF3 with known fhQR estimates to: {ukf3_known_path}")

# Example plot for a single sequence (optional)
seq_index = 13
ukf3_known_fhQR.plot_state_estimate(state_histories_ukf3_known_fhQR[seq_index], ground_truth_np[seq_index])
ukf3_known_fhQR.plot_positions(state_histories_ukf3_known_fhQR[seq_index], ground_truth_np[seq_index])

##############################################################
# 3) UKF2 with unknown f,h but known Q,R
##############################################################
print("=== Running UKF2 with unknown f,h but known Q,R ===")

ukf2_unknown_fh_known_QR = UKF2.UnscentedKalmanFilter(
    mean_ini, P_ini,
    Q, R,       # known Q,R
    modelF,     # learned/approximated f
    modelH      # learned/approximated h
)

state_histories_ukf2_unknown_fh_known_QR = run_ukf(
    ukf2_unknown_fh_known_QR,
    test_measurements,
    test_gt,
    mean_ini, P_ini,
    dev
)

# Save results
ukf2_fh_known_QR_path = os.path.join(base_dir, "unknown_fh_known_QR_state_ests.npy")
np.save(ukf2_fh_known_QR_path, state_histories_ukf2_unknown_fh_known_QR)
print(f"Saved UKF2 with unknown f,h but known Q,R estimates to: {ukf2_fh_known_QR_path}")

##############################################################
# 4) UKF3 with known f,h but unknown Q,R
##############################################################
print("=== Running UKF3 with known f,h and unknown Q,R ===")

ukf3_known_fh_unknown_QR = UKF3.UnscentedKalmanFilter(
    mean_ini, P_ini,
    Q_est, R_est,  # unknown Q,R (estimated)
    F, init_mcs.h  # known f,h
)

state_histories_ukf3_known_fh_unknown_QR = run_ukf(
    ukf3_known_fh_unknown_QR,
    test_measurements,
    test_gt,
    mean_ini, P_ini,
    dev
)

# Save results
ukf3_fh_unknown_QR_path = os.path.join(base_dir, "known_fh_unknown_QR_state_est.npy")
np.save(ukf3_fh_unknown_QR_path, state_histories_ukf3_known_fh_unknown_QR)
print(f"Saved UKF3 with known f,h but unknown Q,R estimates to: {ukf3_fh_unknown_QR_path}")
'''
##############################################################
# LOAD / COMPARE TO KNet (Optional)
##############################################################
file_path_to_load = f'C:/Users/betti/Desktop/two_radar_scenario/KNetFiles_{n_steps}/q{sigma_u}_r{sigma_r}_T{n_steps}/MCSim_test_data_{n_steps}_KNetTraj.npy'
try:
    knet_data_init = np.load(file_path_to_load)
    print(f"KNet data successfully loaded from {file_path_to_load}")
except Exception as e:
    print(f"Error loading KNet trajectory data: {e}")
    raise

# Transpose KNet data from (n_sequences, state_dim, n_steps) to (n_sequences, n_steps, state_dim)
knet_data = knet_data_init.transpose(0, 2, 1)
print('KNet trajectory data shape:', knet_data.shape)
'''
##############################################################
# RMSE CALCULATIONS
##############################################################
print("\n=== RMSE Computations ===")

rmse_ukf2_unknown_fhQR          = compute_rmse(ground_truth_np, state_histories_ukf2_unknown_fhQR,          "UKF2 Unknown f,h,Q,R")
rmse_ukf3_known_fhQR            = compute_rmse(ground_truth_np, state_histories_ukf3_known_fhQR,            "UKF3 Known f,h,Q,R")
rmse_ukf2_unknown_fh_known_QR   = compute_rmse(ground_truth_np, state_histories_ukf2_unknown_fh_known_QR,   "UKF2 Unknown f,h, known Q,R")
rmse_ukf3_known_fh_unknown_QR   = compute_rmse(ground_truth_np, state_histories_ukf3_known_fh_unknown_QR,   "UKF3 Known f,h, unknown Q,R")
#rmse_knet                       = compute_rmse(ground_truth_np, knet_data,                                  "KalmanNet")

# Plot RMSE Over Time
timesteps = np.arange(rmse_ukf2_unknown_fhQR.shape[0])

plt.figure(figsize=(10, 6))
plt.plot(timesteps, rmse_ukf3_known_fhQR,            label='Known f,h,Q,R',                marker='x', color='forestgreen')
plt.plot(timesteps, rmse_ukf3_known_fh_unknown_QR,   label='Known f,h, unknown Q,R',       marker='v', color='purple')
plt.plot(timesteps, rmse_ukf2_unknown_fh_known_QR,   label='Unknown f,h, known Q,R',       marker='^', color='royalblue')
plt.plot(timesteps, rmse_ukf2_unknown_fhQR,          label='Unknown f,h,Q,R',             marker='o', color='darkorange')
#plt.plot(timesteps, rmse_knet,                       label='KalmanNet (Known f,h, unknown Q,R)', marker='s', color='crimson')

# Increase font size for axis labels
plt.xlabel('Timestep', fontsize=16)
plt.ylabel('RMSE', fontsize=16)

# Increase font size for legend and grid
plt.legend(fontsize=14)
plt.grid(True)

plot_save_path = os.path.join(base_dir, f'q{sigma_u}_r{sigma_r}_{n_steps}_step_fig.eps')
plt.savefig(plot_save_path, format='eps')
plt.show()
print(f"RMSE plot saved to {plot_save_path}")

