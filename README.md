# CAN-Kalman-MLE
Code used in the two radar scenario for the CAN-Kalman MLE paper [1]. The paper received the best paper runner-up award at the IEEE 35th International Workshop on Machine Learning for Signal Processing (MLSP).

## Link to paper

[Coordinate Ascent Neural Kalman-MLE for state estimation](https://arxiv.org/pdf/2511.01855v1)

## Code configuration and running instructions
- **main_v2.py**: This is the main file to run training and testing on different relevant models. Contains toggles to control data generation/ loading as required, and training/calling for the various different models (commented throughout), as well UKF testing for all scenarios.

- **parameters.py**: Stores all experiment parameters (sequence length, dataset sizes, true 𝐹, 𝑄, 𝑅 for datagen, initial covariance, etc.)

- **new_sim.py**: Monte-Carlo simulation generating training/CV/test sequences and two-radar range measurements.

- **system_model** This file defines the SystemModel class, which sets up the state-space model and generates synthetic trajectories and measurement batches for training and testing the CAN-Kalman-MLE pipeline.
 
- **F_nn_knownQR.py**: Neural dynamic model assuming known 𝑄; MLP with Mahalanobis loss

- **F_nn.py**: This file contains the CAN-Kalman-MLE algorithm for the dynamic model. Neural dynamic model with unknown 𝑄; iteratively learning $f$ with $Q$ as detailed in 'CAN-Kalman-MLE'.

- **h_known_nn.py**: Neural measurement model assuming known 𝑅; MLP with Mahalanobis loss

- **h_nn.py**: This file contains the CAN-Kalman-MLE algorithm for the measurement model. Neural measurement model with unknown 𝑅; iteratively learning $h$ with $R$ as in 'CAN-Kalman-MLE'.

- **QR_estimator.py**: Maximum-likelihood estimator for 𝑄 and 𝑅 from prediction/measurement residuals (method as detailed in '[A comparison between Kalman-MLE and KalmanNet for state
 estimation with unknown noise parameters](https://livrepository.liverpool.ac.uk/3184477/1/A%20comparison%20between%20Kalman-MLE%20and%20KalmanNet.pdf)')

- **UKF2.py**: UKF using trained PyTorch models in place of $𝑓_𝜃$ and $ℎ_𝜃$

- **UKF3.py**: UKF with known or callable NumPy-based 𝑓 and ℎ

This is the current version of this ever evolving repository. If any specific version/ data is required, or any questions or issues arise, feel free to contact Betti at bhanlon@liverpool.ac.uk

[1] B. Hanlon and Á. F. García-Fernández, "Coordinate Ascent Neural Kalman-MLE for State Estimation," 2025 IEEE 35th International Workshop on Machine Learning for Signal Processing (MLSP), Istanbul, Turkiye, 2025, pp. 1-6, doi: 10.1109/MLSP62443.2025.11204271.
