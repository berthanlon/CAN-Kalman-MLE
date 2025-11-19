# -*- coding: utf-8 -*-
"""
Unknown H mc_sim class
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from system_model import SystemModel
from parameters import F, Q, chol_Q, Q_inv, R_inv, R, chol_R, mean_ini, P_ini, chol_ini, N_E, N_CV, N_T, n_steps 

class MonteCarloSimulation:
    def __init__(
            self,
            nSteps: int, 
            baseDir: str,
            fileName: str,
            seed: int = 0,
            ) -> None:
        self.nSteps = nSteps
        self.X_true = np.array([])
        self.measurements = np.array([])
        self.x_k = np.array([])
        self.P_k = np.array([])
        self.kalman_est = np.array([])
        self.test_target = np.array([])
        self.two_range = np.array([])
        self.folderName = baseDir
        self.seed = seed if seed > 0 else 0
        self.mean_ini = mean_ini
        self.dataFilename = self.folderName + fileName
        self.MLETrajFilename = self.folderName + fileName + '_MLETraj.npy'
        self.KNetTrajFilename = self.folderName + fileName + '_KNetTraj.npy'
        
        today = datetime.today()
        now = datetime.now()
        strToday = today.strftime("%m.%d.%y")
        strNow = now.strftime("%H:%M:%S")
        self.strTime = strToday + "_" + strNow
        print("Current Time =", self.strTime)
        print(f'Initialised MC object, nSteps = {nSteps}, fileName = {fileName}, seed = {seed}')

    def h(self, x: np.array) -> np.array:
        """
        Measurement function that returns two range (distance) measurements
        from a state [x, xdot, y, ydot] to two radars:
          1) Radar A at (0, 0)
          2) Radar B at (200, 0)
    
        Parameters:
        ----------
        x : np.array
            The state vector, assumed to be [x, x_dot, y, y_dot].
    
        Returns:
        --------
        np.array
            A 2D array [r1, r2], where:
              r1 = range from Radar A at (0, 0)
              r2 = range from Radar B at (150, 0)
        """
        # Extract positions
        x_pos = x[0]
        y_pos = x[2]
    
        # Radar A (0, 0)
        r1 = np.sqrt(x_pos**2 + y_pos**2)
    
        # Radar B (200, 0)
        r2 = np.sqrt((x_pos - 150)**2 + y_pos**2)
    
        self.two_range = np.array([r1, r2])
        
        return self.two_range
        
   
    def generateTrajectory(
            self, 
            meanIni: np.array = None
            ) -> None:
        
        if self.seed > 0:
            np.random.seed(self.seed)
            print('Using random seed', self.seed, "reset seed to 0")
            self.seed = 0
            
        self.X_true = np.zeros((4, self.nSteps)) # state vector (x, x_vel, y, y_vel)
        self.measurements = np.zeros((2, self.nSteps))
        self.X_diff_Q_sum = np.array(np.zeros((4,4)))
    
        # Generate ground truth
        if meanIni is None:
            meanIni = self.mean_ini
  
        self.X_true[:, 0] = meanIni + np.dot(chol_ini, np.random.randn(4))
        
        # Generate measurements
        self.measurements[:, 0] = self.h(self.X_true[:, 0]) + np.dot(chol_R, np.random.randn(2))
        
        # Generate the complete trajectory and measurements
        for k in range(1, self.nSteps):
            # Propagate the true state
            
            self.X_true[:, k] = np.dot(F,  self.X_true[:, k-1]) + np.dot(chol_Q, np.random.randn(4))
            
            
            # Generate measurement (adding noise with rms = 1, see r above)
            self.measurements[:, k] = self.h(self.X_true[:, k]) + np.dot(chol_R, np.random.randn(2))
            
               
    def generateSequenceTorch(
            self
            ) -> None:
        self.generateTrajectory()
        
    def getTrajectoryArrays(
            self
            ) -> dict:
        return {
                "X_true": self.X_true, 
                "measurements": self.measurements
                }      
    
    def getTrajectoryArraysTorch(
            self
            ) -> dict:
        X_true_torch = torch.tensor(self.X_true)  #dtype = torch.float32
        measurements_torch = torch.tensor(self.measurements) #dtype = torch.float32
        return {
                "X_true": X_true_torch, 
                "measurements": measurements_torch
                }
        
    def DataGen(self, fileName, randomInit=False):
        
        sys_model = SystemModel(
            torch.tensor(F), #dtype = torch.float32
            torch.tensor(Q), #dtype = torch.float32
            self.h, #_torch, #dtype = torch.float32
            torch.tensor(R), #dtype = torch.float32
            n_steps, n_steps)
        sys_model.InitSequence(mean_ini, P_ini)
        sys_model.SetTrajectoryGenerator(self)
        
        ##################################
        ### Generate Training Sequence ###
        ##################################
        int_n_steps = int(n_steps)
        sys_model.GenerateBatch(N_E, int_n_steps, randomInit=randomInit)
        training_gt = sys_model.gt    #gt = ground truth
        training_measurements = sys_model.meas
    
        ####################################
        ### Generate Validation Sequence ###
        ####################################
        sys_model.GenerateBatch(N_CV, int_n_steps, randomInit=randomInit)
        cv_gt = sys_model.gt    #gt = ground truth
        cv_measurements = sys_model.meas
    
        ##############################
        ### Generate Test Sequence ###
        ##############################
        sys_model.GenerateBatch(N_T, int_n_steps, randomInit=randomInit)
        test_gt = sys_model.gt    #gt = ground truth
        test_measurements = sys_model.meas
    
        #################
        ### Save Data ###
        #################
        torch.save([training_measurements, training_gt, cv_measurements, cv_gt, test_measurements, test_gt,], fileName)
        print('training data saved, loc=')

    def DataLoader(self,
                   fileName):
            
        #print('Loading datafile name', fileName)
        [training_gt, training_measurements, cv_gt, cv_measurements, test_gt, test_measurements] = torch.load(fileName)
        #print('training_input', type(training_input))
        return [training_gt, training_measurements, cv_gt, cv_measurements, test_gt, test_measurements]
    

