# -*- coding: utf-8 -*-
"""
System model
"""
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class SystemModel:

    def __init__(self, F, Q, h, r, T, T_test, outlier_p=0,rayleigh_sigma=10000):

        self.outlier_p = outlier_p
        self.rayleigh_sigma = rayleigh_sigma
        ####################
        ### Motion Model ###
        ####################       
        self.F = F
        self.m = self.F.size()[0]

        self.Q = Q

        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = 2    #bert

        self.r = r
        self.R = r * r * torch.eye(self.n)

        #Assign T and T_test
        self.T = T
        self.T_test = T_test
        
        self.trajGen = None
        
    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        
        
        if self.trajGen is not None:
            self.trajGen.generateSequenceTorch()
            traj = self.trajGen.getTrajectoryArraysTorch()
            self.x = traj["X_true"]
            self.y = traj["measurements"]
            self.x_prev = self.m1x_0 #need to update this if we want continuation sequences
            
            return
        
    def SetTrajectoryGenerator(
            self,
            trajGen
            ) -> None:
        self.trajGen = trajGen
    
   ######################
   ### Generate Batch ###
   ######################
    
    def GenerateBatch(self, size, T, randomInit=False, seqInit=False, T_test=0):
    
        print(' T in generate batch', T)
        # Allocate Empty Array for Input
        self.meas = torch.empty(size, self.n, T)
        # Allocate Empty Array for Target
        self.gt = torch.empty(size, self.m, T)
    
        ### Generate Examples
        initConditions = self.m1x_0
    
        for i in range(0, size):
            # Generate Sequence
    
            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
            if(seqInit):
                initConditions = self.x_prev
                if((i*T % T_test)==0):
                    initConditions = torch.zeros_like(self.m1x_0)
    
            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)
    
            # Training sequence input for h
            self.gt[i, :, :] = self.x
            
            # Training sequence output
            self.meas[i, :, :] = self.y
    
    
    
         
    
        
