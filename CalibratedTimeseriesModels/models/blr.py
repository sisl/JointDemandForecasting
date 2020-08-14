import torch
import torch.nn as nn
import torch.distributions as D
from CalibratedTimeseriesModels.abstractmodels import *
import numpy as np

class BayesianLinearRegression(ExplicitPredictiveModel):
    """ 

    Class for prediction via Bayesian Linear Regression. 
    
    """     
    def __init__(self, input_dim, input_horizon, output_dim, prediction_horizon):
        """ 

        Initializes autoregressive, probabilistic feedforward neural network model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            input_horizon (int): the input horizon T
            output_dim (int): the output dimension
            prediction_horizon (int): the prediction horizon K
        """ 
        super(BayesianLinearRegression, self).__init__()
        self.input_dim = input_dim
        self.T = input_horizon
        self.output_dim = output_dim
        self.K = prediction_horizon
        self.theta = None
        self.b = None
        self.Sigma_chol = None
        
    def forward(self, y, u=None, K=None):
        """ 

        Run a forward pass of data stream x through the model to predict distribution over next K observations.
        Args:
            y (torch.tensor): (B, T, ydim) observations
            u (torch.tensor or None): (B, T+K, udim) inputs
            K (int): horizon to predict 
        Returns:
            dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
        """
        if self.theta is None or self.b is None or self.Sigma_chol is None:
            raise NameError('Yet to fit model.')
        
        B, T, ydim = y.shape
        X = y.reshape((B,self.T*self.input_dim))
        mu = X @ self.theta.T + self.b.unsqueeze(0)
        L = self.Sigma_chol.unsqueeze(0).repeat(B,1,1)
        dist = D.MultivariateNormal(loc=mu, scale_tril=L)
        return dist
   
    def fit(self, y, y_future, u=None):
        """ 

        Fit model given data of past and future observations
        
        Args:
            y (torch.tensor): (B, T, ydim) past observations
            y_future (torch.tensor): (B, K, ydim) future observations conditioned on past observations
            u (torch.tensor or None): (B, T+K, udim) inputs corresponding with past observations
        """
        B, _, _ = y.shape
        
        X = y.reshape((B,self.T*self.input_dim))
        Y = y_future.reshape((B,self.K*self.output_dim))
        muX = X.mean(0)
        muY = Y.mean(0)
        SXX = X.T @ X / B - muX.unsqueeze(1) @ muX.unsqueeze(1).T
        SYY = Y.T @ Y / B - muY.unsqueeze(1) @ muY.unsqueeze(1).T
        SXY = X.T @ Y / B - muX.unsqueeze(1) @ muY.unsqueeze(1).T
        
        self.theta = SXY.T @ SXX.inverse()
        self.b = muY - self.theta @ muX
        self.Sigma_chol = (SYY - self.theta @ SXY).cholesky()