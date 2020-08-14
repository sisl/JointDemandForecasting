import torch
import torch.nn as nn
import torch.distributions as D
from CalibratedTimeseriesModels.abstractmodels import *
import numpy as np
from sklearn.mixture import GaussianMixture

class ConditionalGMM(ExplicitPredictiveModel):
    """ 

    Class for prediction via Conditional GMM. 
    
    """     
    def __init__(self, input_dim, input_horizon, output_dim, prediction_horizon, 
                 n_components=2, random_state=None):
        """ 

        Initializes autoregressive, probabilistic feedforward neural network model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            input_horizon (int): the input horizon T
            output_dim (int): the output dimension
            prediction_horizon (int): the prediction horizon K
            n_components (int): the number of mixture components
            random_state (int): default random_state
        """ 
        super(ConditionalGMM, self).__init__()
        self.input_dim = input_dim
        self.T = input_horizon
        self.output_dim = output_dim
        self.K = prediction_horizon
        
        self.n_components = n_components
        self.random_state = random_state
        
        self.pi_ = None
        self.mu_ = None
        self.var_ = None
        
        self.input_gaussians = None
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
        if self.pi_ is None or self.mu_ is None or self.var_ is None:
            raise NameError('Yet to fit model.')
        
        B, T, ydim = y.shape
        X = y.reshape((B,self.T*self.input_dim))
        
        ps = torch.stack([dist.log_prob(X).exp() for dist in self.input_gaussians],1)
        probs = self.pi_.unsqueeze(0) * ps #(B, k)
        probs = probs / probs.sum(1).unsqueeze(1)
        mix = D.Categorical(probs)
        
        mu = torch.matmul(self.theta.unsqueeze(0),
                         X.unsqueeze(1).unsqueeze(-1)).squeeze(-1) + self.b.unsqueeze(0).expand(B,-1,-1) # (B, k, d) 
        L = self.Sigma_chol.unsqueeze(0).expand(B,-1,-1,-1) # (B, k, d, d)
        comp = D.MultivariateNormal(loc=mu, scale_tril=L)
        dist = D.MixtureSameFamily(mix, comp)
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
        ind = self.T*self.input_dim
        outd = self.K*self.output_dim
        X = y.reshape((B,ind))
        Y = y_future.reshape((B,outd))
        
        gmm = GaussianMixture(n_components = self.n_components, random_state = self.random_state)
        gmm.fit(torch.cat((X,Y), 1).numpy())
        
        self.pi_ = torch.tensor(gmm.weights_).float()
        self.mu_ = torch.tensor(gmm.means_).float()
        self.var_ = torch.tensor(gmm.covariances_).float()
        
        
        
        self.theta = torch.matmul(self.var_[:,ind:,:ind], self.var_[:,:ind,:ind].inverse())
        self.b = self.mu_[:,ind:] - torch.matmul(self.theta, self.mu_[:,:ind].unsqueeze(-1)).squeeze(-1)
        self.input_gaussians = [D.MultivariateNormal(self.mu_[i,:ind], 
                                                     self.var_[i, :ind, :ind]) for i in range(self.n_components)]
        self.Sigma_chol = (self.var_[:,ind:,ind:] - torch.matmul(self.theta, self.var_[:,:ind,ind:])).cholesky()