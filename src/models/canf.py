import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from sklearn.mixture import GaussianMixture

from src.utils import train_nf
from src.models.nf.flows import *
from src.models.nf.models import *
from src.models.cgmm import ConditionalGMM

class ConditionalANF(nn.Module):
    """ 

    Class for prediction via Conditional Approximate Normalizing Flow. 
    
    """     
    def __init__(
        self, 
        input_dim, input_horizon, output_dim, prediction_horizon, 
        hidden_dim=8, n_flows = 4,            
        n_components=2, random_state=None, # CGMM parameters
        ):
        """ 

        Initializes CANF model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            input_horizon (int): the input horizon T
            output_dim (int): the output dimension
            prediction_horizon (int): the prediction horizon K

            n_components (int): the number of mixture components for CGMM
            random_state (int): default random_state
        """ 
        super(ConditionalANF, self).__init__()
        self.input_dim = input_dim
        self.T = input_horizon
        self.output_dim = output_dim
        self.K = prediction_horizon

        self.hidden_dim = hidden_dim
        self.n_flows = n_flows
        self.nf_dims = input_dim*input_horizon+output_dim*prediction_horizon
        flows = [RealNVP(dim=self.nf_dims, 
                        hidden_dim=hidden_dim, 
                        base_network=FCNN) for _ in range(n_flows)]
        prior = D.MultivariateNormal(torch.zeros(self.nf_dims),
                                torch.eye(self.nf_dims))
        self.nf = NormalizingFlowModel(prior, flows, random_state=random_state)
        self.cgmm = ConditionalGMM(input_dim, input_horizon, output_dim, prediction_horizon, 
            n_components=n_components, random_state=random_state)
        
    def forward(self, y, u=None, K=None):
        """ 

        Run a forward pass of data stream x through the model to predict distribution over next K observations.
        Args:
            y (torch.tensor): (B, T, ydim) observations
            u (torch.tensor or None): (B, T+K, udim) inputs
            K (int): horizon to predict 
        Returns:
            dist (torch.Distribution): (B,K*ydim) predictive distribution over next K observations
        """
        return self.cgmm(y, u=u, K=K)
   
    def fit(self, dataset, u=None, n_samples:int=100000, **train_kwargs):
        """ 

        Fit model given data of past and future observations
        
        Args:
            dataset: dataset to pass along to train_nf function
            u (torch.tensor or None): (B, T+K, udim) inputs corresponding with past observations
            n_samples (int): number of samples from nf to fit anf
        """
        
        # train nf
        train_nf(self.nf, dataset, **train_kwargs)
        
        # generate samples
        self.nf.eval()
        samples = self.nf.sample(n_samples)

        # fit cgmmself.input_dim = input_dim
        self.cgmm.fit(samples[:,:self.T*self.input_dim].reshape((-1, self.T, self.input_dim)), 
            samples[:,self.T*self.input_dim:].reshape((-1, self.K, self.output_dim)), u=u)

        
    def log_prob(self, x, y):
        """ 

        Evaluate log pdf of input-output pairs under ANF
        
        Args:
            x (torch.tensor): (B, T, ydim) past observations
            y (torch.tensor): (B, K, ydim) future observations
            
        Returns:
            log_prob (torch.tensor): (B,) log pdf under joint distribution
        """
        return self.cgmm.log_prob(x, y)
    
    def nf_log_prob(self, x, y):
        """ 

        Evaluate log pdf of input-output pairs under NF
        
        Args:
            x (torch.tensor): (B, T, ydim) past observations
            y (torch.tensor): (B, K, ydim) future observations
            
        Returns:
            log_prob (torch.tensor): (B,) log pdf under joint distribution
        """
        return self.nf.log_prob(x, y)
