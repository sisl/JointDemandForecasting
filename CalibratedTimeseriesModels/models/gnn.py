import torch
import torch.nn as nn
from CalibratedTimeseriesModels.abstractmodels import *
from torch.distributions.normal import Normal as normal
from torch.distributions.multivariate_normal import MultivariateNormal as mvnormal
import numpy as np

class GaussianNeuralNet(ExplicitPredictiveModel):
    """ 

    Class for probabilistic feedforward neural network using single MvGaussian. 
    
    """     
    def __init__(self, input_dim, input_horizon, hidden_layer_dims, output_dim, prediction_horizon, full_covariance=False):
        """ 

        Initializes autoregressive, probabilistic feedforward neural network model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            input_horizon (int): the input horizon T
            hidden_layer_dims (list of ints): the hidden layer sizes in the neural network
            output_dim (int): the output dimension
            prediction_horizon (int): the prediction horizon K
            full_covariance (bool): if true predict a full covariance matrix, otherwise a diagonal one
        """ 
        super(GaussianNeuralNet, self).__init__()
        self.input_dim = input_dim
        self.T = input_horizon
        self.hidden_layer_dims = hidden_layer_dims
        self.output_dim = output_dim
        self.K = prediction_horizon
        self.full_covariance = full_covariance
        
        fc_net = []
        fc_sizes = np.append(self.input_dim * self.T, self.hidden_layer_dims)
        for i in range(len(fc_sizes)-1):
            fc_net.append(nn.Linear(in_features=fc_sizes[i], out_features=fc_sizes[i+1]))
            fc_net.append(nn.LeakyReLU())
        
        
        self._num_means = self.output_dim*self.K
        if self.full_covariance:
            n = self.output_dim*self.K
            self._num_cov = int(n*(n+1)/2)
        else:
            self._num_cov = self.output_dim*self.K

        fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=self._num_means+self._num_cov))
        self.fc = nn.Sequential(*fc_net)
        
    def forward(self, y, u=None, K=None):
        """ 

        Run a forward pass of data stream x through the neural network to predict distribution over next K observations.
        Args:
            y (torch.tensor): (B, T, ydim) observations
            u (torch.tensor or None): (B, T+K, udim) inputs
            K (int): horizon to predict 
        Returns:
            dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
        """
        
        B, T, ydim = y.shape
        inputs = y.reshape((B, T*ydim))
        outputs = self.fc(inputs)
        B, outdims = outputs.shape
        mu = outputs[:,:self._num_means]
        
        # full covariance matrix
        if self.full_covariance:
            diag = nn.functional.softplus(outputs[:,self._num_means:2*self._num_means])
            offdiag = outputs[:,2*self._num_means:]
            
            L = torch.zeros(B,self._num_means,self._num_means)
            indices = torch.tril_indices(self._num_means, self._num_means, -1)
            L[:,torch.arange(self._num_means),torch.arange(self._num_means)] = diag
            L[:,indices[0], indices[1]] = offdiag
            
            dist = mvnormal(loc=mu, scale_tril=L)
        
        # isotropic normal distribution
        else:
            sig = nn.functional.softplus(outputs[:,self._num_means:])
            dist = normal(loc=mu, scale=sig)
        return dist
    
class GaussianLSTM(ExplicitPredictiveModel):
    """ 

    Class for sequence-to-sequence probabilistic LSTM using single isotropic MvGaussian. 
    
    """ 
    def __init__(self, input_dim, hidden_dim, fc_hidden_layer_dims, output_dim, prediction_horizon,
                 full_covariance=False, num_layers=1, dropout=0.0, bidirectional=False, random_start=True):
        """ 

        Initializes sequence-to-sequence LSTM model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            hidden_dim (int): number of hidden/cell dimensions
            fc_hidden_layer_dims (list of ints): the hidden layer sizes in the neural network mapping from hidden state to output
            output_dim (int): the dimension of the outputs at each point in the sequence
            prediction_horizon (int): the prediction horizon K
            full_covariance (bool): if true predict a full covariance matrix, otherwise a diagonal one
            num_layers (int): number of layers in a possibly stacked LSTM
            dropout (float): the dropout rate of the lstm
            bidirectional (bool): whether to initialize a bidirectional lstm
            random_start (bool): If true, will initialize the hidden states randomly from a unit Gaussian
        """ 
        super(GaussianLSTM, self).__init__()
        
        # dimensional parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc_hidden_layer_dims = fc_hidden_layer_dims
        self.output_dim = output_dim
        self.K = prediction_horizon
        self.full_covariance = full_covariance
        
        # LSTM parameters
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.random_start = random_start
            
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, batch_first=True,
                          dropout=self.dropout, bidirectional=self.bidirectional)
        
        fc_net = []
        fc_sizes = np.append(hidden_dims, self.fc_hidden_layer_dims)
        for i in range(len(fc_sizes)-1):
            fc_net.append(nn.Linear(in_features=fc_sizes[i], out_features=fc_sizes[i+1]))
            fc_net.append(nn.LeakyReLU())
        
        self._num_means = self.output_dim*self.K
        if self.full_covariance:
            n = self.output_dim*self.K
            self._num_cov = int(n*(n+1)/2)
        else:
            self._num_cov = self.output_dim*self.K
            
        fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=self._num_mean+self._num_cov))
        self.fc = nn.Sequential(*fc_net)
        
    def forward(self, y, u=None, K=None):
        """ 

        Run a forward pass of data stream x through the neural network to predict distribution over next K observations.
        Args:
            y (torch.tensor): (B, T, ydim) observations
            u (torch.tensor or None): (B, T+K, udim) inputs
            K (int): horizon to predict 
        Returns:
            dist (PredictiveDistribution): (B, K*ydim) predictive distribution over next K observations
        """
        
        h_0, c_0 = self.initialize_lstm(x)    
        output_lstm, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # output_lstm has shape (B, T, hidden_dim)
        
        # calculate predictions based on final hidden state
        dist = self.forward_fc(output_lstm[:,-1,:])
        
        return dist
    
    def initialize_lstm(self, x):
        """ 

        Initialize the lstm either randomly or with zeros. 

        Args: 

            x (torch tensor): (batch_size, sequence_length, input_features) tensor of inputs to the lstm. 

        Returns: 

            h_0 (torch tensor): (num_layers*num_directions, batch_size, hidden_size) tensor for initial hidden state
            c_0 (torch tensor): (num_layers*num_directions, batch_size, hidden_size) tensor for initial cell state 

        """ 
        batch_index = 0
        num_direction = 2 if self.bidirectional else 1

        # Hidden state in first seq of the LSTM - use noisy state initialization if random_start is True
        if self.random_start:
            h_0 = torch.randn(self.num_layers * num_direction, x.size(batch_index), self.hidden_size)
            c_0 = torch.randn(self.num_layers * num_direction, x.size(batch_index), self.hidden_size)
        else:
            h_0 = torch.zeros(self.num_layers * num_direction, x.size(batch_index), self.hidden_size)
            c_0 = torch.zeros(self.num_layers * num_direction, x.size(batch_index), self.hidden_size)
        return h_0, c_0
            
    def forward_fc(self, h_n):
        """ 

        Run the hidden states through the forward layer to obtain outputs. 

        Args: 

            h_n (torch tensor): (batch_size, hidden_features) tensor of hidden state values at 
                each point in each sequence. 

        Returns: 
            dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations shaped

        """ 
        outputs = self.fc(h_n)
        # outputs will be (B, 2*K*ydims)
        B, outdims = fc_output.shape
        mu = outputs[:,:self._num_means]
        
        # full covariance matrix
        if self.full_covariance:
            diag = nn.functional.softplus(outputs[:,self._num_means:2*self._num_means])
            offdiag = outputs[:,2*self._num_means:]
            
            L = torch.zeros(B,self._num_means,self._num_means)
            indices = torch.tril_indices(self._num_means, self._num_means, -1)
            L[:,torch.arange(self._num_means),torch.arange(self._num_means)] = diag
            L[:,indices[0], indices[1]] = offdiag
            
            dist = mvnormal(loc=mu, scale_tril=L)
        
        # isotropic normal distribution
        else:
            sig = nn.functional.softplus(outputs[:,self._num_means:])
            dist = normal(loc=mu, scale=sig)
        return dist
        