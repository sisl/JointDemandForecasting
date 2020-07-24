import torch
import torch.nn as nn
import torch.distributions as D
from CalibratedTimeseriesModels.abstractmodels import *
import numpy as np

class GaussianNeuralNet(ExplicitPredictiveModel):
    """ 

    Class for probabilistic feedforward neural network using single MvGaussian. 
    
    """     
    def __init__(self, input_dim, input_horizon, hidden_layer_dims, output_dim, prediction_horizon, 
                 covariance_type='diagonal', rank=2, bands=2, dropout=0.0):
        """ 

        Initializes autoregressive, probabilistic feedforward neural network model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            input_horizon (int): the input horizon T
            hidden_layer_dims (list of ints): the hidden layer sizes in the neural network
            output_dim (int): the output dimension
            prediction_horizon (int): the prediction horizon K
            covariance_type (string): 'diagonal', 'full', 'low-rank', or 'banded'
            rank (int): rank of low-rank covariance matrix
            bands (int): number of off-diagonal bands in banded covariance matrix
            dropout (float): dropout probability
        """ 
        super(GaussianNeuralNet, self).__init__()
        self.input_dim = input_dim
        self.T = input_horizon
        self.hidden_layer_dims = hidden_layer_dims
        self.output_dim = output_dim
        self.K = prediction_horizon
        self.covariance_type = covariance_type
        self.rank = rank
        self.bands = bands
        self.dropout = dropout
        
        fc_net = []
        fc_sizes = np.append(self.input_dim * self.T, self.hidden_layer_dims)
        for i in range(len(fc_sizes)-1):
            fc_net.append(nn.Dropout(p=self.dropout))
            fc_net.append(nn.Linear(in_features=fc_sizes[i], out_features=fc_sizes[i+1]))
            fc_net.append(nn.LeakyReLU())
        
        
        n = self.output_dim*self.K
        self._num_means = n
        if self.covariance_type == 'full':
            self._num_cov = int(n*(n+1)/2)
        elif self.covariance_type == 'diagonal':
            self._num_cov = self.output_dim*self.K
        elif self.covariance_type == 'low-rank':
            self._num_cov = self.output_dim*self.K*(self.rank+1)
        elif self.covariance_type == 'banded':
            if self.bands < 1 or self.bands > self.output_dim*self.K-1:
                raise("Invalid number of bands")
            self._num_cov = 0
            for i in range(self.bands+1):
                self._num_cov += n-i
            
            # determine band indices
            indices = torch.tril_indices(self._num_means, self._num_means, -1)
            good_cols = [i for i in range(indices.shape[1]) if abs(indices[0,i]-indices[1,i]) <= self.bands]
            self._band_indices = indices[:,good_cols]          
        else:
            raise("Invalid covariance type %s." %(self.covariance_type))

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
        if self.covariance_type == 'full':
            diag = torch.exp(outputs[:,self._num_means:2*self._num_means])
            offdiag = outputs[:,2*self._num_means:]
            
            L = torch.zeros(B,self._num_means,self._num_means)
            indices = torch.tril_indices(self._num_means, self._num_means, -1)
            L[:,torch.arange(self._num_means),torch.arange(self._num_means)] = diag
            L[:,indices[0], indices[1]] = offdiag
            
            dist = D.MultivariateNormal(loc=mu, scale_tril=L)
        
        # isotropic normal distribution
        elif self.covariance_type == 'diagonal':
            sig = torch.exp(outputs[:,self._num_means:])
            dist = D.Normal(loc=mu, scale=sig)
            
        # low-rank covariance matrix
        elif self.covariance_type == 'low-rank':
            diag = torch.exp(outputs[:,self._num_means:2*self._num_means])
            factor = outputs[:,2*self._num_means:].reshape(B, self._num_means, self.rank)
            dist = D.LowRankMultivariateNormal(loc=mu, cov_factor=factor, cov_diag=diag)
        
        # banded covariance matrix
        elif self.covariance_type == 'banded':
            diag = torch.exp(outputs[:,self._num_means:2*self._num_means])
            offdiag = outputs[:,2*self._num_means:]

            L = torch.zeros(B,self._num_means,self._num_means)
            L[:,torch.arange(self._num_means),torch.arange(self._num_means)] = diag
            L[:,self._band_indices[0], self._band_indices[1]] = offdiag
            dist = D.MultivariateNormal(loc=mu, scale_tril=L)
        return dist
    
class GaussianLSTM(ExplicitPredictiveModel):
    """ 

    Class for sequence-to-sequence probabilistic LSTM using single MvGaussian. 
    
    """ 
    def __init__(self, input_dim, hidden_dim, fc_hidden_layer_dims, output_dim, prediction_horizon,
                 covariance_type='diagonal', rank=2, bands=2,
                 num_layers=1, dropout=0.0, bidirectional=False, random_start=True):
        """ 

        Initializes sequence-to-sequence LSTM model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            hidden_dim (int): number of hidden/cell dimensions
            fc_hidden_layer_dims (list of ints): the hidden layer sizes in the neural network mapping from hidden state to output
            output_dim (int): the dimension of the outputs at each point in the sequence
            prediction_horizon (int): the prediction horizon K
            covariance_type (string): 'diagonal', 'full', 'low-rank', or 'banded'
            rank (int): rank of low-rank covariance matrix
            bands (int): number of off-diagonal bands in banded covariance matrix
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
        self.covariance_type = covariance_type
        self.rank = rank
        self.bands = bands
              
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
        
        n = self.output_dim*self.K
        self._num_means = n
        if self.covariance_type == 'full':
            self._num_cov = int(n*(n+1)/2)
        elif self.covariance_type == 'diagonal':
            self._num_cov = self.output_dim*self.K
        elif self.covariance_type == 'low-rank':
            self._num_cov = self.output_dim*self.K*(self.rank+1)        
        elif self.covariance_type == 'banded':
            if self.bands < 1 or self.bands > self.output_dim*self.K-1:
                raise("Invalid number of bands")
            self._num_cov = 0
            for i in range(self.bands+1):
                self._num_cov += n-i 
        else:
            raise("Invalid covariance type %s." %(self.covariance_type))
            
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
        if self.covariance_type:
            diag = torch.exp(outputs[:,self._num_means:2*self._num_means])
            offdiag = outputs[:,2*self._num_means:]
            
            L = torch.zeros(B,self._num_means,self._num_means)
            indices = torch.tril_indices(self._num_means, self._num_means, -1)
            L[:,torch.arange(self._num_means),torch.arange(self._num_means)] = diag
            L[:,indices[0], indices[1]] = offdiag
            
            dist = D.MultivariateNormal(loc=mu, scale_tril=L)
        
        # isotropic normal distribution
        elif self.covariance_type == 'full':
            sig = torch.exp(outputs[:,self._num_means:])
            dist = D.Normal(loc=mu, scale=sig)
            
        # low-rank covariance matrix
        elif self.covariance_type == 'low-rank':
            diag = torch.exp(outputs[:,self._num_means:2*self._num_means])
            factor = outputs[:,2*self._num_means:].reshape(B, self._num_means, self.rank)
            dist = D.LowRankMultivariateNormal(loc=mu, cov_factor=factor, cov_diag=diag)

        # banded covariance matrix
        elif self.covariance_type == 'banded':
            diag = torch.exp(outputs[:,self._num_means:2*self._num_means])
            offdiag = outputs[:,2*self._num_means:]
            r = outdims-2*self._num_means
            
            S = torch.zeros(B,self._num_means,self._num_means)
            indices = torch.tril_indices(self._num_means, self._num_means, -1)
            S[:,torch.arange(self._num_means),torch.arange(self._num_means)] = diag
            S[:,indices[0][:r], indices[1][:r]] = offdiag
            S[:,indices[1][:r], indices[0][:r]] = offdiag
            dist = D.MultivariateNormal(loc=mu, covariance_matrix=S)
        
        return dist
        