import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np

class GaussianMixtureNeuralNet(nn.Module):
    """ 

    Class for probabilistic feedforward neural network using Gaussian Mixture Model. 
    
    """     
    def __init__(self, input_dim, input_horizon, output_dim, prediction_horizon,
        hidden_layers=3,
        hidden_dims=40,
        n_components=3,
        covariance_type='diagonal', 
        rank=2, 
        bands=2, 
        tied=False, 
        dropout=0.0):
        """ 

        Initializes autoregressive, probabilistic feedforward neural network model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            input_horizon (int): the input horizon T
            output_dim (int): the output dimension
            prediction_horizon (int): the prediction horizon K
            hidden_layers (int): number of hidden layers in the neural network
            hidden_dims (int): number of hidden dims in each layer
            n_componenets (int): number of components in the mixture model
            covariance_type (string): 'diagonal', 'full', 'low-rank', or 'banded'
            rank (int): rank of low-rank covariance matrix
            bands (int): number of off-diagonal bands in banded covariance matrix
            tied (bool): if True, predict the same covariance for each component in mixture
            dropout (float): dropout probability
        """ 
        super(GaussianMixtureNeuralNet, self).__init__()
        self.input_dim = input_dim
        self.T = input_horizon
        self.hidden_layer_dims = [hidden_dims for _ in range(hidden_layers)]
        self.output_dim = output_dim
        self.K = prediction_horizon
        self.n_components = n_components
        if n_components == 1:
            raise("Number of components = 1, use GaussianNeuralNet class")
        self.covariance_type = covariance_type
        self.rank = rank
        self.bands = bands
        self.tied = tied
        if self.tied:
            raise("Tied covariances not yet implemented")
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

        fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=self.n_components*(1+self._num_means+self._num_cov)))
        self.fc = nn.Sequential(*fc_net)
        
    def forward(self, y, u=None, K=None):
        """ 

        Run a forward pass of data stream x through the neural network to predict distribution over next K observations.
        Args:
            y (torch.tensor): (B, T, ydim) observations
            u (torch.tensor or None): (B, T+K, udim) inputs
            K (int): horizon to predict 
        Returns:
            dist (torch.Distribution): (B,K*ydim) predictive distribution over next K observations
        """
        
        B, T, ydim = y.shape
        inputs = y.reshape((B, T*ydim))
        outputs = self.fc(inputs)
        B, outdims = outputs.shape
        
        probs = nn.functional.softmax(outputs[:,:self.n_components],1)
        mix = D.Categorical(probs)
        
        means_endidx = self.n_components*(self._num_means+1)
        mu = outputs[:,self.n_components:means_endidx].reshape(B, self.n_components, self._num_means)
        
        # full covariance matrix
        if self.covariance_type == 'full':
            n_diags = self.n_components*self._num_means
            diag = torch.exp(outputs[:,means_endidx:means_endidx+n_diags]).reshape(B, self.n_components, self._num_means)
            offdiag = outputs[:,means_endidx+n_diags:].reshape(B, self.n_components, -1)
            
            L = torch.zeros(B, self.n_components, self._num_means, self._num_means)
            indices = torch.tril_indices(self._num_means, self._num_means, -1)
            L[:,:,torch.arange(self._num_means),torch.arange(self._num_means)] = diag
            L[:,:,indices[0], indices[1]] = offdiag
            
            comp = D.MultivariateNormal(loc=mu, scale_tril=L)
        
        # isotropic normal distribution
        elif self.covariance_type == 'diagonal':
            sig = torch.exp(outputs[:,means_endidx:]).reshape(B, self.n_components, self._num_means)
            dist = D.Normal(loc=mu, scale=sig)
            comp = D.Independent(dist,1)
            
        # low-rank covariance matrix
        elif self.covariance_type == 'low-rank':
            n_diags = self.n_components*self._num_means
            diag = torch.exp(outputs[:,means_endidx:means_endidx+n_diags]).reshape(B, self.n_components, self._num_means)
            factor = outputs[:,means_endidx+n_diags:].reshape(B, self.n_components, self._num_means, self.rank)
            comp = D.LowRankMultivariateNormal(loc=mu, cov_factor=factor, cov_diag=diag)
        
        # banded covariance matrix
        elif self.covariance_type == 'banded':
            n_diags = self.n_components*self._num_means
            diag = torch.exp(outputs[:,means_endidx:means_endidx+n_diags]).reshape(B, self.n_components, self._num_means)
            
            offdiag = outputs[:,means_endidx+n_diags:].reshape(B, self.n_components, -1)

            L = torch.zeros(B, self.n_components, self._num_means, self._num_means)
            L[:,:,torch.arange(self._num_means),torch.arange(self._num_means)] = diag
            L[:,:,self._band_indices[0], self._band_indices[1]] = offdiag
            comp = D.MultivariateNormal(loc=mu, scale_tril=L)
        
        
        dist = D.MixtureSameFamily(mix, comp)
        return dist
    
class GaussianMixtureLSTM(nn.Module):
    """ 

    Class for sequence-to-sequence probabilistic LSTM using a Gaussian Mixture Model. 
    
    """ 
    def __init__(self, input_dim, output_dim, prediction_horizon,
        hidden_dim=20,
        fc_hidden_layers=2,
        fc_hidden_dims=20,
        n_components=3, 
        covariance_type='diagonal', 
        rank=2, 
        tied=False,
        num_layers=1, 
        dropout=0.0,
        bidirectional=False, 
        random_start=True):
        """ 

        Initializes sequence-to-sequence LSTM model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            output_dim (int): the dimension of the outputs at each point in the sequence
            prediction_horizon (int): the prediction horizon K
            hidden_dim (int): number of hidden/cell dimensions
            fc_hidden_layers (int): number of hidden layers in the decoder network
            fc_hidden_dims (int): number of hidden dims in each layer
            n_components (int): number of components in the mixture model
            covariance_type (string): 'diagonal', 'full', or 'low-rank'
            rank (int): rank of low-rank covariance matrix
            tied (bool): if True, predict the same covariance for each component in mixture
            num_layers (int): number of layers in a possibly stacked LSTM
            dropout (float): the dropout rate of the lstm
            bidirectional (bool): whether to initialize a bidirectional lstm
            random_start (bool): If true, will initialize the hidden states randomly from a unit Gaussian
        """ 
        super(GaussianMixtureLSTM, self).__init__()
        
        # dimensional parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc_hidden_layer_dims = [fc_hidden_dims for _ in range(fc_hidden_layers)]
        self.output_dim = output_dim
        self.K = prediction_horizon
        self.n_components = n_components
        if n_components == 1:
            raise("Number of components = 1, use GaussianNeuralNet class")
        self.covariance_type = covariance_type
        self.rank = rank
        self.tied = tied
        if self.tied:
            raise("Tied covariances not yet implemented")
        
        # LSTM parameters
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.random_start = random_start
            
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, batch_first=True,
                          dropout=self.dropout, bidirectional=self.bidirectional)
        
        fc_net = []
        fc_sizes = np.append(self.hidden_dim, self.fc_hidden_layer_dims)
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
        else:
            raise("Invalid covariance type %s." %(self.covariance_type))
            
        fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=self.n_components*(1+self._num_means+self._num_cov)))
        self.fc = nn.Sequential(*fc_net)
        
    def forward(self, y, u=None, K=None):
        """ 

        Run a forward pass of data stream x through the neural network to predict distribution over next K observations.
        Args:
            y (torch.tensor): (B, T, ydim) observations
            u (torch.tensor or None): (B, T+K, udim) inputs
            K (int): horizon to predict 
        Returns:
            dist (torch.Distribution): (B, K*ydim) predictive distribution over next K observations
        """
        
        h_0, c_0 = self.initialize_lstm(y)    
        output_lstm, (h_n, c_n) = self.lstm(y, (h_0, c_0))
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

            h_0 (torch tensor): (num_layers*num_directions, batch_size, hidden_dim) tensor for initial hidden state
            c_0 (torch tensor): (num_layers*num_directions, batch_size, hidden_dim) tensor for initial cell state 

        """ 
        batch_index = 0
        num_direction = 2 if self.bidirectional else 1
        device = torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")

        # Hidden state in first seq of the LSTM - use noisy state initialization if random_start is True
        if self.random_start:
            h_0 = torch.randn(self.num_layers * num_direction, x.size(batch_index), self.hidden_dim).to(device)
            c_0 = torch.randn(self.num_layers * num_direction, x.size(batch_index), self.hidden_dim).to(device)
        else:
            h_0 = torch.zeros(self.num_layers * num_direction, x.size(batch_index), self.hidden_dim).to(device)
            c_0 = torch.zeros(self.num_layers * num_direction, x.size(batch_index), self.hidden_dim).to(device)
        return h_0, c_0
            
    def forward_fc(self, h_n):
        """ 

        Run the hidden states through the forward layer to obtain outputs. 

        Args: 

            h_n (torch tensor): (batch_size, hidden_features) tensor of hidden state values at 
                each point in each sequence. 

        Returns: 
            dist (torch.Distribution): (B,K*ydim) predictive distribution over next K observations shaped

        """ 
        outputs = self.fc(h_n)
        B, outdims = outputs.shape
        
        probs = nn.functional.softmax(outputs[:,:self.n_components],1)
        mix = D.Categorical(probs)
        
        means_endidx = self.n_components*(self._num_means+1)
        mu = outputs[:,self.n_components:means_endidx].reshape(B, self.n_components, self._num_means)
        
        # full covariance matrix
        if self.covariance_type == 'full':
            n_diags = self.n_components*self._num_means
            diag = torch.exp(outputs[:,means_endidx:means_endidx+n_diags]).reshape(B, self.n_components, self._num_means)
            offdiag = outputs[:,means_endidx+n_diags:].reshape(B, self.n_components, -1)
            
            L = torch.zeros(B, self.n_components, self._num_means, self._num_means)
            indices = torch.tril_indices(self._num_means, self._num_means, -1)
            L[:,:,torch.arange(self._num_means),torch.arange(self._num_means)] = diag
            L[:,:,indices[0], indices[1]] = offdiag
            
            comp = D.MultivariateNormal(loc=mu, scale_tril=L)
        
        # isotropic normal distribution
        elif self.covariance_type == 'diagonal':
            sig = torch.exp(outputs[:,means_endidx:]).reshape(B, self.n_components, self._num_means)
            dist = D.Normal(loc=mu, scale=sig)
            comp = D.Independent(dist,1)
            
        # low-rank covariance matrix
        elif self.covariance_type == 'low-rank':
            n_diags = self.n_components*self._num_means
            diag = torch.exp(outputs[:,means_endidx:means_endidx+n_diags]).reshape(B, self.n_components, self._num_means)
            factor = outputs[:,means_endidx+n_diags:].reshape(B, self.n_components, self._num_means, self.rank)
            comp = D.LowRankMultivariateNormal(loc=mu, cov_factor=factor, cov_diag=diag)
                           
        dist = D.MixtureSameFamily(mix, comp)
        return dist
        