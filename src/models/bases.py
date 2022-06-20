import torch
import torch.nn as nn
import torch.distributions as D

class GaussianResnet(nn.Module):
    """ 

    Class for Resnet outputting to Gaussian. 
    Similar to `Short-Term Load Forecasting with Deep Residual Networks`
    
    """     
    def __init__(self, input_dim, input_horizon, output_dim, prediction_horizon, 
        hidden_layers=3, 
        hidden_dims=40,
        large_skip_every=2,
        in_out_skip=True,
        clamp_min=0.,
        clamp_max=1e7,
        sig_eps=1e-2):
        """ 

        Initializes autoregressive, probabilistic feedforward neural network model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            input_horizon (int): the input horizon T
            output_dim (int): the output dimension
            prediction_horizon (int): the prediction horizon K
            hidden_layers (int): number of hidden layers in the neural network
            hidden_dims (int): number of hidden dims in each layer
            large_skip_every (int): a large skip connection every n modules
        """ 
        super(GaussianResnet, self).__init__()
        
        # forecasting properties
        self.input_dim = input_dim
        self.T = input_horizon
        self.output_dim = output_dim
        self.K = prediction_horizon
        assert self.K==1, 'GaussianResnet only designed for single-step prediction'
        
        # resnet properties
        self.hidden_layers = hidden_layers
        self.hidden_dims = hidden_dims
        self.large_skip_every = large_skip_every
        self.in_out_skip = in_out_skip
        
        # output to diagonal gaussian
        indims = self.input_dim*self.T
        self.resnets = [nn.Sequential(torch.nn.Linear(in_features=indims, out_features=hidden_dims),
            torch.nn.SELU(),
            torch.nn.Linear(in_features=hidden_dims, out_features=indims)) for _ in range(hidden_layers)]
        assert len(self.resnets) >= 1, 'too few hidden layers'
        self._num_means = self.output_dim*self.K
        self._num_cov = self.output_dim*self.K 
        self.last_linear = nn.Linear(in_features=indims, out_features=self._num_means+self._num_cov)
        
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.sig_eps = sig_eps
        
    def forward(self, y):
        """ 

        Run a forward pass of data stream x through the neural network to predict distribution over next K observations.
        Args:
            y (torch.tensor): (B, T, ydim) observations
        Returns:
            dist (torch.Distribution): (B,K*ydim) predictive distribution over next K observations
        """
        B, T, ydim = y.shape
        start = y.reshape((B, T*ydim))
        shortcut = start
        x = start + self.resnets[0](start)
        for i in range(1,len(self.resnets)):
            x = x + self.resnets[i](x)
            if (i+1) % self.large_skip_every == 0:
                x = x + shortcut
                shortcut = x
        if self.in_out_skip:
            x = x + start
        outputs = self.last_linear(x)
        B, outdims = outputs.shape
        mu = torch.clamp(outputs[:,:self._num_means], min=self.clamp_min, max=self.clamp_max)
        sig = torch.clamp(torch.exp(outputs[:,self._num_means:]), min=self.sig_eps, max=self.clamp_max)
        dist = D.Normal(loc=mu, scale=sig)
        return dist

class QuantileResnet(nn.Module):
    """ 

    Class for Resnet outputting to quantiles. 
    Similar to `Improving Probabilistic Load Forecasting Using Quantile Regression NN with Skip Connections`
    
    """     
    def __init__(self, input_dim, input_horizon, output_dim, prediction_horizon, 
        hidden_layers=3, 
        hidden_dims=40,
        large_skip_every=2,
        in_out_skip=True,
        n_quantiles=5,
        clamp_min=0.,
        clamp_max=1e7,
        del_eps=1e-2):
        """ 

        Initializes autoregressive, probabilistic feedforward neural network model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            input_horizon (int): the input horizon T
            output_dim (int): the output dimension
            prediction_horizon (int): the prediction horizon K
            hidden_layers (int): number of hidden layers in the neural network
            hidden_dims (int): number of hidden dims in each layer
            large_skip_every (int): a large skip connection every n modules
            quantiles (int): number of quantiles to output to
        """ 
        super(QuantileResnet, self).__init__()
        
        # forecasting properties
        self.input_dim = input_dim
        self.T = input_horizon
        self.output_dim = output_dim
        self.K = prediction_horizon
        assert self.K==1, 'QuantileResnet only supporting single-step prediction'
        assert self.output_dim==1, 'QuantileResnet only supporting single dimension time-series'

        # resnet properties
        self.hidden_layers = hidden_layers
        self.hidden_dims = hidden_dims
        self.large_skip_every = large_skip_every
        self.in_out_skip = in_out_skip
        
        # output to quantiles
        indims = self.input_dim*self.T
        self.resnets = [nn.Sequential(torch.nn.Linear(in_features=indims, out_features=hidden_dims),
            torch.nn.SELU(),
            torch.nn.Linear(in_features=hidden_dims, out_features=indims)) for _ in range(hidden_layers)]
        assert len(self.resnets) >= 1, 'too few hidden layers'
        self.n_quantiles = n_quantiles
        if n_quantiles == 5:
            self.quantiles = [.01, .25, .5, .75, .99]
        elif n_quantiles == 7:
            self.quantiles = [.01, .1, .25, .5, .75, .9, .99]
        elif n_quantiles == 11:
            self.quantiles = [.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99]  
        else:
            raise NotImplementedError
        self.quantiles = torch.tensor(self.quantiles)
        self.last_linear = nn.Linear(in_features=indims, out_features=self.n_quantiles)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.del_eps = del_eps

    def forward_quantiles(self, y):
        """ 

        Run a forward pass of data stream x through the neural network to predict quantiles over the next step.
        Args:
            y (torch.tensor): (B, T, ydim) observations
        Returns:
            quantiles (torch.tensor): (B, 1, n_quantiles) predictions of quantiles
        """
        
        B, T, ydim = y.shape
        start = y.reshape((B, T*ydim))
        shortcut = start
        x = start + self.resnets[0](start)
        for i in range(1,len(self.resnets)):
            x = x + self.resnets[i](x)
            if (i+1) % self.large_skip_every == 0:
                x = x + shortcut
                shortcut = x
        if self.in_out_skip:
            x = x + start
        outputs = self.last_linear(x)
        B, outdims = outputs.shape
        quantiles = outputs[:,[0]].expand(-1, outdims)
        quantiles = torch.clamp(quantiles, min=self.clamp_min, max=self.clamp_max) 
        dels = torch.exp(outputs[:,1:])
        dels = torch.clamp(dels, min=self.del_eps, max=self.clamp_max)
        quantiles[:,1:] += torch.cumsum(dels, dim=-1) # (B, n_quantiles)
        return quantiles.unsqueeze(1)

    def forward(self, y):
        """
        Run a forward pass of data stream x through the neural network to predict distribution over the next observation.
        Args:
            y (torch.tensor): (B, T, ydim) observations
        Returns:
            dist (torch.Distribution): (B,1) predictive distribution over the next observations
        """
        output = self.forward_quantiles(y) #(B, 1, nq)
        B, _, nq = output.shape
        output = torch.transpose(output,1,2) #(B, nq, 1)

        # use MixtureSameFamily to make a piecewise uniform distribution
        # mixture probability is categorical of size (B, nq-1) based on quantile values
        prob = (self.quantiles[1:] - self.quantiles[:-1]).unsqueeze(0).expand(B,-1) 
        prob = prob*100/98 # adjust to ignore bottom and top 1%
        mix = D.Categorical(prob)

        # upper and lower are each (B, nq-1, 1) based on output
        lower, upper = output[:,:-1], output[:,1:]+self.del_eps
        comp = D.Independent(D.Uniform(lower, upper), 1)
        dist = D.MixtureSameFamily(mix, comp)
        
        return dist

class QuantileLSTM(nn.Module):
    """
    Class for LSTM that predicts quantiles.
    Similar to `Probabilistic Individual Load Forecasting Using Pinball Loss Guided LSTM`
    """
    def __init__(self, input_dim, output_dim, prediction_horizon,
        hidden_dim=20,
        fc_hidden_layers=2,
        fc_hidden_dims=20,
        num_layers=1, 
        dropout=0.0,
        bidirectional=False, 
        random_start=False,
        n_quantiles=5,
        clamp_min=0.,
        clamp_max=1e7,
        del_eps=1e-2):
        """ 

        Initializes sequence-to-sequence LSTM model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            output_dim (int): the dimension of the outputs at each point in the sequence
            prediction_horizon (int): the prediction horizon K
            hidden_dim (int): number of hidden/cell dimensions
            fc_hidden_layers (int): number of hidden layers in the decoder network
            fc_hidden_dims (int): number of hidden dims in each layer
            num_layers (int): number of layers in a possibly stacked LSTM
            dropout (float): the dropout rate of the lstm
            bidirectional (bool): whether to initialize a bidirectional lstm
            random_start (bool): If true, will initialize the hidden states randomly from a unit Gaussian
        """ 
        super(QuantileLSTM, self).__init__()
        
        # dimensional parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc_hidden_layer_dims = [fc_hidden_dims for _ in range(fc_hidden_layers)]
        self.output_dim = output_dim
        self.K = prediction_horizon
              
        # LSTM parameters
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.random_start = random_start
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.del_eps = del_eps    
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, batch_first=True,
                          dropout=self.dropout, bidirectional=self.bidirectional)
        
        fc_net = []
        fc_sizes = [self.hidden_dim] + self.fc_hidden_layer_dims
        for i in range(len(fc_sizes)-1):
            fc_net.append(nn.Linear(in_features=fc_sizes[i], out_features=fc_sizes[i+1]))
            fc_net.append(nn.LeakyReLU())
        
        self.n_quantiles = n_quantiles
        if n_quantiles == 5:
            self.quantiles = [.01, .25, .5, .75, .99]
        elif n_quantiles == 7:
            self.quantiles = [.01, .1, .25, .5, .75, .9, .99]
        elif n_quantiles == 11:
            self.quantiles = [.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99]  
        else:
            raise NotImplementedError
        self.quantiles = torch.tensor(self.quantiles)
        fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=self.n_quantiles))
        self.fc = nn.Sequential(*fc_net)
        
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
        
    def forward_quantiles(self, x, y=None):
        """ 

        Run a forward pass of data stream x through the neural network to predict quantiles over the next step.
        Args:
            x (torch.tensor): (B, T, ydim) input observations
            y (Optional[torch.tensor]): (B, K, ydim) output observations if doing many to many training
        Returns:
            quantiles (torch.tensor): (B, K, n_quantiles) predictions of quantiles
        """
        B,T,D = x.shape
        h_0, c_0 = self.initialize_lstm(x)   
        
        if y is None:
            output_lstm, _ = self.lstm(x, (h_0, c_0))
            o = output_lstm[:,[-1]] #(B, 1, hidden_dim)
        else: 
            B2,K,D2 = y.shape
            assert (B==B2 and D==D2), "m2m input sizes not compatible"
    
            output_lstm, _ = self.lstm(
                torch.cat((x,y[:,:-1]), 1), 
                (h_0, c_0)) #(B, T+K-1, hidden_dim)
            o = output_lstm[:,-K:] #(B, K, hidden_dim)
        outputs = self.fc(o)
        _, _, outdims = outputs.shape
        quantiles = outputs[:,:,[0]].expand(-1,-1,outdims)
        quantiles = torch.clamp(quantiles, min=self.clamp_min, max=self.clamp_max) 
        dels = torch.exp(outputs[:,:,1:])
        dels = torch.clamp(dels, min=self.del_eps, max=self.clamp_max)
        quantiles[:,:,1:] += torch.cumsum(dels, dim=-1) # (B, K, n_quantiles)
        return quantiles

    def forward(self, x, y=None):
        """ 

        Run a forward pass of data stream x to predict distribution over next K observations.
        Args:
            x (torch.tensor): (B, T, dim) input observations
            y (Optional[torch.tensor]): (B, K, dim) output observations (for many to many forward pass)
        Returns:
            dist (torch.Distribution): (B, K*dim) predictive distribution over next K observations
        """
        
        output = self.forward_quantiles(x, y=y) # (B, K, n_quantiles) quantiles
        B, K, nq = output.shape
        output = torch.transpose(output,1,2) #(B, nq, K)

        # use MixtureSameFamily to make a piecewise uniform distribution
        # mixture probability is categorical of size (B, nq-1) based on quantile values
        prob = (self.quantiles[1:] - self.quantiles[:-1]).unsqueeze(0).expand(B,-1) 
        prob = prob*100/98 # adjust to ignore bottom and top 1%
        mix = D.Categorical(prob)

        # upper and lower are each (B, nq-1, 1) based on output
        lower, upper = output[:,:-1], output[:,1:]+self.del_eps
        comp = D.Independent(D.Uniform(lower, upper), 1)
        dist = D.MixtureSameFamily(mix, comp)
        return dist

    def forward_fc(self, h_n):
        """ 

        Run the hidden states through the forward layer to obtain outputs. 

        Args: 

            h_n (torch tensor): (B, hidden_dim) tensor of hidden state values at 
                each point in each sequence. 

        Returns: 
            dist (torch.Distribution): (B,1) predictive distribution over next observation

        """ 
        outputs = self.fc(h_n) # quantiles
        B, outdims = outputs.shape
        quantiles = outputs[:,[0]].expand(-1,outdims)
        quantiles = torch.clamp(quantiles, min=self.clamp_min, max=self.clamp_max) 
        dels = torch.exp(outputs[:,1:])
        dels = torch.clamp(dels, min=self.del_eps, max=self.clamp_max)
        quantiles[:,1:] += torch.cumsum(dels, dim=-1) # (B, n_quantiles)
        output = quantiles.unsqueeze(-1) #(B, nq, 1)

        # use MixtureSameFamily to make a piecewise uniform distribution
        # mixture probability is categorical of size (B, nq-1) based on quantile values
        prob = (self.quantiles[1:] - self.quantiles[:-1]).unsqueeze(0).expand(B,-1) 
        prob = prob*100/98 # adjust to ignore bottom and top 1%
        mix = D.Categorical(prob)

        # upper and lower are each (B, nq-1, 1) based on output
        lower, upper = output[:,:-1], output[:,1:]+self.del_eps
        comp = D.Independent(D.Uniform(lower, upper), 1)
        dist = D.MixtureSameFamily(mix, comp)
        return dist