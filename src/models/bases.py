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
        quantiles=5):
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
        
    def forward_quantiles(self, y):
        """ 

        Run a forward pass of data stream x through the neural network to predict quantiles over the next step.
        Args:
            y (torch.tensor): (B, T, ydim) observations
        Returns:
            quantiles (torch.tensor): (B, n_quantiles) predictions of quantiles
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
        dels = torch.exp(outputs[:,1:])
        quantiles[:,1:] += torch.cumsum(dels, dim=-1)
        return quantiles

    def forward(self, y):
        """
        Run a forward pass of data stream x through the neural network to predict distribution over the next observation.
        Args:
            y (torch.tensor): (B, T, ydim) observations
        Returns:
            dist (torch.Distribution): (B,1) predictive distribution over the next observations
        """
        output = self.forward_quantiles(y) #(B, nq)
        B, nq = output.shape

        # use MixtureSameFamily to make a piecewise uniform distribution
        # mixture probability is categorical of size (B, nq-1) based on quantile values
        prob = (self.quantiles[1:] - self.quantiles[:-1]).unsqueeze(0).expand(B,-1) 
        prob *= 100/98 # adjust to ignore bottom and top 1%
        mix = D.Categorical(prob)

        # upper and lower are each (B, nq-1) based on output
        lower, upper = output[:,:-1], output[:,1:]
        comp = D.Uniform(lower, upper) # does this require D.Independent??

        dist = D.MixtureSameFamily(mix, comp)
        return dist


class QuantileLSTM(nn.Module):
    """
    Class for LSTM that predicts quantiles.
    Similar to `Probabilistic Individual Load Forecasting Using Pinball Loss Guided LSTM`
    """
    def __init__(self):
        pass
    def forward(self):
        pass