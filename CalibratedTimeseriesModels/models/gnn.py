import torch
import torch.nn as nn
from CalibratedTimeseriesModels.abstractmodels import *
from torch.distributions.normal import Normal as normal

class GaussianNeuralNet(ExplicitPredictiveModel):
    
    def __init__(self, input_dim, input_horizon, hidden_layer_sizes, output_dim, output_horizon):
        """ 

        Initializes sequence-to-sequence LSTM model. 

        Args: 

            input_dim (int): number of input dimensions at each step in the series 
            input_horizon (int): the input horizon T
            hidden_layer_sizes (list of ints): the hidden layer sizes in the neural network
            output_dim (int): the output dimension
            prediction_horizon (int): the prediction horizon K

        """ 
        super(GaussianNeuralNet, self).__init__()
        self.input_size = input_dim
        self.T = input_horizon
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_dim
        self.K = prediction_horizon
        
        fc_net = []
        fc_sizes = np.append(self.input_size * self.T, self.hidden_layer_sizes)
        for i in range(len(fc_sizes)-1):
            fc_net.append(nn.Linear(in_features=fc_sizes[i], out_features=fc_sizes[i+1]))
            fc_net.append(nn.LeakyReLU())
        
        fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=2*self.output_size*self.K))
        self.fc = nn.Sequential(*fc_net)
        
    def forward(self, y, u, K):
        """ 

        Run a forward pass of data stream x through the neural network to predict distribution over next K observations.
        Args:
            y (torch.tensor): (B, T, ydim) observations
            u (torch.tensor or None): (B, T, udim) inputs
            K (int): horizon to predict 
        Returns:
            dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations shaped
        """
        
        B, T, ydim = y.shape()
        inputs = y.reshape((B, T*ydim))
        outputs = self.fc(inputs)
        B, outdims = outputs.shape
        mu = outputs[:,:outdims//2]
        sig = nn.functional.softplus(outputs[:,outdims//2:])
        
        # isotropic normal distribution
        dist = normal(loc=mu, scale=sig)
        return dist
    
class GaussianLSTM(ExplicitPredictiveModel):
    pass