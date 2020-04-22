import numpy as np
import torch
import torch.nn as nn

class LSTMS2S(nn.Module):
    """ 

    Class for sequence-to-sequence LSTM. 
    
    """ 
    def __init__(self, input_size, hidden_size, fc_hidden_layer_sizes, output_size, num_layers=1,
                 dropout=0.0, batch_first=True, bidirectional=False,
                 random_start=True, prob=False):
        """ 

        Initializes sequence-to-sequence LSTM model. 

        Args: 

            input_size (int): number of input dimensions at each step in the series 
            hidden_size (int): number of hidden/cell dimensions
            fc_hidden_layer_sizes (list of ints): the hidden layer sizes in the neural network mapping from hidden state to output
            output_size (int): the dimension of the outputs at each point in the sequence
            num_layers (int): number of layers in a possibly stacked LSTM
            dropout (float): the dropout rate of the lstm
            batch_first (bool): whether the batch is in the first dimension, which it must be)
            bidirectional (bool): whether to initialize a bidirectional lstm
            random_start (bool): If true, will initialize the hidden states randomly from a unit Gaussian
            prob (bool): If true, will output a mean and sigma rather than just a mean

        """ 
        super(LSTMS2S, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc_hidden_layer_sizes = fc_hidden_layer_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.random_start = random_start
        self.prob = prob
        if not self.batch_first:
            raise('Error: Batch second not implemented, feed batch first')
            
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=self.batch_first,
                          dropout=self.dropout, bidirectional=self.bidirectional)
        
        fc_net = []
        fc_sizes = np.append(hidden_size, self.fc_hidden_layer_sizes)
        for i in range(len(fc_sizes)-1):
            fc_net.append(nn.Linear(in_features=fc_sizes[i], out_features=fc_sizes[i+1]))
            fc_net.append(nn.LeakyReLU())
        
        # for probabilistic, output twice the features for mu, sigma
        if self.prob:
            fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=2*self.output_size))
        else:
            fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=self.output_size))
        self.fc = nn.Sequential(*fc_net)
        
    def forward(self, x):
        """ 

        Run a forward pass of data stream x through the LSTM. 

        Args: 

            x (torch tensor): (batch_size, sequence_length, input_features) tensor of inputs to the lstm. 
                An output will be created for each element in each sequence.

        Returns: 

            (torch tensor): (batch_size, sequence_length, output) tensor of model outputs. If running a 
                probabilistic model, first half of the elements along the output dimension will be means, 
                the last half will be sigmas

        """ 
        h_0, c_0 = self.initialize_lstm(x)    
        output_lstm, h_n = self.lstm(x, (h_0, c_0))
        # output_gru has shape (batch_size, seq_len, hidden_dimensions)
        # nn.Linear operates on the last dimension of its input
        return self.forward_fc(output_lstm)
    
    def initialize_lstm(self, x):
        """ 

        Initialize the lstm either randomly or with zeros. 

        Args: 

            x (torch tensor): (batch_size, sequence_length, input_features) tensor of inputs to the lstm. 

        Returns: 

            h_0 (torch tensor): (num_layers*num_directions, batch_size, hidden_size) tensor for initial hidden state
            c_0 (torch tensor): (num_layers*num_directions, batch_size, hidden_size) tensor for initial cell state 

        """ 
        batch_index = 0 if self.batch_first else 1
        num_direction = 2 if self.bidirectional else 1

        # Hidden state in first seq of the LSTM - use noisy state initialization
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

            h_n (torch tensor): (batch_size, sequence_length, hidden_features) tensor of hidden state values at 
                each point in each sequence. 

        Returns: 

            (torch tensor): (batch_size, sequence_length, output_size) tensor for model outputs at each stage
                of the sequence. If probabilistic, the first half of elements along the output_size dimension
                are means, while the latter half are sigmas. 

        """ 
        fc_output = self.fc(h_n)
        # fc_output will be (batch_size, seq_len, num_classes)
        if self.prob:
            sh = fc_output.shape
            axes = len(sh)
            outdims = sh[-1]
            transposed = torch.transpose(fc_output,0,axes-1)
            mut, sigt = transposed[:outdims//2], nn.functional.softplus(transposed[outdims//2:])
            return torch.cat((mut,sigt)).transpose(0,axes-1)
        else:
            return fc_output
    
class ARNN(nn.Module):
    """ 

    Class for autoregressive neural network. 
    
    """     
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout=0.0, prob=False):
        """ 

        Initializes sequence-to-sequence LSTM model. 

        Args: 

            input_size (int): number of input dimensions at each step in the series 
            hidden_layer_sizes (list of ints): the hidden layer sizes in the neural network
            output_size (int): the output dimension
            dropout (float): the dropout rate of the lstm
            random_start (bool): If true, will initialize the hidden states randomly from a unit Gaussian
            prob (bool): If true, will output a mean and sigma rather than just a mean

        """ 
        super(ARNN, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.prob = prob
        
        fc_net = []
        
        fc_sizes = np.append(self.input_size, self.hidden_layer_sizes)
        for i in range(len(fc_sizes)-1):
            fc_net.append(nn.Linear(in_features=fc_sizes[i], out_features=fc_sizes[i+1]))
            fc_net.append(nn.LeakyReLU())
        
        # for probabilistic, output twice the features for mu, sigma
        if self.prob:
            fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=2*self.output_size))
        else:
            fc_net.append(nn.Linear(in_features=fc_sizes[-1], out_features=self.output_size))
        self.fc = nn.Sequential(*fc_net)
        
    def forward(self, x):
        """ 

        Run a forward pass of data stream x through the neural network. 

        Args: 

            x (torch tensor): (batch_size, *, input_features) tensor of input data.

        Returns: 

            (torch tensor): (batch_size, *, output_size) tensor for model outputs. If probabilistic, the first 
                half of elements along the output_size dimension are means, while the latter half are sigmas. 

        """ 
        if self.prob:
            outputs = self.fc(x)
            sh = outputs.shape
            axes = len(sh)
            outdims = sh[-1]
            transposed = torch.transpose(outputs,0,axes-1)
            mut, sigt = transposed[:outdims//2], nn.functional.softplus(transposed[outdims//2:])
            return torch.cat((mut,sigt)).transpose(0,axes-1)
        else:
            return self.fc(x)