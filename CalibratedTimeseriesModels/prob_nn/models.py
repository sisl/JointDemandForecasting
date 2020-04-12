import numpy as np
import torch
import torch.nn as nn

class LSTMS2S(nn.Module):
    def __init__(self, input_size, hidden_size, fc_hidden_layer_sizes, output_size, num_layers=1,
                 dropout=0.0, batch_first=True, bidirectional=False, sequence_length=12, 
                 random_start=True, prob=False):

        super(LSTMS2S, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc_hidden_layer_sizes = fc_hidden_layer_sizes
        self.output_size = output_size
        self.sequence_length = sequence_length
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
        h_0, c_0 = self.initialize_lstm(x)    
        output_lstm, h_n = self.lstm(x, (h_0, c_0))
        # output_gru has shape (batch_size, seq_len, hidden_dimensions)
        # nn.Linear operates on the last dimension of its input
        return self.forward_fc(output_lstm)
    
    def initialize_lstm(self, x):
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
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout=0.0, prob=False):

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