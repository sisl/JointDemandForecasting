import torch.nn as nn
import torch
import math
from codebase.models.nns.BBBLayer import BBBLayer
from torch.nn.utils.rnn import PackedSequence

class BBBRNN(BBBLayer):

    def __init__(self, mode, sharpen, input_size, hidden_size,
                 num_layers=1, 
                 batch_first=False,
                 dropout=0, 
                 bidirectional=False, 
                 *args, **kwargs):

        super(BBBRNN, self).__init__(*args, **kwargs)
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        self.smoohing = sharpen

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self.means = []
        self.logvars = []
        self.eta = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = \
                    input_size if layer == 0 else hidden_size * num_directions

                w_ih_mean = nn.Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh_mean = nn.Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih_mean = nn.Parameter(torch.Tensor(gate_size))
                b_hh_mean = nn.Parameter(torch.Tensor(gate_size))
                self.means += [w_ih_mean, w_hh_mean, b_ih_mean, b_hh_mean]

                if self.BBB is True:
                    w_ih_logvar = \
                        nn.Parameter(torch.Tensor(gate_size, layer_input_size))
                    w_hh_logvar = nn.Parameter(torch.Tensor(gate_size, hidden_size))
                    b_ih_logvar = nn.Parameter(torch.Tensor(gate_size))
                    b_hh_logvar = nn.Parameter(torch.Tensor(gate_size))
                    self.logvars += \
                        [w_ih_logvar, w_hh_logvar, b_ih_logvar, b_hh_logvar]

                # set weight to be attribute
                if self.BBB is True:
                    layer_params = (
                        w_ih_mean, w_ih_logvar,
                        w_hh_mean, w_hh_logvar,
                        b_ih_mean, b_ih_logvar,
                        b_hh_mean, b_hh_logvar
                    )
                    suffix = '_reverse' if direction == 1 else ''
                    param_names = ['weight_ih_mean_l{}{}', 
                                    'weight_ih_logvar_l{}{}', 
                                    'weight_hh_mean_l{}{}', 
                                    'weight_hh_logvar_l{}{}']
                    param_names += ['bias_ih_mean_l{}{}',  
                                    'bias_ih_logvar_l{}{}', 
                                    'bias_hh_mean_l{}{}',  
                                    'bias_hh_logvar_l{}{}']
                else:
                    layer_params = (
                            w_ih_mean,
                            w_hh_mean,
                            b_ih_mean,
                            b_hh_mean
                            )
                    suffix = '_reverse' if direction == 1 else ''
                    param_names = ['weight_ih_mean_l{}{}', 'weight_hh_mean_l{}{}']
                    param_names += ['bias_ih_mean_l{}{}',  'bias_hh_mean_l{}{}']

                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)

                if self.smoohing:
                    w_ih_eta = nn.Parameter(torch.Tensor(gate_size, layer_input_size))
                    w_hh_eta = nn.Parameter(torch.Tensor(gate_size, hidden_size))
                    b_ih_eta = nn.Parameter(torch.Tensor(gate_size))
                    b_hh_eta = nn.Parameter(torch.Tensor(gate_size))
                    self.eta += [w_ih_eta, w_hh_eta, b_ih_eta, b_hh_eta]
                    layer_params_sharpen = (
                            w_ih_eta,
                            w_hh_eta,
                            b_ih_eta,
                            b_hh_eta
                            )
                    suffix = '_reverse' if direction == 1 else ''
                    param_names = ['weight_ih_eta_l{}{}', 'weight_hh_eta_l{}{}']
                    param_names += ['bias_ih_eta_l{}{}',  'bias_hh_eta_l{}{}']
                    param_names = [x.format(layer, suffix) for x in param_names]
                    for name, param in zip(param_names, layer_params_sharpen):
                        setattr(self, name, param)

        self.reset_parameters()


    def reset_parameters(self):
        """
        init parameters
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        logvar_init = math.log(stdv) * 2
        for mean in self.means:
            mean.data.uniform_(-stdv, stdv)
        if self.BBB is True:
            for logvar in self.logvars:
                logvar.data.fill_(logvar_init)

            if self.smoohing:
                for eta in self.eta:
                    eta.data.uniform_(-stdv, stdv)

    def _apply(self, fn):
        ret = super(BBBRNN, self)._apply(fn)
        return ret

    def get_all_weights(self, weights):
        """
        a helper function that transform a list of weights
        to pytorch RNN backend weight
        """
        start = 0
        all_weights = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                w_ih = weights[start]
                w_hh = weights[start+1]
                b_ih = weights[start+2]
                b_hh = weights[start+3]
                start += 4
                all_weights.extend([w_ih, w_hh, b_ih, b_hh])

        return all_weights

    def forward(self, input, hx=None, grads=None):
        if grads is not None:
            self.resample_with_sharpening(grads, self.eta)
            weights = self.sampled_sharpen_weights
        elif self.training and self.BBB is True:
            self.sample()
            weights = self.sampled_weights
        else:
            weights = self.means

        # modify weights to pytorch format
        self.all_weights = self.get_all_weights(weights)
        # Adopted from pytorch RNN base code
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)

            if self.mode == 'LSTM':
                hx = (zeros, zeros) # (h, c)
            else:
                hx = zeros

        if batch_sizes is None:
            result = nn._VF.lstm(input, hx, self.all_weights, self.bias, 
                self.num_layers, self.dropout, self.training, 
                self.bidirectional, self.batch_first)
        else:
            result = nn._VF.lstm(input, batch_sizes, hx, self.all_weights, 
                self.bias, self.num_layers, self.dropout, self.training, 
                self.bidirectional)

        output, hidden = result[0], result[1:]

        if is_packed:
            output = PackedSequence(output, batch_sizes)

        return output, hidden



