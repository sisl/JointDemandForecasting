import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append("../../")
import JointDemandForecasting

from JointDemandForecasting.utils import *
from JointDemandForecasting.models.gmnn import *
from JointDemandForecasting.models.blr import *

from load_data import load_data
from charging_utils import *

### Experiment Settings (uncomment one of these)

# location: 1=Bakersfield,9=SLC
# past_dims, fut_dims: past and future time step, either (24,8) or (8,12)
# epoch2: number of training epochs in the rnn model
# seed1: seeds used in final IFNN experiments (arbitrarily arrived at during rapid prototyping)
# seed2: seeds used in final IRNN experiments 

#loc, past_dims, fut_dims, epoch2, seed1, seed2 = (1, 24, 8, 140, 0, 3)
#loc, past_dims, fut_dims, epoch2, seed1, seed2 = (9, 24, 8, 140, 1, 3)

loc, past_dims, fut_dims, epoch2, seed1, seed2 = (1, 8, 12, 200, 0, 0)
loc, past_dims, fut_dims, epoch2, seed1, seed2 = (9, 8, 12, 200, 0, 0)
loc, past_dims, fut_dims, epoch2, seed1, seed2 = (1, 16, 12, 200, 0, 0)
loc, past_dims, fut_dims, epoch2, seed1, seed2 = (9, 16, 12, 200, 0, 0)

loc, past_dims, fut_dims, epoch2, seed1, seed2 = (1, 24, 12, 200, 0, 0)
loc, past_dims, fut_dims, epoch2, seed1, seed2 = (9, 24, 12, 200, 1, 1)


### Load Data

X_train, Y_train_full, X_test, Y_test_full = load_data(loc, past_dims, fut_dims)
Y_train = Y_train_full[:,[0],:]
Y_test = Y_test_full[:,[0],:]
X_batches, Y_batches = batch(X_train, Y_train, batch_size = 128)

### Single Step Modeling

# ARMA-1
lin_reg = BayesianLinearRegression(1, past_dims, 1, 1)
lin_reg.fit(X_train, Y_train)

# FNN-1
hidden_layers = [40, 40, 40]
ss_gmnn = GaussianMixtureNeuralNet(1, past_dims, hidden_layers, 1, 1, 
                                   n_components=3, random_state=seed1)
train(ss_gmnn, X_batches, Y_batches, num_epochs=150, learning_rate=.005)

# RNN-1
hidden_layers = [20, 20, 20]
hidden_dim = 40
ss_gmmlstm = GaussianMixtureLSTM(1, hidden_dim, hidden_layers, 1, 1, 
                                 n_components=3, random_state=seed2)
train(ss_gmmlstm, X_batches, Y_batches, num_epochs=epoch2, learning_rate=.005)

### Propagation Test Metrics

# decision problem
min_indices = 4
obj_fn = lambda x: var(x, 0.8)

# ARMA-K
print('ARMA-K Metrics:')
samples_lr = sample_forward(lin_reg, X_test, fut_dims)
for f in [mape, wape, rmse, rwse]:
    print(f(samples_lr,Y_test_full,sampled=True))
print(index_allocation(samples_lr, min_indices, 
                       obj_fn, Y_test_full, 0.8))

# FNN-K
print('FNN-K Metrics:')
samples_ss_gmnn = sample_forward(ss_gmnn, X_test, fut_dims)
for f in [mape, wape, rmse, rwse]:
    print(f(samples_ss_gmnn,Y_test_full,sampled=True))
print(index_allocation(samples_ss_gmnn, min_indices, 
                       obj_fn, Y_test_full, 0.8))
    
# RNN-K
print('RNN-K Metrics:')
samples_ss_gmmlstm = sample_forward_lstm(ss_gmmlstm, X_test, fut_dims)
for f in [mape, wape, rmse, rwse]:
    print(f(samples_ss_gmmlstm,Y_test_full,sampled=True))
print(index_allocation(samples_ss_gmmlstm, min_indices, 
                       obj_fn, Y_test_full, 0.8))

