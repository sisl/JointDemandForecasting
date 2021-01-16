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

#loc, past_dims, fut_dims = (1, 8, 12)
#loc, past_dims, fut_dims = (9, 8, 12)
#loc, past_dims, fut_dims = (1, 16, 12)
#loc, past_dims, fut_dims = (9, 16, 12)
#loc, past_dims, fut_dims = (1, 24, 12)
loc, past_dims, fut_dims = (9, 24, 12)

ntrials = 10

# decision problem
min_indices = 4
obj_fn = lambda x: var(x, 0.8)

### Load Data

X_train, Y_train_full, X_test, Y_test_full = load_data(loc, past_dims, fut_dims)
Y_train = Y_train_full[:,[0],:]
Y_test = Y_test_full[:,[0],:]

### Single Step Modeling

# ARMA-1
lin_reg = BayesianLinearRegression(1, past_dims, 1, 1)
lin_reg.fit(X_train, Y_train)

# ARMA-K
print('ARMA Metrics:')
samples = sample_forward(lin_reg, X_test, fut_dims)

print("WAPE = ", wape(samples,Y_test_full,sampled=True)[0])
print("RWSE = ", rwse(samples,Y_test_full,sampled=True)[0])
print("DScore = ", index_allocation(samples, min_indices, 
                                    obj_fn, Y_test_full, 0.8))

# FNN-1
wapes, rwses, dscores = [], [], []
for seed1 in range(ntrials):
    X_batches, Y_batches = batch(X_train, Y_train, batch_size = 64, random_state = seed1)
    hidden_layers = [40, 40, 40]
    ss_gmnn = GaussianMixtureNeuralNet(1, past_dims, hidden_layers, 1, 1, 
                                       n_components=3, random_state=seed1)

    try:
        train(ss_gmnn, X_batches, Y_batches, num_epochs=150, learning_rate=.005, verbose=False)    
        samples = sample_forward(ss_gmnn, X_test, fut_dims)
    except:
        continue
    
    wapes.append(wape(samples,Y_test_full,sampled=True)[0])
    rwses.append(rwse(samples,Y_test_full,sampled=True)[0])
    dscores.append(index_allocation(samples, min_indices, 
                                    obj_fn, Y_test_full, 0.8))
    print('Seed ', seed1, ' done')
print('IFNN Metrics:')
print("WAPEs = ", torch.stack(wapes))
print("RWSEs = ", torch.stack(rwses))
print("DScores = ", torch.stack(dscores))

# RNN-1
wapes, rwses, dscores = [], [], []
for seed2 in range(ntrials):
    X_batches, Y_batches = batch(X_train, Y_train, batch_size = 64, random_state = seed2)
    hidden_layers = [20, 20, 20]
    hidden_dim = 40
    ss_gmmlstm = GaussianMixtureLSTM(1, hidden_dim, hidden_layers, 1, 1, 
                                     n_components=3, random_state=seed2, random_start=False)

    try:
        train(ss_gmmlstm, X_batches, Y_batches, num_epochs=200, learning_rate=.005, verbose=False)
        samples = sample_forward_lstm(ss_gmmlstm, X_test, fut_dims)
    except:
        continue
        
    wapes.append(wape(samples,Y_test_full,sampled=True)[0])
    rwses.append(rwse(samples,Y_test_full,sampled=True)[0])
    dscores.append(index_allocation(samples, min_indices, 
                                    obj_fn, Y_test_full, 0.8))
    print('Seed ', seed2, ' done')
    
print('IRNN Metrics:')
print("WAPEs = ", torch.stack(wapes))
print("RWSEs =  ", torch.stack(rwses))
print("DScores = ", torch.stack(dscores))
