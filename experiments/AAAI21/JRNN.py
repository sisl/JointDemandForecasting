import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append("../../")
import JointDemandForecasting
from JointDemandForecasting.utils import *
from JointDemandForecasting.models.gmnn import *

from load_data import load_data
from charging_utils import *

### Experiment Settings (uncomment one of these)

# location: 1=Bakersfield,9=SLC
# past_dims, fut_dims: past and future time step, either (24,8) or (8,12)
# epochs: number of epochs through training data used in paper experiments
# seed: seeds used in final experiments (arbitrarily arrived at during rapid prototyping)
 
#loc, past_dims, fut_dims, epochs, seed = (1, 24, 8, 300, 2020)
#loc, past_dims, fut_dims, epochs, seed = (9, 24, 8, 300, 2020)
#loc, past_dims, fut_dims, epochs, seed = (1, 8, 12, 300, 0)
loc, past_dims, fut_dims, epochs, seed = (9, 8, 12, 200, 1)

# use_cuda: whether to use a gpu (training LSTM way faster on gpu)
use_cuda = torch.cuda.is_available()

### Load Data

if use_cuda:
    torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")

X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)
X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

X_batches, Y_batches = batch(X_train, Y_train, batch_size = 64)

### JRNN

#train
hidden_layers = [40,40,40]
hidden_dim = 40
ss_gmmlstm = GaussianMixtureLSTM(1, hidden_dim, hidden_layers,  1, fut_dims, 
                                 covariance_type='low-rank', rank=3, n_components=2, 
                                 random_state=seed).to(device)
train(ss_gmmlstm, X_batches, Y_batches, num_epochs=epochs, learning_rate=.005)

# test metrics
for f in [mape, wape, rmse, rwse, nll]:
    print(f(ss_gmmlstm(X_test),Y_test))
    
# decision problem
min_indices = 4
samples = ss_gmmlstm(X_test).sample((1000,)).cpu()
obj_fn = lambda x: var(x, 0.8)
print(index_allocation(samples, min_indices, 
                       obj_fn, Y_test.cpu(), 0.8))