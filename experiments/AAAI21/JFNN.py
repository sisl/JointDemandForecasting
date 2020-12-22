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

#loc, past_dims, fut_dims = (1, 24, 8)
#loc, past_dims, fut_dims = (9, 24, 8)
#loc, past_dims, fut_dims = (1, 8, 12)
loc, past_dims, fut_dims = (9, 8, 12)


### Load Data

X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)
X_batches, Y_batches = batch(X_train, Y_train, batch_size = 64)

### JFNN

# train
hidden_layers = [40, 40, 40]
model = GaussianMixtureNeuralNet(1, past_dims, hidden_layers, 1, fut_dims, 
                                 n_components=2, covariance_type='low-rank',
                                 rank=2, random_state=0)
train(model, X_batches, Y_batches, num_epochs=300, learning_rate=.002)

# test
model.eval()
for f in [mape, wape, rmse, rwse, nll]:
    print(f(model(X_test),Y_test))
    
# decision problem
min_indices = 4
samples = model(X_test).sample((1000,))
obj_fn = lambda x: var(x, 0.8)
print(index_allocation(samples, min_indices, 
                       obj_fn, Y_test, 0.8))    
