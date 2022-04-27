import os
import sys
import numpy as np
import torch
import torch.nn as nn

import src
from src.utils import *
from src.models.gmnn import *
from experiments.charging_utils import *

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

X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)
### JFNN

wapes, rwses, nlls, trnlls, dscores = [], [], [], [], []
for seed in range(ntrials):

    # train
    hidden_layers = [40, 40, 40]
    model = GaussianMixtureNeuralNet(1, past_dims, hidden_layers, 1, fut_dims, 
                                     n_components=2, covariance_type='low-rank',
                                     rank=2, random_state=seed)
    X_batches, Y_batches = batch(X_train, Y_train, batch_size = 64, random_state = seed)
    try:
        train(model, X_batches, Y_batches, num_epochs=300, learning_rate=.002, verbose = False)
    except:
        continue
        
    model.eval()  
    # test metrics
    dist = model(X_test)
    dist_tr = model(X_train)
    wapes.append(wape(dist,Y_test)[0])
    rwses.append(rwse(dist,Y_test)[0])
    nlls.append(nll(dist,Y_test)[0].detach())
    trnlls.append(nll(dist_tr,Y_train)[0].detach())

    samples = dist.sample((1000,))
    dscores.append(index_allocation(samples, min_indices, 
                                    obj_fn, Y_test, 0.8))
    print('Seed ', seed, ' done')

print('JFNN Metrics:')
print("WAPEs = ", torch.stack(wapes))
print("RWSEs = ", torch.stack(rwses))
print("NLLs = ", torch.stack(nlls))
print("TrNLLs = ", torch.stack(trnlls))
print("DScores = ", torch.stack(dscores))
