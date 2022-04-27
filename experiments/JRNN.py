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
# epochs: number of epochs through training data used in paper experiments
# seed: seeds used in final experiments (arbitrarily arrived at during rapid prototyping)

loc, past_dims, fut_dims = (1, 8, 12)
#loc, past_dims, fut_dims = (9, 8, 12)
#loc, past_dims, fut_dims = (1, 16, 12)
#loc, past_dims, fut_dims = (9, 16, 12)
#loc, past_dims, fut_dims = (1, 24, 12)
#loc, past_dims, fut_dims = (9, 24, 12)

ntrials = 10

# decision problem
min_indices = 4
obj_fn = lambda x: var(x, 0.8)
    
# use_cuda: whether to use a gpu (training LSTM way faster on gpu)
use_cuda = False # torch.cuda.is_available()

### Load Data

if use_cuda:
    torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")

X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)
X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

### JRNN
wapes, rwses, nlls, trnlls, dscores = [], [], [], [], []
for seed in range(ntrials):

    #train
    X_batches, Y_batches = batch(X_train, Y_train, batch_size = 64, random_state = seed)
    hidden_layers = [40,40,40]
    hidden_dim = 40
    model = GaussianMixtureLSTM(1, hidden_dim, hidden_layers,  1, fut_dims,
                                covariance_type='low-rank', rank=2, n_components=2,
                                random_state=seed, random_start=False).to(device)
    try:
        train(model, X_batches, Y_batches, num_epochs=200, learning_rate=.005, verbose=False)
    except:
        continue
        
    model.eval()  
    # test metrics
    dist = model(X_test)
    dist_tr = model(X_train)
    wapes.append(wape(dist,Y_test)[0].cpu())
    rwses.append(rwse(dist,Y_test)[0].cpu())
    nlls.append(nll(dist,Y_test)[0].detach().cpu())
    trnlls.append(nll(dist_tr,Y_train)[0].detach().cpu())

    samples = dist.sample((1000,)).cpu()
    dscores.append(index_allocation(samples, min_indices, 
                                    obj_fn, Y_test.cpu(), 0.8))
    print('Seed ', seed, ' done')

print('JRNN Metrics:')
print("WAPEs = ", torch.stack(wapes))
print("RWSEs = ", torch.stack(rwses))
print("NLLs = ", torch.stack(nlls))
print("TrNLLs = ", torch.stack(trnlls))
print("DScores = ", torch.stack(dscores))
