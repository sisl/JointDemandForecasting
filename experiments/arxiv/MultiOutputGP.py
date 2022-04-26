import os
import sys
import numpy as np
import torch
import torch.nn as nn
import gpytorch

sys.path.append("../../")
import JointDemandForecasting
from JointDemandForecasting.utils import *
from JointDemandForecasting.models.mogp import *

from load_data import load_data
from charging_utils import *

### Experiment Settings (uncomment one of these)

# location: 1=Bakersfield,9=SLC
# past_dims, fut_dims: past and future time step, either (24,8) or (8,12)
# epochs: number of epochs through training data used in paper experiments

loc, past_dims, fut_dims, epochs = (1, 8, 12, 55)
#loc, past_dims, fut_dims, epochs = (9, 8, 12, 55)
#loc, past_dims, fut_dims, epochs = (1, 16, 12, 65)
#loc, past_dims, fut_dims, epochs = (9, 16, 12, 65)
#loc, past_dims, fut_dims, epochs = (1, 24, 12, 75)
#loc, past_dims, fut_dims, epochs = (9, 24, 12, 75)

ntrials = 10

# decision problem
min_indices = 4
obj_fn = lambda x: var(x, 0.8)

# use_cuda: whether to use a gpu (training LSTM way faster on gpu)
use_cuda = torch.cuda.is_available()

### Load Data

if use_cuda:
    torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")

X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)
train_x, train_y = X_train.reshape(X_train.size(0),-1).to(device), Y_train.reshape(Y_train.size(0),-1).to(device)
test_x, test_y = X_test.reshape(X_test.size(0),-1).to(device), Y_test.reshape(Y_test.size(0),-1).to(device)

### MOGP

# train
kernels = {'rbf': gpytorch.kernels.RBFKernel(),
           'ind_rbf': gpytorch.kernels.RBFKernel(ard_num_dims=past_dims),
           'matern': gpytorch.kernels.MaternKernel(ard_num_dims=past_dims),
           'rq': gpytorch.kernels.RQKernel(ard_num_dims=past_dims),
           'spectral': gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2,ard_num_dims=24)
          }

covar_kernel = kernels['rq']*kernels['matern']
index_rank = 8

wapes, rwses, nlls, trnlls, dscores = [], [], [], [], []
for seed in range(ntrials):
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=fut_dims).to(device)
    model = MultiOutputGP(train_x, train_y, likelihood, covar_kernel=covar_kernel, 
                          index_rank=index_rank, random_state=seed).to(device)
    train_mogp(model, likelihood, train_x, train_y, epochs, verbose=False)

    # train nll    
    model.eval()
    likelihood.eval()
    train_dist = model.forward_independent(train_x)

    # test nll
    test_dist = model.forward_independent(test_x)

    # other test metrics
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        dist = model(test_x)

    wapes.append(wape(dist,Y_test.to(device))[0])
    rwses.append(rwse(dist,Y_test.to(device))[0])
    nlls.append(nll(test_dist, Y_test.to(device))[0])    
    trnlls.append(nll(train_dist, Y_train.to(device))[0])

    samples = dist.sample(torch.Size([1000]))
    dscores.append(index_allocation(samples, min_indices, 
                                    obj_fn, Y_test, 0.8))
    print('Seed ', seed, ' done')

print('MOGP Metrics:')
print("WAPEs = ", torch.stack(wapes).cpu())
print("RWSEs = ", torch.stack(rwses).cpu())
print("NLLs = ", torch.stack(nlls).cpu())
print("TrNLLs = ", torch.stack(trnlls).cpu())
print("DScores = ", torch.stack(dscores).cpu())