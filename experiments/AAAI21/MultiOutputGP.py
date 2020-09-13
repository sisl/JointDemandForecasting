import os
import sys
import numpy as np
import torch
import torch.nn as nn
import gpytorch

sys.path.append("../../")
import CalibratedTimeseriesModels
from CalibratedTimeseriesModels.utils import *
from CalibratedTimeseriesModels.models.mogp import *

from load_data import load_data

### Experiment Settings (uncomment one of these)

# location: 1=Bakersfield,9=SLC
# past_dims, fut_dims: past and future time step, either (24,8) or (8,12)
# epochs: number of epochs through training data used in paper experiments
# seed: seeds used in final experiments (arbitrarily arrived at during rapid prototyping)
 
loc, past_dims, fut_dims, epochs, seed = (1, 24, 8, 80, 2)
#loc, past_dims, fut_dims, epochs, seed = (9, 24, 8, 80, 2)
#loc, past_dims, fut_dims, epochs, seed = (1, 8, 12, 55, 0)
#loc, past_dims, fut_dims, epochs, seed = (9, 8, 12, 55, 0)

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
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=fut_dims).to(device)
model = MultiOutputGP(train_x, train_y, likelihood, covar_kernel=covar_kernel, 
                      index_rank=index_rank, random_state=seed).to(device)

train_mogp(model, likelihood, train_x, train_y, epochs, verbose=False)


# train nll    
model.eval()
likelihood.eval()
train_dist = model.forward_independent(train_x)
print(nll(train_dist, Y_train.to(device)))

# test nll
test_dist = model.forward_independent(test_x)
print(nll(test_dist, Y_test.to(device)))

# other test metrics
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    dist = model(test_x)
for f in [mape, wape, rmse, rwse]:
    print(f(dist,Y_test.to(device)))