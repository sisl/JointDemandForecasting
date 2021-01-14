import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append("../../")
import JointDemandForecasting

from JointDemandForecasting.utils import *
from JointDemandForecasting.models.cgmm import *
from JointDemandForecasting.models.blr import *

from load_data import load_data
from charging_utils import *

### Experiment Settings (uncomment one of these)

# location: 1=Bakersfield,9=SLC
# past_dims, fut_dims: past and future time step, either (24,8) or (8,12)
# ncomp: number of components used in mixture, more used for longer output

loc, past_dims, fut_dims, ncomp = (1, 8, 12, 5)
#loc, past_dims, fut_dims, ncomp = (9, 8, 12, 5)
#loc, past_dims, fut_dims, ncomp = (1, 16, 12, 5)
#loc, past_dims, fut_dims, ncomp = (9, 16, 12, 5)
#loc, past_dims, fut_dims, ncomp = (1, 24, 12, 4)
#loc, past_dims, fut_dims, ncomp = (9, 24, 12, 4)

ntrials = 10

# decision problem
min_indices = 4
obj_fn = lambda x: var(x, 0.8)

### Load Data

X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)

### Conditional Gaussian

# train
lin_reg = BayesianLinearRegression(1, past_dims, 1, fut_dims)
lin_reg.fit(X_train, Y_train)

# test metrics
print('CondG Metrics:')
dist = lin_reg(X_test)
dist_tr = lin_reg(X_train)
print("WAPE = ", wape(dist,Y_test)[0])
print("RWSE = ", rwse(dist,Y_test)[0])
print("NLL = ", nll(dist,Y_test)[0])
print("TrNLL = ", nll(dist_tr,Y_train)[0])

samples = dist.sample((1000,))
print("DScore = ", index_allocation(samples, min_indices, 
                                    obj_fn, Y_test, 0.8))
### Conditional GMM

# train
wapes, rwses, nlls, trnlls, dscores = [], [], [], [], []
for seed in range(ntrials):
    cgmm = ConditionalGMM(1, past_dims, 1, fut_dims, 
                          n_components=ncomp, random_state=seed)
    try:
        cgmm.fit(X_train, Y_train)
    except:
        continue

    # test metrics
    dist = cgmm(X_test)
    dist_tr = cgmm(X_train)
    wapes.append(wape(dist,Y_test)[0])
    rwses.append(rwse(dist,Y_test)[0])
    nlls.append(nll(dist,Y_test)[0])
    trnlls.append(nll(dist_tr,Y_train)[0])

    samples = dist.sample((1000,))
    dscores.append(index_allocation(samples, min_indices, 
                                    obj_fn, Y_test, 0.8))
    print('Seed ', seed, ' done')

print('CondGMM Metrics:')
print("WAPEs = ", torch.stack(wapes))
print("RWSEs = ", torch.stack(rwses))
print("NLLs = ", torch.stack(nlls))
print("TrNLLs = ", torch.stack(trnlls))
print("DScores = ", torch.stack(dscores))