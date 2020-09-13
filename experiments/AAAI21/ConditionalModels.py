import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append("../../")
import CalibratedTimeseriesModels

from CalibratedTimeseriesModels.utils import *
from CalibratedTimeseriesModels.models.cgmm import *
from CalibratedTimeseriesModels.models.blr import *

from load_data import load_data

### Experiment Settings (uncomment one of these)

# location: 1=Bakersfield,9=SLC
# past_dims, fut_dims: past and future time step, either (24,8) or (8,12)
# ncomp: number of components used in mixture, more used for longer output
# seed: seeds used in final experiments (arbitrarily arrived at during rapid prototyping)

loc, past_dims, fut_dims, ncomp, seed = (1, 24, 8, 4, 4)
#loc, past_dims, fut_dims, ncomp, seed = (9, 24, 8, 4, 5)
#loc, past_dims, fut_dims, ncomp, seed = (1, 8, 12, 5, 4)
#loc, past_dims, fut_dims, ncomp, seed = (9, 8, 12, 5, 4)


### Load Data

X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)

### Conditional Gaussian

# train
lin_reg = BayesianLinearRegression(1, past_dims, 1, fut_dims)
lin_reg.fit(X_train, Y_train)

# test metrics
for f in [mape, wape, rmse, rwse, nll]:
    print(f(lin_reg(X_test),Y_test))

# train nll
print(nll(lin_reg(X_train),Y_train))

### Conditional GMM

# train
cgmm = ConditionalGMM(1, past_dims, 1, fut_dims, 
                      n_components=ncomp, random_state=seed)
cgmm.fit(X_train, Y_train)

# test metrics
for f in [mape, wape, rmse, rwse, nll]:
    print(f(cgmm(X_test),Y_test))

# train nll
print(nll(cgmm(X_train),Y_train))





