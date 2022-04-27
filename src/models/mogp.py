import torch
import torch.nn as nn
import torch.distributions as D
import gpytorch
import numpy as np

class MultiOutputGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 covar_kernel=gpytorch.kernels.RBFKernel(), 
                 index_rank=2, random_state=None):
        
        B, num_tasks = train_y.shape
        B2, past_dims = train_x.shape
        assert B==B2, "nonmatching data batch size"
        self.past_dims = past_dims
        self.num_tasks = num_tasks
        self.index_rank = index_rank
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        super(MultiOutputGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=self.num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            covar_kernel, num_tasks=self.num_tasks, rank=self.index_rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def forward_independent(self, x, verbose=True):
        # builds mvg assuming data points independent (for calculating nll)
        # assumes x is a (B, d) tensor
        # returns a d-dim joint MVG with batch_dim B
        
        idxs = x.size(0)
        device = torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")
        
        locs = torch.zeros(idxs, self.num_tasks).to(device)
        covar = torch.zeros(idxs, self.num_tasks, self.num_tasks).to(device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(idxs):
                single_sample = x[[i],:]
                d_single = self(single_sample)
                locs[i,:] = d_single.loc
                covar[i,:,:] = d_single.covariance_matrix
                if verbose and i % 500 == 0:
                    print('Point %d out of %d' %(i,idxs))
    
        dist = D.MultivariateNormal(locs, covar)
        return dist
        