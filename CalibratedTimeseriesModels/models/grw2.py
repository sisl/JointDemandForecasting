#
# grw.py
#

# Gaussian Random Walk model

from CalibratedTimeseriesModels.abstractmodels import *
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvn

import numpy as np

def bfill_lowertriangle(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.tril_indices(A.size(-2), k=-1, m=A.size(-1))
    A[..., ii, jj] = vec
    return A


def bfill_diagonal(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    A[..., ii, jj] = vec
    return A

class GaussianRandomWalkModel(ExplicitPredictiveModel):

    def __init__(self, drift, cov):
        """
        Args:
            drift (torch.tensor): (xdim,) RW drift
            cov (torch.tensor): (xdim, xdim) RW covariance
        """

        super().__init__()

        self._ydim = len(drift)
        self._drift = nn.Parameter(drift)
        
        cov_chol = cov.cholesky()
        self._cov_chol_logdiag = nn.Parameter(cov_chol.diag().log())
        self._tril_inds = np.tril_indices(cov.shape[0], -1)
        self._cov_chol_tril = nn.Parameter(cov_chol[self._tril_inds])


    @property
    def _cov_chol(self):
        cov_chol_diag = self._cov_chol_logdiag.exp()
        cov_tril = self._cov_chol_tril

        cov_chol = cov_chol_diag.new_zeros(self._ydim, self._ydim)
        cov_chol = bfill_lowertriangle(cov_chol, cov_tril)
        cov_chol = bfill_diagonal(cov_chol, cov_chol_diag)

        return cov_chol
    

    def forward_mu(self, y, u, K):
        """
        Compute mean function of GRW
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T+K,udim) inputs
            K (int): horizon to predict 
        Returns:
            mu (torch.tensor): (B, K, ydim) mean estimates 
        """

        B,_,ydim = y.shape

        drift = self._drift
        ydim = self._ydim

        t = torch.sqrt(torch.arange(K,dtype=torch.get_default_dtype())+1.)

        mu = (t.unsqueeze(1) * drift.unsqueeze(0)).unsqueeze(0) 
        
        mu = mu + y[:,-1:]

        return mu 

    def forward_jac_x(self, y, u, K):
        """
        Compute cov function of GRW
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T+K,udim) inputs
            K (int): horizon to predict 
        Returns:
            jac_x (torch.tensor): (B, K, ydim, ydim) dynamic jacobians
        """
        cov_chol = self._cov_chol
        B,_,ydim = y.shape

        return torch.eye(ydim).view(1,1,ydim,ydim).expand(B,K,ydim,ydim)

    def forward_mu_jac_x(self, y, u, K):
        """
        Compute mean function and cov of GRW
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T+K,udim) inputs
            K (int): horizon to predict 
        Returns:
            mu (torch.tensor): (B, K, ydim) mean estimates 
            jac_x (torch.tensor): (B, K, ydim, ydim) jacobians
        """

        return self.forward_mu(y, u, K), self.forward_jac_x(y, u, K)


    def forward(self, y, u, K):
        """
        Predicts distribution over next K observations.
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T+K,udim) inputs
            K (int): horizon to predict 
        Returns:
            dist (PredictiveDistribution): predictive distribution over next K observations shaped (B,K*ydim)
        Notes:
            We've basically just extended 1D Brownian motion to N-D using
            the cholesky of the covariance matrix. 
            See Eqn 6.5: http://statweb.stanford.edu/~owen/mc/Ch-processes.pdf
        """

        B,_,ydim = y.shape

        mu, jac_x = self.forward_mu_jac_x(y, u, K)
        mu = mu.reshape(B,ydim*K) 
        ydim = self._ydim
        
        cov_chol = self._cov_chol.unsqueeze(0).expand(B,ydim,ydim)

        Sig_chol = y.new_zeros(B,K*ydim,K*ydim)

        for i in range(K):
            mat = cov_chol
            for j in range(i+1):
                xoff = i*ydim
                yoff = (i-j)*ydim
                Sig_chol[:, xoff:xoff+ydim, yoff:yoff+ydim] = mat
                mat = jac_x[:,j] @ mat

        dist = mvn(loc=mu, scale_tril=Sig_chol)

        return dist

class GenerativeGRWModel(GaussianRandomWalkModel, GenerativePredictiveModel):

    def __init__(self, drift, cov):
        """
        Args:
            drift (torch.tensor): (xdim,) RW drift
            cov (torch.tensor): (xdim, xdim) RW covariance
        """
        super().__init__(drift, cov)

    def dist(self, y, u, K):
        return super().forward(y,u,K)

    def forward(self, y, u, nsamps, K):
        """
        Samples from predictive distribution over next K observations.
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T+K,udim) inputs
            nsamps (int): number of samples
            K (int): horizon to predict 
        Returns:
            ypredsamps (torch.tensor): (nsamps,B,K,ydim) samples of predicted observations
        """

        # Note: this method is slower than it could be because
        # we are instantiating a GP every time we sample. However
        # this is good for subclasses where we override the mean function.

        B, _ ,ydim = y.shape

        dist = self.dist(y,u,K)
        ypredsamps = dist.sample((nsamps,)).reshape(nsamps,B,K,ydim)

        return ypredsamps

