#
# ukfgrw.py
#

"""
GRW for nonlinear models using UKF
"""

import torch
from torch import nn
from CalibratedTimeseriesModels.models.grw import *
import numpy as np

def bmm(A,b):
    """
    Args:
        A (torch.tensor): (*,n,n) matrix
        b (torch.tensor): (*,n) vector
    Returns:
        c (torch.tensor): (*,n) mvp
    """
    return (A @ b.unsqueeze(-1)).squeeze(-1)

class UKFGRWModel(GaussianRandomWalkModel):

    def __init__(self, system, Q):
        """
        Args:
            system (nn.Module): system.step should map (t,yt,ut) -> yt+1
            Q (torch.tensor): (xdim, xdim) covariance
        """
        self._ydim = Q.shape[0]
        self._xdim = Q.shape[0]
        super().__init__(torch.zeros((self._xdim,)), Q)

        self._model = system

        self._alpha = 1e-3
        self._kappa = 1.0
        self._beta = 2.0

    def forward_mu_cov_chol(self, y, u, K):
        """
        Compute mean function and cov of GRW
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T+K,udim) inputs
            K (int): horizon to predict 
        Returns:
            mu (torch.tensor): (B, K, ydim) mean estimates 
            cov_chol (torch.tensor): (B, K, ydim, ydim) cov estimates 
        Notes:
            https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter
        """


        B,_,ydim = y.shape
        al, ka, be = self._alpha, self._kappa, self._beta
        Q = self._cov_chol @ self._cov_chol.T

        N = 2*ydim + 1

        cov_chols = []
        mus = []
        sigma = y[:,-1:].repeat(1,N,1).view(-1,1,ydim)

        Wa = torch.tensor([1. - ydim/(al**2 * ka)] + [1./(2*al**2 * ka)]*(2*ydim))
        Wa = Wa.reshape(1,N,1)
        Wc = torch.tensor([2. - ydim/(al**2 * ka) - al**2 - be] + [1./(2*al**2 * ka)]*(2*ydim))
        Wc = Wc.reshape(1,N,1)

        for t in range(K):
            # propegate
            tinp = torch.tensor([t],dtype=torch.get_default_dtype())
            tinp = tinp.view(1,1).expand(B*N,1)
            if u is not None:
                utm1 = u[:,-K-1+t:]
                utm1 = utm1.repeat(1,N,1).view(-1,1,self._udim)
            else:
                utm1 = None

            yj = self._model.step(tinp, sigma, utm1)

            # recompute sigma points
            yj = yj.reshape(B,N,ydim)
            mu = (Wa * yj).sum(1, keepdims=True)
            mus.append(mu)
            yj = yj - mu
            covt = (Wc * yj).transpose(-2,-1) @ yj + Q.unsqueeze(0)

            try:
                A = covt.cholesky()
            except RuntimeError:
                import ipdb
                ipdb.set_trace()
            cov_chols.append(A)

            sigma = torch.cat([torch.zeros(B,1,ydim),
                            al*np.sqrt(ka)*A.transpose(-2,-1),
                           -al*np.sqrt(ka)*A.transpose(-2,-1)],dim=1) + mu

            sigma = sigma.view(-1,1,ydim)

        mus = torch.cat(mus, dim=1)
        cov_chols = torch.stack(cov_chols, dim=1)

        return mus, cov_chols
