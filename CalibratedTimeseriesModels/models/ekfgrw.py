#
# ukfgrw.py
#

"""
GRW for nonlinear models using EKF
"""

import torch
from torch import nn
from CalibratedTimeseriesModels.models.grw2 import *
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

class EKFGRWModel(GaussianRandomWalkModel):

    def __init__(self, system, Q):
        """
        Args:
            system (nn.Module): system.step should map (t,yt,ut) -> yt+1
            Q (torch.tensor): (xdim, xdim) covariance
        Notes:
            https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter
            Implementation uses the mean set sigma points under these recommendations:
            https://nbviewer.jupyter.org/github/sbitzer/UKF-exposed/blob/master/UKF.ipynb

        """
        self._ydim = Q.shape[0]
        self._xdim = Q.shape[0]
        super().__init__(torch.zeros((self._xdim,)), Q)

        self._model = system

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


        B,_,ydim = y.shape

        # rollout
        ys = []
        ny = y[:,-1:]
        ts = []
        for t in range(K):
            tinp = torch.tensor([[t]], dtype=torch.get_default_dtype()).repeat(B,1)
            uinp = u[:,T+t:T+t+1] if u is not None else None
            ny = self._model.step(tinp, ny, uinp)
            ts.append(tinp)
            ys.append(ny)


        ts = torch.cat(ts, dim=1).detach()
        mu = torch.cat(ys, dim=1)
        uinp =  u[:,T:T+K] if u is not None else None

        # compute jac
        jac_x = self._model.jac_step_x(ts, mu, uinp)

        return mu, jac_x
