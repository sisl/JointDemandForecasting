#
# lineargrw.py
#

"""
Linear Gaussian Random Walk Model
"""

import torch
from torch import nn
from CalibratedTimeseriesModels.models.grw import *

def bmm(A,b):
    """
    Args:
        A (torch.tensor): (*,n,n) matrix
        b (torch.tensor): (*,n) vector
    Returns:
        c (torch.tensor): (*,n) mvp
    """
    return (A @ b.unsqueeze(-1)).squeeze(-1)

class LinearGRWModel(GaussianRandomWalkModel):

    def __init__(self, A, B, Q):
        self._ydim = A.shape[0]
        self._xdim = A.shape[0]
        self._udim = B.shape[0] if B is not None else None
        super().__init__(torch.zeros((self._xdim,)), Q)

        self._A = nn.Parameter(A.unsqueeze(0))
        self._B = nn.Parameter(B.unsqueeze(0)) if B is not None else None

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

        yprev = y[:,-1]
        ys = []

        for t in range(K):
            yt = bmm(self._A, yprev) 
            if self._B is not None:
                yt += + bmm(self._B, u[:,-K-1+t])
            ys.append(yt)
            yprev = yt

        return torch.stack(ys, dim=1)

    def forward_cov_chol(self, y, u, K):
        """
        Compute cov function of GRW
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T+K,udim) inputs
            K (int): horizon to predict 
        Returns:
            cov_chol (torch.tensor): (B, K, ydim, ydim) cov estimates 
        """

        B,_,ydim = y.shape

        Q = self._cov_chol @ self._cov_chol.T
        Q = Q.unsqueeze(0).expand(B,ydim,ydim)

        Ps = [Q]

        for t in range(1,K):
            P = self._A @ Ps[-1] @ self._A.transpose(-2,-1) + Q
            Ps.append(P)

        Ps = torch.stack(Ps, dim=1)

        cov_chol = Ps.cholesky()

        return cov_chol