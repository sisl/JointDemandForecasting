#
# ukfgrw.py
#

"""
GRW for nonlinear models using UKF
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

class UKFGRWModel(GaussianRandomWalkModel):

    def __init__(self, system, Q, w0=1./3):
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

        self._w0 = w0

    def forward_mu_jac_x(self, y, u, K):
        """
        Compute mean function and cov of GRW
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T+K,udim) inputs
            K (int): horizon to predict 
        Returns:
            mu (torch.tensor): (B, K, ydim) mean estimates 
            jac_x (torch.tensor): (B, K, ydim, ydim) jacobian estimates 
        """


        B,_,ydim = y.shape
        Q = self._cov_chol @ self._cov_chol.T

        N = 2*ydim + 1

        jac_x = []
        mus = []
        # sigma = y[:,-1:].repeat(1,N,1).view(-1,1,ydim)

        Wa = torch.tensor([self._w0] + [(1.-self._w0)/(2.*ydim)]*(2*ydim))
        Wa = Wa.reshape(1,N,1)
        Wc = torch.tensor([self._w0] + [(1.-self._w0)/(2.*ydim)]*(2*ydim))
        Wc = Wc.reshape(1,N,1)

        A = self._cov_chol.unsqueeze(0).repeat(B,1,1)

        fac = np.sqrt(ydim / (1. - self._w0))
        sigma = torch.cat([torch.zeros(B,1,ydim),
                        fac*A.transpose(-2,-1),
                       -fac*A.transpose(-2,-1)],dim=1) + y[:,-1:]

        sigma = sigma.view(-1,1,ydim)

        covt = (self._cov_chol @ self._cov_chol.T).unsqueeze(0)

        for t in range(K):
            covt_prev = covt

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
            yj_ = yj - mu
            covt = (Wc * yj_).transpose(-2,-1) @ yj_ + Q.unsqueeze(0)

            try:
                A = covt.cholesky()
            except RuntimeError:
                import ipdb
                ipdb.set_trace()

            # approximate jacobian using least squares
            sigma_ = sigma.view(B,N,ydim)
            X = (sigma_ - sigma_[:,0:1])[:,1:]
            Y = yj_[:,1:]
            Xt = X.transpose(-2,-1)
            J = torch.solve(Xt @ Y, Xt @ X)[0]

            jac_x.append(J)

            fac = np.sqrt(ydim / (1. - self._w0))
            sigma = torch.cat([torch.zeros(B,1,ydim),
                            fac*A.transpose(-2,-1),
                           -fac*A.transpose(-2,-1)],dim=1) + mu

            sigma = sigma.view(-1,1,ydim)

        mus = torch.cat(mus, dim=1)
        jac_x = torch.stack(jac_x, dim=1)

        return mus, jac_x