#
# ukfgrw.py
#

"""
EKF-GRW with parameter uncertainty
"""

import torch
from torch import nn
from CalibratedTimeseriesModels.models.grw2 import *
import numpy as np
from CalibratedTimeseriesModels.utils import *
from torch.nn.utils import parameters_to_vector as ptv
from torch.nn.utils import vector_to_parameters as vtp


class PUEKFGRWModel(GaussianRandomWalkModel):

    def __init__(self, system, Q, T):
        """
        Args:
            system (nn.Module): system.step should map (t,yt,ut) -> yt+1
            Q (torch.tensor): (xdim, xdim) covariance
            T (torch.tensor): (paramdim, paramdim) covariance
        Notes:
            https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter
            Implementation uses the mean set sigma points under these recommendations:
            https://nbviewer.jupyter.org/github/sbitzer/UKF-exposed/blob/master/UKF.ipynb

        """
        self._ydim = Q.shape[0]
        self._xdim = Q.shape[0]
        super().__init__(torch.zeros((self._xdim,)), Q)

        self._model = system
        self._meanprm = ptv(system.parameters())

        self._paramdim = len(self._meanprm)
        
        T_chol = T.cholesky()
        self._T_chol_logdiag = nn.Parameter(T_chol.diag().log())
        self._tril_inds = np.tril_indices(T.shape[0], -1)
        self._T_chol_tril = nn.Parameter(T_chol[self._tril_inds])

    @property
    def _prm_cov_chol(self):
        cov_chol_diag = self._T_chol_logdiag.exp()
        cov_tril = self._T_chol_tril

        cov_chol = cov_chol_diag.new_zeros(self._paramdim, self._paramdim)
        cov_chol = bfill_lowertriangle(cov_chol, cov_tril)
        cov_chol = bfill_diagonal(cov_chol, cov_chol_diag)

        return cov_chol

    def forward_mu_jac_x_th(self, y, u, K):
        """
        Compute mean function and cov of GRW
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T+K,udim) inputs
            K (int): horizon to predict 
        Returns:
            mu (torch.tensor): (B, K, ydim) mean estimates 
            jac_x (torch.tensor): (B, K, ydim, ydim) jacobians
            jac_th (torch.tensor): (B, K, ydim, paramdim)
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

        jac_th_d = self._model.jac_step_theta(ts, mu, uinp)

        jac_th = []

        for k,v in jac_th_d.items():
            jac_th.append(v.view(*v.shape[:-1], -1))

        jac_th = torch.cat(jac_th, dim=-1)

        return mu, jac_x, jac_th

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

        mu, jac_x, jac_th = self.forward_mu_jac_x_th(y, u, K)
        mu = mu.reshape(B,ydim*K) 
        mu = torch.cat([self._meanprm.view(1,self._paramdim).expand(B,self._paramdim),
                        mu], dim=1)
        ydim = self._ydim
        
        cov_chol = self._cov_chol.unsqueeze(0).expand(B,ydim,ydim)

        prm_cov_chol = self._prm_cov_chol.unsqueeze(0).expand(B,ydim,ydim)

        # compute covariances resulting from theta

        leftcol = [prm_cov_chol, jac_th[:,0] @ prm_cov_chol]

        prev = jac_th[:,0]

        for i in range(1, K):
            prev = jac_x[:,i] @ prev + jac_th[:,i] @ prm_cov_chol
            leftcol.append(prev)

        leftcol = torch.cat(leftcol, dim=1)

        Sig_chol = y.new_zeros(B,K*ydim,K*ydim)

        for i in range(K):
            mat = cov_chol
            for j in range(i+1):
                xoff = i*ydim
                yoff = (i-j)*ydim
                Sig_chol[:, xoff:xoff+ydim, yoff:yoff+ydim] = mat
                mat = jac_x[:,j] @ mat

        Sig_chol = torch.cat([torch.zeros(B,self._paramdim, K*ydim), Sig_chol], dim=1)

        Sig_chol = torch.cat([leftcol, Sig_chol], dim=-1)

        dist = mvn(loc=mu, scale_tril=Sig_chol)

        return dist
