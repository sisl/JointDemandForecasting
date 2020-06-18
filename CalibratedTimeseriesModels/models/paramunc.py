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


class EKFGRWModel(GaussianRandomWalkModel):

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

        jac_th = self._model.jac_step_theta(ts, mu, uinp).values()

        for i in range(len(jac_th)):
            jac_th[i] = jac_th[i].view(*jac_th[i].shape[:-1], -1)

        jac_th = torch.cat(jac_th[-1], dim=-1)

        return mu, jac_x, jac_th
