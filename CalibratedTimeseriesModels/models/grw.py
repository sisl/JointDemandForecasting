#
# grw.py
#

# Gaussian Random Walk model

from CalibratedTimeseriesModels.abstractmodels import *
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvn

class GaussianRandomWalkModel(ExplicitPredictiveModel):

    def __init__(self, drift, cov):
        """
        Args:
            drift (torch.tensor): (xdim,) RW drift
            cov (torch.tensor): (xdim, xdim) RW covariance
        """

        super().__init__()

        self._drift = nn.Parameter(drift)
        self._cov_chol = nn.Parameter(cov.cholesky())
        self._ydim = len(drift)

    def forward_mu(self, y, u, K):
        B,_,ydim = y.shape

        drift = self._drift
        ydim = self._ydim

        t = torch.sqrt(torch.arange(K,dtype=torch.get_default_dtype())+1.)

        mu = (t.unsqueeze(1) * drift.unsqueeze(0)).unsqueeze(0) 
        mu = mu + y[:,-1:]
        mu = mu.reshape(B,ydim*K) 

        return mu 

    def forward(self, y, u, K):
        """
        Predicts distribution over next K observations.
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T,udim) inputs
            K (int): horizon to predict 
        Returns:
            dist (PredictiveDistribution): predictive distribution over next K observations shaped (B,K*ydim)
        Notes:
            We've basically just extended 1D Brownian motion to N-D using
            the cholesky of the covariance matrix. 
            See Eqn 6.5: http://statweb.stanford.edu/~owen/mc/Ch-processes.pdf
        """

        B,_,ydim = y.shape

        mu = self.forward_mu(y, u, K)

        cov_chol = self._cov_chol
        ydim = self._ydim
        
        timematrix = torch.tril(torch.ones(K,K))

        Sig_chol = timematrix.view(K,1,K,1) * cov_chol.view(1,ydim,1,ydim)

        Sig_chol = Sig_chol.reshape(K*ydim, K*ydim)

        Sig_chol = Sig_chol.unsqueeze(0).expand(B,*Sig_chol.shape)

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

    def forward(self, y, u, nsamps, K):
        """
        Samples from predictive distribution over next K observations.
        Args:
            y (torch.tensor): (B,T,ydim) observations
            u (torch.tensor or None): (B,T,udim) inputs
            nsamps (int): number of samples
            K (int): horizon to predict 
        Returns:
            ypredsamps (torch.tensor): (nsamps,B,K,ydim) samples of predicted observations
        """

        # Note: this method is slower than it could be because
        # we are instantiating a GP every time we sample. However
        # this is good for subclasses where we override the mean function.

        B, _ ,ydim = y.shape

        dist = super().forward(y,u,K)
        ypredsamps = dist.sample((nsamps,)).reshape(nsamps,B,K,ydim)

        return ypredsamps
