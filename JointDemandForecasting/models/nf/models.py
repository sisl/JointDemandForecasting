# from https://github.com/tonyduan/normalizing-flows

import torch
import torch.nn as nn


class NormalizingFlowModel(nn.Module):

    def __init__(self, prior, flows, random_state=None):
        super().__init__()
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(self.random_state)
        self.prior = prior
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z)
        return x
    
    def log_prob(self, x, y):
        """ 

        Evaluate log pdf of input-output pairs
        
        Args:
            x (torch.tensor): (B, T, ydim) past observations
            y (torch.tensor): (B, K, ydim) future observations
            
        Returns:
            log_prob (torch.tensor): (B,) log pdf under joint distribution
        """
        device = torch.device("cuda" if y.is_cuda else "cpu")
        
        B = x.shape[0]
        B2 = y.shape[0]
        assert B==B2, "batch dimension mismatch"
        
        X = x.reshape((B,-1))
        Y = y.reshape((B,-1))
        joint = torch.cat((X,Y), 1)
        z, prior_logprob, log_det = self.forward(joint)
        log_prob = prior_logprob + log_det
        return log_prob