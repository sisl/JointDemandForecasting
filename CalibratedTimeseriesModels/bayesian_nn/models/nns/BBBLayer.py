import torch.nn as nn
import torch
from torch.autograd import Variable
from codebase import utils as ut

class BBBLayer(nn.Module):
    """
    a base class for all BBB layer with gaussian mixture prior
    """
    def __init__(self, pi, std1, std2, gpu, BBB, bias=True):
        super(BBBLayer, self).__init__()
        self.pi = pi
        self.std1 = std1
        self.std2 = std2
        self.gpu = gpu
        self.BBB = BBB
        self.bias = bias

        self.sampled_weights = []
        self.sampled_sharpen_weights = []
        self.means = []
        self.logvars = []
        self.h_post_means = []

    def sample(self):
        assert self.BBB is True

        self.sampled_weights = [] # clear samples
        for i in range(len(self.means)):
            mean = self.means[i]
            logvar = self.logvars[i]
            eps = torch.randn_like(logvar)
            if self.gpu:
                eps = eps.cuda()

            # eps.normal_()
            std = logvar.mul(0.5).exp()
            weight = mean + eps * std
            self.sampled_weights.append(weight)

    def resample_with_sharpening(self, grads, eta, std=0.02):
        self.sampled_sharpen_weights = []
        self.h_post_means = []
        for i in range(len(self.sampled_weights)):
            w = self.sampled_weights[i]
            # Random number
            eps = torch.randn_like(w)
            if self.gpu:
                eps = eps.cuda()

            # eps.normal_()
            g = grads[i].detach()
            # Sample fron normal wih posterior sharpening
            h_post_means = (w - eta[i] * g)
            weight = h_post_means + eps * std
            self.h_post_means.append(h_post_means)
            self.sampled_sharpen_weights.append(weight)

    def get_kl_sharpening(self, sigma=0.02):
        kl = 0
        for i in range(len(self.sampled_weights)):
            sharp_w = self.sampled_sharpen_weights[i]
            w = self.sampled_weights[i].detach()
            # without constant term
            kl += torch.sum((sharp_w - w).pow(2) / (2 * sigma**2))

        return kl

    def get_kl(self):
        """
        Use the current sampled weights to calculate the KL divergence 
        from posterior to prior.
        :return: One point estimate for KL, computed through the sum of the 
            elementwise log-likelihood of all (this sampled) weights.
        """
        assert len(self.sampled_weights) != 0 # make sure we sample weights

        log_posterior = ut.log_normal_for_weights(
            weights=self.sampled_weights,
            means=[mean.detach() for mean in self.means],
            logvars=[logvar.detach() for logvar in self.logvars])

        log_prior = ut.log_scale_gaussian_mix_prior(self.sampled_weights, 
                                pi=self.pi, std1=self.std1, std2=self.std2)

        kl = log_posterior - log_prior
        return kl


