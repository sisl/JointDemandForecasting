import numpy as np
import matplotlib.pyplot as plt
#import matplotlib2tikz as mpl2t
import itertools
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

sys.path.append("../../")
import JointDemandForecasting
from JointDemandForecasting.utils import *
from JointDemandForecasting.models.cgmm import *
from JointDemandForecasting.models.gmnn import *
from JointDemandForecasting.models.nf.flows import *
from JointDemandForecasting.models.nf.models import *

from sklearn import datasets

# get scikitlearn datasets
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

n_train, n_cv, n_test = 30, 1000, 1000
n_samples = n_train+n_cv+n_test
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)

#dataset, name = torch.tensor(noisy_moons[0]).float(), 'moons'
dataset, name = torch.tensor(noisy_circles[0]).float(), 'circles'

past_dims, fut_dims = 1, 1
train, cv, test = dataset[:n_train], dataset[n_train:n_train+n_cv], dataset[n_train+n_cv:]
X_train, Y_train = train[:,:past_dims].unsqueeze(-1), train[:,past_dims:].unsqueeze(-1)
X_test, Y_test = test[:,:past_dims].unsqueeze(-1), test[:,past_dims:].unsqueeze(-1)


# Train 3 Joint Models: GMM, NF, ANF

ncomp = 5
nflows, hdim, lr, iters  = (10, 10, 0.005, 500)
print_every = 20
nsamples = 100000
ncomps = 10
use_cuda = torch.cuda.is_available() # False
log_level = logging.DEBUG # logging.DEBUG for verbose training, logging.ERROR for just errors


# fit condGMM
gmm = ConditionalGMM(1, past_dims, 1, fut_dims, 
                      n_components=ncomp, random_state=seed)
gmm.fit(X_train, Y_train)
gmm_log_probs = gmm.log_prob(X_test, Y_test).detach()

# Fit NF
flows = [RealNVP(dim=train.shape[1], hidden_dim = hdim, 
                    base_network=FCNN) for _ in range(nflows)]
prior = D.MultivariateNormal(torch.zeros(train.shape[1]),
                            torch.eye(train.shape[1]))
model = NormalizingFlowModel(prior, flows, random_state=seed)

if use_cuda:
    torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")

logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for i in range(iters):
    optimizer.zero_grad()
    z, prior_logprob, log_det = model(train)
    logprob = prior_logprob + log_det
    loss = -torch.mean(prior_logprob + log_det)
    loss.backward()
    optimizer.step()
    if (i+1) % print_every == 0:
        _, prior_logprob_test, log_det_test = model(test)
        logprob_test = prior_logprob_test + log_det_test
        logger.info(f"Iter: {i+1}\t" +
                    f"Logprob: {logprob.mean().data:.2f}\t" +
                    f"Prior: {prior_logprob.mean().data:.2f}\t" +
                    f"LogDet: {log_det.mean().data:.2f}\t" + 
                    f"Logprob_test: {logprob_test.mean().data:.2f}")

model.eval()


# sample model
many_samples = model.sample(nsamples).detach()
X_train_sampled = many_samples[:,:past_dims].unsqueeze(-1)
Y_train_sampled = many_samples[:,past_dims:].unsqueeze(-1)

# fit approximate norm flow
anf = ConditionalGMM(1, past_dims, 1, fut_dims, 
                      n_components=ncomps, random_state=seed)
anf.fit(X_train_sampled, Y_train_sampled)

# log probs
test_flow = model.log_prob(X_test, Y_test).detach()
test_anf = anf.log_prob(X_test, Y_test).detach()

#### PLOTS
#log_level = logging.ERROR
#logging.basicConfig(level=log_level)
#logger = logging.getLogger(__name__)

plt.figure()
plt.scatter(X_train[:,0,0], Y_train[:,0,0], c='k')
plt.savefig('results/simple_'+name+'_train_points.png')

n=100
x, y = np.meshgrid(np.linspace(-1.5, 2.5, n), np.linspace(-1, 1.5, n))
x = torch.tensor(x).float()
y = torch.tensor(y).float()
xrs = x.reshape((-1,1,1))
yrs = y.reshape((-1,1,1))
gmm_lp = gmm.log_prob(xrs, yrs).detach().reshape((n,n))
nf_lp = model.log_prob(xrs, yrs).detach().reshape((n,n))
anf_lp = anf.log_prob(xrs, yrs).detach().reshape((n,n))

# gmm
fig, ax = plt.subplots()
c=ax.pcolormesh(x, y, gmm_lp, vmin=-15)
ax.scatter(X_test[:,0,0], Y_test[:,0,0], c='k')
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
plt.savefig('results/simple_'+name+'_gmm.png')

# nf
fig, ax = plt.subplots()
c=ax.pcolormesh(x, y, nf_lp, vmin=-15)
ax.scatter(X_test[:,0,0], Y_test[:,0,0], c='k')
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
plt.savefig('results/simple_'+name+'_nf.png')

# anf
fig, ax = plt.subplots()
c=ax.pcolormesh(x, y, anf_lp, vmin=-15)
ax.scatter(X_test[:,0,0], Y_test[:,0,0], c='k')
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
plt.savefig('results/simple_'+name+'_anf.png')


# print test log probs
print('GMM LogProb = {}'.format(gmm_log_probs.mean()))
print('NF LogProb = {}'.format(test_flow.mean()))
print('ANF LogProb = {}'.format(test_anf.mean()))
