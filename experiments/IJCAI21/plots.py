import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz as mpl2t

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

from load_data import load_data
from charging_utils import *

save_tex = True # whether to save the .tex files or .pngs
if not os.path.isdir('figs'):
    os.mkdir('figs')

########## PLOTS ###############

### 1. Per-index error plot

# Taken from running default models for 12 |8 in Bakersfield, CA, with seeds set to 0. 
# The second return of the JointDemandForecasting.utils.rwse is per-index RWSE.

cgmmrwse = np.array([0.0200, 0.0699, 0.1258, 0.1887, 0.2758, 0.3668, 0.4395, 0.4754, 0.4792,
        0.4469, 0.3779, 0.2983])
fnnrwse = np.array([0.0119, 0.0465, 0.0975, 0.1674, 0.2639, 0.3735, 0.4721, 0.5437, 0.5895,
        0.6148, 0.6265, 0.6335])
canfrwse = np.array([0.0109, 0.0341, 0.0586, 0.0889, 0.1278, 0.1699, 0.1959, 0.2054, 0.2096,
        0.2000, 0.1761, 0.1529])

x = np.arange(1,13)
width = 0.17
gap = .04
plt.figure()
fig, ax = plt.subplots()
fr = ax.bar(x - width, fnnrwse, width, label='FNN-12')
cr = ax.bar(x        , cgmmrwse, width, label='CondGMM')
fr = ax.bar(x + width, canfrwse, width, label='CANF')
ax.set_ylim([-0.02, 0.8])
ax.legend(mode='expand', ncol=3)
ax.set_aspect(5)
plt.xlabel('Future time index')
plt.ylabel('Error metric')
plt.grid(True)
if save_tex:
    mpl2t.save('figs/errors.tex')
else:
    plt.savefig('figs/errors.png')
    plt.show()

### 2. Sample Forecast Plot
past_dims, fut_dims, loc = (8, 12, 9)
X_train, Y_train_full, X_test, Y_test_full = load_data(loc, past_dims, fut_dims)
Y_train = Y_train_full[:,[0],:]
Y_test = Y_test_full[:,[0],:]

X_batches, Y_batches = batch(X_train, Y_train, batch_size = 128)

#rnn12
hidden_layers = [20, 20, 20]
hidden_dim = 40
rnn12 = GaussianMixtureLSTM(1, hidden_dim, hidden_layers, 1, 1, 
                                 n_components=3, random_state=0)
train(rnn12, X_batches, Y_batches, num_epochs=200, learning_rate=.005)
dtest_rnn12 = rnn12(X_test)
samples_rnn12 = sample_forward_lstm(rnn12, X_test, fut_dims, n_samples=25)

#canf
nflows, hdim, iters = (10, 32, 4000)
seed=0


# GNN Approximation hyperparameters
nsamples = 100000
ncomps = 25

exp_string = 'loc_'+str(loc)+'.tot_dims_'+str(past_dims+fut_dims) + '.nflows_' + str(nflows) + '.hdim_' + str(hdim) + '.iters_' + str(iters) + '.seed_' + str(seed)

an = ActNorm(dim=past_dims+fut_dims)
flows = [RealNVP(dim=past_dims+fut_dims, hidden_dim = hdim, 
                 base_network=FCNN) for _ in range(nflows)]
prior = D.MultivariateNormal(torch.zeros(past_dims+fut_dims),
                            torch.eye(past_dims+fut_dims))
model = NormalizingFlowModel(prior, flows, random_state=seed)
path = 'models/'+  exp_string +'.pt'
assert os.path.isfile(path), 'Model not yet trained'
print('Model Loaded')
model.load_state_dict(torch.load(path))
model.eval()

# sample
many_samples = model.sample(nsamples).detach()
X_train_sampled = many_samples[:,:past_dims].unsqueeze(-1)
Y_train_sampled = many_samples[:,past_dims:].unsqueeze(-1)

# fit approximate normflow
cgmm = ConditionalGMM(1, past_dims, 1, fut_dims, 
                      n_components=ncomps, random_state=seed)
cgmm.fit(X_train_sampled, Y_train_sampled)
print('GMM FIT')

dtest_cgmm = cgmm(X_test)
samples_cgmm = dtest_cgmm.sample((25,))


# plot samples
plt.figure()
idxs = [100, 200, 300, 400]
fig, axs = plt.subplots(2,2, sharex='col', sharey='row')

for i in range(len(idxs)):
    r = i // 2
    c = i % 2
    axs[r,c].plot(range(-past_dims+1,fut_dims+1),
                torch.cat((X_test[idxs[i],:,0],
                           Y_test_full[idxs[i],:,0]),0), 
                color='k', label='True Demand')
    cgmm_lower = torch.min(samples_cgmm[:,idxs[i],:],dim=0)[0].squeeze()
    cgmm_upper = torch.max(samples_cgmm[:,idxs[i],:],dim=0)[0].squeeze()
    rnn12_lower = torch.min(samples_rnn12[:,idxs[i],:],dim=0)[0].squeeze()
    rnn12_upper = torch.max(samples_rnn12[:,idxs[i],:],dim=0)[0].squeeze()
    axs[r,c].fill_between(range(1,fut_dims+1),rnn12_lower, rnn12_upper,
                          color='r',alpha=0.2,
                        label='IRNN')
    axs[r,c].fill_between(range(1,fut_dims+1),cgmm_lower, 
                        cgmm_upper,alpha=0.3,
                        label='CANF')

fig.text(0.5, 0.04, 'Time Index', ha='center', va='center')
fig.text(0.06, 0.5, 'Demand', ha='center', va='center', rotation='vertical')
handles, labels = axs[1,1].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center',ncol=3)
axs[1,1].legend(loc='upper left')
if save_tex:
    mpl2t.save('figs/samples.tex')
else:
    plt.savefig('figs/samples.png')
    plt.show()

### 3. Full Trajectory NLL

# Settting Parameters
loc, past_dims, fut_dims = (1, 8, 12)

# Flow Parameters
nflows, hdim, iters = (10, 32, 4000)

# CondGMM Parameters
ncomp, seed = (5, 4)

# GMM Approximation hyperparameters
nsamples = 100000
ncomps = 25

# Cuda
use_cuda = torch.cuda.is_available() # False

# String
seed=0
exp_string = 'loc_'+str(loc)+'.tot_dims_'+str(past_dims+fut_dims) + '.nflows_' + str(nflows) + '.hdim_' + str(hdim) + '.iters_' + str(iters) + '.seed_' + str(seed)

# Load Data

X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)
X_batches, Y_batches = batch(X_train, Y_train, batch_size = 64)

combined = torch.cat((X_train[:,:,0], Y_train[:,:,0]), -1)
x = combined[torch.randperm(combined.shape[0]),:]

combined_test = torch.cat((X_test[:,:,0], Y_test[:,:,0]), -1)
x_test = combined_test[torch.randperm(combined_test.shape[0]),:]

# set up and load norm flow
an = ActNorm(dim=combined.shape[1])
flows = [RealNVP(dim=combined.shape[1], hidden_dim = hdim, 
                 base_network=FCNN) for _ in range(nflows)]
prior = D.MultivariateNormal(torch.zeros(combined.shape[1]),
                            torch.eye(combined.shape[1]))
model = NormalizingFlowModel(prior, flows)

path = 'models/'+  exp_string +'.pt'
assert os.path.isfile(path), "no pre-existing model as defined"
print('Model Loaded')
model.load_state_dict(torch.load(path))
model.eval()

# sample model
many_samples = model.sample(nsamples).detach()
X_train_sampled = many_samples[:,:past_dims].unsqueeze(-1)
Y_train_sampled = many_samples[:,past_dims:].unsqueeze(-1)

# fit approximate norm flow
anf = ConditionalGMM(1, past_dims, 1, fut_dims, 
                      n_components=ncomps, random_state=seed)
anf.fit(X_train_sampled, Y_train_sampled)

# fit condGMM
cgmm = ConditionalGMM(1, past_dims, 1, fut_dims, 
                      n_components=ncomp, random_state=seed)
cgmm.fit(X_train, Y_train)

# log probs
test_flow = model.log_prob(X_test, Y_test).detach()
test_cgmm = cgmm.log_prob(X_test, Y_test).detach()
test_anf = anf.log_prob(X_test, Y_test).detach()

# plot
plt.figure()
ax = plt.gca()

varp = 0.5
aph = 0.6
nbins = 30 

color = next(ax._get_lines.prop_cycler)['color']
plt.hist(test_cgmm, bins=nbins,alpha=aph,color=color, density=True, label='CGMM')
plt.axvline(x=var(test_cgmm,varp),color=color, label='CGMM Median')

color = next(ax._get_lines.prop_cycler)['color']
plt.hist(test_flow, bins=nbins,alpha=aph,color=color, density=True, label='Flow')
plt.axvline(x=var(test_flow,varp),color=color, label='NormFlow Median')

color = next(ax._get_lines.prop_cycler)['color']
plt.hist(test_anf, bins=nbins,alpha=aph,color=color, density=True,label='Approx Flow')
plt.axvline(x=var(test_anf,varp),color=color, label='CANF Median')

plt.title('Distribution over Test LL')
plt.xlim(-75,75)
plt.xlabel('Test log-likelihood')
plt.ylabel('Density')
plt.grid(True)
plt.legend(loc='upper left')
if save_tex:
    mpl2t.save('figs/likelihoods.tex')
else:
    plt.savefig('figs/likelihoods.png')
    plt.show()