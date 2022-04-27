import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz as mpl2t

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import src
from src.utils import *
from src.models.cgmm import *
from src.models.gmnn import *
from src.models.nf.flows import *
from src.models.nf.models import *
from charging_utils import *

save_tex = True # whether to save the .tex files or .pngs
if not os.path.isdir('experiments/figs'):
    os.makedirs('experiments/figs')

########## PLOTS ###############

### 0. Full Trajectory NLL

# Settting Parameters
loc, past_dims, fut_dims = (1, 8, 12)

# Flow Parameters
nflows, hdim, iters = (10, 32, 4000)

# CondGMM Parameters
ncomp = 5

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

combined = torch.cat((X_train[:,:,0], Y_train[:,:,0]), -1)
combined_test = torch.cat((X_test[:,:,0], Y_test[:,:,0]), -1)

x = combined[torch.randperm(combined.shape[0]),:]
x_test = combined_test[torch.randperm(combined_test.shape[0]),:]

# set up and load norm flow
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
    mpl2t.save('experiments/figs/likelihoods.tex')
else:
    plt.savefig('experiments/figs/likelihoods.png')
    plt.show()
    
### 1. Per-index error plot

# Taken from running default models for 12 |8 in Bakersfield, CA, with seeds set to 0. 
# The second return of the src.utils.rwse is per-index RWSE.

# fit IFNN and sample
Y_train_single = Y_train[:,[0],:]
X_batches, Y_batches = batch(X_train, Y_train_single, batch_size = 64, random_state = 0)
hidden_layers = [40, 40, 40]
ss_gmnn = GaussianMixtureNeuralNet(1, past_dims, hidden_layers, 1, 1,
                                    n_components=3, random_state=0)
train(ss_gmnn, X_batches, Y_batches, num_epochs=150, learning_rate=.005, verbose=False)    
ifnn_samples = sample_forward(ss_gmnn, X_test, fut_dims)

# generate CANF samples                                   
dist = anf(X_test)
canf_samples = []
for _ in range(1000//200):
    s = dist.sample((200,))
    canf_samples.append(s) 
canf_samples = torch.cat(canf_samples,0)                                   
                                   
cgmmrwse = rwse(cgmm(X_test), Y_test)[1]
fnnrwse = rwse(ifnn_samples, Y_test,sampled=True)[1]
canfrwse = rwse(canf_samples, Y_test,sampled=True)[1]

x = torch.arange(1,13)
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
    mpl2t.save('experiments/figs/errors.tex')
else:
    plt.savefig('experiments/figs/errors.png')
    plt.show()

### 2. Sample Forecast Plot
past_dims, fut_dims, loc = (8, 12, 9)
X_train, Y_train_full, X_test, Y_test_full = load_data(loc, past_dims, fut_dims)
Y_train = Y_train_full[:,[0],:]
Y_test = Y_test_full[:,[0],:]

X_batches, Y_batches = batch(X_train, Y_train, batch_size = 128, random_state=0)

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
handles, labels = axs[0,0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center',ncol=3)
axs[0,0].legend(loc='upper left')
if save_tex:
    mpl2t.save('experiments/figs/samples.tex')
else:
    plt.savefig('experiments/figs/samples.png')
    plt.show()

