import os
import sys
import itertools
import logging
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import src
from src.utils import *
from src.models.cgmm import *
from src.models.nf.flows import *
from src.models.nf.models import *

from load_data import load_data
from experiments.charging_utils import *

### Experiment Settings (uncomment one of these)

# location: 1=Bakersfield,9=SLC
# past_dims, fut_dims: past and future time step, either (24,8) or (8,12)
# nflows, hdim, iters: # of flows in RealNVP, # of hidden dims in each coupling layer, training epochs

loc, past_dims, fut_dims = (1, 8, 12)
#loc, past_dims, fut_dims = (9, 8, 12)
#loc, past_dims, fut_dims = (1, 16, 12)
#loc, past_dims, fut_dims = (9, 16, 12)
#loc, past_dims, fut_dims = (1, 24, 12)
#loc, past_dims, fut_dims = (9, 24, 12)

nflows, hdim, iters = (10, 32, 4000)

ntrials = 10

# GNN Approximation hyperparameters
nsamples = 100000
ncomps = 25

plot = False
use_cuda = torch.cuda.is_available() # False
log_level = logging.ERROR # logging.DEBUG for verbose training

# decision problem
min_indices = 4
obj_fn = lambda x: var(x, 0.8)

### Load Data
if use_cuda:
    torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")

X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)
combined = torch.cat((X_train[:,:,0], Y_train[:,:,0]), -1)
combined_test = torch.cat((X_test[:,:,0], Y_test[:,:,0]), -1)

wapes, rwses, nlls, trnlls, dscores = [], [], [], [], []
for seed in range(ntrials):
    exp_string = 'loc_'+str(loc)+'.tot_dims_'+str(past_dims+fut_dims) + '.nflows_' + str(nflows) + '.hdim_' + str(hdim) + '.iters_' + str(iters) + '.seed_' + str(seed)
    
    torch.manual_seed(seed)
    x = combined[torch.randperm(combined.shape[0]),:]
    x_test = combined_test[torch.randperm(combined_test.shape[0]),:]
    
    # Setup model
    flows = [RealNVP(dim=combined.shape[1], hidden_dim = hdim, 
                     base_network=FCNN) for _ in range(nflows)]
    prior = D.MultivariateNormal(torch.zeros(combined.shape[1]),
                                torch.eye(combined.shape[1]))
    model = NormalizingFlowModel(prior, flows, random_state=seed)

    # Train or Load
    path = 'models/'+  exp_string +'.pt'
    if os.path.isfile(path):
        print('Model Loaded')
        model.load_state_dict(torch.load(path))
        model.eval()
    else:
        logging.basicConfig(level=log_level)
        logger = logging.getLogger(__name__)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        for i in range(iters):
            optimizer.zero_grad()
            z, prior_logprob, log_det = model(x)
            logprob = prior_logprob + log_det
            loss = -torch.mean(prior_logprob + log_det)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                _, prior_logprob_test, log_det_test = model(x_test)
                logprob_test = prior_logprob_test + log_det_test
                logger.info(f"Iter: {i+1}\t" +
                            f"Logprob: {logprob.mean().data:.2f}\t" +
                            f"Prior: {prior_logprob.mean().data:.2f}\t" +
                            f"LogDet: {log_det.mean().data:.2f}\t" + 
                            f"Logprob_test: {logprob_test.mean().data:.2f}")
        if not os.path.isdir('models'):
            os.mkdir('models')
        print('Model Saved')
        torch.save(model.state_dict(), path)
        model.eval()


    if plot:

        # plot samples
        samples = model.sample(10).detach().numpy()
        plt.plot(range(samples.shape[1]),samples.T)
        plt.show()

        # plot true
        plt.plot(range(combined.shape[1]),
             combined[torch.randperm(combined.shape[0])[:10]].T)
        plt.show()

    # sample flow
    many_samples = model.sample(nsamples).detach()
    X_train_sampled = many_samples[:,:past_dims].unsqueeze(-1)
    Y_train_sampled = many_samples[:,past_dims:].unsqueeze(-1)

    # train ANF
    cgmm = ConditionalGMM(1, past_dims, 1, fut_dims, 
                          n_components=ncomps, random_state=seed)
    cgmm.fit(X_train_sampled, Y_train_sampled)

    print('GMM FIT')

    # test metrics
    dist = cgmm(X_test)
    dist_tr = cgmm(X_train)

    samples = []
    for _ in range(1000//200):
        s = dist.sample((200,))
        samples.append(s) 
    samples = torch.cat(samples,0)
    
    wapes.append(wape(samples,Y_test,sampled=True)[0])
    rwses.append(rwse(samples,Y_test,sampled=True)[0])
    nlls.append(nll(dist,Y_test)[0])
    trnlls.append(nll(dist_tr,Y_train)[0])
    dscores.append(index_allocation(samples, min_indices, 
                                    obj_fn, Y_test, 0.8))
    print('Seed ', seed, ' done')

print('CANF Metrics:')
print("WAPEs = ", torch.stack(wapes))
print("RWSEs = ", torch.stack(rwses))
print("NLLs = ", torch.stack(nlls))
print("TrNLLs = ", torch.stack(trnlls))
print("DScores = ", torch.stack(dscores))


