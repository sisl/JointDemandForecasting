import numpy as np
import matplotlib.pyplot as plt
#import matplotlib2tikz as mpl2t
import itertools
import logging
import os
import sys
import json
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

log_level = logging.ERROR # logging.DEBUG for verbose training, logging.ERROR for just errors
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available() # False
if use_cuda:
    torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")

def fit_nf_model(model, optimizer, train, cv, iters=1000, save_every=50):
    """
    Train a normalizing flow model, given a model, optimizer, train set, cv set
    Args:
        model: NF model
        optimizer: torch optimizer
        train: train set
        cv: cv set
        iters: number of iterations
        save_every: number of iterations to test cvs
    Returns:
        lp_list (list): list of mean log probs
        iter_list (list): list of iterations
    """ 
    lp_list = []
    iter_list = []
    for i in range(iters):
        optimizer.zero_grad()
        z, prior_logprob, log_det = model(train)
        logprob = prior_logprob + log_det
        loss = -torch.mean(prior_logprob + log_det)
        loss.backward()
        optimizer.step()
        if (i+1) % save_every == 0:
            _, prior_logprob_test, log_det_test = model(cv)
            logprob_test = prior_logprob_test + log_det_test
            iter_list.append(i+1)
            lp_list.append(logprob_test.mean().data)
            logger.info(f"Iter: {i+1}\t" +
                        f"Logprob: {logprob.mean().data:.2f}\t" +
                        f"Prior: {prior_logprob.mean().data:.2f}\t" +
                        f"LogDet: {log_det.mean().data:.2f}\t" + 
                        f"Logprob_test: {logprob_test.mean().data:.2f}")
    return lp_list, iter_list

#   hyperparams = {
#        'gmm_ncomp': [6],
#        'nflows': [10],
#        'hdim': [10],
#        'lr':[0.005],
#        'iters':[500], # will do early stopping to choose best iteration number
#        'flow_samples': [10000], #only training one 
#        'anf_ncomp': [6],
#    }

def hyp_tune_gmm(train, cv, ncomps, nseeds=10):
    """
    Hyperparameter tune a gmm
    Args:
        train
        cv
        ncomps (list of int): different numbers of components to train gmm with
        nseeds (int): number of times to try gmm with different seeds
    Returns:
        best_ncomp (int): best gmm ncomponenets
        best_ll (int): best mean ll
    """
    X_train, Y_train = train[:,:1].unsqueeze(-1), train[:,1:].unsqueeze(-1)
    X_cv, Y_cv = cv[:,:1].unsqueeze(-1), cv[:,1:].unsqueeze(-1)
    best_ll = -float('inf')
    means = []
    for ncomp in ncomps:
        ll = []
        for i in range(nseeds):
            try:
                gmm = ConditionalGMM(1, 1, 1, 1, 
                    n_components=ncomp, random_state=i)
                gmm.fit(X_train, Y_train)
                ll.append(gmm.log_prob(X_cv, Y_cv).detach().mean())
            except:
                pass
        ll_mean = torch.tensor(ll).mean().item()
        means.append(ll_mean)
        if ll_mean > best_ll:
            best_ll = ll_mean
            best_ncomp = ncomp
    print('GMM LL means: {}'.format(means))
    return best_ncomp, best_ll

def hyp_tune_nf(train, cv, hyp):
    """
    Hyperparameter tune a normflow model
    Args:
        train
        cv
        hyp (dict): dictionary of hyperparameter choices
    Returns:
        best_nflows (int): best number of flows
        best_hdim (int): best number of hidden dimensions
        best_lr (float): best learning rate
        best_iters (int): best number of iterations
        best_nf: best norm flow model
    """
    best_ll = -float('inf')
    iters = hyp['iters'][0]
    dim = train.shape[1]

    for nflows in hyp['nflows']:
        for hdim in hyp['hdim']:
            for lr in hyp['lr']:
                lps = []
                for seed in range(5):
                    flows = [RealNVP(dim=dim, hidden_dim = hdim, 
                            base_network=FCNN) for _ in range(nflows)]
                    prior = D.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
                    nf = NormalizingFlowModel(prior, flows, random_state=seed)
                    optimizer = torch.optim.Adam(nf.parameters(), lr=lr)
                    lp_list, iter_list = fit_nf_model(nf, optimizer, train, cv, iters)
                    lps.append(lp_list)
                lps = torch.tensor(lps).mean(dim=0)
                idx = torch.argmax(lps)
                max_iter, max_ll = iter_list[idx], lps[idx]
                print(nflows, hdim, lr, max_iter, max_ll)
                if max_ll > best_ll:
                    best_ll = max_ll
                    best_nflows = nflows
                    best_hdim = hdim
                    best_lr = lr
                    best_iters = max_iter

    # retrain best model
    flows = [RealNVP(dim=dim, hidden_dim = best_hdim, 
            base_network=FCNN) for _ in range(best_nflows)]
    prior = D.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    best_nf = NormalizingFlowModel(prior, flows, random_state=0)
    optimizer = torch.optim.Adam(best_nf.parameters(), lr=best_lr)
    _ = fit_nf_model(best_nf, optimizer, train, cv, best_iters)
    best_nf.eval()
    return best_nflows, best_hdim, best_lr, best_iters, best_nf

def hyp_tune(train, cv, hyp, filestr):
    """
    Tune hyperparameters based on train set, cv set, and hyperparameter options.
    Saves best hyperparameter set

    Args:
        train: train set
        cv: cv set
        hyp (dict): dictionary of lists of hyperparameter options
        filestr (dict): filestr to save best_hyperparams to
    """ 
    # gmm
    best_gmm_ncomp, _ = hyp_tune_gmm(train, cv, hyp['gmm_ncomp'])

    # nf
    best_nflows, best_hdim, best_lr, best_iters, best_nf = hyp_tune_nf(train, cv, hyp)

    # anf
    best_anf_ll = -float('inf')
    for f in hyp['flow_samples']:
        many_samples = best_nf.sample(f).detach()
        anf_ncomp, anf_ll = hyp_tune_gmm(many_samples, cv, hyp['anf_ncomp'])
        if anf_ll > best_anf_ll:
            best_anf_ll = anf_ll
            best_anf_ncomp = anf_ncomp
            best_flow_samples = f
    
    best_hyperparams = {
        'gmm_ncomp': best_gmm_ncomp,
        'nflows': best_nflows,
        'hdim': best_hdim,
        'lr': best_lr,
        'iters': best_iters, 
        'flow_samples': best_flow_samples, 
        'anf_ncomp': best_anf_ncomp,
    }

    # save best hyperparams
    with open(filestr+'_best_hyperparams.json', 'w') as outfile:
        json.dump(best_hyperparams, outfile)

def test_models(dataset_list, hyp, filestr, save_plot_number):
    """
    Test tuned hyperparameters on newly sampled datasets

    Args:
        dataset_list (list of tuple): list of nseeds (train, test) dataset pairs
        hyp (dict): dictionary of best hyperparameters
        filestr (dict): filestr to save plots to
        save_plot_number (int): index of plot to save
    """
    nseeds = len(dataset_list)
    gmm_lps, nf_lps, anf_lps = [], [], []
    
    for i, (train, test) in enumerate(dataset_list):
        # fit models
        X_train, Y_train = train[:,:1].unsqueeze(-1), train[:,1:].unsqueeze(-1)
        X_test, Y_test = test[:,:1].unsqueeze(-1), test[:,1:].unsqueeze(-1)
        
        # GMM
        try:
            gmm = ConditionalGMM(1, 1, 1, 1, 
                n_components=hyp['gmm_ncomp'], random_state=i+1)
            gmm.fit(X_train, Y_train)
            gmm_lps.append(gmm.log_prob(X_test, Y_test).detach().mean())
        except:
            print('GMM seed {} failed'.format(i))

        # NF
        flows = [RealNVP(dim=train.shape[1], hidden_dim = hyp['hdim'], 
                        base_network=FCNN) for _ in range(hyp['nflows'])]
        prior = D.MultivariateNormal(torch.zeros(train.shape[1]),
                                torch.eye(train.shape[1]))
        nf = NormalizingFlowModel(prior, flows, random_state=i+1)
        optimizer = torch.optim.Adam(nf.parameters(), lr=hyp['lr'])
        _ = fit_nf_model(nf, optimizer, train, test, hyp['iters'])
        nf.eval()
        nf_lps.append(nf.log_prob(X_test, Y_test).detach().mean())

        # ANF
        many_samples = nf.sample(hyp['flow_samples']).detach()
        X_train_sampled, Y_train_sampled = many_samples[:,:1].unsqueeze(-1), many_samples[:,1:].unsqueeze(-1)
        try:
            anf = ConditionalGMM(1, 1, 1, 1,
                    n_components=hyp['anf_ncomp'], random_state=i+1)
            anf.fit(X_train_sampled, Y_train_sampled)
            anf_lps.append(anf.log_prob(X_test, Y_test).detach().mean())
        except:
            print('ANF seed {} failed'.format(i))

        print('Seed %i done' %(i))

        # save correct plot number
        if i == save_plot_number:
            
            plot_models = {'gmm':gmm,'nf':nf, 'anf':anf}
            plot_train, plot_test = train.clone(), test.clone()

    # Print
    gmm_lps, nf_lps, anf_lps = torch.tensor(gmm_lps), torch.tensor(nf_lps), torch.tensor(anf_lps)
    
    # Log probs
    print('GMM LogProbs = {}'.format(gmm_lps))
    print('NF LogProbs = {}'.format(nf_lps))
    print('ANF LogProbs = {}'.format(anf_lps))

    # Statistics
    print('GMM LogProb Mean = %.03f \pm %.03f' %(torch.mean(gmm_lps), torch.std(gmm_lps)/(len(gmm_lps)**0.5)))
    print('NF LogProb Mean = %.03f \pm %.03f' %(torch.mean(nf_lps), torch.std(nf_lps)/(len(nf_lps)**0.5)))
    print('ANF LogProb Mean = %.03f \pm %.03f' %(torch.mean(anf_lps), torch.std(anf_lps)/(len(anf_lps)**0.5)))
    
    # Save plots
    save_plots(plot_models, plot_train, plot_test, filestr)

def save_plots(models, train, test, filestr, tikz=False):
    """
    Make and save plots given train and test set
    Args:
        models: dictionary of models
        train (torch.tensor): (n_train, 2) train set
        test (torch.tensor): (n_test, 2) test set
        filestr: base filestring to save plots to
        tikz (bool): whether or not to save to tikz
    """
    X_train, Y_train = train[:,:1].unsqueeze(-1), train[:,1:].unsqueeze(-1)
    X_test, Y_test = test[:,:1].unsqueeze(-1), test[:,1:].unsqueeze(-1)
    
    plt.figure()
    plt.scatter(X_train[:,0,0], Y_train[:,0,0], c='k')
    plt.savefig(filestr+'_train_points.png')

    n=100
    x, y = np.meshgrid(np.linspace(-1.5, 2.5, n), np.linspace(-1, 1.5, n))
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    xrs = x.reshape((-1,1,1))
    yrs = y.reshape((-1,1,1))
    gmm_lp = models['gmm'].log_prob(xrs, yrs).detach().reshape((n,n))
    nf_lp = models['nf'].log_prob(xrs, yrs).detach().reshape((n,n))
    anf_lp = models['anf'].log_prob(xrs, yrs).detach().reshape((n,n))

    # gmm
    fig, ax = plt.subplots()
    c=ax.pcolormesh(x, y, gmm_lp, vmin=-15)
    ax.scatter(X_test[:,0,0], Y_test[:,0,0], c='k')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.savefig(filestr+'_gmm.png')

    # nf
    fig, ax = plt.subplots()
    c=ax.pcolormesh(x, y, nf_lp, vmin=-15)
    ax.scatter(X_test[:,0,0], Y_test[:,0,0], c='k')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.savefig(filestr+'_nf.png')

    # anf
    fig, ax = plt.subplots()
    c=ax.pcolormesh(x, y, anf_lp, vmin=-15)
    ax.scatter(X_test[:,0,0], Y_test[:,0,0], c='k')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.savefig(filestr+'_anf.png')

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
def make_dataset(nsamples, noise):
    #data1 = torch.tensor(datasets.make_moons(n_samples=nsamples, noise=noise)[0]).float()
    #data2 = torch.tensor(datasets.make_moons(n_samples=nsamples, noise=noise)[0]).float()
    data3 = torch.rand(nsamples, 2).float()
    return data3

if __name__=='__main__':
    # parse arguments


    import argparse
    parser = argparse.ArgumentParser(description='Run simple experiment with noisy moons dataset')

    parser.add_argument("--tune", help="tune hyperparameters",
        action="store_true")
    parser.add_argument("--test", help="test stored hyperparameters",
        action="store_true")
    parser.add_argument('-nseeds', default=10, type=int,
        help='number of test seeds')    
    parser.add_argument('-ntrain', default=1000, type=int,
        help='number of training samples') 
    parser.add_argument('-ncv', default=200, type=int,
        help='number of cv samples')
    parser.add_argument('-ntest', default=1000, type=int,
        help='number of test samples')
    parser.add_argument('-plotidx', default=0, type=int,
        help='index of seed to plot')                 
    args = parser.parse_args()

    n_train, n_cv, n_test = args.ntrain, args.ncv, args.ntest
    save_plot_number = args.plotidx
    nseeds = args.nseeds
    filestr = 'results/simple_moons'
    
    
    # early experiments: nflows, hdim, lr, iters  = (10, 10, 0.005, 500)

    hyperparams = {
        'gmm_ncomp': [7,8,9, 10, 11],
        'nflows': [5, 10, 15],
        'hdim': [8, 12, 24],
        'lr':[0.001, 0.005],
        'iters':[2000], # will NOT do early stopping to choose best iteration number
        'flow_samples': [100000],
        'anf_ncomp': [5, 10, 15, 20],
    }

    noise_level = 0.05
    if args.tune:

        set_seeds(0)
        n_samples = n_train + n_cv + n_test
        dataset = make_dataset(n_samples, noise_level)
        train, cv, test = dataset[:n_train], dataset[n_train:n_train+n_cv], dataset[n_train+n_cv:]
        hyp_tune(train, cv, hyperparams, filestr)

    if args.test:
        
        with open(filestr+'_best_hyperparams.json') as json_file:
            best_hyperparams = json.load(json_file)
        dataset_list = []
        for i in range(nseeds):
            set_seeds(i+1)
            n_samples = n_train + n_cv + n_test
            dataset = make_dataset(n_samples, noise_level)
            dataset_list.append((dataset[:n_train], dataset[n_train+n_cv:]))

        test_models(dataset_list, best_hyperparams, filestr, save_plot_number)
        