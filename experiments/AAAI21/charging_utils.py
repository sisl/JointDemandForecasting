import os
import sys
#import numpy as np
import torch
#import torch.nn as nn

#sys.path.append("../../")
#import JointDemandForecasting
#from JointDemandForecasting.utils import *


def generate_candidate_indices(samples, min_indices):
    """ 
    Generate list of candidate minimum indices given samples of forecast trajectories. 

    Args: 
        samples (torch.tensor): (n,K*ydim) tensor of samples
        min_indices (int): number of indices to allocate

    Returns: 
        candidate_actions (torch.tensor): (c, min_indices) tensor of candidate ordered minimum indices
    """
    _, indices = torch.sort(samples)
    indices = indices[:,:min_indices]
    sorted_indices, _ = torch.sort(indices)
    candidate_actions = torch.unique(sorted_indices, dim=0)
    return candidate_actions

def generate_candidate_utility_distributions(samples, candidate_actions):
    """ 
    Generate list of utility distributions measuring action utility on samples of forecast trajectories. 

    Args: 
        samples (torch.tensor): (n,K*ydim) tensor
        candidate_actions (torch.tensor): (c, min_indices) tensor of candidate ordered minimum indices

    Returns: 
        candidate_utilities (torch.tensor): (c, n): tensor of candidate sum-min-load distributions
    """
    
    candidate_action_loads = samples[:,candidate_actions] # (n,c,min_indices)
    candidate_utilities = torch.sum(candidate_action_loads,-1).T
    return candidate_utilities

def optimal_action(candidate_actions, candidate_utilities, obj_fn):
    """ 
    Return the optimal action given candidates, their utility distribution, and an objective function. 

    Args: 
        candidate_actions (torch.tensor): (c, min_indices) tensor of candidate ordered minimum indices
        candidate_utilities (torch.tensor): (c, n): tensor of candidate sum-min-load distributions
        obj_fn (function): objective function to minimize. function should be (c, n) -> (c,)
        
    Returns: 
        best_action (torch.tensor): (min_indices): tensor of candidate sum-min-load distributions
    """    
    action_scores = obj_fn(candidate_utilities)
    best_action = candidate_actions[torch.argmin(action_scores)]
    return best_action

def best_action_in_test(y_test, min_indices):
    """ 
    Generate tensor of best actions at test time (with hindsight). 

    Args: 
        y_test (torch.tensor): (B, K, ydim) tensor of true data labels.
        min_indices (int): number of indices to allocate

    Returns: 
        best_actions (torch.tensor): (B, min_indices) tensor of indices of best actions
    """
    y = y_test.reshape(y_test.shape[0],-1)
    _, indices = torch.sort(y)
    indices = indices[:,:min_indices]
    best_actions, _ = torch.sort(indices)
    return best_actions

def power_utility(x, lam=0.5, negate=False):
    """ 
    Runs tensor through power utility. 

    Args: 
        x (torch.tensor): (*dims) tensor
        lam (float): power to apply power utility
        negate (bool): whether to negate utility (if minimizing)

    Returns: 
        U_x (torch.tensor): x transformed by power utility
    """
    
    assert abs(lam) != 1.0, "lambda set to 1"
    #assert lam >= 0.0, "lambda negative"
    
    U_x = (torch.pow(x, 1-lam) - 1) / (1-lam)
    return -U_x if negate else U_x
    
def expectation(x):
    """ 
    Performs expectation along final dimension. 

    Args: 
        x (torch.tensor): (*dims) tensor

    Returns: 
        E_x (torch.tensor): expectation of x
    """    
    
    E_x = torch.mean(x,-1)
    return E_x

def var(x, alpha=0.95):
    """ 
    Performs value-at-risk along final dimension. 

    Args: 
        x (torch.tensor): (*dims) tensor
        alpha (float): quantile for value-at-risk

    Returns: 
        var_x (torch.tensor): alpha quantile of probability distribution
    """    
    assert alpha > 0.0 and alpha < 1.0, "alpha out of domain"
    n = x.shape[-1]
    a = int(alpha*n)
    sort_x, _ = torch.sort(x)
    var_x = sort_x[...,a]
    return var_x

def cvar(x, alpha=0.95):
    """ 
    Finds conditional value-at-risk along final dimension. 

    Args: 
        x (torch.tensor): (*dims) tensor
        alpha (float): quantile for conditional value-at-risk

    Returns: 
        cvar_x (torch.tensor): expectation of top 1-alpha quantile of distribution
    """ 
    assert alpha > 0.0 and alpha < 1.0, "alpha out of domain"
    n = x.shape[-1]
    a = int(alpha*n)
    sort_x, _ = torch.sort(x)
    cvar_x = torch.mean(sort_x[...,a:],-1)
    return cvar_x

# scoring functions

def correct_action_score(charge_indices, true_traj):
    """ 
    Measures proportion of correct actions. 

    Args: 
        charge_indices (torch.tensor): (B, min_indices) tensor of charge indices
        true_traj (torch.tensor): (B, K, ydim) tensor of true trajectory

    Returns: 
        score (float): proportion of correct actions
    """ 
    n, min_indices = charge_indices.shape
    best_action_test = best_action_in_test(true_traj, min_indices)
    score = sum([all(best_action_test[i] == charge_indices[i]) for i in range(n)])/n
    return score

def linear_score(charge_indices, true_traj, return_scores=False):
    """ 
    Measures average proportional load over optimal (decision-true)/true. 

    Args: 
        charge_indices (torch.tensor): (B, min_indices) tensor of charge indices
        true_traj (torch.tensor): (B, K, ydim) tensor of true trajectory
        return_scores (bool): if True, return score distribution rather than mean

    Returns: 
        score (float): average proportional load over optimal
    """ 
    n, min_indices = charge_indices.shape
    best_action_test = best_action_in_test(true_traj, min_indices)
    traj = true_traj.reshape(n,-1)
    best_score = torch.gather(traj, 1, best_action_test).sum(-1)
    action_score = torch.gather(traj, 1, charge_indices).sum(-1)
    scores = (action_score-best_score)/best_score
    score = float(torch.mean(scores))
    return scores if return_scores else score

def exponential_score(charge_indices, true_traj, return_scores=False):
    """ 
    Measures average exponential proportional load over optimal exp((decision-true)/true). 

    Args: 
        charge_indices (torch.tensor): (B, min_indices) tensor of charge indices
        true_traj (torch.tensor): (B, K, ydim) tensor of true trajectory
        return_scores (bool): if True, return score distribution rather than mean

    Returns: 
        score (float): average proportional load over optimal
    """ 
    n, min_indices = charge_indices.shape
    best_action_test = best_action_in_test(true_traj, min_indices)
    traj = true_traj.reshape(n,-1)
    best_score = torch.gather(traj, 1, best_action_test).sum(-1)
    action_score = torch.gather(traj, 1, charge_indices).sum(-1)
    scores = torch.exp((action_score-best_score)/best_score) 
    score = float(torch.mean(scores))
    return scores if return_scores else score