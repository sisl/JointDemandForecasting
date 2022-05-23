import torch
from itertools import combinations

def index_allocation(samples, min_indices, obj_fn, y_test, alpha, return_scores=False):
    """ 
    Perform index allocation decision problem. 

    Args: 
        samples (torch.tensor): (n_samples, n_test, K*ydim) tensor of samples
        min_indices (int): number of indices to allocate
        obj_fn (function): objective function to minimize. function should be (c, n) -> (c,)
        y_test (torch.tensor): (n_test, K, ydim) tensor of true data labels
        alpha (float): final alpha-quantile to report
        return_scores (bool): if True, return score distribution rather than mean
        
    Returns: 
        score (float): final score
    """
    _, ntest, fut_dims = samples.shape
    
    # generate all candidate actions
    candidates = torch.stack([torch.tensor(inds) for inds in combinations(range(fut_dims), min_indices)]) # (c, min_indices)
    
    # generate all sum utilites for each candidate
    utilities = samples[:,:,candidates].sum(-1) # (n_samples, n_test, c)
    
    # get action candidate scores in batch
    # permute: (n_samples, n_test, c) -> (n_test, c, n_samples) 
    # obj_fn: (n_test, c, n_samples) -> (n_test, c)
    action_scores = obj_fn(torch.permute(utilities, (1, 2, 0)))
    
    # get best actions in batch
    # argmin: (n_test, c) -> (n_test)
    # index: (n_test), (c, min_indices)  -> (n_test, min_indices)
    best_actions = candidates[torch.argmin(action_scores, dim=-1)].long() 
    
    # score best actions
    scores = linear_score(best_actions, y_test, return_scores=True)
    score = var(scores, alpha)
    return (score, scores) if return_scores else score

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
    traj = true_traj.reshape(n,-1).double()
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