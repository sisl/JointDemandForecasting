#
# utils.py
#

import numpy as np
import torch

def bfill_lowertriangle(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.tril_indices(A.size(-2), k=-1, m=A.size(-1))
    A[..., ii, jj] = vec
    return A


def bfill_diagonal(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    A[..., ii, jj] = vec
    return A

def bmm(A,b):
    """
    Args:
        A (torch.tensor): (*,n,n) matrix
        b (torch.tensor): (*,n) vector
    Returns:
        c (torch.tensor): (*,n) mvp
    """
    return (A @ b.unsqueeze(-1)).squeeze(-1)

def batch(X,Y,batch_size=32):
    """ 
    Batches the data along the first dimension. 

    Args: 
        X (torch tensor): (num_sequences, *) tensor of data input streams.
        Y (torch tensor): (num_sequences, *) tensor of data labels.
        batch_size (int): batch size.

    Returns: 
        X_batches (list of torch tensor): list of (batch_size, *) tensor input data.
        Y_batches (list of torch tensor): list of (batch_size, *) tensor data labels.
    """
    X_batches = []
    Y_batches = []
    num_batches = int(X.shape[0]/batch_size)
    for i in range(num_batches):
        X_batches.append(X[i*batch_size:(i+1)*batch_size])
        Y_batches.append(Y[i*batch_size:(i+1)*batch_size])
    return (X_batches, Y_batches)

def train(model, X_batches, Y_batches, num_epochs=20, learning_rate=0.01,
          verbose=True, weighting=None):
    """ 
    Train a model. 

    Args: 
        model: pytorch model to train.
        X_batches (list of torch tensor): list of (batch_size, *) tensor input data.
        Y_batches (list of torch tensor): list of (batch_size, *) tensor data labels.
        num_epochs (int): number of times to iterate through all batches
        learning_rate (float): learning rate for Adam optimizer
        verbose (bool): if true, print epoch losses
    """                                                        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, y in zip(X_batches, Y_batches):
            optimizer.zero_grad()
            
            dist = model(x)
            loss, _ = nll(dist, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()/len(X_batches)
        if verbose:
            print ("epoch : %d, loss: %1.4f" %(epoch+1, epoch_loss))
    if verbose:
        print ("Learning finished!")

def mape(dist, target):
    """ 
    Compute MAPE Loss. 

    Args: 
        dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
        target (torch tensor): (B, K, ydim) tensor of true data labels.

    Returns: 
        mape (float): MAPE loss
        mape_mean (torch.tensor): (K*ydim) MAPE along each output dimension
        mape_std (torch.tensor): (K*ydim) std of MAPE along each output dimension
    """
    
    output = dist.mean
    target_shape = target.shape
    target = target.reshape(*target_shape[:-2],-1)
    mape_mean = torch.mean(torch.abs((target - output) / target), 0)
    mape_std = torch.std(torch.abs((target - output) / target), 0)
    mape = torch.mean(mape_mean)
    return mape, mape_mean, mape_std

def rmse(dist, target):
    """ 
    Compute RMSE Loss. 

    Args: 
        dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
        target (torch tensor): (B, K, ydim) tensor of true data labels.

    Returns: 
        rmse (float): RMSE loss
        mse_mean (torch.tensor): (K*ydim) MSE along each output dimension
        mse_std (torch.tensor): (K*ydim) std of MSE along each output dimension
    """
    
    output = dist.mean
    target_shape = target.shape
    target = target.reshape(*target_shape[:-2],-1)
    
    se = (target - output)**2
    mse_mean = torch.mean(se, 0)
    mse_std = torch.std(se, 0)
    rmse = torch.mean(mse_mean)**0.5
    return rmse, mse_mean, mse_std

def rwse(dist, target, n=1000):
    """ 
    Compute RWSE Loss. 

    Args: 
        dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
        target (torch tensor): (B, K, ydim) tensor of true data labels.
        n (int): number of samples to compute RWSE over

    Returns: 
        rwse (float): RWSE loss
        wse_mean (torch.tensor): (K*ydim) WSE along each output dimension
        wse_std (torch.tensor): (K*ydim) std of WSE along each output dimension
    """
    samples = dist.sample((n,))
    target_shape = target.shape
    target = target.reshape(1,*target_shape[:-2],-1)

    se = torch.mean((target - samples)**2,0)
    wse_mean = torch.mean(se, 0)
    wse_std = torch.std(se, 0)
    rwse = torch.mean(wse_mean)**0.5
    return rwse, wse_mean, wse_std

def nll(dist, target):
    """ 
    Compute negative log likelihood of target given distribution. 

    Args: 
        dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
        target (torch tensor): (B, K, ydim) tensor of true data labels.

    Returns: 
        nll (float): negative log likelihood
        nlls (torch.tensor): (B) nll of each sample in batch
    """
    target_shape = target.shape
    target = target.reshape(*target_shape[:-2],-1)
    nlls = -dist.log_prob(target)
    if len(nlls.shape) > 1:
        nlls = torch.sum(nlls,1)
    nll = nlls.mean()
    return nll, nlls