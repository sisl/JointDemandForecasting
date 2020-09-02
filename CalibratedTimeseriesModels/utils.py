#
# utils.py
#

import numpy as np
import torch
import random
import math

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

def electric_train_test_split(X, Y, test_size=0.25, random_state=1, disp_idx=36):
    """
    Performs train/test split for electric dataset by avoiding overlaps in rolling windows.
    
    Args:
        X (torch.tensor): (num_locs,n,in_features) tensor of inputs
        Y (torch.tensor): (num_locs,n,out_features) tensor of corresponding outputs
        test_size (float)
        random_state (int): seed for the random number generator
        disp_idx (int): index by which to displace indices to avoid test/train overlap
    Returns:
        X_train (torch.tensor): (num_locs,m,in_features) tensor of training inputs
        X_test (torch.tensor): (num_locs,t,in_features) tensor of testing inputs
        Y_train (torch.tensor): (num_locs,m,out_features) tensor of training inputs
        Y_test (torch.tensor): (num_locs,t,out_features) tensor of testing inputs
    """    
    _, n, _ = X.shape
    random.seed(random_state)
    week_idx = 24*7
    X_train, X_test, Y_train, Y_test = [], [], [], []
    prev_batch_train = True
    for i in range(math.ceil(n/week_idx)):
        start_idx = i*week_idx
        end_idx = min((i+1)*week_idx,n)
        
        # add to training
        if random.random()>test_size:
            
            # if switching batch, add to start index
            if not prev_batch_train:
                start_idx += disp_idx
                prev_batch_train = True
            X_train.append(X[:,start_idx:end_idx,:])
            Y_train.append(Y[:,start_idx:end_idx,:])
            
        # add to testing
        else:
            
            # if switching batch, add to start index
            if prev_batch_train:
                start_idx += disp_idx
                prev_batch_train = False
            X_test.append(X[:,start_idx:end_idx,:])
            Y_test.append(Y[:,start_idx:end_idx,:])
    X_train = torch.cat(X_train, dim=1)
    X_test = torch.cat(X_test, dim=1)
    Y_train = torch.cat(Y_train, dim=1)
    Y_test = torch.cat(Y_test, dim=1)
    return X_train, X_test, Y_train, Y_test

def batch(X, Y, batch_size=32, random_state=0):
    """ 
    Batches the data along the first dimension. 

    Args: 
        X (torch tensor): (num_sequences, *) tensor of data input streams.
        Y (torch tensor): (num_sequences, *) tensor of data labels.
        batch_size (int): batch size.
        random_state (int): seed to set random number generator for batching data

    Returns: 
        X_batches (list of torch tensor): list of (batch_size, *) tensor input data.
        Y_batches (list of torch tensor): list of (batch_size, *) tensor data labels.
    """
    n = X.shape[0]
    torch.random.manual_seed(random_state)
    r = torch.randperm(n)
    X = X[r,...]
    Y = Y[r,...]
    
    X_batches = []
    Y_batches = []
    num_batches = int(X.shape[0]/batch_size)
    for i in range(num_batches):
        X_batches.append(X[i*batch_size:(i+1)*batch_size])
        Y_batches.append(Y[i*batch_size:(i+1)*batch_size])
    return (X_batches, Y_batches)

def train(model, X_batches, Y_batches, num_epochs=20, optimizer = torch.optim.Adam,
          learning_rate=0.01, verbose=True, weighting=None):
    """ 
    Train a model. 

    Args: 
        model: pytorch model to train.
        X_batches (list of torch tensor): list of (batch_size, *) tensor input data.
        Y_batches (list of torch tensor): list of (batch_size, *) tensor data labels.
        num_epochs (int): number of times to iterate through all batches
        optimizer (object): torch optimizer
        learning_rate (float): learning rate for Adam optimizer
        verbose (bool): if true, print epoch losses
    """                                                        
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    checkpoint = model.state_dict()
    
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
        if np.isnan(epoch_loss): #or epoch_loss > 10^10:
            model.load_state_dict(checkpoint)
            print("Training Error, re-loading checkpoint")
            continue
        if epoch % 2 == 0: 
            checkpoint = model.state_dict()
        if verbose and (num_epochs < 100 or epoch % 10 == 0):
            print ("epoch : %d, loss: %1.4f" %(epoch+1, epoch_loss))
            
    if verbose:
        print ("Learning finished!")

def mape(dist, target, sampled=False):
    """ 
    Compute MAPE Loss. 

    Args: 
        dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
            (torch tensor): if sampled is True, dist must be a (n,B,K*ydim) tensor
        target (torch tensor): (B, K, ydim) tensor of true data labels.
        sampled (bool): if True, dist is a tensor of samples

    Returns: 
        mape (float): MAPE loss
        mape_mean (torch.tensor): (K*ydim) MAPE along each output dimension
        mape_std (torch.tensor): (K*ydim) std of MAPE along each output dimension
    """
    
    if sampled:
        output = dist.mean(0)
    else:
        output = dist.mean 
    
    target_shape = target.shape
    target = target.reshape(*target_shape[:-2],-1)
    mape_mean = torch.mean(torch.abs((target - output) / target), 0)
    mape_std = torch.std(torch.abs((target - output) / target), 0)
    mape = torch.mean(mape_mean)
    return mape, mape_mean, mape_std

def wape(dist, target, n=1000, sampled=False):
    """ 
    Compute WAPE Loss. 

    Args: 
        dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
            (torch tensor): if sampled is True, dist must be a (n,B,K*ydim) tensor
        target (torch tensor): (B, K, ydim) tensor of true data labels.
        sampled (bool): if True, dist is a tensor of samples

    Returns: 
        wape (float): WAPE loss
        wape_mean (torch.tensor): (K*ydim) WAPE along each output dimension
        wape_std (torch.tensor): (K*ydim) std of WAPE along each output dimension
    """
    
    if sampled:
        samples = dist
    else:
        samples = dist.sample(torch.Size([n]))
    
    target_shape = target.shape
    target = target.reshape(1,*target_shape[:-2],-1)  
    
    ape = torch.mean(torch.abs((target - samples) / target),0)
    wape_mean = torch.mean(ape, 0)
    wape_std = torch.std(ape, 0)
    wape = torch.mean(wape_mean)
    return wape, wape_mean, wape_std

def rmse(dist, target, sampled=False):
    """ 
    Compute RMSE Loss. 

    Args: 
        dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
            (torch tensor): if sampled is True, dist must be a (n,B,K*ydim) tensor
        target (torch tensor): (B, K, ydim) tensor of true data labels.
        sampled (bool): if True, dist is a tensor of samples

    Returns: 
        rmse (float): RMSE loss
        mse_mean (torch.tensor): (K*ydim) MSE along each output dimension
        mse_std (torch.tensor): (K*ydim) std of MSE along each output dimension
    """
    
    if sampled:
        output = dist.mean(0)
    else:
        output = dist.mean 
    target_shape = target.shape
    target = target.reshape(*target_shape[:-2],-1)
    
    se = (target - output)**2
    mse_mean = torch.mean(se, 0)
    mse_std = torch.std(se, 0)
    rmse = torch.mean(mse_mean)**0.5
    return rmse, mse_mean, mse_std

def rwse(dist, target, n=1000, sampled=False):
    """ 
    Compute RWSE Loss. 

    Args: 
        dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations
            (torch tensor): if sampled is True, dist must be a (n,B,K*ydim) tensor
        target (torch tensor): (B, K, ydim) tensor of true data labels.
        n (int): number of samples to compute RWSE over
        sampled (bool): if True, dist is a tensor of samples

    Returns: 
        rwse (float): RWSE loss
        wse_mean (torch.tensor): (K*ydim) WSE along each output dimension
        wse_std (torch.tensor): (K*ydim) std of WSE along each output dimension
    """
    
    if sampled:
        samples = dist
    else:
        samples = dist.sample(torch.Size([n]))
    
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

def sample_forward(model, y, prediction_horizon, n_samples=1000):
    
    samples = []
    for iS in range(n_samples):
        # initial step
        new_sample = model(y).sample()
        sequence = [new_sample] #(B, 1)
        
        fut_y = y
        for _ in range(1, prediction_horizon):

            # append to end of input sequence (OPENEI DATA)
            fut_y = torch.cat((fut_y[:,1:,:],sequence[-1].unsqueeze(-1)),1)

            # run through model
            dist = model(fut_y)

            # generate next time series
            next_step = dist.sample()
            sequence.append(next_step)
        samples.append(torch.cat(sequence,1))    
    samples = torch.stack(samples,0)
    return samples

def sample_forward_lstm(model, y, prediction_horizon, n_samples=1000):
    samples = []
    for i in range(n_samples):
        
        ## MUST GO THROUGH LSTM BY HAND FOR HIDDEN STATES
        # initial step
        h_0, c_0 = model.initialize_lstm(y)    
        output_lstm, (h_n, c_n) = model.lstm(y, (h_0, c_0))
        dist = model.forward_fc(output_lstm[:,-1,:])
        new_sample = dist.sample()
        sequence = [new_sample] #(B, 1)   
        for _ in range(1, prediction_horizon):

            # put last sample through lstm
            output_lstm, (h_n, c_n) = model.lstm(sequence[-1].unsqueeze(-1), (h_n, c_n))
            
            # run through model
            dist = model.forward_fc(output_lstm[:,-1,:])

            # generate next time series
            next_step = dist.sample()
            sequence.append(next_step)
            
        samples.append(torch.cat(sequence,1)) 
        
    samples = torch.stack(samples,0)
    return samples