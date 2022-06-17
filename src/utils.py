#
# utils.py
#

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
import os

class SequenceDataset(Dataset):
  """
  Implements a torch dataset for sequence data
  """
  def __init__(self, data:Dict[str,torch.Tensor]):
    """
    Args:
      data (dict): dict with keys
        x (torch.tensor): (B, past_steps, dim) input time series
        y (torch.tensor): (B, fut_steps, dim2) target time series
    """
    self.data_keys = ['x','y']
    assert all([k in data.keys() for k in self.data_keys]), 'missing data keys'
    assert all([data[k].ndim==3 for k in self.data_keys]), 'improper data dimensions'
    self.data = data
    self.B = len(data['x'])
    assert all([self.B == len(data[k]) for k in self.data_keys]), 'batch length mismatch'
  
  def __len__(self):
    return self.B

  def __getitem__(self, idx):
    return {key: self.data[key][idx] for key in self.data_keys}

def set_seed(seed:int):
    """
    Set numpy and torch seed manually from int
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

def bmm(A,b):
    """
    Args:
        A (torch.tensor): (*,n,n) matrix
        b (torch.tensor): (*,n) vector
    Returns:
        c (torch.tensor): (*,n) mvp
    """
    return (A @ b.unsqueeze(-1)).squeeze(-1)

def mape(dist, target, sampled=False):
    """ 
    Compute MAPE Loss. 

    Args: 
        dist (torch.Distribution): (B,K*ydim) predictive distribution over next K observations
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
        dist (torch.Distribution): (B,K*ydim) predictive distribution over next K observations
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
        dist (torch.Distribution): (B,K*ydim) predictive distribution over next K observations
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
        dist (torch.Distribution): (B,K*ydim) predictive distribution over next K observations
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

def sample_forward(model, y, prediction_horizon, n_samples=1000):
    """
    Iteratively sample forward from a model in batch

    Args:
        model (nn.Module): torch model that returns a dist on forward call
        y (torch.tensor): (B, input_horizon, O) shaped data stream
        prediction_horizon (int): horizon to iteratively go forward
        n_samples (int): number of samples to generate for each data stream
    Returns
        samples (torch.tensor): (n_samples, B, prediction_horizon*O) tensor of samples
    """

    model.eval()
    B, T, O = y.shape
    inp = y.unsqueeze(0).expand(n_samples,-1, -1, -1).reshape(B*n_samples, T, O)
    out = torch.zeros(B*n_samples, prediction_horizon, O)

    for i in range(prediction_horizon):
        dist = model(inp)
        out[:,[i]] = dist.sample().reshape(B*n_samples, 1, O)
        inp = torch.cat((inp[:,1:], out[:,[i]]), dim=1)
    samples = out.reshape(n_samples, B, prediction_horizon*O)
    return samples

def sample_forward_lstm(model, y, prediction_horizon, n_samples=1000):
    """
    Iteratively sample forward from a model in batch through an lstm

    Args:
        model (nn.Module): torch model that returns a dist on forward call
        y (torch.tensor): (B, input_horizon, O) shaped data stream
        prediction_horizon (int): horizon to iteratively go forward
        n_samples (int): number of samples to generate for each data stream
    Returns
        samples (torch.tensor): (n_samples, B, prediction_horizon*O) tensor of samples
    """
    model.eval()
    B, T, O = y.shape
    inp = y.unsqueeze(0).expand(n_samples,-1, -1, -1).reshape(B*n_samples, T, O)
    out = torch.zeros(B*n_samples, prediction_horizon, O)
    h_0, c_0 = model.initialize_lstm(inp)   

    output_lstm, (h_n, c_n) = model.lstm(inp, (h_0, c_0))
    dist = model.forward_fc(output_lstm[:,-1,:])
    out[:,[0]] = dist.sample().reshape(B*n_samples, 1, O)
    for i in range(1, prediction_horizon):
        output_lstm, (h_n, c_n) = model.lstm(out[:,[i-1]], (h_n, c_n))
        dist = model.forward_fc(output_lstm[:,-1,:])
        out[:,[i]] = dist.sample().reshape(B*n_samples, 1, O)
    samples = out.reshape(n_samples, B, prediction_horizon*O)
    return samples

def sample_forward_singlepoint(model, y, prediction_horizon, n_samples=1000):   
    # initial step
    y = y.expand(n_samples,-1,-1)
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
    samples = torch.cat(sequence,1)    
    return samples

def sample_forward_lstm_singlepoint(model, y, prediction_horizon, n_samples=1000):
    ## MUST GO THROUGH LSTM BY HAND FOR HIDDEN STATES
    # initial step
    y = y.expand(n_samples,-1,-1)
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

    samples = torch.cat(sequence,1) 
    return samples