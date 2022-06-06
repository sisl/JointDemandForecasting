import os
import random
import math
import torch
import numpy as np
import src
from src.utils import SequenceDataset
from typing import Dict

BASE_PATH = os.path.normpath(os.path.join(os.path.dirname(src.__file__), '..'))

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

def random_week_train_test_split(X, test_prop=0.25, random_state=1, window=36):
    """
    Performs train/test split for electric dataset by avoiding overlaps in weekly rolling windows.
    
    Args:
        X (torch.tensor): (n,) data stream
        test_prop (float)
        random_state (int): seed for the random number generator
        window (int): window for joint input/output sequences
    Returns:
        train (torch.tensor): (ntrain, window, 1) tensor of training sequences
        test (torch.tensor): (ntest, window, 1) tensor of testing sequences
    """
    _seq_idxs = np.arange(window)[np.newaxis]
    random.seed(random_state)
    n = len(X)
    train_stidxs, test_stidxs = [], []
    weeklen = 24 * 7
    add_train = True
    for i_week in range(n//weeklen):
        if random.random() > test_prop: # train
            if add_train: # continuing training
                train_stidxs += list(range(max(0, i_week*weeklen-window+1), (i_week+1)*weeklen-window+1)) 
            else: # switching from test to train week
                add_train = True
                train_stidxs += list(range(i_week*weeklen,(i_week+1)*weeklen-window+1))
        else: # test
            if add_train: # switching from test to train week
                add_train = False
                test_stidxs += list(range(i_week*weeklen,(i_week+1)*weeklen-window+1))
            else: # continuing testing
                test_stidxs += list(range(max(0, i_week*weeklen-window+1), (i_week+1)*weeklen-window+1)) 

    train_stidxs, test_stidxs = np.array(train_stidxs), np.array(test_stidxs)
    train_idxs = train_stidxs[:,np.newaxis] + _seq_idxs
    test_idxs = test_stidxs[:,np.newaxis] + _seq_idxs 

    train = X[train_idxs].unsqueeze(-1)
    test = X[test_idxs].unsqueeze(-1)
    return train, test  

def time_series_train_test_split(X, test_prop=0.25, random_state=1, window=36):
    """
    Performs train/test split by reserving last test_prop portion of sequence for testing.
    
    Args:
        X (torch.tensor): (n,) data stream
        test_prop (float)
        random_state (int): seed for the random number generator
        window (int): window for joint input/output sequences
    Returns:
        train (torch.tensor): (ntrain, window, 1) tensor of training sequences
        test (torch.tensor): (ntest, window, 1) tensor of testing sequences
    """
    _seq_idxs = np.arange(window)[np.newaxis]
    n = len(X)
    last_train_stidx = int((1-test_prop)*n - window)
    train_stidxs = np.arange(last_train_stidx)
    test_stidxs = np.arange(last_train_stidx+window, n-window+1)

    train_idxs = train_stidxs[:,np.newaxis] + _seq_idxs
    test_idxs = test_stidxs[:,np.newaxis] + _seq_idxs 

    train = X[train_idxs].unsqueeze(-1)
    test = X[test_idxs].unsqueeze(-1)
    return train, test


def load_data(past_dims, fut_dims, 
    dataset='openei', 
    loc=0, 
    window=36, 
    split_fn=random_week_train_test_split) -> Dict[str,SequenceDataset]:
    """
    Load x and y openEI from path, and perform sequence length preprocessing and train/test split.

    Returns:
        datasets (dict): dictionary with the following fields:
            train (SequenceDataset): training dataset
            val (SequenceDataset): validation dataset
            test (SequenceDataset): testing dataset
    """
    assert past_dims+fut_dims <= window, "too many total dimensions in sequence lengths" 
    if dataset=='openei':
        path = os.path.join(BASE_PATH,"datasets/processed/openEI/openei_011_subset.pt")
    elif dataset=='iso-ne':
        pass
    elif dataset=='nsw':
        pass
    else:
        raise NotImplementedError


    # Data setup:
    X_orig = torch.load(path).float() # (locs, len)

    train_joint, test_joint = split_fn(X_orig[loc], window=36) #(N, window, 1), (M, window, 1)

    # Re-split into appropriate length input and output
    datasets = {
        'train': SequenceDataset({'x':train_joint[:,:past_dims], 'y':train_joint[:,past_dims:past_dims+fut_dims]}),
        'val': SequenceDataset({'x':test_joint[:,:past_dims], 'y':test_joint[:,past_dims:past_dims+fut_dims]}),
        'test': SequenceDataset({'x':test_joint[:,:past_dims], 'y':test_joint[:,past_dims:past_dims+fut_dims]}),
    } 
    return datasets

def load_data_old(loc, past_dims, fut_dims, path_x=None, path_y=None) -> Dict[str,SequenceDataset]:
    """
    Load x and y openEI from path, and perform sequence length preprocessing and train/test split.

    Returns:
        datasets (dict): dictionary with the following fields:
            train (SequenceDataset): training dataset
            val (SequenceDataset): validation dataset
            test (SequenceDataset): testing dataset
    """
    assert past_dims+fut_dims <= 36, "too many total dimensions in sequence lengths" 
    if path_x is None:
        path_x = os.path.join(BASE_PATH,"datasets/processed/openEI/X_openei_011_subset_multitask.pt")
    if path_y is None:
        path_y = os.path.join(BASE_PATH,"datasets/processed/openEI/Y_openei_011_subset_multitask.pt")


    # Data setup:
    X_orig = torch.load(path_x)
    Y_orig = torch.load(path_y)

    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = electric_train_test_split(X_orig, Y_orig, disp_idx=24+12)

    X_train = X_train_orig[loc,:,:24].reshape((-1,24)).unsqueeze(-1).float()
    Y_train = Y_train_orig[loc,:,:12].reshape((-1,12)).unsqueeze(-1).float()
    X_test = X_test_orig[loc,:,:24].reshape((-1,24)).unsqueeze(-1).float()
    Y_test = Y_test_orig[loc,:,:12].reshape((-1,12)).unsqueeze(-1).float()

    # Combine processing output into single-strand sequences 
    train_joint = torch.cat((X_train, Y_train),1)
    test_joint = torch.cat((X_test, Y_test),1)

    # Re-split into appropriate lengths
    datasets = {
        'train': SequenceDataset({'x':train_joint[:,:past_dims], 'y':train_joint[:,past_dims:past_dims+fut_dims]}),
        'val': SequenceDataset({'x':test_joint[:,:past_dims], 'y':test_joint[:,past_dims:past_dims+fut_dims]}),
        'test': SequenceDataset({'x':test_joint[:,:past_dims], 'y':test_joint[:,past_dims:past_dims+fut_dims]}),
    } 
    return datasets