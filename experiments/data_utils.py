import os
import random
import math
import torch
import numpy as np
import src
from src.utils import SequenceDataset
from typing import Dict

BASE_PATH = os.path.normpath(os.path.join(os.path.dirname(src.__file__), '..'))

def random_week_train_test_split(X, test_prop=0.3, random_state=1, window=36, split_test=False):
    """
    Performs train/test split for electric dataset by avoiding overlaps in weekly rolling windows.
    
    Args:
        X (torch.tensor): (n,) data stream
        test_prop (float)
        random_state (int): seed for the random number generator
        window (int): window for joint input/output sequences
        split_test (bool): if true, split test evenly into test/val, else use test for both test and val
    Returns:
        train (torch.tensor): (ntrain, window, 1) tensor of training sequences
        val (torch.tensor): (nval, window, 1) tensor of validation sequences
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

    # split test into val/test if desired
    if split_test:
        idxs = list(range(len(test)))
        random.shuffle(idxs)
        val = test[idxs[:len(idxs)//2]]
        test = test[idxs[len(idxs)//2:]]
    else:
        val = test.clone()

    return train, val, test  

def time_series_train_test_split(X, test_prop=0.3, random_state=1, window=36, split_test=False):
    """
    Performs train/test split by reserving last test_prop portion of sequence for testing.
    
    Args:
        X (torch.tensor): (n,) data stream
        test_prop (float)
        random_state (int): seed for the random number generator
        window (int): window for joint input/output sequences
        split_test (bool): if true, split test evenly into test/val, else use test for both test and val
    Returns:
        train (torch.tensor): (ntrain, window, 1) tensor of training sequences
        val (torch.tensor): (nval, window, 1) tensor of validation sequences
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
    
    # split test into val/test if desired
    if split_test:
        idxs = list(range(len(test)))
        random.shuffle(idxs)
        val = test[idxs[:len(idxs)//2]]
        test = test[idxs[len(idxs)//2:]]
    else:
        val = test.clone()

    return train, val, test


def load_data(past_dims, fut_dims, 
    dataset='openei', 
    loc=0, 
    window=36, 
    split_fn=random_week_train_test_split,
    split_test=False) -> Dict[str,SequenceDataset]:
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
        #path = os.path.join(BASE_PATH,"datasets/processed/iso-ne/iso-ne_8760.pt")
        #path = os.path.join(BASE_PATH,"datasets/processed/iso-ne/iso-ne_17520.pt")
        #path = os.path.join(BASE_PATH,"datasets/processed/iso-ne/iso-ne_26280.pt")
        path = os.path.join(BASE_PATH,"datasets/processed/iso-ne/iso-ne_35064.pt")
    elif dataset=='nau':
        path = os.path.join(BASE_PATH,"datasets/processed/nau/nau.pt")
    else:
        raise NotImplementedError


    # Data setup:
    X_orig = torch.load(path).float() # (locs, len)

    train, val, test = split_fn(X_orig[loc], window=window, split_test=split_test) #(N, window, 1), (M, window, 1)

    # Re-split into appropriate length input and output
    datasets = {
        'train': SequenceDataset({'x':train[:,:past_dims], 'y':train[:,past_dims:past_dims+fut_dims]}),
        'val': SequenceDataset({'x':val[:,:past_dims], 'y':val[:,past_dims:past_dims+fut_dims]}),
        'test': SequenceDataset({'x':test[:,:past_dims], 'y':test[:,past_dims:past_dims+fut_dims]}),
    } 
    return datasets