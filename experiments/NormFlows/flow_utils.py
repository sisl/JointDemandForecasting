import os
import sys
#import numpy as np
import torch
#import torch.nn as nn


class Normalize(object):
    """ 
    Class to normalize a 2D tensor
    """
    
    def __init__(self, data):
        """ 
        Initialize transformation parameters. 

        Args: 
            data (torch.tensor): (B, nfeatures) tensor
        """
        
        assert len(data.shape)==2, "Improper data size"
        
        self.mean = torch.mean(data,0)
        self.std = torch.mean(data,0)
   
    def normalize(self, data):
        """ 
        Apply mean-std normalization. 

        Args: 
            data (torch.tensor): (B, nfeatures) tensor
        Returns:
            trans (torch.tensor): (B, nfeatures) transformed tensor
        """
        
        assert len(data.shape)==2, "Improper data size"
        trans = (data-self.mean.unsqueeze(0))/self.std.unsqueeze(0)
        return trans
        
    
    def unnormalize(self, data):
        """ 
        Unapply mean-std normalization. 

        Args: 
            data (torch.tensor): (B, nfeatures) tensor
        Returns:
            trans (torch.tensor): (B, nfeatures) transformed tensor
        """
        
        assert len(data.shape)==2, "Improper data size"
        trans = self.std.unsqueeze(0)*data + self.mean.unsqueeze(0)
        return trans

class PCA(object):
    """
    Perform principal component analysis.
    """
    
    def __init__(self, data):
        """
        Perform PCA and initialize transformation parameters.
        
        Args:
            data (torch.tensor): (npoints, nfeatures) normalized data tensor 
        """
        assert len(data.shape) == 2, "Improper data size"
        self.npoints, self.nfeatures = data.shape
        Sig = data.T @ data / self.npoints
        self.U, self.S, _ = torch.svd(Sig)
        self.explained_var = torch.cumsum(self.S,dim=0)/torch.sum(self.S)
        
    def compress(self, data, r):
        """
        Compress data according to principal components.
        
        Args:
            data (torch.tensor): (npoints, nfeatures) normalized data tensor
            r (torch.tensor): rank of compression
            
        Returns:
            compressed (torch.tensor): (npoints, r) compressed data tensor
        """
        assert len(data.shape) == 2, "Improper data size"
        assert r <= self.nfeatures and r > 0, "Improper rank"
        compressed = data @ self.U[:,:r]
        return compressed
    
    def decompress(self, data):
        """
        Compress data according to principal components.
        
        Args:
            data (torch.tensor): (npoints, r) compressed data tensor
            
        Returns:
            decompressed (torch.tensor): (npoints, nfeatures) uncompressed data tensor
        """
        assert len(data.shape) == 2, "Improper data size"
        r = data.shape[1]
        decompressed = data @ self.U[:,:r].T
        return decompressed
    
    
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