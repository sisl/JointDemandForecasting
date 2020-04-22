import numpy as np
import torch

def symmetric(matrix):
    return (matrix+matrix.t())/2

def predict(mu, sigma, A, Q):
    mup = torch.matmul(A,mu)
    sigmap = symmetric(torch.matmul(A,sigma).matmul(A.t()) + Q)
    return mup, sigmap

def measure(mu, sigma, obs, C, R):
    obse, innov = expected_observation(mu, sigma, C, R)
    resid = obs - obse
    K = torch.matmul(sigma, C.t()).matmul(innov.inverse())
    
    mun = mu + torch.matmul(K,resid)
    sigman = (torch.eye(sigma.shape[0]) - torch.matmul(K, C)).matmul(sigma)
    sigman = symmetric(sigman)
    return mun, sigman

def expected_observation(mu, sigma, C, R):
    obse = torch.matmul(C,mu)
    obssigma = symmetric(torch.matmul(C,sigma).matmul(C.t()) + R)
    return obse, obssigma

def kalman_update(mu, sigma, obs, A, C, Q, R):
    mup, sigmap = predict(mu, sigma, A, Q)
    mun, sigman = measure(mup, sigmap, obs, C, R)
    return mun, sigman

def kalman_fit_predict(data, A, C, Q=None, R=None):
    # data of shape (# time steps, # observations)
    # default kf values
    if Q is None:
        Q = torch.eye(A.shape[0])
    if R is None:
        R = torch.zeros(C.shape[0],C.shape[0])
    
    # initial belief
    mu = torch.zeros(A.shape[0], 1)
    sigma = torch.eye(A.shape[0])
    mu_history = [mu]
    sigma_history = [sigma]
    obse_history = []
    obssigma_history = []
    
    # Run Kalman Filter
    for i in range(data.shape[0]):
        obs = data[[i],:].t()
        mup, sigmap = predict(mu, sigma, A, Q)
        obse, obssigma = expected_observation(mup, sigmap, C, R)
        mu, sigma = measure(mup, sigmap, obs, C, R)
        
        mu_history.append(mu) 
        sigma_history.append(sigma)
        obse_history.append(obse)
        obssigma_history.append(obssigma)
        
    # predict next hidden state and observation
    mup, sigmap = predict(mu, sigma, A, Q)
    obse, obssigma = expected_observation(mup, sigmap, C, R)
    
    return obse, obssigma, mu_history, sigma_history, obse_history, obssigma_history