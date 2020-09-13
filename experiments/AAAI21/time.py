import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append("../../")
import CalibratedTimeseriesModels
from CalibratedTimeseriesModels.utils import *

from CalibratedTimeseriesModels.models.cgmm import *
from CalibratedTimeseriesModels.models.gmnn import *
from CalibratedTimeseriesModels.models.blr import *
from CalibratedTimeseriesModels.models.mogp import *

from load_data import load_data

def main(model_name, n_samples, device):
       
    loc, past_dims, fut_dims = (9, 8, 12) # Loc: SLC, p(+12|-8)
    
    # Data setup
    X_train, Y_train, X_test, Y_test = load_data(loc, past_dims, fut_dims)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    test_point = X_test[[0],:,:].to(device) # (1, i, 1)
    test_output = Y_test[[0],:,:].to(device)
    
    # Model-specific training and sample timing
    if model_name == 'arma':
        model = BayesianLinearRegression(1, past_dims, 1, 1).to(device)
        model.fit(X_train, Y_train[:,[0],:])
        start = time.time()
        samples = sample_forward(model, test_point, fut_dims, 
                                 n_samples=n_samples[0])
        end = time.time()
    
    elif model_name == 'ifnn':
        X_batches, Y_batches = batch(X_train, Y_train[:,[0],:], batch_size = 64)
        hidden_layers = [40, 40, 40]
        model = GaussianMixtureNeuralNet(1, past_dims, hidden_layers, 1, 1, 
                                   n_components=3, random_state=0).to(device)
        train(model, X_batches, Y_batches, num_epochs=150, learning_rate=.005)
        start = time.time()
        samples = sample_forward(model, test_point, fut_dims, 
                                 n_samples=n_samples[0])
        end = time.time()
    
    elif model_name == 'irnn':
        X_batches, Y_batches = batch(X_train, Y_train[:,[0],:], batch_size = 64)
        hidden_layers = [20, 20, 20]
        hidden_dim = 40
        model = GaussianMixtureLSTM(1, hidden_dim, hidden_layers, 1, 1,
                                    n_components=3, random_state=0).to(device)
        train(model, X_batches, Y_batches, num_epochs=200, learning_rate=.005)
        start = time.time()
        samples = sample_forward_lstm_singlepoint(model, test_point, fut_dims, 
                                 n_samples=n_samples[0])
        end = time.time()
        samples = samples.unsqueeze(1)
    
    elif model_name == 'condg':
        model = BayesianLinearRegression(1, past_dims, 1, fut_dims).to(device)
        model.fit(X_train, Y_train)
        start = time.time()
        dist = model(test_point)
        samples = dist.sample(n_samples)
        end = time.time()
    
    elif model_name == 'condgmm':
        model = ConditionalGMM(1, past_dims, 1, fut_dims, 
                               n_components=5, random_state=4).to(device)
        model.fit(X_train, Y_train)
        start = time.time()
        dist = model(test_point)
        samples = dist.sample(n_samples)
        end = time.time()
    
    elif model_name == 'mogp':
        train_x = X_train.reshape(X_train.size(0),-1)
        train_y = Y_train.reshape(Y_train.size(0),-1)
        test_x = test_point.reshape(test_point.size(0),-1)
        kernels = {'matern': gpytorch.kernels.MaternKernel(ard_num_dims=past_dims),
                   'rq': gpytorch.kernels.RQKernel(ard_num_dims=past_dims)}
        covar_kernel = kernels['rq']*kernels['matern']
        index_rank = 8 
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=fut_dims).to(device)
        model = MultiOutputGP(train_x, train_y, likelihood, 
                              covar_kernel=covar_kernel,
                              index_rank=index_rank,random_state=0).to(device)
        
        train_mogp(model, likelihood, train_x, train_y, 55, verbose=False)
        
        model.eval()
        likelihood.eval()            
        start = time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = model(test_x)
        samples = dist.sample(n_samples)
        end = time.time()
    
    elif model_name == 'jfnn':
        X_batches, Y_batches = batch(X_train, Y_train, batch_size = 64)
        hidden_layers = [40, 40, 40]
        model = GaussianMixtureNeuralNet(1, past_dims, hidden_layers, 
                                         1, fut_dims, n_components=2, 
                                         covariance_type='low-rank',
                                         rank=2, random_state=0).to(device)
        train(model, X_batches, Y_batches, num_epochs=300, learning_rate=.002)
        start = time.time()
        dist = model(test_point)
        samples = dist.sample(n_samples)
        end = time.time()
    
    elif model_name == 'jrnn':
        X_batches, Y_batches = batch(X_train, Y_train, batch_size = 64)
        hidden_layers = [40,40,40]
        hidden_dim = 40
        model = GaussianMixtureLSTM(1, hidden_dim, hidden_layers,  1, fut_dims,
                                    covariance_type='low-rank', rank=3, 
                                    n_components=2, random_state=1).to(device)
        train(model, X_batches, Y_batches, num_epochs=200, learning_rate=.005)
        start = time.time()
        dist = model(test_point)
        samples = dist.sample(n_samples)
        end = time.time()
    
    print('Elapsed Time: %f' %((end-start)))
    print(samples.shape)
    print(mape(samples, test_output, sampled=True)[0])
    
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    use_cuda = torch.cuda.is_available()
    parser.add_argument('--model', default='cgmm')
    parser.add_argument('--n', default=10000, type=int)
    parser.add_argument('--use_cuda', default=use_cuda, type=bool)
    args = parser.parse_args()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    print(args)
    main(args.model, torch.Size([args.n]), device)
    