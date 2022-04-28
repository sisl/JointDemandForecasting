from tqdm import tqdm 
import torch
import torch.nn as nn
import gpytorch
import src
from src.utils import *
from src.train_utils import train, train_mogp

from experiments.charging_utils import *
from src.models import *
from typing import Optional, Dict
MODELS = ['ARMA', 'IFNN', 'IRNN', 'CG', 'JFNN', 'JRNN', 'MOGP', 'CGMM', 'CANF']
#        kernels = {'rbf': gpytorch.kernels.RBFKernel(),
#           'ind_rbf': gpytorch.kernels.RBFKernel(ard_num_dims=past_dims),
#           'matern': gpytorch.kernels.MaternKernel(ard_num_dims=past_dims),
#           'rq': gpytorch.kernels.RQKernel(ard_num_dims=past_dims),
#           'spectral': gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2,ard_num_dims=24)
#          }

def initialize_model(
    model_name:str, 
    past_dims:int, 
    fut_dims:int, 
    seed:int=None, 
    mogp_data:Optional[Dict[str, torch.Tensor]]=None,
    **model_kwargs):
    
    if model_name=='ARMA':
        model = BayesianLinearRegression(1, past_dims, 1, 1)
    
    elif model_name=='IFNN':
        hidden_layers = [40, 40, 40]
        ncomps = 3
        model = GaussianMixtureNeuralNet(1, past_dims, hidden_layers, 1, 1, 
                                       n_components=ncomps, random_state=seed)
    
    elif model_name=='IRNN':
        hidden_layers = [20, 20, 20]
        hidden_dim = 40
        model = GaussianMixtureLSTM(1, hidden_dim, hidden_layers, 1, 1, 
                                     n_components=3, random_state=seed, random_start=False)
    
    elif model_name=='CG':
        model = BayesianLinearRegression(1, past_dims, 1, fut_dims)
    
    elif model_name=='JFNN':
        hidden_layers = [40, 40, 40]
        ncomps = 2
        covtype = 'low-rank'
        rank = 2
        model = GaussianMixtureNeuralNet(1, past_dims, hidden_layers, 1, fut_dims, 
                                     n_components=ncomps, covariance_type=covtype,
                                     rank=rank, random_state=seed)
    
    elif model_name=='JRNN':
        hidden_layers = [40,40,40]
        hidden_dim = 40
        ncomps = 2
        covtype = 'low-rank'
        rank = 2
        model = GaussianMixtureLSTM(1, hidden_dim, hidden_layers,  1, fut_dims,
                                n_components=ncomps, covariance_type=covtype, rank=rank, 
                                random_state=seed, random_start=False)
    
    elif model_name=='MOGP':
        assert mogp_data is not None, "No train_x, train_y passed"
        covar_kernel = gpytorch.kernels.RQKernel(ard_num_dims=past_dims) * gpytorch.kernels.MaternKernel(ard_num_dims=past_dims)
        index_rank = 8
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=fut_dims)
        model = MultiOutputGP(mogp_data['train_x'], mogp_data['train_y'], likelihood, covar_kernel=covar_kernel, 
                          index_rank=index_rank, random_state=seed)
    
    elif model_name=='CGMM':
        if past_dims==8:
            ncomp = 4
        elif past_dims==24:
            ncomp = 5
        else:
            raise NotImplementedError
        model = ConditionalGMM(1, past_dims, 1, fut_dims, 
                          n_components=ncomp, random_state=seed)
    
    elif model_name=='CANF':
        ncomp = 25
        nflows = 10
        hdim = 32
        model = ConditionalANF(1, past_dims, 1, fut_dims, 
            hidden_dim=hdim, n_flows=nflows,     
            n_components=ncomp, random_state=seed)
    else:
        raise NotImplementedError
    return model

def train_model(
    model_name:str, 
    model:torch.nn.Module, 
    dataset, 
    seed:Optional[int]=None, 
    mogp_data:Optional[Dict[str, torch.Tensor]]=None,
    **train_kwargs):

    if model_name=='ARMA':
        model.fit(dataset['train'][:]['x'], dataset['train'][:]['y'][:,[0],:])
    
    elif model_name=='IFNN':
        X_batches, Y_batches = batch(dataset['train'][:]['x'], dataset['train'][:]['y'][:,[0],:], batch_size = 64, random_state = seed)
        batches = {'X_batches':X_batches, 'Y_batches':Y_batches}
        train(model, batches, epochs=150, learning_rate=.005)
    
    elif model_name=='IRNN':
        X_batches, Y_batches = batch(dataset['train'][:]['x'], dataset['train'][:]['y'][:,[0],:], batch_size = 64, random_state = seed)
        batches = {'X_batches':X_batches, 'Y_batches':Y_batches}
        train(model, batches, epochs=200, learning_rate=.005)
    
    elif model_name=='CG':
        model.fit(dataset['train'][:]['x'], dataset['train'][:]['y'])
    
    elif model_name=='JFNN':
        X_batches, Y_batches = batch(dataset['train'][:]['x'], dataset['train'][:]['y'], batch_size = 64, random_state = seed)
        batches = {'X_batches':X_batches, 'Y_batches':Y_batches}
        train(model, batches, epochs=300, learning_rate=.002)
    
    elif model_name=='JRNN':
        X_batches, Y_batches = batch(dataset['train'][:]['x'], dataset['train'][:]['y'], batch_size = 64, random_state = seed)
        batches = {'X_batches':X_batches, 'Y_batches':Y_batches}
        train(model, batches, epochs=200, learning_rate=.005)
    
    elif model_name=='MOGP':
        assert mogp_data is not None, "No train_x, train_y passed"     
        if model.past_dims==8:
            ep = 55
        elif model.past_dims==24:
            ep = 75
        else:
            raise NotImplementedError
        train_mogp(model, mogp_data, epochs=ep)
    
    elif model_name=='CGMM':
        model.fit(dataset['train'][:]['x'], dataset['train'][:]['y'])
    
    elif model_name=='CANF':
        nsamp = 100000
        ep = 500 # 4000
        lr = 0.005
        model.fit(dataset, n_samples=nsamp, 
            epochs=ep, seed=seed, val_every=100, lr=lr) # train_kwargs
    else:
        raise NotImplementedError
    return model

def generate_samples(model_name, model, dataset, mogp_data=None, n_samples=1000):
    fut_dims = dataset['train'][:]['y'].shape[1]
    model.eval()
    if model_name in ['ARMA','IFNN']:
        samples = sample_forward(model, dataset['test'][:]['x'], fut_dims, n_samples=n_samples)
    
    elif model_name=='IRNN':
        samples = sample_forward_lstm(model, dataset['test'][:]['x'], fut_dims, n_samples=n_samples)
    
    elif model_name in ['CG', 'JFNN', 'JRNN', 'CGMM']:
        samples = model(dataset['test'][:]['x']).sample((n_samples,))

    elif model_name=='MOGP':
        assert mogp_data is not None, "No train_x, train_y passed"     
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            samples = model(mogp_data['test_x']).sample(torch.Size([n_samples]))
    
    elif model_name=='CANF':
        dist = model(dataset['test'][:]['x'])
        samples = []
        for _ in range(n_samples//200):
            s = dist.sample((200,))
            samples.append(s) 
        samples = torch.cat(samples,0)
    else:
        raise NotImplementedError
    return samples

def tune():
    pass

def test(model_name, loc, past_dims, fut_dims, nseeds):

    # get dataset
    dataset = load_data(loc, past_dims, fut_dims)

    mogp_data = {
        'train_x': dataset['train'][:]['x'].reshape(dataset['train'][:]['x'].size(0),-1).contiguous(),
        'train_y': dataset['train'][:]['y'].reshape(dataset['train'][:]['y'].size(0),-1).contiguous(),
        'test_x': dataset['test'][:]['x'].reshape(dataset['test'][:]['x'].size(0),-1).contiguous(),
        'test_y': dataset['test'][:]['y'].reshape(dataset['test'][:]['y'].size(0),-1).contiguous(),
        } if model_name=='MOGP' else None

    # loop through and save wapes, rwses, dscores [ignoring nlls, train nlls]
    metrics = {'WAPE':[], 'RWSE':[], 'DScore':[]}
    for seed in tqdm(range(nseeds)):
        
        # define model
        model = initialize_model(model_name, past_dims, fut_dims, seed=seed, mogp_data=mogp_data)

        # train
        train_model(model_name, model, dataset, seed=seed, mogp_data=mogp_data)
        #try:
        #    train_model(model_name, model, dataset)
        #except:
        #    continue
        
        # test
        samples = generate_samples(model_name, model, dataset, n_samples=1000, mogp_data=mogp_data)
        metrics['WAPE'].append(wape(samples,dataset['test'][:]['y'], sampled=True)[0])
        metrics['RWSE'].append(rwse(samples,dataset['test'][:]['y'], sampled=True)[0])
        #nlls.append(nll(dist,Y_test)[0])
        #trnlls.append(nll(dist_tr,Y_train)[0])
        min_indices = 4
        obj_fn = lambda x: var(x, 0.8)
        metrics['DScore'].append(index_allocation(samples, min_indices, obj_fn, dataset['test'][:]['y'], 0.8))
    
    print(f'{model_name} Metrics:')
    for metric_key in metrics.keys():
        metrics[metric_key] = torch.tensor(metrics[metric_key])
        print(f'{metric_key}s: {metrics[metric_key]}')
        print(f'{metric_key}: {metrics[metric_key].mean()} \pm {metrics[metric_key].std()}')



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run model testing')

    #parser.add_argument("--tune", help="tune hyperparameters",
    #    action="store_true")
    #parser.add_argument("--test", help="test stored hyperparameters",
    #    action="store_true")
    parser.add_argument('-ntestseeds', default=10, type=int,
        help='number of test seeds')
    #parser.add_argument('-ntuneseeds', default=10, type=int,
    #    help='number of tuning seeds')      
    parser.add_argument('-model', choices=MODELS, 
        default='ARMA', help='model to run')
    parser.add_argument('-loc', default=1, type=int,
        help='location in OpenEI TMY3 Dataset')
    parser.add_argument('-input', default=8, type=int,
        help='input time-series length')
    parser.add_argument('-output', default=12, type=int,
        help='output time-series length')                  
    args = parser.parse_args()
    test(args.model, args.loc, args.input, args.output, args.ntestseeds)


