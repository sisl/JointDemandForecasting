from tqdm import tqdm 
import torch
import torch.nn as nn
import gpytorch
from ray import tune

import src
from src.utils import *
from src.train_utils import train, train_mogp

from experiments.charging_utils import *
from experiments.get_config import get_config, get_config_ray
from src.models import *
from typing import Optional, Dict
from functools import partial

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
    mogp_data:Optional[Dict[str, torch.Tensor]]=None,
    **model_kwargs):
    
    if model_name=='ARMA':
        model = BayesianLinearRegression(1, past_dims, 1, 1)
    
    elif model_name=='IFNN':
        if model_kwargs['n_components'] == 1:
            del model_kwargs['n_components']
            model = GaussianNeuralNet(1, past_dims, 1, 1, **model_kwargs)
        else:
            model = GaussianMixtureNeuralNet(1, past_dims, 1, 1, **model_kwargs)
    
    elif model_name=='IRNN':
        if model_kwargs['n_components'] == 1:
            del model_kwargs['n_components']
            model = GaussianLSTM(1, 1, 1, **model_kwargs)
        else:
            model = GaussianMixtureLSTM(1, 1, 1, **model_kwargs)          
    
    elif model_name=='CG':
        model = BayesianLinearRegression(1, past_dims, 1, fut_dims)
    
    elif model_name == 'JFNN':
        if model_kwargs['n_components'] == 1:
            del model_kwargs['n_components']
            model = GaussianNeuralNet(1, past_dims, 1, fut_dims, **model_kwargs)
        else:
            model = GaussianMixtureNeuralNet(1, past_dims, 1, fut_dims, **model_kwargs)
    
    elif model_name=='JRNN':
        if model_kwargs['n_components'] == 1:
            del model_kwargs['n_components']
            model = GaussianLSTM(1, 1, fut_dims, **model_kwargs)
        else:
            model = GaussianMixtureLSTM(1, 1, fut_dims, **model_kwargs)    
    
    elif model_name=='MOGP':
        assert mogp_data is not None, "No train_x, train_y passed"
        covar_kernel = gpytorch.kernels.RQKernel(ard_num_dims=past_dims) * gpytorch.kernels.MaternKernel(ard_num_dims=past_dims)
        model = MultiOutputGP(mogp_data['train']['x'], mogp_data['train']['y'], 
            covar_kernel=covar_kernel, **model_kwargs)
    
    elif model_name=='CGMM':
        model = ConditionalGMM(1, past_dims, 1, fut_dims, **model_kwargs)      
    
    elif model_name=='CANF':
        model = ConditionalANF(1, past_dims, 1, fut_dims, **model_kwargs)

    else:
        raise NotImplementedError
    return model

def train_model(
    model_name:str, 
    model:torch.nn.Module, 
    dataset, 
    mogp_data:Optional[Dict[str, torch.Tensor]]=None,
    **train_kwargs):
    
    # update training/val data for single step in iterative models
    if model_name in ['ARMA', 'IFNN', 'IRNN']:
        for key in ['train','val']:
            dataset[key].data['y'] = dataset[key].data['y'][:,[0]]

    # train models
    if model_name in ['ARMA', 'CG', 'CGMM']:
        model.fit(dataset['train'][:]['x'], dataset['train'][:]['y'])
    
    elif model_name in ['IFNN','IRNN','JFNN','JRNN']:
        train(model, dataset, **train_kwargs)
    
    elif model_name=='MOGP':
        assert mogp_data is not None, "No train_x, train_y passed"     
        train_mogp(model, mogp_data, **train_kwargs)
    
    elif model_name=='CANF':
        model.fit(dataset, **train_kwargs)
    
    else:
        raise NotImplementedError
    return model

def generate_samples(model_name, model, dataset, mogp_data=None, n_samples=1000):
    fut_dims = dataset['test'][:]['y'].shape[1]
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
            samples = model(mogp_data['test']['x']).sample(torch.Size([n_samples]))
    
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

def calc_metrics(samples, test_y):
        min_indices = 4
        obj_fn = lambda x: var(x, 0.8)
        metrics = {
            'WAPE': wape(samples,test_y, sampled=True)[0].item(), 
            'RWSE': rwse(samples,test_y, sampled=True)[0].item(),
            #nlls.append(nll(dist,Y_test)[0])
            #trnlls.append(nll(dist_tr,Y_train)[0])
            'DScore':index_allocation(samples, min_indices, obj_fn, test_y, 0.8).item(),
        }
        return metrics

def test(config, model_name=None, loc=None, past_dims=None, fut_dims=None, ray=False):
    assert None not in [model_name, loc, past_dims, fut_dims]
    # get dataset and config
    dataset = load_data(loc, past_dims, fut_dims)

    mogp_data = {
        d:{
            l:dataset[d][:][l].reshape(len(dataset[d][:][l]),-1).contiguous() for l in ['x','y',]
            } for d in ['train','val','test']
        } if model_name=='MOGP' else None

    # train model and report metrics
    seed = config['seed']
        
    # set seed
    set_seed(seed)

    # define model
    model = initialize_model(model_name, past_dims, fut_dims, mogp_data=mogp_data, **config['model'])

    # train
    train_model(model_name, model, dataset, mogp_data=mogp_data, ray=ray, **config['train'])
    #try:
    #    train_model(model_name, model, dataset)
    #except:
    #    continue
    
    # test
    samples = generate_samples(model_name, model, dataset, n_samples=1000, mogp_data=mogp_data)

    # get and print metrics
    metrics = calc_metrics(samples, dataset['test'][:]['y'])
    if ray:
        tune.report(**metrics)
    else:
        print(f'{model_name} Metrics:')
        for key in metrics.keys():
            print(f'{key}: {metrics[key]}')

# print(f'{model_name} Metrics:')
#for metric_key in metrics_all[0].keys():
#    ms = torch.tensor([m[metric_key] for m in metrics_all])
#    print(f'{metric_key}s: {ms}')
#    print(f'{metric_key}: {ms.mean()} \pm {ms.std()}')

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
    parser.add_argument("--ray", help="run ray tune",
        action="store_true", default=False)
    parser.add_argument('-cpus_per_trial', default=1, type=int,
        help='number of cpus per trial')                
    args = parser.parse_args()

    if args.ray:
        config = get_config_ray(args.model, args.loc, args.input, args.output, args.ntestseeds)
        tune.run(partial(test, 
                model_name=args.model, 
                loc=args.loc, 
                past_dims=args.input, 
                fut_dims=args.output,  
                ray=args.ray), 
            config=config, resources_per_trial={"cpu":args.cpus_per_trial})
    else:
        config = get_config(args.model, args.loc, args.input, args.output)
        test(config, model_name=args.model, loc=args.loc, past_dims=args.input, fut_dims=args.output,  ray=args.ray)


