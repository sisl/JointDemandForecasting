from tqdm import tqdm 
import torch
import torch.nn as nn
import gpytorch
from ray import tune
import json
import src
from src.utils import *
from src.train_utils import train, train_mogp, train_pinball
from experiments.charging_utils import *
from experiments.data_utils import load_data
from experiments.get_config import get_config, get_config_ray
from src.models import *
from typing import Optional, Dict,  List
from functools import partial
BASEPATH = os.path.dirname(os.path.abspath(__file__))

MODELS = ['ARMA', 'IFNN', 'IRNN', 'CG', 'JFNN', 'JRNN', 'MOGP', 'CGMM', 'CANF', 
    'EncDec', 'ResNN','QResPinb', 'QRNNPinb', 'QRNNDecPinb']
DATASETS=['openei', 'nau', 'iso-ne1', 'iso-ne2', 'iso-ne3', 'iso-ne4']
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
    
    elif model_name=='ResNN':
        model = GaussianResnet(1, past_dims, 1, 1, **model_kwargs)
    
    elif model_name in ['QResPinb']:
        model = QuantileResnet(1, past_dims, 1, 1, **model_kwargs)

    elif model_name in ['IRNN','EncDec']:
        if model_kwargs['n_components'] == 1:
            del model_kwargs['n_components']
            model = GaussianLSTM(1, 1, 1, **model_kwargs)
        else:
            model = GaussianMixtureLSTM(1, 1, 1, **model_kwargs)

    elif model_name in ['QRNNPinb','QRNNDecPinb']:
        model = QuantileLSTM(1, 1, 1, **model_kwargs)

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
    

    # train models
    if model_name in ['ARMA', 'CG', 'CGMM']:
        model.fit(dataset['train'][:]['x'], dataset['train'][:]['y'][:,:model.K])
    
    elif model_name in ['IFNN','IRNN','JFNN','JRNN','EncDec', 'ResNN']:
        train(model, dataset, **train_kwargs)
    
    elif model_name in ['QResPinb','QRNNPinb','QRNNDecPinb']:
        train_pinball(model, dataset, **train_kwargs)

    elif model_name=='MOGP':
        assert mogp_data is not None, "No train_x, train_y passed"     
        train_mogp(model, mogp_data, **train_kwargs)
    
    elif model_name=='CANF':
        model.fit(dataset, **train_kwargs)
    
    else:
        raise NotImplementedError
    return model

def generate_samples(model_name, model, dataset, mogp_data=None, n_samples=1000):
    """
    Generate samples from minibatches of data
    """
    
    B, fut_dims, O = dataset['y'].shape
    model.eval()

    if model_name in ['ARMA','IFNN','ResNN','QResPinb']:
        samples = sample_forward(model, dataset['x'], fut_dims, n_samples=n_samples)

    elif model_name in ['IRNN','EncDec','QRNNPinb','QRNNDecPinb']:
        samples = sample_forward_lstm(model, dataset['x'], fut_dims, n_samples=n_samples)

    elif model_name in ['CG', 'JFNN', 'JRNN', 'CGMM','CANF']:
        samples = model(dataset['x']).sample((n_samples,))

    elif model_name=='MOGP':
        assert mogp_data is not None, "No train_x, train_y passed"     
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            samples = model(mogp_data['x']).sample(torch.Size([n_samples]))

    else:
        raise NotImplementedError

    return samples

def calc_metrics(model_name, model, dataset, mogp_data=None, n_samples=1000, batch_size=200):
    """
    Calculate metrics in minibatches and aggregate them
    """
    # batch datasets - use indices rather than dataloader because of mogp data
    B = len(dataset)
    idx_list = np.array_split(np.arange(B), B//batch_size)
    
    # iterate over batches
    batched_metrics, batch_sizes = [], []
    for idxs in idx_list:
        batch = dataset[idxs]
        mogp_data_batch = None
        if mogp_data is not None:
            mogp_data_batch = {l:mogp_data[l][idxs] for l in ['x','y']}

        # generate samples
        samples = generate_samples(model_name, model, batch, mogp_data=mogp_data_batch, n_samples=1000)
        
        # generate batched metrics
        batched_metrics.append({
            'WAPE': wape(samples, batch['y'], sampled=True)[0].item(), 
            'RWSE': rwse(samples, batch['y'], sampled=True)[0].item(),
            'QS': quantile_score(samples, batch['y'])[0].item(),
            #nlls.append(nll(dist,Y_test)[0])
            #trnlls.append(nll(dist_tr,Y_train)[0])
            'DScore':index_allocation(samples, 4, lambda x: var(x, 0.8), batch['y'], 0.8).item(),
        })
        batch_sizes.append(len(batch))

    metrics = {}
    for key in batched_metrics[0].keys():
        if key == 'RWSE': # rws combine
            metrics[key] = rws_combine([m[key] for m in batched_metrics], batch_sizes)
        else: # mean combine
            metrics[key] = mean_combine([m[key] for m in batched_metrics], batch_sizes)
    return metrics

def rws_combine(batch_metrics:List[float], batch_sizes:List[int]):
    """
    Combine metrics that need to be root-weighted-squared
    Args:
        batch_metrics (List[float]): list of float metrics
        batch_sizes (List[int]): list of batch sizes
    Returns:
        metric (float): averaged metric
    """
    x, lens = np.array(batch_metrics), np.array(batch_sizes)
    wsquares = x**2 * lens
    metric = (np.sum(wsquares)/np.sum(lens)) ** 0.5
    return metric

def mean_combine(batch_metrics:List[float], batch_sizes:List[int]):
    """
    Combine metrics by weighted mean
    Args:
        batch_metrics (List[float]): list of float metrics
        batch_sizes (List[int]): list of batch sizes
    Returns:
        metric (float): averaged metric
    """
    x, lens = np.array(batch_metrics), np.array(batch_sizes)
    wmeans = x * lens
    metric = np.sum(wmeans) /np.sum(lens)
    return metric

def train_test(config, 
    model_name=None,
    dataset=None, 
    loc=None, 
    past_dims=None, 
    fut_dims=None, 
    ray=False, 
    split_test=True):
    
    assert None not in [model_name, dataset, loc, past_dims, fut_dims]
    print(f'Running {model_name} on {dataset}-{loc} for {fut_dims} given {past_dims} and split_test {split_test}')
    
    # get dataset and config
    dataset = load_data(dataset, past_dims, fut_dims, loc=loc, split_test=split_test)
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
    print('Initializing model')
    model = initialize_model(model_name, past_dims, fut_dims, mogp_data=mogp_data, **config['model'])

    # train
    print('Training model')
    train_model(model_name, model, dataset, mogp_data=mogp_data, ray=ray, **config['train'])
    
    # Generate val and test metrics 
    metrics = {}
    for data_type in ['val', 'test']:
        print(f'Generating {data_type} metrics')

        """
        # gen samples
        samples = generate_samples(model_name, model, dataset[data_type][:], 
            n_samples=1000, 
            mogp_data=mogp_data[data_type] if mogp_data is not None else None)

        # get and print metrics
        _met = calc_metrics(samples, dataset[data_type][:]['y'])
        """

        # calculate metrics
        _met = calc_metrics(model_name, model, dataset[data_type],
            n_samples=1000,
            mogp_data=mogp_data[data_type] if mogp_data is not None else None)

        metrics.update({f'{data_type}_{key}': value for key, value in _met.items()})

    # print metrics and report if ray
    print(f'{model_name} Metrics:')
    for key in metrics.keys():
        print(f'{key}: {metrics[key]}')

    if ray:
        tune.report(**metrics)
        with open('metrics.json', 'w') as fp:
            json.dump(metrics, fp)

# print(f'{model_name} Metrics:')
#for metric_key in metrics_all[0].keys():
#    ms = torch.tensor([m[metric_key] for m in metrics_all])
#    print(f'{metric_key}s: {ms}')
#    print(f'{metric_key}: {ms.mean()} \pm {ms.std()}')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run model testing')

    parser.add_argument('-nseeds', default=5, type=int,
        help='number of seeds of each model to run') 
    parser.add_argument('-model', choices=MODELS, 
        default='ARMA', help='model to run')
    parser.add_argument('-dataset', choices=DATASETS,
        default='openei', help='dataset to run')
    parser.add_argument('-loc', default=0, type=int,
        help='location in batch of time series')
    parser.add_argument('-input', default=8, type=int,
        help='input time-series length')
    parser.add_argument('-output', default=12, type=int,
        help='output time-series length')   
    parser.add_argument("--ray", help="run ray tune",
        action="store_true", default=False)
    parser.add_argument("--train", help="train model and store config with best RWSE",
        action="store_true", default=False)
    parser.add_argument("--val_on_test", help="validate on test set if specified. otherwise split holdout into 50-50 val-test",
        action="store_true", default=False)
    parser.add_argument('-cpus_per_trial', default=1, type=int,
        help='number of cpus per trial')                
    args = parser.parse_args()

    if args.ray: # Main training / testing scripts
        
        # path to best config
        dataname = args.dataset if args.loc==0 else f'{args.dataset}-{args.loc}'
        results_path = os.path.join(BASEPATH,'results', dataname, 
            f'in{args.input}_out{args.output}_valontest{args.val_on_test}',args.model)

        if args.train: # get hyperparameter sweep ray config
            config = get_config_ray(args.model, args.loc, args.input, args.output, args.nseeds)

        else: # retrieve stored best config with ray sweep over test seeds
            if not os.path.exists(os.path.join(results_path,'best_config.json')):
                raise Exception('No best config found, run with --train first')
            with open(os.path.join(results_path,'best_config.json')) as fp:
                config = json.load(fp)
            config['seed'] = tune.grid_search(list(range(100,100+args.nseeds)))

        analysis = tune.run(partial(train_test, 
                model_name=args.model, 
                dataset=args.dataset,
                loc=args.loc, 
                past_dims=args.input, 
                fut_dims=args.output,  
                ray=args.ray,
                split_test=not args.val_on_test), 
            config=config, 
            resources_per_trial={"cpu":args.cpus_per_trial},
            log_to_file='out.log')

        if args.train: # save best config file
            best_config = analysis.get_best_config('val_RWSE','min')
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            with open(os.path.join(results_path,'best_config.json'), 'w') as fp:
                json.dump(best_config, fp)

        else: # aggregate metrics across test seeds 
            results = analysis.results
            seed_keys = list(results.keys())
            metric_keys = [[key for key in results[skey].keys() if 'test' in key] for skey in seed_keys] # look for 'test' keys
            metric_keys = list(set(sum(metric_keys,[]))) # take union of all metric keys since some seeds may have failed
            
            # aggregate metrics
            metrics = {mkey:[results[skey][mkey] for skey in seed_keys if mkey in results[skey].keys()] for mkey in metric_keys}
            
            # produce summary metrics
            summary = {}
            for key, val in metrics.items():
                vec =  np.array(val)
                mu, sig = np.mean(vec).item(), np.std(vec).item()
                summary['mean_'+key] = mu
                summary['std_'+key] = sig

            # try to remove outliers - indices outside 1.5*IQR from Q1, Q3
            try:
                vec = np.array(metrics['test_RWSE'])
                q1, q2, q3 = np.quantile(vec,.25), np.quantile(vec,.5), np.quantile(vec,.75)
                iqr = q3-q1
                lf, uf = q1-1.5*iqr, q3+1.5*iqr
                good_idxs = np.where((vec < uf) & (vec > lf))
                summary['outlierprop'] = 1. - len(good_idxs)/len(vec)
                
                for key, val in metrics.items():
                    vec =  np.array(val)[good_idxs]
                    mu, sig = np.mean(vec).item(), np.std(vec).item()
                    summary['nooutliermean_'+key] = mu
                    summary['nooutlierstd_'+key] = sig
            except:
                pass
            metrics.update(summary)
            
            # save to file
            with open(os.path.join(results_path,'test_results.json'), 'w') as fp:
                json.dump(metrics, fp)

        
    else: # Performing rapid testing of single runs
        config = get_config(args.model, args.loc, args.input, args.output)
        train_test(config, 
            model_name=args.model,
            dataset=args.dataset, 
            loc=args.loc, 
            past_dims=args.input, 
            fut_dims=args.output,  
            ray=args.ray,
            split_test=not args.val_on_test)


