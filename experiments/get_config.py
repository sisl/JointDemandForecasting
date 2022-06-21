from ray import tune
# MODELS = ['ARMA', 'IFNN', 'IRNN', 'CG', 'JFNN', 'JRNN', 'MOGP', 'CGMM', 'CANF', 
#     'EncDec', 'ResNN', 'QRes','QResPinb', 'QRNN', 'QRNNPinb', 'QRNNDec']

def get_config_ray(model_name:str, loc, past_dims, fut_dims, nseeds):
    config = {'seed':tune.grid_search(list(range(nseeds))), 'model':{}, 'train':{}}    
    if model_name=='ARMA':
        pass
    
    elif model_name=='IFNN':
        config['model'].update(dict(hidden_layers=tune.grid_search([2,3]), 
            hidden_dims=tune.grid_search([20,40]), 
            n_components=tune.grid_search([2,3])))
        config['train'].update(dict(epochs=tune.grid_search([100,200,300]), 
            learning_rate=.005, 
            batch_size=64))

    elif model_name=='ResNN':
        config['model'].update(dict(hidden_layers=tune.grid_search([2,3,4]), 
            hidden_dims=tune.grid_search([20,40]), 
            large_skip_every=2,
            in_out_skip=True))
        config['train'].update(dict(epochs=tune.grid_search([100,200,300]), 
            learning_rate=.005, 
            batch_size=64))
    
    elif model_name=='QRes':
        config['model'].update(dict(hidden_layers=4, 
            hidden_dims=20, 
            large_skip_every=2,
            in_out_skip=True,
            n_quantiles=5))
        config['train'].update(dict(epochs=150, 
            learning_rate=.005, 
            batch_size=64))

    elif model_name=='QResPinb':
        config['model'].update(dict(hidden_layers=tune.grid_search([2,3,4]), 
            hidden_dims=tune.grid_search([20,40]), 
            large_skip_every=2,
            in_out_skip=True,
            n_quantiles=tune.grid_search([5,7,11])))
        config['train'].update(dict(epochs=tune.grid_search([100,200,300]), 
            learning_rate=.005, 
            batch_size=64))

    elif model_name=='IRNN':
        config['model'].update(dict(hidden_dim=tune.grid_search([20,40]), 
            fc_hidden_layers=tune.grid_search([1,2,3]), 
            fc_hidden_dims=tune.grid_search([16,32]), 
            n_components=tune.grid_search([1,2,3]), 
            random_start=False))
        config['train'].update(dict(epochs=tune.grid_search([100,200,300]), 
            learning_rate=tune.grid_search([.005]),
            batch_size=tune.grid_search([64])))
    
    elif model_name=='EncDec':
        config['model'].update(dict(hidden_dim=tune.grid_search([20,40]), 
            fc_hidden_layers=tune.grid_search([1,2,3]), 
            fc_hidden_dims=tune.grid_search([16,32]), 
            n_components=tune.grid_search([1,2,3]), 
            random_start=False))
        config['train'].update(dict(epochs=tune.grid_search([100,200,300]), 
            learning_rate=.005, 
            batch_size=64, 
            m2m=True))
    
    elif model_name=='QRNNPinb':
        config['model'].update(dict(hidden_dim=tune.grid_search([20,40]), 
            fc_hidden_layers=tune.grid_search([1,2,3]), 
            fc_hidden_dims=tune.grid_search([16,32]), 
            n_quantiles=tune.grid_search([5,7,11]), 
            random_start=False))
        config['train'].update(dict(epochs=tune.grid_search([100,200,300]), 
            learning_rate=.005, 
            batch_size=64))

    elif model_name=='QRNNDecPinb':
        config['model'].update(dict(hidden_dim=tune.grid_search([20,40]), 
            fc_hidden_layers=tune.grid_search([1,2,3]), 
            fc_hidden_dims=tune.grid_search([16,32]), 
            n_quantiles=tune.grid_search([5,7,11]), 
            random_start=False))
        config['train'].update(dict(epochs=tune.grid_search([100,200,300]), 
            learning_rate=.005, 
            batch_size=64, 
            m2m=True))

    elif model_name=='CG':
        pass
    
    elif model_name=='JFNN':
        config['model'].update(dict(hidden_layers=tune.grid_search([1,2,3]), 
            hidden_dims=tune.grid_search([20,40]), 
            n_components=tune.grid_search([1,2,3]), 
            covariance_type='low-rank', 
            rank=2))
        config['train'].update(dict(epochs=tune.grid_search([100,200,300]), 
            learning_rate=.002, 
            batch_size=64))
    
    elif model_name=='JRNN':
        config['model'].update(dict(hidden_dim=tune.grid_search([20,40]), 
            fc_hidden_layers=tune.grid_search([1,2,3]), 
            fc_hidden_dims=tune.grid_search([20,40]),
            n_components=tune.grid_search([1,2,3]), 
            covariance_type='low-rank', 
            rank=2, 
            random_start=False))
        config['train'].update(dict(epochs=tune.grid_search([100,200,300]), 
            learning_rate=.005, 
            batch_size=64))
    
    elif model_name=='MOGP':
        config['model'].update(dict(index_rank=8))
        config['train'].update(dict(epochs=tune.grid_search([30,60,80])))
    
    elif model_name=='CGMM':
        config['model'].update(dict(n_components=tune.grid_search([3,4,5,6,7,8,9])))
    
    elif model_name=='CANF':
        config['model'].update(dict(hidden_dim=tune.grid_search([16,32]), 
            n_flows=tune.grid_search([6,10,16]),     
            n_components=tune.grid_search([15,20,25,30])))
        config['train'].update(dict(n_samples=tune.grid_search([50000,100000]), 
            epochs=tune.grid_search([2000,4000,8000]), 
            val_every=100, 
            lr=0.005))
    
    else:
        raise NotImplementedError
    
    return config

def get_config(model_name:str, loc, past_dims, fut_dims):
    config = {'seed':0, 'model':{}, 'train':{}}    
    if model_name=='ARMA':
        pass
    
    elif model_name=='IFNN':
        config['model'].update(dict(hidden_layers=3, hidden_dims=40, n_components=3))
        config['train'].update(dict(epochs=150, learning_rate=.005, batch_size=64))
    
    elif model_name=='ResNN':
        config['model'].update(dict(hidden_layers=4, hidden_dims=20))
        config['train'].update(dict(epochs=150, learning_rate=.005, batch_size=64))

    elif model_name=='QRes':
        config['model'].update(dict(hidden_layers=4, hidden_dims=20, n_quantiles=5))
        config['train'].update(dict(epochs=150, learning_rate=.005, batch_size=64))

    elif model_name=='QResPinb':
        config['model'].update(dict(hidden_layers=4, hidden_dims=20, n_quantiles=5))
        config['train'].update(dict(epochs=150, learning_rate=.005, batch_size=64))

    elif model_name=='IRNN':
        config['model'].update(dict(hidden_dim=40, fc_hidden_layers=3, fc_hidden_dims=20, n_components=1, random_start=False))
        config['train'].update(dict(epochs=200, learning_rate=.005, batch_size=64))
    
    elif model_name=='EncDec':
        config['model'].update(dict(hidden_dim=40, fc_hidden_layers=3, fc_hidden_dims=20, n_components=1, random_start=False)) #ncomps =1 
        config['train'].update(dict(epochs=100, learning_rate=.005, batch_size=64, m2m=True))

    elif model_name=='QRNNDecPinb':
        config['model'].update(dict(hidden_dim=40, fc_hidden_layers=3, fc_hidden_dims=20, n_quantiles=5, random_start=False)) #ncomps =1 
        config['train'].update(dict(epochs=100, learning_rate=.005, batch_size=64, m2m=True))

    elif model_name=='QRNNPinb':
        config['model'].update(dict(hidden_dim=40, fc_hidden_layers=3, fc_hidden_dims=20, n_quantiles=5, random_start=False))  
        config['train'].update(dict(epochs=100, learning_rate=.005, batch_size=64))

    elif model_name=='CG':
        pass
    
    elif model_name=='JFNN':
        config['model'].update(dict(hidden_layers=3, hidden_dims=40, n_components=2, covariance_type='low-rank', rank=2))
        config['train'].update(dict(epochs=300, learning_rate=.002, batch_size=64))
    
    elif model_name=='JRNN':
        config['model'].update(dict(hidden_dim=40, fc_hidden_layers=3, fc_hidden_dims=40,
            n_components=2, covariance_type='low-rank', rank=2, random_start=False))
        config['train'].update(dict(epochs=200, learning_rate=.005, batch_size=64))
    
    elif model_name=='MOGP':
        config['model'].update(dict(index_rank=8))
        config['train'].update(dict(epochs=55))
    
    elif model_name=='CGMM':
        config['model'].update(dict(n_components=4))
    
    elif model_name=='CANF':
        config['model'].update(dict(hidden_dim=32, n_flows=10,     
            n_components=25))
        config['train'].update(dict(n_samples=100000, epochs=4000, val_every=100, lr=0.005))
    
    else:
        raise NotImplementedError
    
    return config