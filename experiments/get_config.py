
def get_config(model_name:str, loc, past_dims, fut_dims):
    config = {'seed':0, 'model':{}, 'train':{}}    
    if model_name=='ARMA':
        pass
    
    elif model_name=='IFNN':
        config['model'].update(dict(hidden_layers=3, hidden_dims=40, n_components=3))
        config['train'].update(dict(epochs=150, learning_rate=.005, batch_size=64))
    
    elif model_name=='IRNN':
        config['model'].update(dict(hidden_dim=40, fc_hidden_layers=3, fc_hidden_dims=20, n_components=3, random_start=False))
        config['train'].update(dict(epochs=200, learning_rate=.005, batch_size=64))
    
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