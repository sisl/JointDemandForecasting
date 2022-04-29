from src.utils import SequenceDataset
from typing import Dict
import torch
from torch.utils.data import DataLoader
import gpytorch
import logging
from ray import tune
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# loss functions
def nll(dist, target):
    """ 
    Compute negative log likelihood of target given distribution. 

    Args: 
        dist (torch.Distribution): (B,K*ydim) predictive distribution over next K observations
        target (torch tensor): (B, K, ydim) tensor of true data labels.

    Returns: 
        nll (float): negative log likelihood
        nlls (torch.tensor): (B) nll of each sample in batch
    """
    target_shape = target.shape
    target = target.reshape(*target_shape[:-2],-1)
    nlls = -dist.log_prob(target)
    if len(nlls.shape) > 1:
        nlls = torch.sum(nlls,1)
    nll = nlls.mean()
    return nll, nlls

# training functions
def train(model, dataset, 
    epochs:int=20, 
    optimizer = torch.optim.Adam,
    learning_rate:float=0.01,
    batch_size:int=64,
    val_every:int=10, 
    ray=False):
    """ 
    Train a regular model. 

    Args: 
        model: pytorch model to train.
        dataset
        num_epochs (int): number of times to iterate through all batches
        optimizer (object): torch optimizer
        learning_rate (float): learning rate for Adam optimizer
        val_every (int): logging interval
    """                                                        
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset['val'], batch_size=batch_size)
    
    # train the model
    for i in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            dist = model(batch['x'])
            loss, _ = nll(dist, batch['y'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*len(batch['x'])
        epoch_loss /= len(dataset['train'])

        # validate and log scores
        if (i+1) % val_every == 0:
            with torch.no_grad():
                val_loss = 0
                for batch in val_loader:
                    vloss,_ = nll(model(batch['x']), batch['y'])
                    val_loss += vloss.item()*len(batch['x'])
                val_loss /= len(dataset['val'])

            logger.info(f"Iter: {i+1}/{epochs}\t" +
                "Train Loss: %1.4f\t" %(epoch_loss) +
                "Val Loss: %1.4f\t" %(val_loss))
            if ray:
                tune.report(epoch=i+1,train_loss=epoch_loss, val_loss=val_loss)
            
def train_mogp(model, dataset, 
    epochs=40,
    optimizer = torch.optim.Adam, 
    lr=0.05, 
    val_every=5, 
    ray=False):
    """ 
    Train a Multi-Output Gaussian Process model. 

    Args: 
        model: gpytorch model to train.
        train_x (torch tensor): (B,in_length)-sized tensor of training inputs
        train_y (torch tensor): (B,out_length)-sized tensor of training inputs
        epochs (int): number of times to iterate through data
        optimizer (object): torch optimizer
        learning_rate (float): learning rate for Adam optimizer
        verbose (bool): if true, print epoch losses
    """
    model.train(), model.likelihood.train()
    optimizer = optimizer([  
        {'params': model.parameters()}], lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    
    # train the model
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(dataset['train']['x'])
        loss = -mll(output, dataset['train']['y'])
        loss.backward()
        optimizer.step()

        # validate and log scores
        if (i+1) % val_every==0:
            model.eval(), model.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                val_loss = -mll(model(dataset['val']['x']), 
                    dataset['val']['y'])
            model.train(), model.likelihood.train()
            epoch_loss = loss.item()/len(dataset['train']['x'])
            val_loss = val_loss.item()/len(dataset['val']['x'])
            logger.info(f"Iter: {i+1}/{epochs}\t" +
                "Train Loss: %.3f\t" % (epoch_loss) +
                "Val Loss: %.3f" % (val_loss))    
            if ray:
                tune.report(epoch=i+1,train_loss=epoch_loss, val_loss=val_loss)   

def train_nf(model, dataset:Dict[str,SequenceDataset], 
    epochs=100, 
    val_every=100, 
    lr=0.005, 
    optimizer=torch.optim.Adam, 
    ray=False):
    """ 
    Train a normalizing flow model model. 

    Args: 
        model: normalizing flow model to train
        dataset: dataset with input output pairs to fit joint nf over
        epochs (int): number of epochs for training
        val_every (int): interval for validation
        learning_rate (float): learning rate for Adam optimizer
        optimizer (torch.optim): torch optimizer
    """
    # make joint train/val data
    B_train, T, ind = dataset['train'][:]['x'].shape
    _, K, outd = dataset['train'][:]['y'].shape
    B_val = dataset['val'][:]['x'].shape[0] 
    combined = torch.cat((dataset['train'][:]['x'].reshape((B_train,-1)), dataset['train'][:]['y'].reshape((B_train,-1))), -1)
    combined_val = torch.cat((dataset['val'][:]['x'].reshape((B_val,-1)), dataset['val'][:]['y'].reshape((B_val,-1))), -1)
    
    x = combined[torch.randperm(B_train)]
    x_val = combined_val[torch.randperm(B_val)]

    optimizer = optimizer(model.parameters(), lr=lr)
    
    # train the model
    for i in range(epochs):
        optimizer.zero_grad()
        z, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.mean(prior_logprob + log_det)
        loss.backward()
        optimizer.step()

        # validate and log scores
        if (i+1) % val_every == 0:
            _, prior_logprob_val, log_det_val = model(x_val)
            logprob_val = prior_logprob_val + log_det_val
            epoch_loss = -logprob.mean().data
            val_loss = -logprob_val.mean().data
            logger.info(f"Iter: {i+1}/{epochs}\t" +
                        f"Train Loss (NLL): {epoch_loss:.2f}\t" +
                        f"Prior: {prior_logprob.mean().data:.2f}\t" +
                        f"LogDet: {log_det.mean().data:.2f}\t" + 
                        f"Val Loss: {val_loss:.2f}")
            if ray:
                tune.report(epoch=i+1,train_loss=epoch_loss, val_loss=val_loss,
                    prior=prior_logprob.mean().data, logdet=log_det.mean().data) 
    model.eval()