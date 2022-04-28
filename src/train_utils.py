from src.utils import SequenceDataset
from typing import Dict
import torch
import gpytorch
import logging
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
    epochs=20, 
    optimizer = torch.optim.Adam,
    learning_rate=0.01,
    log_every=10, 
    ):
    """ 
    Train a regular model. 

    Args: 
        model: pytorch model to train.
        X_batches (list of torch tensor): list of (batch_size, *) tensor input data.
        Y_batches (list of torch tensor): list of (batch_size, *) tensor data labels.
        num_epochs (int): number of times to iterate through all batches
        optimizer (object): torch optimizer
        learning_rate (float): learning rate for Adam optimizer
    """                                                        
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in zip(dataset['X_batches'], dataset['Y_batches']):
            optimizer.zero_grad()
            
            dist = model(x)
            loss, _ = nll(dist, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()/len(dataset['X_batches'])

        if (epoch+1) % log_every == 0:
            logger.info("epoch : %d, loss: %1.4f" %(epoch+1, epoch_loss))
            
        
def train_mogp(model, dataset, 
    epochs=40,
    optimizer = torch.optim.Adam, 
    lr=0.05, 
    log_every=5):
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
    model.train()
    model.likelihood.train()
    optimizer = optimizer([  
        {'params': model.parameters()}], lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(dataset['train_x'])
        loss = -mll(output, dataset['train_y'])
        loss.backward()
        if (i+1) % log_every==0:
            logger.info('Iter %d/%d - Mean Loss: %.3f' % (i + 1, 
                epochs, loss.item()/dataset['train_x'].size(0)))
        optimizer.step()

def train_nf(model, dataset:Dict[str,SequenceDataset], 
    epochs=100, 
    seed=0, 
    val_every=100, 
    lr=0.005, 
    optimizer=torch.optim.Adam):
    """ 
    Train a normalizing flow model model. 

    Args: 
        model: normalizing flow model to train
        dataset: dataset with input output pairs to fit joint nf over
        epochs (int): number of epochs for training
        seed (int): seed for shuffling data
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
    
    torch.manual_seed(seed)
    x = combined[torch.randperm(B_train)]
    x_val = combined_val[torch.randperm(B_val)]

    optimizer = optimizer(model.parameters(), lr=lr)
    for i in range(epochs):
        optimizer.zero_grad()
        z, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.mean(prior_logprob + log_det)
        loss.backward()
        optimizer.step()
        if (i+1) % val_every == 0:
            _, prior_logprob_val, log_det_val = model(x_val)
            logprob_val = prior_logprob_val + log_det_val
            logger.info(f"Iter: {i+1}\t" +
                        f"Logprob: {logprob.mean().data:.2f}\t" +
                        f"Prior: {prior_logprob.mean().data:.2f}\t" +
                        f"LogDet: {log_det.mean().data:.2f}\t" + 
                        f"Logprob_val: {logprob_val.mean().data:.2f}")
    model.eval()
