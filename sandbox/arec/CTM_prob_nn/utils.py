import numpy as np
import torch
import torch.nn as nn
import sklearn
from sklearn.neighbors import KernelDensity

def kde(x,xplot):
    """ 

    Use kernel density estimation to estimate a density. 

    Args: 
        x (np.array): (data_points,) array of data points
        xplot (np.array): (plot_points,) array of points to plot the density over

    Returns: 

        (np.array): (plot_points) array of estimated density values. 

    """ 
    xplot = xplot[:,np.newaxis]
    kde = KernelDensity().fit(x)
    log_dens = kde.score_samples(xplot)
    return np.exp(log_dens)

def create_rnn_XY(X, Y, tw=12,scale=False):
    """ 
    Create rnn sequences from raw inputs using the sliding window method. 

    Args: 
        X (torch tensor): (data_points, input_features) tensor of data points
        Y (torch tensor): (data_points, output_features) tensor of labels
        tw (int): length of each sequence (time window)
        scale (bool): whether or not to scale the data

    Returns: 
        (torch tensor): (num_sequences, sequence_length, input_features) tensor of data input streams.
        (torch tensor): (num_sequences, sequence_length, output_features) tensor of data labels.
        sf (float): scale factor

    """
    x_out, y_out = [], []
    L = X.shape[0]
    for i in range(L-tw):
        train_seq = X[i:i+tw,:].float()
        train_label = Y[i:i+tw,:].float()
        x_out.append(train_seq)
        y_out.append(train_label)
    if scale:
        sf = X[:,0].max()
        return torch.stack(x_out)/sf, torch.stack(y_out)/sf, sf
    else:
        return torch.stack(x_out), torch.stack(y_out)

def create_nn_XY(X,Y,tw=12,scale=False):
    """ 
    Create neural net datasets from raw inputs using the sliding window method. 

    Args: 
        X (torch tensor): (data_points, input_features) tensor of data points
        Y (torch tensor): (data_points, output_features) tensor of labels
        tw (int): length of each sequence (time window)
        scale (bool): whether or not to scale the data

    Returns: 
        (torch tensor): (num_sequences, sequence_length) tensor of data input streams.
        (torch tensor): (num_sequences, output_features) tensor of data labels.
        sf (float): scale factor

    """
    x_out, y_out = [], []
    L = X.shape[0]
    for i in range(L-tw):
        train_seq = X[i:i+tw,0].float()
        train_label = Y[i+tw-1,:].float()
        x_out.append(train_seq)
        y_out.append(train_label)
    if scale:
        sf = X[:,0].max()
        return torch.stack(x_out)/sf, torch.stack(y_out)/sf, sf
    else:
        return torch.stack(x_out), torch.stack(y_out)

def train_test_split_rnn(X,Y,proportion_test=0.3):
    """ 
    Split the rnn data between training and test data, making sure there are no overlaps in the sets. 

    Args: 
        X (torch tensor): (num_sequences, sequence_length, input_features) tensor of data input streams.
        Y (torch tensor): (num_sequences, sequence_length, output_features) tensor of data labels.
        proportion_test (float): proportion to put in the test set

    Returns: 
        X_train (torch tensor): (num_train, sequence_length, input_features) tensor of training data input streams.
        Y_train (torch tensor): (num_train, sequence_length, output_features) tensor of training data labels.
        X_test (torch tensor): (num_test, sequence_length, input_features) tensor of test data input streams.
        Y_test (torch tensor): (num_test, sequence_length, output_features) tensor of test data labels.
    """
    n, tw, input_size  = X.shape
    
    # form test indices
    # test sequences = n * prop
    test_idxs = np.random.choice(n, int(n * proportion_test), replace = False)
    
    train_idxs = []
    for i in range(n):
        add = True
        for test in test_idxs: 
            if i > test - tw and i < test + tw:
                add = False
        if add:
            train_idxs.append(i)
    
    # shuffle training indices
    train_idxs = np.random.permutation(np.array(train_idxs))
    
    # X_train, Y_train, X_test, Y_test
    return X[train_idxs,:,:], Y[train_idxs,:,:], X[test_idxs,:,:], Y[test_idxs,:,:]
    
def train_test_split_nn(X,Y,proportion_test=0.3):
    """ 
    Split the nn data between training and test data, making sure there are no overlaps in the sets. 

    Args: 
        X (torch tensor): (num_sequences, sequence_length) tensor of data input streams.
        Y (torch tensor): (num_sequences, output_features) tensor of data labels.
        proportion_test (float): proportion to put in the test set

    Returns: 
        X_train (torch tensor): (num_train, sequence_length) tensor of training data input streams.
        Y_train (torch tensor): (num_train, output_features) tensor of training data labels.
        X_test (torch tensor): (num_test, sequence_length) tensor of test data input streams.
        Y_test (torch tensor): (num_test, output_features) tensor of test data labels.
    """
    n, tw  = X.shape
    
    # form test indices
    # test sequences = n * prop
    test_idxs = np.random.choice(n, int(n * proportion_test), replace = False)
    
    train_idxs = []
    for i in range(n):
        add = True
        for test in test_idxs: 
            if i > test - tw and i < test + tw:
                add = False
        if add:
            train_idxs.append(i)
    
    # shuffle training indices
    train_idxs = np.random.permutation(np.array(train_idxs))
    
    # X_train, Y_train, X_test, Y_test
    return X[train_idxs,:], Y[train_idxs,:], X[test_idxs,:], Y[test_idxs,:]

"""
def train_test_split(X,Y,proportion_test=0.3):
    n, f  = X.shape
    
    # form test indices
    # test sequences = n * prop
    idxs = np.random.permutation(np.arange(n))
    test_idxs = idxs[:int(n*proportion_test)]
    train_idxs = idxs[int(n*proportion_test):]
    
    # X_train, Y_train, X_test, Y_test
    return X[train_idxs,:], Y[train_idxs,:], X[test_idxs,:], Y[test_idxs,:]
"""

def batch(X,Y,batch_size=32):
    """ 
    Batches the data along the first dimension. 

    Args: 
        X (torch tensor): (num_sequences, *) tensor of data input streams.
        Y (torch tensor): (num_sequences, *) tensor of data labels.
        batch_size (int): batch size.

    Returns: 
        X_batches (list of torch tensor): list of (batch_size, *) tensor input data.
        Y_batches (list of torch tensor): list of (batch_size, *) tensor data labels.
    """
    X_batches = []
    Y_batches = []
    num_batches = int(X.shape[0]/batch_size)
    for i in range(num_batches):
        X_batches.append(X[i*batch_size:(i+1)*batch_size])
        Y_batches.append(Y[i*batch_size:(i+1)*batch_size])
    return (X_batches, Y_batches)

def train(model, X_batches, Y_batches, num_epochs=20, learning_rate=0.01, loss_on_last_half=False, 
          verbose=True, weighting=None,prob=False):
    """ 
    Train a model. 

    Args: 
        model: pytorch model to train.
        X_batches (list of torch tensor): list of (batch_size, *) tensor input data.
        Y_batches (list of torch tensor): list of (batch_size, *) tensor data labels.
        num_epochs (int): number of times to iterate through all batches
        learning_rate (float): learning rate for Adam optimizer
        loss_on_last_half (bool): if true, only apply loss to last half of each sequence
        verbose (bool): if true, print epoch losses
        weighting (list of floats or None): if included, relative weights on different output indices
        prob (bool): if true, using a probabilistic model that should be trained with NLL
    """                                                        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, labels in zip(X_batches, Y_batches):
            optimizer.zero_grad()
            
            outputs = model(x)
            if loss_on_last_half:
                tw = x.shape[1]
                outputs, labels = outputs[:,(tw//2):,:], labels[:,(tw//2):,:]
            
            if prob:
                loss = NLL(outputs,labels)
            else:
                loss = MSELoss(outputs, labels, weighting)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()/len(X_batches)
        if verbose:
            print ("epoch : %d, loss: %1.4f" %(epoch+1, epoch_loss))
    if verbose:
        print ("Learning finished!")
    
def MAPELoss(output, target, offset=None):
    """ 
    Compute MAPE Loss. 

    Args: 
        output (torch tensor): (*) tensor of model outputs.
        target (torch tensor): (*) tensor of true data labels.
        offset (torch tensor or None): if not None, applies offset to output and targets.

    Returns: 
        (float): MAPE loss
    """
    if offset is not None:
        return torch.mean(torch.abs((target - output) / (target+offset))), torch.std(torch.abs((target - output) / (target+offset)))
    return torch.mean(torch.abs((target - output) / target)), torch.std(torch.abs((target - output) / target))   

def MSELoss(output, target, weighting=None):
    """ 
    Compute MSE Loss. 

    Args: 
        output (torch tensor): (*, output_size) tensor of model outputs.
        target (torch tensor): (*, output_size) tensor of true data labels.
        weighting (list of floats or None): if not None, applies relative weighting over output dimension.

    Returns: 
        (float): MSE loss
    """
    mse = torch.nn.MSELoss()
    
    # MSE
    if weighting is None:
        return mse(output, target)
    
    # Custom Weighted MSE
    o = outputs.shape[-1]
    outputs = torch.transpose(outputs,0,len(outputs.shape)-1)
    labels = torch.transpose(labels,0,len(outputs.shape)-1)

    if len(weighting) is not o:
        raise("Wrong weighting size")
    loss = 0
    for i in range(o):
        loss += weighting[i]/sum(weighting)*mse(outputs[i],labels[i])
    return loss

def NLL(output, target):
    """ 
    Compute negative log likelihood of targets given Gaussians paramaterized by output. 

    Args: 
        output (torch tensor): (*, 2*output_size) tensor of model output mean and standard deviations.
        target (torch tensor): (*, output_size) tensor of true data labels.

    Returns: 
        (float): negative log likelihood
    """
    outdims = output.shape[-1]
    axes = len(output.shape)
    output = output.transpose(0,axes-1)
    mu, sig = output[:outdims//2].transpose(0,axes-1), output[outdims//2:].transpose(0,axes-1)
    gaussian = torch.distributions.normal.Normal(mu,sig)
    log_probs = gaussian.log_prob(target)
    loss = -torch.sum(log_probs)
    return loss

def RMSE(output,target):
    """ 
    Compute root mean squared error. 

    Args: 
        output (torch tensor): (*, output_size) tensor of model outputs.
        target (torch tensor): (*, output_size) tensor of true data labels.

    Returns: 
        (float): root mean squared error.
    """
    return MSELoss(output,target)**0.5

"""
def RWSE(output,target):
    outdims = output.shape[-1]
    mu, sig = output[:outdims//2], output[outdims//2:]
    gaussian = torch.distributions.normal.Normal(mu,sig)
    return 0
"""

def test(model, X_test, Y_test, last_only=True, residuals=False, nn=False):
    """ 
    Test model on test set. 

    Args: 
        model: pytorch model being tested.
        X_test (torch tensor): (test_sequences, sequence_length, *) tensor input data.
        Y_test (torch tensor): (test_sequences, *, output_size) tensor data labels.
        last_only (bool): if true, only compute loss on last element in sequences.
        residuals (bool): if true, test labels are residuals from previous points.
        nn (bool): if true, testing a feedforward neural network, otherwise an rnn.
    """
    with torch.no_grad():
        forward = model(X_test) 
    
    # only test on final y in sequence
    if last_only:
        outputs = forward[:,-1,:]
        targets = Y_test[:,-1,:]
    
    # test after index 2 in sequence
    else:
        outputs = forward
        targets = Y_test
    
    mse = []
    # MSE per column
    for icol in range(targets.shape[1]):
        if last_only or nn:
            new_mse = MSELoss(outputs[:,icol],targets[:,icol]).item()
        else:
            new_mse = MSELoss(outputs[:,:,icol],targets[:,:,icol]).item()
        print("column : %d, mse loss: %1.3f" %(icol+1, new_mse))
        mse.append(new_mse)
        
    # MSE overall
    print("overall mse loss: %1.3f" %(np.array(mse).mean()))
    
    mape = []
    # MAPE per column
    offset = None
    if residuals:
        if last_only:
            offset = X_test[:,-1,0]
        else:
            offset = X_test[:,:,0]
            
    for icol in range(targets.shape[1]):
        if last_only or nn:
            new_mape, new_mape_std = MAPELoss(outputs[:,icol],targets[:,icol], offset)
        else:
            new_mape, new_mape_std = MAPELoss(outputs[:,:,icol],targets[:,:,icol], offset)
        print("column : %d, mape loss: %1.3f pm %1.4f" %(icol+1, new_mape, new_mape_std))
        mape.append(new_mape)
        
    # MAPE overall
    print("overall mape loss: %1.3f" %(np.array(mape).mean()))
    
def test_naive(X_test, Y_test, last_only=True):
    """ 
    Test a naive model on test set (predict next point = last point). 

    Args: 
        X_test (torch tensor): (test_sequences, sequence_length, *) tensor input data.
        Y_test (torch tensor): (test_sequences, *, output_size) tensor data labels.
        last_only (bool): if true, only compute loss on last element in sequences.
    """
    forward = X_test[:,:,[0]]
    targets = Y_test[:,:,[0]]
    
    # only test on final y in sequence
    if last_only:
        outputs = forward[:,-1,:]
        targets = targets[:,-1,:]
    
    mse = []
    # MSE per column
    for icol in range(targets.shape[1]):
        if last_only:
            new_mse = MSELoss(outputs[:,icol],targets[:,icol]).item()
        else:
            new_mse = MSELoss(outputs[:,:,icol],targets[:,:,icol]).item()
        print("column : %d, naive mse loss: %1.3f" %(icol+1, new_mse))
        mse.append(new_mse)
        
    # MSE overall
    print("overall naive mse loss: %1.3f" %(np.array(mse).mean()))
    
    mape = []
    # MAPE per column
    offset = None
    for icol in range(targets.shape[1]):
        if last_only:
            new_mape = MAPELoss(outputs[:,icol],targets[:,icol], offset).item()
        else:
            new_mape = MAPELoss(outputs[:,:,icol],targets[:,:,icol], offset).item()
        print("column : %d, naive mape loss: %1.3f" %(icol+1, new_mape))
        mape.append(new_mape)
        
    # MAPE overall
    print("overall naive mape loss: %1.3f" %(np.array(mape).mean()))