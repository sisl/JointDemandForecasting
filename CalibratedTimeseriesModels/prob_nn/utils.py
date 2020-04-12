import numpy as np
import torch
import torch.nn as nn
import sklearn
from sklearn.neighbors import KernelDensity

def kde(x,xplot):
    xplot = xplot[:,np.newaxis]
    kde = KernelDensity().fit(x)
    log_dens = kde.score_samples(xplot)
    return np.exp(log_dens)

def create_rnn_XY(X,Y,tw=12,scale=False):
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

def train_test_split(X,Y,proportion_test=0.3):
    n, f  = X.shape
    
    # form test indices
    # test sequences = n * prop
    idxs = np.random.permutation(np.arange(n))
    test_idxs = idxs[:int(n*proportion_test)]
    train_idxs = idxs[int(n*proportion_test):]
    
    # X_train, Y_train, X_test, Y_test
    return X[train_idxs,:], Y[train_idxs,:], X[test_idxs,:], Y[test_idxs,:]

def batch(X,Y,batch_size=32):
    X_batches = []
    Y_batches = []
    num_batches = int(X.shape[0]/batch_size)
    for i in range(num_batches):
        X_batches.append(X[i*batch_size:(i+1)*batch_size])
        Y_batches.append(Y[i*batch_size:(i+1)*batch_size])
    return (X_batches, Y_batches)

def train(model, X_batches, Y_batches, num_epochs=20, learning_rate=0.01, loss_on_last_half=False, 
          verbose=True, weighting=None,prob=False):
                                                        
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
    if offset is not None:
        return torch.mean(torch.abs((target - output) / (target+offset))), torch.std(torch.abs((target - output) / (target+offset)))
    return torch.mean(torch.abs((target - output) / target)), torch.std(torch.abs((target - output) / target))   

def MSELoss(output, target, weighting=None):
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
    outdims = output.shape[-1]
    axes = len(output.shape)
    output = output.transpose(0,axes-1)
    mu, sig = output[:outdims//2].transpose(0,axes-1), output[outdims//2:].transpose(0,axes-1)
    gaussian = torch.distributions.normal.Normal(mu,sig)
    log_probs = gaussian.log_prob(target)
    loss = -torch.sum(log_probs)
    return loss

def RMSE(output,target):
    return MSELoss(output,target)**0.5

def RWSE(output,target):
    outdims = output.shape[-1]
    mu, sig = output[:outdims//2], output[outdims//2:]
    gaussian = torch.distributions.normal.Normal(mu,sig)
    return 0

def test(model, X_test, Y_test, last_only=True, residuals=False, nn=False):
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