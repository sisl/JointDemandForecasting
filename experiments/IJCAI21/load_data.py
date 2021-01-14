import torch
import sys
sys.path.append("../../")
import JointDemandForecasting
from JointDemandForecasting.utils import *

def load_data(loc, past_dims, fut_dims, path_x=None, path_y=None):
    """
    Load x and y openEI from path, and perform sequence length preprocessing and train/test split.
    """
     
    if path_x is None:
        path_x = "../../datasets/processed/openEI/X_openei_011_subset_multitask.pt"
    if path_y is None:
        path_y = "../../datasets/processed/openEI/Y_openei_011_subset_multitask.pt"
        
    # Data setup:
    X_orig = torch.load(path_x)
    Y_orig = torch.load(path_y)

    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = electric_train_test_split(X_orig, Y_orig, disp_idx=24+12)

    X_train = X_train_orig[loc,:,:24].reshape((-1,24)).unsqueeze(-1).float()
    Y_train = Y_train_orig[loc,:,:12].reshape((-1,12)).unsqueeze(-1).float()
    X_test = X_test_orig[loc,:,:24].reshape((-1,24)).unsqueeze(-1).float()
    Y_test = Y_test_orig[loc,:,:12].reshape((-1,12)).unsqueeze(-1).float()

    # Combine processing output into single-strand sequences 
    train_joint = torch.cat((X_train, Y_train),1)
    test_joint = torch.cat((X_test, Y_test),1)

    # Re-split into appropriate lengths
    X_train = train_joint[:,:past_dims,:] #(B, i, 1)
    Y_train = train_joint[:,past_dims:past_dims+fut_dims,:] #(B, o, 1)
    X_test = test_joint[:,:past_dims,:] #(B, i, 1)
    Y_test = test_joint[:,past_dims:past_dims+fut_dims,:] #(B, o, 1)
    return X_train, Y_train, X_test, Y_test