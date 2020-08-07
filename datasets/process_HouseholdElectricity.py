import os
import numpy as np
import torch

# Define files to process
pathdir = './raw/HouseholdElectricity/data_sets/'
files = ['h_house%i_total.txt'%(i) for i in range(1,11)]+['temp.txt','week.txt']

# Load data
data = [np.loadtxt(os.path.join(pathdir, file)) for file in files]

# Tensor data
X = torch.tensor(data)

# Save data
output_dir = './processed/HouseholdElectricity'
if not os.path.exists(output_dir):
        os.mkdir(output_dir)   
torch.save(X, os.path.join(output_dir, 'X.pt'))