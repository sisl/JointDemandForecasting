import os
import sys
import numpy as np
import tqdm
import pandas as pd
import torch

from process_utils.read_csv import *

### Define locations to process
pathdir = './raw/EPLUS_TMY2_RESIDENTIAL_BASE/'
chosenfiles = 10
startidx = 5 # capture Big Delta, AK

# Load in features from datasets
files = sorted([f for f in os.listdir(pathdir) if os.path.isfile(os.path.join(pathdir, f))])
nfiles = len(files)
load_features = ['DT', 'ELEC']
options = {}
Xs, Ys, filename = [], [], []
loadnum = 0
with tqdm.tqdm(total=chosenfiles) as pbar:
    for i,file in enumerate(files):
        if (i-startidx) % (nfiles//chosenfiles) is not 0:
            continue
        path = os.path.join(pathdir, file)
        print(path)
        df = read_load_csv(path,load_features)
        data, labels = df_to_numpy(df)
        x, y = encode_features(data, options)
        Xs.append(x)
        Ys.append(y)
        filename.append(file)
        loadnum += 1
        pbar.update(1)
X = np.vstack(Xs)
Y = np.vstack(Ys)

## Save as torch tensors

X_torch = torch.tensor(X)
Y_torch = torch.tensor(Y)

output_dir = 'processed/openEI'
postfix = '_openei_%03d_subset_multitask.pt' %(loadnum)
if not os.path.exists(output_dir):
        os.mkdir(output_dir)   
        
torch.save(X_torch, os.path.join(output_dir, 'X'+postfix))
torch.save(Y_torch, os.path.join(output_dir, 'Y'+postfix))