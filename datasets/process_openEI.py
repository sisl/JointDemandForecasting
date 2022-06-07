import os
import numpy as np
import tqdm
import pandas as pd
import torch

from process_utils.read_csv import *

### Define locations to process
pathdir = './raw/EPLUS_TMY2_RESIDENTIAL_BASE/'
chosenfiles = 10
startidx = 5 # capture Big Delta, AK
rw_length = 36 # rolling window length

# Load in features from datasets
files = sorted([f for f in os.listdir(pathdir) if os.path.isfile(os.path.join(pathdir, f))])
nfiles = len(files)
load_features = ['DT', 'ELEC']
Xs, filename = [], []
loadnum = 0
with tqdm.tqdm(total=chosenfiles) as pbar:
    for i,file in enumerate(files):
        if (i-startidx) % (nfiles//chosenfiles) is not 0:
            continue
        path = os.path.join(pathdir, file)
        print(path)
        df = read_load_csv(path, load_features)
        data, labels = df_to_numpy(df)
        Xs.append(data[:,0])
        filename.append(file)
        loadnum += 1
        pbar.update(1)
X = torch.tensor(np.stack(Xs))

## Save

output = 'processed'
if not os.path.exists(output):
	os.mkdir(output)

output_dir = 'processed/openEI'
fname = 'openei_%03d_subset.pt' %(loadnum)
if not os.path.exists(output_dir):
        os.mkdir(output_dir)   
        
torch.save(X, os.path.join(output_dir, fname))
