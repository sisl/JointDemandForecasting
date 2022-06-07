import os
import numpy as np
import tqdm
import pandas as pd
import torch

### Define locations to process
pathdir = './raw/nau/output.txt'
df = pd.read_csv(pathdir,header=None, sep='\s+')
ts_raw = df.iloc[:-1,1:].to_numpy().flatten()
# this dataset has a bunch of zeros. replace zeros with average between previous and next timesteps
ts = ts_raw.copy()
for i in range(len(ts_raw)):
    if ts_raw[i] < 100:
        ts[i] = (ts[i-1]+ts[i+1])/2
        print(f'Changing index {i} from {ts_raw[i]} to {ts[i]}')
X = torch.tensor(ts[np.newaxis])

## Save

output_dir = 'processed/nau'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fname = 'nau.pt'   
torch.save(X, os.path.join(output_dir, fname))
