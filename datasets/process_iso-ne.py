import os
import numpy as np
import tqdm
import pandas as pd
import torch

### Define locations to process
pathdir = './raw/iso-ne.csv'
df = pd.read_csv(pathdir, skiprows=5)

# note when we look through the data there are 4 dates that have two values of the same hour (each in november)
# as well as 4 dates that have an hour missing (each in april). 
# we ignore all of this in the time series as this is due to daylight savings time
# There is an additional date which has 3 dates missing: 5/24/2016 is missing values at 17, 18, 19h. 
# For these values, we copy the values from a day before at the same times
ts = df['MWh'].to_numpy()
front = ts[:3471]
back = ts[3471:-1] 

X = torch.tensor(np.concatenate((front, front[-23:-20], back))[np.newaxis])

## Save different length files

output_dir = 'processed/iso-ne'
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

for x in [X, X[...,-24*365*3:], X[...,-24*365*2:], X[...,-24*365:]]: #(4, 3, 2, and 1 year)
    fname = 'iso-ne_%d.pt' %(x.shape[-1])       
    torch.save(x, os.path.join(output_dir, fname))
