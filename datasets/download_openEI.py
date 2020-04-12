import os
import requests, zipfile, io

dataset = 'https://openei.org/datasets/files/961/pub/EPLUS_TMY2_RESIDENTIAL_BASE.zip'
pathdir = './raw/'

if not os.path.exists('./raw/EPLUS_TMY2_RESIDENTIAL_BASE/'):
    r = requests.get(dataset)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(pathdir)