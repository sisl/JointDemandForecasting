import os
import requests, zipfile, io

pathdir = './raw/'
os.makedirs(pathdir, exist_ok=True)

fileID = '16cNbON7nQh2CIkr8d5BJMjvfKdFrOSBD'
filename = 'INTERACTION-Dataset-DR-v1_0.zip'

URL = "https://docs.google.com/uc?export=download"
session = requests.Session()
r = session.get(URL, params = {'id': fileID}, stream = True)

token = None
for key, value in r.cookies.items():
    if key.startswith('download_warning'):
        token = value
if token:
    r = session.get(URL, params = {'id': fileID, 'confirm': token}, stream = True)    

z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(pathdir)