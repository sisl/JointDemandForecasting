import os, zipfile

dataset = './raw/HouseholdElectricity/data_sets.zip'
pathdir = './raw/HouseholdElectricity'

if not os.path.exists(dataset):
    raise NameError('Must download dataset to %s' %(dataset))

if not os.path.exists(dataset[:-4]):
    with zipfile.ZipFile(dataset, 'r') as zip_ref:
        zip_ref.extractall(pathdir)