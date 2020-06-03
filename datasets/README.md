# Datasets
This README provides instructions for acquiring all datasets used in experiments.

## COVID-19 data
Assuming the `datasets` folder is the current working directory, the `COVID-19` dataset can be obtained by running the following command:
```
git clone https://github.com/CSSEGISandData/COVID-19.git raw/COVID-19
```

## OpenEI data
Assuming the `datasets` folder is the current working directory, the `OpenEI` dataset can be obtained by running the following command:
```
python3 download_openEI.py
```

It can then be processed by editing and running:
```
python3 process_openEI.py
```

## Interaction data
Assuming the `datasets` folder is the current working directory, the `Interaction` dataset can be obtained by running the following command:
```
python3 download_interaction.py
```

It can then be processed by running:
```
python3 process_interaction.py
```

## Gaussian Random Walk data
Test dataset generated from a correlated 2D Gaussian random walk.
Generate using:
```
python3 datasets/generate_randomwalk_data.py
```
Stored in a torch datafile:
```
datasets/gauss_rw.data
```
Load as follows:
```
data = torch.load('datasets/gauss_rw.data')
t = data['t'] # (1000, 100) torch.tensor
x = data['y'] # (1000, 100, 2) torch.tensor
```

