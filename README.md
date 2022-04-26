# JointDemandForecasting.py
A framework for multi-step probabilistic time-series/demand forecasting models. Code for "[Conditional Approximate Normalizing Flows for Joint Multi-Step Probabilistic Forecasting with Application to Electricity Demand](https://arxiv.org/pdf/2201.02753.pdf)". If you find this repository useful, please cite the following:

```
@article{jamgochian2022conditional,
	author = {Arec Jamgochian and Di Wu and Kunal Menda and Soyeon Jung and Mykel J. Kochenderfer},
	title = {Conditional Approximate Normalizing Flows for Joint Multi-Step Probabilistic Forecasting with Application to Electricity Demand},
	journal = {arXiv:2201.02753 [cs]},
	year = {2022}
}
```


## File stucture
- `JointDemandForecasting` contains the source code for the package
- `datasets` contains datasets pertaining to experiments.
	- `datasets/raw` contains raw datasets
	- `datasets/processed` contains processed datasets
	- `datasets/process_utils` contains utilities for processing utilities
	- `datasets/README.md` contains instructions for sourcing at processing all datasets.
- `experiments` contains any experiments conducted in the submitted manuscript.

## Environment
A conda environment for exact experiment reproduction can be set up and uses with:
```
conda create -y --name jdf python==3.7
conda install -y --name jdf --file requirements.txt
conda activate jdf
...
conda deactivate
```

To enable CUDA capabilities, rerun:
```
conda install --name jdf pytorch cudatoolkit=10.2 -c pytorch
```

## Running Experiments

To run experiments from the associated paper:

1. Install and activate the environment.
2. Follow directions in the `datasets` folder to download and process the OPENEI dataset.
3. Navigate the the `experiments/arxiv/` folder and run the appropriate scripts:
	- Toy Density Estimation:
		- the best hyperparameter config from our experiments is saved in `experiments/arxiv/results/simple_square_best_hyperparams.json`
		- to run the experiment from the paper with this config, run `python simple_experiment.py --test`
		- to retune hyperparameters based on a grid search with a grid defined in-file, run `python simple_experiment.py --tune`
	- Electricity Demand Application
		- with, for example, `python ConditionalModels.py`
			- ARMA, IFNN, IRNN models are in `IterativeModels.py`, CG and CGMM are in `ConditionalModels.py`, MOGP is in `MultiOutputGP.py`, JFNN is in `JFNN.py`, JRNN is in `JRNN.py`, and CANF is in`CANF.py`
			- pick appropriate setting (location, input length, output length, etc.) by uncommenting appropriate line
		- our multi-trial results, as well as run-rejection for outlier trials can be seen in `experiments/arxiv/results` (no CANF trials were rejected)
		- plots can be generated with `python plots.py`
