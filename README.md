# JointDemandForecasting.py
A framework for calibrating probabilistic time-series models

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

For a looser set of requirements, use:
```
conda install --name jdf --file loose_requirements.txt
```

## Running Experiments

To run experiments from the associated paper:

1. Install and activate the environment.
2. Follow directions in the `datasets` folder to download and process the OPENEI dataset.
3. Navigate the the `experiments/AAAI21/` folder and run the appropriate scripts:
	- with, for example, `python ConditionalModels.py`
	- pick correct settings for metric-generating scripts by uncommenting appropriate lines in each script
	- pick correct settings for timing script with `--model`  and `--n` settings
		- e.g. `python time.py --model condgmm --n 10000` 
