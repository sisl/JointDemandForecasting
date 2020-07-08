# CalibratedTimeseriesModels.py
A framework for calibrating probabilistic time-series models

## File stucture
- `CalibratedTimeseriesModels` contains the source code for the package
- `datasets` contains datasets pertaining to experiments.
	- `datasets/raw` contains raw datasets
	- `datasets/processed` contains processed datasets
	- `datasets/process_utils` contains utilities for processing utilities
	- `datasets/README.md` contains instructions for sourcing at processing all datasets.
- `experiments` contains any experiments conducted in the submitted manuscript, including instructions for exactly regenerating all figures.
- `sandbox` contains scratch folders for each developer of this package.

## Environment
A conda environment for this repo can be set up and uses with:
```
conda create -y --name ctm python==3.7
conda install -y --name ctm -c conda-forge pytorch --file requirements.txt
conda activate ctm
...
conda deactivate
```
