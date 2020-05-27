from CalibratedTimeseriesModels.models.lineargrw import *
import torch

from CalibratedTimeseriesModels.dataset import TimeseriesDataset

from CalibratedTimeseriesModels.evaluators import ExplicitEvaluator

from torch.utils import data

import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import expm

def main():

	torch.set_default_dtype(torch.float64)


	model = LinearGRWModel(torch.tensor(expm([[0., 1.0],[-0.5, -0.05]])), None,
			0.1*torch.eye(2))

	y_inp = 100.*torch.randn(1,1,2)

	dist = model(y_inp, None, 100)

	B = 10
	y_out_pred = dist.sample((10,)).reshape(10,100,2).detach()

	# y_out_pred = model(y_inp, None, 1, 20)[0]

	# evaluator = ExplicitEvaluator()

	# metrics = evaluator(model, test_d)

	for i in range(10):
		plt.plot(y_out_pred[i,:,0],y_out_pred[i,:,1], alpha=0.5)

	plt.show()
	

if __name__ == '__main__':
	main()