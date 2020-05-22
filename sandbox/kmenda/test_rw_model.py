from CalibratedTimeseriesModels.models.grw import *
import torch

from CalibratedTimeseriesModels.dataset import TimeseriesDataset

from CalibratedTimeseriesModels.evaluators import ExplicitEvaluator

from torch.utils import data

import matplotlib.pyplot as plt

def main():

	torch.set_default_dtype(torch.float64)

	dat = torch.load('datasets/gauss_rw.data')
	t = dat['t']
	y = dat['y']

	dataset = TimeseriesDataset(t,y,20)

	B = len(dataset)

	trs = int(0.8 * B)
	tes = B - trs

	train_d, test_d = data.random_split(dataset, [trs, tes])

	t_inp, y_inp, t_out, y_out = train_d[:]

	model = GaussianRandomWalkModel(torch.tensor([0.,0.]),
			torch.eye(2))

	dist = model(y_inp, None, 20)
	y_out_pred = dist.sample().reshape(y_out.shape).detach()

	# y_out_pred = model(y_inp, None, 1, 20)[0]

	evaluator = ExplicitEvaluator()

	metrics = evaluator(model, test_d)

	plt.subplot(2,1,1)
	plt.plot(t_inp[0], y_inp[...,0].T)
	plt.plot(t_out[0], y_out_pred[...,0].T, alpha=0.5)

	plt.subplot(2,1,2)
	plt.plot(t_inp[0], y_inp[...,1].T)
	plt.plot(t_out[0], y_out_pred[...,1].T, alpha=0.5)

	plt.show()
	

if __name__ == '__main__':
	main()