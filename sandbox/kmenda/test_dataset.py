import torch
from CalibratedTimeseriesModels.dataset import TimeseriesDataset

from torch.utils import data

import matplotlib.pyplot as plt

def main():

	dat = torch.load('datasets/gauss_rw.data')
	t = dat['t']
	y = dat['x']

	dataset = TimeseriesDataset(t,y,20)

	B = len(dataset)

	trs = int(0.8 * B)
	tes = B - trs

	train_d, test_d = data.random_split(dataset, [trs, tes])

	t_inp, y_inp, t_out, y_out = train_d[:]
	
	plt.subplot(2,1,1)
	plt.plot(t_inp[0], y_inp[...,0].T)
	plt.plot(t_out[0], y_out[...,0].T, alpha=0.5)

	plt.subplot(2,1,2)
	plt.plot(t_inp[0], y_inp[...,1].T)
	plt.plot(t_out[0], y_out[...,1].T, alpha=0.5)

	plt.show()
	

if __name__ == '__main__':
	main()
