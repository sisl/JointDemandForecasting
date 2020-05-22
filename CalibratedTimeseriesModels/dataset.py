import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

class TimeseriesDataset(data.Dataset):
	
	def __init__(self, t, y, K):
		"""
		Args:
			t (torch.tensor): (B,T) time indices
			y (torch.tensor): (B,T,xdim) observations
			K (int): prediction horizon
		"""
		self._t = t
		self._y = y
		self._K = K

		assert self._t.shape[:2] == self._y.shape[:2], 'B or T not the same.'

	def __getitem__(self, ind):
		"""
		Args:
			ind (int, list, or torch.LongTensor): indices
		Returns:
			t_inp (torch.tensor): input times
			y_inp (torch.tensor): input observations
			t_out (torch.tensor): output times
			y_out (torch.tensor): output observations
		"""
		t = self._t
		y = self._y
		K = self._K
		return t[ind, :-K], y[ind, :-K], t[ind, -K:], y[ind, -K:]

	def __len__(self):
		return self._t.shape[0]


