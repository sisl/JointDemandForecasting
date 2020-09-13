import torch
from torch import nn


class PredictiveDistribution(nn.Module):

	def __init__(self):

		super().__init__()

	def sample(self, nsamps):
		"""
		Sample from predictive distribution
		Args:
			nsamps (int): number of samples
		Returns:
			samples (torch.tensor): (nsamps,*) samples
		"""
		raise NotImplementedError

	def log_probs(self, samples):
		"""
		Returns log probability of samples
		Args:
			samples (torch.tensor): (nsamps,*) samples
		Returns:
			log_probs (torch.tensor): (nsamps,) log probabilties
		"""
		raise NotImplementedError


class GenerativePredictiveModel(nn.Module):

	def __init__(self):

		super().__init__()

	def forward(self, y, u, nsamps, K):
		"""
		Samples from predictive distribution over next K observations.
		Args:
			y (torch.tensor): (B,T,ydim) observations
			u (torch.tensor or None): (B,T+K,udim) inputs
			nsamps (int): number of samples
			K (int): horizon to predict 
		Returns:
			ypredsamps (torch.tensor): (nsamps,B,K,ydim) samples of predicted observations
		"""
		raise NotImplementedError

	def sample(self, y, u, nsamps, K, **kwargs):
		# method to distinguish forward for GenerativePredictiveModel 
		# and ExplictPredictiveModel
		return self(y, u, nsamps, K, **kwargs)


class ExplicitPredictiveModel(nn.Module):

	def __init__(self):

		super().__init__()

	def forward(self, y, u, K):
		"""
		Predicts distribution over next K observations.
		Args:
			y (torch.tensor): (B,T,ydim) observations
			u (torch.tensor or None): (B,T+K,udim) inputs
			K (int): horizon to predict 
		Returns:
			dist (PredictiveDistribution): (B,K*ydim) predictive distribution over next K observations shaped
		"""
		raise NotImplementedError

	def dist(self, y, u, K, **kwargs):
		# method to distinguish forward for GenerativePredictiveModel 
		# and ExplictPredictiveModel
		return self(y, u, K, **kwargs)



