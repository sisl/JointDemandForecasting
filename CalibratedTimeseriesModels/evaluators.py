import torch


class Evaluator:

	def __call__(self, model, eval_dataset):
		"""
		Evaluates the calibration of a model
		Args:
			model (models.PredictiveDistribution but TODO): predictive model
			eval_dataset (datasets.Dataset): evaluation dataset
		Returns:
			evaluation_metrics (dict): metrics evaluating calibration
		"""
		raise NotImplementedError


class ExplicitEvaluator:

	def __call__(self, model, eval_dataset):

		t_inp, y_inp, t_out, y_out = eval_dataset[:]

		B, K, _ = y_out.shape

		dist = model(y_inp, None, K)

		lps = dist.log_prob(y_out.reshape(B,-1))

		evaluation_metrics = {'logprobs': lps}

		return lps