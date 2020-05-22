import torch


class Evaluator:

	def evaluate(self, model, eval_dataset):
		"""
		Evaluates the calibration of a model
		Args:
			model (models.PredictiveDistribution but TODO): predictive model
			eval_dataset (datasets.Dataset): evaluation dataset
		Returns:
			evaluation_metrics (dict): metrics evaluating calibration
		"""
		raise NotImplementedError
