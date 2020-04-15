import torch


class Calibrator:

	def calibrate(self, model, calib_dataset):
		"""
		Calibrate model using dataset
		Args:
			model (models.PredictiveDistribution but TODO): predictive model
			calib_dataset (datasets.Dataset): calibration dataset
		Return:
			calibrated_model (models.PredictiveDistribution but TODO): calibrated predictive model
		"""
		raise NotImplementedError

		