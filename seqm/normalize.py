
import numpy as np
import pandas as pd

class DummyNormalizer:
	def __init__(self):
		self.p_scale=1 # scalar
	def fit(self,data):
		return self
	def transform(self,data):
		return data
	def inverse_transform(self,data):
		return data

class Normalizer:
	def __init__(self):
		self.mean = None
		self.std = None		

	@property
	def p_scale(self):
		if self.std is None:
			return 1
		else:
			return np.sqrt(np.sum(np.power(1/self.std,2)))
	
	def fit(self, data):
		"""Compute the mean and standard deviation of the data."""
		self.mean = np.mean(data, axis=0)
		self.std = np.std(data, axis=0)

	def transform(self, data):
		"""Normalize the data."""
		if self.mean is None or self.std is None:
			raise ValueError("Normalizer must be fitted before transforming data.")
		return (data - self.mean) / self.std

	def inverse_transform(self, data):
		"""Reverse the normalization process."""
		if self.mean is None or self.std is None:
			raise ValueError("Normalizer must be fitted before inverse transforming data.")
		return (data * self.std) + self.mean


if __name__=='__main__':
	pass

