
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseTransform(ABC):
	
	@abstractmethod
	def fit(self,arr: np.ndarray) -> 'BaseTransform':
		"""Subclasses must implement this method"""
		pass

	@abstractmethod
	def transform(self,arr: np.ndarray) -> np.ndarray:
		"""Subclasses must implement this method"""
		pass
	
	@abstractmethod
	def inverse_transform(self,arr: np.ndarray) -> np.ndarray:
		"""Subclasses must implement this method"""
		pass

	@property
	@abstractmethod
	def p_scale(self) -> float:
		"""Subclasses must implement this method"""
		pass

class IdleTransform(BaseTransform):
	def __init__(self):
		pass
	@property
	def p_scale(self):
		return 1	
	def fit(self,data):
		return self
	def transform(self,data):
		return data
	def inverse_transform(self,data):
		return data

class MeanScaleTransform(BaseTransform):
	def __init__(self,demean=True):
		self.demean = demean
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
		if self.demean: 
			return (data - self.mean) / self.std
		else:
			return data / self.std

	def inverse_transform(self, data):
		"""Reverse the normalization process."""
		if self.mean is None or self.std is None:
			raise ValueError("Normalizer must be fitted before inverse transforming data.")
		if self.demean:
			return (data * self.std) + self.mean
		else:
			return data * self.std

if __name__=='__main__':
	pass

