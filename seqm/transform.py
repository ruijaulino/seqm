
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseTransform(ABC):
	
	@abstractmethod
	def view(self) -> None:
		pass

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

	@abstractmethod
	def pw(self, arr: np.ndarray) -> float:
		"""Subclasses must implement this method"""
		pass

class IdleTransform(BaseTransform):
	def __init__(self):
		pass		
	def view(self):
		print('** IdleTransform **')		
	def pw(self,y):
		return 1	
	def fit(self,data):
		return self
	def transform(self,data):
		return data
	def inverse_transform(self,data):
		return data

class MeanScaleTransform(BaseTransform):
	def __init__(self):
		self.mean = None
		self.std = None		

	def view(self):
		print('** MeanScaleTransform **')
		print('Mean: ', self.mean)
		print('Scale: ', self.std)

	def pw(self,data):
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



class InvVolPwTransform(seqm.BaseTransform):
	def __init__(self):
		self.std = None		
	def view(self):
		print('** ScaleTransform **')
		print('Scale: ', self.std)
	def pw(self,data):
		return np.sqrt(np.sum(np.power(1/self.std,2)))
	def fit(self, data):
		"""Compute the mean and standard deviation of the data."""
		self.std = np.std(data, axis=0)
	def transform(self,data):
		return data
	def inverse_transform(self,data):
		return data





class ScaleTransform(BaseTransform):
	def __init__(self):
		self.std = None		

	def view(self):
		print('** ScaleTransform **')
		print('Scale: ', self.std)

	def pw(self,data):
		return np.sqrt(np.sum(np.power(1/self.std,2)))
	
	def fit(self, data):
		"""Compute the mean and standard deviation of the data."""
		self.std = np.std(data, axis=0)

	def transform(self, data):
		"""Normalize the data."""
		if self.std is None:
			raise ValueError("Normalizer must be fitted before transforming data.")		 
		return data / self.std
		
	def inverse_transform(self, data):
		"""Reverse the normalization process."""
		if self.std is None:
			raise ValueError("Normalizer must be fitted before inverse transforming data.")
		return data * self.std

class RollPWScaleTransform(BaseTransform):
	def __init__(self,window=10,delay=0,demean=False):
		self.window = window
		self.delay = delay
		self.demean = demean
		self.std = None		
		self.mean = None

	def view(self):
		print('** RollPWScaleTransform **')
		print('Fixed Scale: ', self.std)
		if self.demean:
			print('Mean: ', self.mean)	

	def pw(self,data):
		if data.shape[0] > self.window:
			tmp = np.std(data[-self.window:data.shape[0]-self.delay], axis=0)
			if np.sum(tmp) == 0: tmp=self.std
			return np.sqrt( np.sum( np.power( 1 / tmp, 2 ) ) )	
		else:
			return np.sqrt( np.sum( np.power( 1 / self.std, 2 ) ) )
	
	def fit(self, data):
		"""Compute the mean and standard deviation of the data."""
		self.mean = np.mean(data, axis=0)
		self.std = np.std(data, axis=0)

	def transform(self, data):
		"""Normalize the data."""
		if self.std is None:
			raise ValueError("Normalizer must be fitted before transforming data.")		 
		if self.demean:
			return (data - self.mean) / self.std
		else:
			return data / self.std
			
	def inverse_transform(self, data):
		"""Reverse the normalization process."""
		if self.std is None:
			raise ValueError("Normalizer must be fitted before inverse transforming data.")
		if self.demean:
			return (data * self.std) + self.mean
		else:
			return data * self.std

if __name__=='__main__':
	pass

