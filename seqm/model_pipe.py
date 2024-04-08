
# TO DO: change name to ModelPipe??

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy

try:
	from .transform import BaseTransform,IdleTransform
	from .constants import *
except ImportError:
	from transform import BaseTransform,IdleTransform
	from constants import *

# class with a model and a transformer
class ModelPipe:
	
	def __init__(
				self,
				model = None, 
				x_transform:BaseTransform = None, 
				y_transform:BaseTransform = None,
				key: str = ''
				):
		self.model=copy.deepcopy(model)
		self.x_transform = copy.deepcopy(x_transform) if x_transform is not None else IdleTransform()  # Store the class, not an instance
		self.y_transform = copy.deepcopy(y_transform) if y_transform is not None else IdleTransform()  # Store the class, not an instance
		self.key = key or 'Dataset'

	def view(self):
		print('** Model Wrapper **')
		self.model.view()
		self.x_transform.view()
		self.y_transform.view()
		
	# setters
	def set_model(self, model):
		self.model=copy.deepcopy(model)
		return self

	def set_x_transform(self,x_transform:BaseTransform):
		self.x_transform=x_transform
		return self

	def set_y_transform(self,y_transform:BaseTransform):
		self.y_transform=y_transform
		return self

	# fit transform
	def fit_x_transform(self,x_train):
		self.x_transform.fit(x_train)
		return self

	def fit_y_transform(self,y_train):
		self.y_transform.fit(y_train)
		return self

	# apply transform
	def transform_x(self, x, copy_array = True):
		if copy_array: x = np.array(x)
		return self.x_transform.transform(x)
	
	def transform_y(self, y, copy_array = True):
		if copy_array: y = np.array(y)
		return self.y_transform.transform(y)

	# estimate model
	def estimate(self, x, y, **kwargs):
		"""Train the model using the training data contained within this DataElement."""
		self.fit_x_transform(x).fit_y_transform(y)
		x=self.transform_x(x)
		y=self.transform_y(y)
		self.model.estimate(x=x, y=y)
		return self

	# get weight
	def get_weight(self, xq, x, y, apply_transform_x = True, apply_transform_y = True):
		# process inputs
		if apply_transform_y: y = self.transform_y(y,True)
		if x is not None:
			if apply_transform_x: x = self.transform_x(x,True)
		if xq is not None:
			if apply_transform_x: xq = self.transform_x(xq,True)
		return self.model.get_weight(**{'y': y, 'x': x, 'xq': xq})

	# get portfolio weights from the transform
	def get_pw(self,y):
		return self.y_transform.pw(y)		


# dict of ModelWrappers
class ModelPipes:
	def __init__(self, ):
		self.models = {}

	def add(self, key, item: ModelPipe):
		if not isinstance(item, ModelPipe):
			raise TypeError("Item must be an instance of ModelPipe")
		self.models[key] = copy.deepcopy(item)

	def __getitem__(self, key):
		return self.models[key]

	def __setitem__(self, key, item: ModelPipe):
		if not isinstance(item, ModelPipe):
			raise TypeError("Item must be an instance of ModelPipe")
		self.models[key] = copy.deepcopy(item)

	def __len__(self):
		return len(self.models)

	def __iter__(self):
		return iter(self.models)

	def keys(self):
		return self.models.keys()

	def values(self):
		return self.models.values()

	def items(self):
		return self.models.items()

	# more methods
	def has_keys(self,keys:list):
		for key in keys:
			if key not in self.keys():
				return False
		return True

