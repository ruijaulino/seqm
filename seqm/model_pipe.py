import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy

try:
	from .transform import BaseTransform, IdleTransform
	from .constants import *
except ImportError:
	from transform import BaseTransform, IdleTransform
	from constants import *

# class with a model and a transformer
# train, test, live of model

def does_cls_have_method(cls_instance, method: str) -> bool:
	instance_method = getattr(cls_instance, method, None)
	if instance_method is None:
		return False
	else:
		if callable(instance_method):
			return True
		else:
			return False

def round_weight(w:np.ndarray, w_precision:float = 0.0001) -> np.ndarray:
	return np.sign(w) * np.round( np.abs(w) / w_precision ) * w_precision

# container for computations
# apply transformations, estimate and evaluate models
class BaseModelPipe:
	
	def __init__(
				self,
				model = None, 
				x_transform:BaseTransform = None, 
				y_transform:BaseTransform = None,
				w_precision = 0.0001,
				key: str = 'no key defined'
				):
		self.model = copy.deepcopy(model)
		self.x_transform = copy.deepcopy(x_transform) if x_transform is not None else IdleTransform()  # Store the class, not an instance
		self.y_transform = copy.deepcopy(y_transform) if y_transform is not None else IdleTransform()  # Store the class, not an instance
		self.w_precision = w_precision
		self.key = key or 'Dataset'
		self.s, self.w, self.pw, self.targets = None, None, None, None
		self.train_data, self.test_data = None, None
		self.train_pw = None

	def view(self):
		print('-- Model Pipe %s --'%self.key)
		print(' = Portfolio weight = ')
		print(self.train_pw)
		print(' = Model = ')
		self.model.view()

	def get_s_df(self):
		if self.s is not None:
			return pd.DataFrame(self.s,columns=[STRATEGY_COLUMN],index=self.test_data.get_ts_as_timestamp())
		else:
			return None

	def get_targets_df(self):
		if self.targets is not None:
			return pd.DataFrame(self.targets,columns=[TARGET_PREFIX_COLUMNS+c for c in self.test_data.y_cols],index=self.test_data.get_ts_as_timestamp())
		else:
			return None			
	
	def get_w_df(self):
		if self.w is not None:
			return pd.DataFrame(self.w,columns=[WEIGHT_PREFIX_COLUMNS+c for c in self.test_data.y_cols],index=self.test_data.get_ts_as_timestamp())
		else:
			return None	
	
	def get_pw_df(self):
		if self.pw is not None:
			return pd.DataFrame(self.pw,columns=[PORTFOLIO_WEIGHT_COLUMN],index=self.test_data.get_ts_as_timestamp())
		else:
			return None

	# ------------
	# setters
	def set_model(self, model):
		self.model = copy.deepcopy(model)
		return self

	def set_x_transform(self, x_transform:BaseTransform):
		self.x_transform = x_transform
		return self

	def set_y_transform(self, y_transform:BaseTransform):
		self.y_transform = y_transform
		return self

	def set_train_data(self, train_data):		
		self.train_data = train_data
		# when the train data is set, fit the transform
		self.fit_transform()
		return self

	def set_test_data(self, test_data):
		self.test_data = test_data
		return self

	def set_data(self, train_data, test_data):
		self.set_train_data(train_data).set_test_data(test_data)
		return self

	# fit transforms
	# ------------
	def fit_x_transform(self):
		if self.train_data.has_x: self.x_transform.fit(self.train_data.x)
		return self

	def fit_y_transform(self):
		self.y_transform.fit(self.train_data.y)
		return self

	def fit_transform(self):
		self.fit_x_transform().fit_y_transform()
		# apply transform
		if self.train_data.has_x: self.train_data.x = self.transform_x(self.train_data.x)
		self.train_data.y = self.transform_y(self.train_data.y)

	# apply transform
	# ------------
	def transform_x(self, x:np.ndarray, copy_array:bool = True):
		if copy_array: x = np.array(x)
		return self.x_transform.transform(x)
	
	def transform_y(self, y:np.ndarray, copy_array:bool = True):
		if copy_array: y = np.array(y)
		return self.y_transform.transform(y)

	# estimate model
	def estimate(self, **kwargs):
		"""Train the model using the training data contained within this DataElement."""
		aux = self.train_data.build_train_inputs()
		self.model.estimate(**aux)
		self.train_pw = self.get_pw(aux.get('y'))
		return self

	# get weight
	def get_weight(self, xq, x, y, z, apply_transform_x = True, apply_transform_y = True):
		# process inputs
		if apply_transform_y: y = self.transform_y(y, True)
		if x is not None:
			if apply_transform_x: x = self.transform_x(x, True)
		if xq is not None:
			if apply_transform_x: xq = self.transform_x(xq, True)
		return self.model.get_weight(**{'y': y, 'x': x, 'xq': xq, 'z':z})
	
	# get portfolio weights from the transform
	def get_pw(self,y):
		return self.y_transform.pw(y)		

	def evaluate(self):
		"""Evaluate the model using the test data and return performance metrics."""
		
		n = self.test_data.y.shape[0]
		p = self.test_data.y.shape[1]
		
		idx = self.test_data.converted_idx()
		if idx is None:
			idx=np.array([[0,n]],dtype=int)		
		n_seq=idx.shape[0]

		self.s = np.zeros(n, dtype=np.float64)		
		self.w = np.zeros((n, p), dtype=np.float64)
		self.pw = np.ones(n, dtype=np.float64)
		self.targets = np.zeros((n,p), dtype=np.float64)

		# for models that can benefict from incremental computations
		# we can call a method set_start_evaluate to tell the model class
		# to store previous computations and evaluate faster
		if does_cls_have_method(self.model, 'set_start_evaluate'): self.model.set_start_evaluate()

		for l in range(n_seq): 
			for i in range(idx[l][0],idx[l][1]):
				# normalize y for input (make copy of y)
				# make sure this array does not get modified
				# the x input is already normalized!
				xq_ = None
				x_ = None
				z_ = None
				if self.test_data.has_x:
					xq_ = self.test_data.x[i]
					x_ = self.test_data.x[idx[l][0]:i]
				if self.test_data.has_z:
					z_ = self.test_data.z[i]
				w = self.get_weight(
									xq = xq_, 
									x = x_, 
									y = self.test_data.y[idx[l][0]:i], 
									z = z_,
									apply_transform_x = True, 
									apply_transform_y = True
									)
				w = round_weight(w = w, w_precision = self.w_precision)
				self.w[i] = w
				self.targets[i] = self.test_data.y[i]
				self.s[i] = np.dot(self.test_data.y[i], w)
				self.pw[i] = self.get_pw(self.test_data.y[:i])

		# when this is called the class should reset the variables used for incremental evaluation
		if does_cls_have_method(self.model, 'set_end_evaluate'): self.model.set_end_evaluate()

# dict of BaseModelPipe to keep logic separated
class ModelPipe:

	def __init__(self, master_model = None):
		# if a master model is defined it is trained with 
		# all training data and set on all individual pipes
		# as it's model
		self.master_model = copy.deepcopy(master_model)
		self.base_model_pipes = {}

	def view(self):
		for k,v in self.items():
			v.view()
			print()
			print()

	def add(self, model = None, x_transform:BaseTransform = None, y_transform:BaseTransform = None, w_precision = 0.0001, key: str = 'no key defined'):
		'''
		Create a new instance of BaseModelPipe
		'''
		self.base_model_pipes[key] = BaseModelPipe(model = model, x_transform = x_transform, y_transform = y_transform, w_precision = w_precision, key = key)
		
	def __getitem__(self, key):
		return self.base_model_pipes.get(key, None)

	def __setitem__(self, key, item: BaseModelPipe):
		'''
		Should not be used..
		'''
		if not isinstance(item, BaseModelPipe):
			raise TypeError("Item must be an instance of BaseModelPipe")
		self.base_model_pipes[key] = copy.deepcopy(item)
		self.base_model_pipes[key].key = key

	def __len__(self):
		return len(self.base_model_pipes)

	def __iter__(self):
		return iter(self.base_model_pipes)

	def keys(self):
		return self.base_model_pipes.keys()

	def values(self):
		return self.base_model_pipes.values()

	def items(self):
		return self.base_model_pipes.items()

	def remove(self, key):
		"""
		Removes a ModelPipe instance associated with the specified key.

		Parameters:
		- key: The key of the ModelPipe instance to be removed.

		Raises:
		- KeyError: If the key is not found in the models.
		"""
		if key in self.base_model_pipes:
			del self.base_model_pipes[key]
		else:
			pass

	# more methods
	def has_keys(self, keys:Union[list,str]):
		if isinstance(keys,str):
			keys=[keys]
		for key in keys:
			if key not in self.keys():
				return False
		return True

	def set_data(self, key:str, train_data, test_data):
		assert key in self.keys(), f"ModelPipe for key {key} not defined"
		self.base_model_pipes[key].set_data(train_data, test_data)

	def set_train_data(self, key:str, train_data):
		assert key in self.keys(), f"ModelPipe for key {key} not defined"
		self.base_model_pipes[key].set_train_data(train_data)

	def set_test_data(self, key:str, test_data):
		assert key in self.keys(), f"ModelPipe for key {key} not defined"
		self.base_model_pipes[key].set_test_data(test_data)

	def estimate(self):
		
		if self.master_model is not None:
			# if the model is shared then we train a single model
			# on all data joined
			data = None
			for k, e in self.items():
				if data is None:
					data = copy.deepcopy(e.train_data)
				else:
					data.stack(e.train_data, allow_both_empty = True)
			if data.empty: raise Exception('data is empty. should not happen')
			self.master_model.estimate(**data.build_train_inputs())
			# set individual copies			
			for k, e in self.items(): e.set_model(self.master_model)
		else:
			for k, m in self.items(): m.estimate()
		return self

	def evaluate(self):
		for k, e in self.items(): e.evaluate()
		return self


# Path is a collection/list of ModelPipe to join backtest results
class Path:
	def __init__(self):
		self.model_pipes = []
		self.results = {}
		self.s, self.w, self.pw = {}, {}, {}
		self.joined = False

	@property
	def keys(self):
		out = []
		for e in self.model_pipes: 
			out += list(e.keys())
		out = list(set(out))
		return out

	def add(self, item:ModelPipe):
		if isinstance(item, ModelPipe):
			self.model_pipes.append(item)
		else:
			raise TypeError("Item must be an instance of ModelPipe")

	def __getitem__(self, index):
		return self.model_pipes[index]

	def __len__(self):
		return len(self.model_pipes)

	def __iter__(self):
		return iter(self.model_pipes)

	# join path results by name
	def join(self):
		# join strategy performance for
		# all series in all model_pipes
		self.s = {}
		self.w = {}
		self.targets = {}
		self.pw = {}
		for key in self.keys:
			self.s.update({key:[]})
			self.w.update({key:[]})
			self.pw.update({key:[]})
			self.targets.update({key:[]})
		# iterate model_pipes
		for model_pipes in self:
			tmp = {}
			# iterate series in element
			for k, base_model_pipes in model_pipes.items():
				# get performance dataframe
				self.s[k].append(base_model_pipes.get_s_df())
				# get weight dataframe
				self.w[k].append(base_model_pipes.get_w_df())
				# get targets dataframe
				self.targets[k].append(base_model_pipes.get_targets_df())
				# get portfolio weight
				self.pw[k].append(base_model_pipes.get_pw_df())
		for k, v in self.s.items():
			tmp = pd.concat(self.s[k], axis = 0)
			tmp = tmp.sort_index()
			self.s[k] = tmp
		for k, v in self.w.items():
			tmp = pd.concat(self.w[k], axis = 0)
			tmp = tmp.sort_index()
			self.w[k] = tmp
		for k, v in self.targets.items():
			tmp = pd.concat(self.targets[k], axis = 0)
			tmp = tmp.sort_index()
			self.targets[k] = tmp			
		for k, v in self.pw.items():
			tmp = pd.concat(self.pw[k], axis = 0)
			tmp = tmp.sort_index()			
			self.pw[k] = tmp		
		self.joined = True
		return self
	
	def get_results(self):
		if not self.joined: self.join()
		self.results = {}
		for key in self.keys:			
			self.results.update({key:pd.concat([self.s.get(key), self.w.get(key), self.pw.get(key), self.targets.get(key)], axis = 1)})
		return self.results		



class ModelPipes():
	pass

