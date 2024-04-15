
# TO DO: change name to ModelPipe??

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy

try:
	from .transform import BaseTransform,IdleTransform
	from .arrays import Arrays
	from .constants import *
except ImportError:
	from transform import BaseTransform,IdleTransform
	from arrays import Arrays
	from constants import *

# class with a model and a transformer
# train, test, live of model
class ModelPipe:
	
	def __init__(
				self,
				x_transform:BaseTransform = None, 
				y_transform:BaseTransform = None,
				model = None, 
				key: str = ''
				):
		self.model=copy.deepcopy(model)
		self.x_transform = copy.deepcopy(x_transform) if x_transform is not None else IdleTransform()  # Store the class, not an instance
		self.y_transform = copy.deepcopy(y_transform) if y_transform is not None else IdleTransform()  # Store the class, not an instance
		self.key = key or 'Dataset'
		self.s,self.w,self.pw = None,None,None
		self.x_cols,self.y_cols = None,None
		self.x_train,self.y_train = None,None
		self.x_test,self.y_test = None,None
		self.train_arrays,self.test_arrays = None,None
		self.ts = None

	def view(self):
		print('** Element **')
		if self.x_train is not None:
			print('- Train')
			print(pd.DataFrame(np.hstack((self.x_train,self.y_train)),columns=self.x_cols+self.y_cols))
		if self.x_test is not None:
			print('- Test')
			print(pd.DataFrame(np.hstack((self.x_test,self.y_test)),columns=self.x_cols+self.y_cols,index=self.ts))
		if self.s is not None:
			print('- Strategy')
			print(self.get_s())
			print('- Weight')
			print(self.get_pw())

	def get_s_df(self):
		if self.s is not None:
			return pd.DataFrame(self.s,columns=[STRATEGY_COLUMN],index=self.test_arrays.ts)
		else:
			print('ola none s')
			return None
	def get_w_df(self):
		if self.w is not None:
			return pd.DataFrame(self.w,columns=[WEIGHT_PREFIX_COLUMNS+c for c in self.test_arrays.y_cols],index=self.test_arrays.ts)
		else:
			print('ola none w')
			return None	
	def get_pw_df(self):
		if self.pw is not None:
			return pd.DataFrame(self.pw,columns=[PORTFOLIO_WEIGHT_COLUMN],index=self.test_arrays.ts)
		else:
			print('ola none pw')
			return None

	def view(self):
		print()
		print('** Model Pipe **')
		self.model.view()
		print()
		self.x_transform.view()
		print()
		self.y_transform.view()
		print()
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

	def fit_transform(self):
		self.fit_x_transform().fit_y_transform()
		# apply transform
		if self.train_arrays.has_x: self.train_arrays.x = self.transform_x(self.train_arrays.x)
		self.train_arrays.y = self.transform_y(self.train_arrays.y)

	# fit transform
	def fit_x_transform(self):
		if self.train_arrays.has_x: self.x_transform.fit(self.train_arrays.x)
		return self

	def fit_y_transform(self):
		self.y_transform.fit(self.train_arrays.y)
		return self

	# apply transform
	def transform_x(self, x, copy_array = True):
		if copy_array: x = np.array(x)
		return self.x_transform.transform(x)
	
	def transform_y(self, y, copy_array = True):
		if copy_array: y = np.array(y)
		return self.y_transform.transform(y)


	# estimate model
	def estimate(self, **kwargs):
		"""Train the model using the training data contained within this DataElement."""
		# self.fit_x_transform(x).fit_y_transform(y)
		# x=self.transform_x(x)
		# y=self.transform_y(y)
		# self.model.estimate(x=self.train, y=self.y_train)
		self.model.estimate( **{'x' : self.train_arrays.x, 'y' : self.train_arrays.y, 'z' : self.train_arrays.z} )
		return self

	# --------------------------
	# main external methods
	# --------------------------
	# set data


	# TO REMOVE
	# ------------
	def set_cols(self, x_cols:list, y_cols:list):
		self.x_cols,self.y_cols = x_cols,y_cols
		return self

	def check_cols(self, x_cols:list, y_cols:list):
		assert x_cols==self.x_cols,"x columns do not match the existent model pipe"
		assert y_cols==self.y_cols,"y columns do not match the existent model pipe"
		return self

	def set_train_data(self, x_train, y_train,copy_array = True):		
		self.x_train = np.array(x_train) if copy_array else x_train
		self.y_train = np.array(y_train) if copy_array else y_train
		# when the train data is set, fit the transform
		self.fit_transform()
		return self

	def set_test_data(self, x_test, y_test, ts, copy_array = True):
		self.x_test = np.array(x_test) if copy_array else x_test
		self.y_test = np.array(y_test) if copy_array else y_test
		self.ts = copy.deepcopy(ts) if copy_array else ts
		# do not transform here training data to make
		# the implementation more clear when testing
		return self

	def set_data(self, x_train, y_train, x_test, y_test, ts):
		self.set_train_data(x_train, y_train).set_test_data(x_test, y_test, ts)
		return self
	# -------

	def set_train_arrays(self, arrays:Arrays):
		self.train_arrays = arrays
		self.fit_transform()
		return self
	def set_test_arrays(self, arrays:Arrays):
		self.test_arrays = arrays
		return self
	def set_arrays(self, train_arrays:Arrays, test_arrays:Arrays):
		self.set_train_arrays(train_arrays).set_test_arrays(test_arrays)
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
		n = self.test_arrays.y.shape[0]
		p = self.test_arrays.y.shape[1]
		self.s = np.zeros(n, dtype=np.float64)
		self.w = np.zeros((n, p), dtype=np.float64)
		self.pw = np.ones(n, dtype=np.float64)

		for i in range(n):
			# normalize y for input (make copy of y)
			# make sure this array does not get modified
			# the x input is already normalized!
			xq_ = None
			x_ = None
			z_ = None
			if self.test_arrays.has_x:
				xq_ = self.test_arrays.x[i]
				x_ = self.test_arrays.x[:i]
			if self.test_arrays.has_z:
				z_ = self.test_arrays.z[i]
			w = self.get_weight(
								xq = xq_, 
								x = x_, 
								y = self.test_arrays.y[:i], 
								z = z_,
								apply_transform_x = True, 
								apply_transform_y = True
								)
			self.w[i] = w
			self.s[i] = np.dot(self.test_arrays.y[i], w)
			self.pw[i] = self.get_pw(self.test_arrays.y[:i])

# dict of ModelWrappers
class ModelPipes:

	def __init__(self, master_model = None):
		# if a master model is defined it is trained with 
		# all training data and set on all individual pipes
		# as it's model
		self.master_model = copy.deepcopy(master_model)
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

	def remove(self, key):
		"""
		Removes a ModelPipe instance associated with the specified key.

		Parameters:
		- key: The key of the ModelPipe instance to be removed.

		Raises:
		- KeyError: If the key is not found in the models.
		"""
		if key in self.models:
			del self.models[key]
		else:
			pass

	# more methods
	def has_keys(self,keys:Union[list,str]):
		if isinstance(keys,str):
			keys=[keys]
		for key in keys:
			if key not in self.keys():
				return False
		return True

	def estimate(self):
		
		if self.master_model is not None:
			# if the model is shared then we train a single model
			# on all data joined
			# it is assumed that the individual model pipes
			# have suitable normalizations
			# maybe this could be a method...
			arrays = None
			for k,e in self.items():
				if arrays is None:
					arrays = copy.deepcopy(e.train_arrays)
				else:
					arrays.stack(e.train_arrays)					
			# x = np.vstack([e.train_arrays.x for k,e in self.items()])
			# y = np.vstack([e.train_arrays.y for k,e in self.items()])
			self.master_model.estimate(**{'x':arrays.x,'y':arrays.y,'z':arrays.z})
			# set individual copies			
			for k,e in self.items(): e.set_model(self.master_model)
		else:
			for k,m in self.items():m.estimate()
		return self

	def evaluate(self):
		for k,e in self.items(): e.evaluate()
		return self


# Path is a collection/list of ModelPipes
class Path:
	def __init__(self):
		self.model_pipes=[]
		self.results={}
		self.s,self.w,self.pw={},{},{}
		self.joined=False

	@property
	def keys(self):
		out = []
		for e in self.model_pipes: 
			out+=list(e.keys())
		out=list(set(out))
		return out
		#if len(self.model_pipes)!=0:
		#	return list(self.model_pipes[0].keys())
		#else:
		# 	return []

	def add(self,item:ModelPipes):
		if isinstance(item, ModelPipes):
			self.model_pipes.append(item)
		else:
			raise TypeError("Item must be an instance of ModelPipes")

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
		self.s={}
		self.w={}
		self.pw={}
		for key in self.keys:
			self.s.update({key:[]})
			self.w.update({key:[]})
			self.pw.update({key:[]})
		# iterate model_pipes
		for mps in self:
			tmp={}
			# iterate series in element
			for k,mp in mps.items():
				# get performance dataframe
				self.s[k].append(mp.get_s_df())
				# get weight dataframe
				self.w[k].append(mp.get_w_df())
				# get portfolio weight
				self.pw[k].append(mp.get_pw_df())
		for k,v in self.s.items():
			tmp=pd.concat(self.s[k],axis=0)
			tmp=tmp.sort_index()
			self.s[k]=tmp
		for k,v in self.w.items():
			tmp=pd.concat(self.w[k],axis=0)
			tmp=tmp.sort_index()
			self.w[k]=tmp
		for k,v in self.pw.items():
			tmp=pd.concat(self.pw[k],axis=0)
			tmp=tmp.sort_index()			
			self.pw[k]=tmp		
		self.joined=True
		return self
	
	def get_results(self):
		if not self.joined: self.join()
		self.results={}
		for key in self.keys:			
			self.results.update({key:pd.concat([self.s.get(key),self.w.get(key),self.pw.get(key)],axis=1)})
		return self.results		