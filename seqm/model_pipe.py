
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

	def get_s(self):
		if self.s is not None:
			return pd.DataFrame(self.s,columns=[STRATEGY_COLUMN],index=self.ts)
		else:
			return pd.DataFrame(columns=[STRATEGY_COLUMN])
	def get_w(self):
		return pd.DataFrame(self.w,columns=[WEIGHT_PREFIX_COLUMNS+c for c in self.y_cols],index=self.ts)
		
	def get_pw(self):
		return pd.DataFrame(self.pw,columns=[PORTFOLIO_WEIGHT_COLUMN],index=self.ts)

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
		self.fit_x_transform(self.x_train).fit_y_transform(self.y_train)
		self.x_train=self.transform_x(self.x_train)
		self.y_train=self.transform_y(self.y_train)

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
		# self.fit_x_transform(x).fit_y_transform(y)
		# x=self.transform_x(x)
		# y=self.transform_y(y)
		self.model.estimate(x=self.x_train, y=self.y_train)
		return self

	# --------------------------
	# main external methods
	# --------------------------
	# set data

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

	def evaluate(self):
		"""Evaluate the model using the test data and return performance metrics."""
		if self.y_test is not None:
			n = self.y_test.shape[0]
			p = self.y_test.shape[1]
			self.s = np.zeros(n, dtype=np.float64)
			self.w = np.zeros((n, p), dtype=np.float64)
			self.pw = np.ones(n, dtype=np.float64)

			for i in range(n):
				# normalize y for input (make copy of y)
				# make sure this array does not get modified
				# the x input is already normalized!
				w = self.get_weight(
									xq = self.x_test[i], 
									x = self.x_test[:i], 
									y = self.y_test[:i], 
									apply_transform_x = True, 
									apply_transform_y = True
									)
				self.w[i] = w
				self.s[i] = np.dot(self.y_test[i], w)
				self.pw[i] = self.get_pw(self.y_test[:i])

# dict of ModelWrappers
class ModelPipes:
	def __init__(self, model = None):
		# model to be used when we are sharing the model
		self.model = copy.deepcopy(model)
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

	def estimate(self,share_model = True):
		
		if share_model:
			# if the model is shared then we train a single model
			# on all data joined
			# it is assumed that the individual model pipes
			# have suitable normalizations

			# TO DO: WHY CAN x_train BE None WHEN DOING CVBT??
			x = np.vstack([e.x_train for k,e in self.items() if e.x_train is not None])
			y = np.vstack([e.y_train for k,e in self.items() if e.y_train is not None])
			self.model.estimate(**{'x':x,'y':y})
			# set individual copies			
			for k,e in self.items(): e.set_model(self.model)
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
		if len(self.model_pipes)!=0:
			return list(self.model_pipes.keys())
		else:
			return []

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
		for es in self:
			tmp={}
			# iterate series in element
			for e in es:
				if e.s is not None:
					# get performance dataframe
					self.s[e.key].append(e.get_s())
					# get weight dataframe
					self.w[e.key].append(e.get_w())
					# get portfolio weight
					self.pw[e.key].append(e.get_pw())
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
		if len(self.keys)==1:
			self.results=self.results.get(self.keys[0])
		return self.results		