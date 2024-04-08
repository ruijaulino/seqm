
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy

try:
	from .transform import BaseTransform,IdleTransform
	from .model_pipe import ModelPipe,ModelPipes
	from .constants import *
except ImportError:
	from transform import BaseTransform,IdleTransform
	from model_pipe import ModelPipe,ModelPipes
	from constants import *


# just a train/test data container
class Element:
	def __init__(
			self, 
			x_train: np.ndarray = None, 
			y_train: np.ndarray = None, 
			x_test: np.ndarray = None, 
			y_test: np.ndarray = None, 
			ts=None, 
			x_cols = None, 
			y_cols = None,
			key: str = '', 
			):
		self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
		self.ts = pd.date_range(DEFAULT_START_TS, freq=DEFAULT_FREQ, periods=self.x_test.shape[0]) if ts is None else ts
		self.key = key or 'Dataset'
		self.x_cols = x_cols or ['%s_%s'%(FEATURE_PREFIX,i+1) for i in range(self.x_train.shape[1])]
		self.y_cols = y_cols or ['%s_%s'%(TARGET_PREFIX,i+1) for i in range(self.y_train.shape[1])]
		self.s,self.w,self.pw=None,None,None		

	def view(self):
		print('** Element **')
		print('- Train')
		print(pd.DataFrame(np.hstack((self.x_train,self.y_train)),columns=self.x_cols+self.y_cols))
		print('- Test')
		print(pd.DataFrame(np.hstack((self.x_test,self.y_test)),columns=self.x_cols+self.y_cols,index=self.ts))
		if self.s is not None:
			print('- Strategy')
			print(self.get_s())
			print('- Weight')
			print(self.get_pw())

	def get_s(self):
		return pd.DataFrame(self.s,columns=[STRATEGY_COLUMN],index=self.ts)

	def get_w(self):
		return pd.DataFrame(self.w,columns=[WEIGHT_PREFIX_COLUMNS+c for c in self.y_cols],index=self.ts)

	def get_pw(self):
		return pd.DataFrame(self.pw,columns=[PORTFOLIO_WEIGHT_COLUMN],index=self.ts)
			
	def set_model_wrapper(self,model_wrapper):
		# force a model
		self.model_wrapper=copy.deepcopy(model_wrapper)
		return self

	def set_model_in_wrapper(self, model):
		# set the model class that is on the wrapper
		assert self.model_wrapper is not None,"model_wrapper not defined"
		self.model_wrapper.set_model(model)

	def get_model_in_wrapper(self):
		assert self.model_wrapper is not None,"model_wrapper not defined"
		return self.model_wrapper.model

	def estimate(self, model, **kwargs):
		"""Train the model using the training data contained within this DataElement."""
		self.model=copy.deepcopy(model)  		
		self.model.estimate(x=self.x_train, y=self.y_train)

	def evaluate(self):
		"""Evaluate the model using the test data and return performance metrics."""
		if self.model is None:
			raise Exception("Model has not been estimated.")		
		n = self.y_test.shape[0]
		p = self.y_test.shape[1]
		self.s = np.zeros(n, dtype=np.float64)
		self.w = np.zeros((n, p), dtype=np.float64)
		self.pw = np.ones(n, dtype=np.float64)
		# self.pw = self.y_transform.p_scale*np.ones(n, dtype=np.float64)
		
		for i in range(n):
			# normalize y for input (make copy of y)
			# make sure this array does not get modified
			y_test_copy = np.array(self.y_test[:i])
			y_hat = self.y_transform.transform(y_test_copy)
			# the x input is already normalized!
			w = self.model.get_weight(**{'y': y_hat, 'x': self.x_test[:i], 'xq': self.x_test[i]})
			self.w[i] = w
			self.s[i] = np.dot(self.y_test[i], w)
			self.pw[i] = self.y_transform.pw(y_test_copy)


# dict of ModelWrappers
class Elements:
	def __init__(self):
		self.elements = {}

	def add(self, key, item: Element):
		if not isinstance(item, Element):
			raise TypeError("item must be an instance of Element")
		self.elements[key] = item

	def __getitem__(self, key):
		return self.elements[key]

	def __setitem__(self, key, item: Element):
		if not isinstance(item, Element):
			raise TypeError("item must be an instance of Element")
		self.elements[key] = item

	def __len__(self):
		return len(self.elements)

	def __iter__(self):
		return iter(self.elements)

	def keys(self):
		return self.elements.keys()

	def values(self):
		return self.elements.values()

	def items(self):
		return self.elements.items()
	# more methods on the dict
	def view(self):
		for k,e in self.items():e.view()

	@property
	def all_cols_equal(self):
		x_cols,y_cols = None,None
		for k,v in self.items():
			if x_cols is None:
				x_cols,y_cols = v.x_cols,v.y_cols
			else:
				if x_cols != v.x_cols or y_cols != v.y_cols:
					return False 
		return True
	
	def estimate(self,model_pipes:ModelPipes, share_training_data = True):

		if share_training_data:
			pass

		if single_model:
			x=np.vstack([e.x_train for e in self])
			y=np.vstack([e.y_train for e in self])
			model.estimate(**{'x':x,'y':y})
			self.model = copy.deepcopy(model)
			if view_models:
				model.view(False)
			# set same model for all element
			for e in self: e.set_model(model)
		else:
			for k,e in self.items():
				model_pipes[k].estimate(x=self.x_train,y=self.y_train)
		return self

	def evaluate(self,model_pipes:ModelPipes):
		print('not yet evaluate')
		print(sdfsdf)
		for e in self: e.evaluate()
		return self


# Elements is a collection/list of Element
class Elements_:
	def __init__(self):
		self.element=[]
		self.model = None

	@property
	def keys(self):
		return [e.key for e in self]
	
	def set_model_wrapper(self, model_wrapper):
		# set the same to all elements
		for e in self: e.set_model_wrapper(model_wrapper)
		return self

	def get_model(self):
		if self.model is not None:
			return self.model
		else:
			out = {}
			for e in self: out.update({e.key : e.model})
			return out

	def view(self):
		for element in self.element:
			element.view()

	def add(self,item:Element):
		if isinstance(item, Element):
			self.element.append(item)
		else:
			raise TypeError("Item must be an instance of Element")

	def __getitem__(self, index):
		return self.element[index]

	def __len__(self):
		return len(self.element)

	def __iter__(self):
		return iter(self.element)

	def estimate(self,single_model=False,view_models=False):
		if single_model:
			x=np.vstack([e.x_train for e in self])
			y=np.vstack([e.y_train for e in self])
			model.estimate(**{'x':x,'y':y})
			self.model = copy.deepcopy(model)
			if view_models:
				model.view(False)
			# set same model for all element
			for e in self: e.set_model(model)
		else:
			for e in self: e.estimate(model)
		return self

	def evaluate(self):
		for e in self: e.evaluate()
		return self

# Path is a collection/list of Elements
class Path:
	def __init__(self):
		self.elements=[]
		self.results={}
		self.s,self.w,self.pw={},{},{}
		self.joined=False

	@property
	def keys(self):
		tmp=[]
		for es in self.elements:
			tmp+=es.keys
		tmp=list(set(tmp))
		tmp.sort()
		return tmp
	
	def view(self):
		for e in self: e.view()

	def add(self,item:Elements):
		if isinstance(item, Elements):
			self.elements.append(item)
		else:
			raise TypeError("Item must be an instance of Elements")

	def __getitem__(self, index):
		return self.elements[index]

	def __len__(self):
		return len(self.elements)

	def __iter__(self):
		return iter(self.elements)

	# join path results by name
	def join(self):
		# join strategy performance for
		# all series in all elements
		self.s={}
		self.w={}
		self.pw={}
		for key in self.keys:
			self.s.update({key:[]})
			self.w.update({key:[]})
			self.pw.update({key:[]})
		# iterate elements
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

if __name__=='__main__':
	pass