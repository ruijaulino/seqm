
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
# base class that contains a data element in numpy array form
# models can be applied to it

class Element:
	def __init__(
			self, 
			x_train: np.ndarray, 
			y_train: np.ndarray, 
			x_test: np.ndarray, 
			y_test: np.ndarray, 
			ts=None, 
			x_transform: BaseTransform = None, 
			y_transform: BaseTransform = None, 
			key: str = '', 
			x_cols = None, 
			y_cols = None
			):
		self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
		self.ts = pd.date_range('1950-01-01', freq='D', periods=self.x_test.shape[0]) if ts is None else ts
		self.key = key or 'Dataset'
		self.x_cols = x_cols or [f'x_{i+1}' for i in range(self.x_train.shape[1])]
		self.y_cols = y_cols or [f'y_{i+1}' for i in range(self.y_train.shape[1])]

		assert self.x_train.shape[0] == self.y_train.shape[0], "x_train and y_train must have the same number of observations"
		assert self.x_test.shape[1] == self.x_train.shape[1], "x_train and x_test must have the same number of variables"
		assert self.y_test.shape[1] == self.y_train.shape[1], "y_train and y_test must have the same number of variables"		

		self.x_transform = x_transform if x_transform is not None else IdleTransform()  # Store the class, not an instance
		self.y_transform = y_transform if y_transform is not None else IdleTransform()  # Store the class, not an instance

		self.model = None  # Placeholder for a trained model
		self.s,self.w,self.pw=None,None,None
		
		self._verify_input_data()
		self._fit_transform()

	def get_s(self):
		return pd.DataFrame(self.s,columns=[STRATEGY_COLUMN],index=self.ts)

	def get_w(self):
		return pd.DataFrame(self.w,columns=[WEIGHT_PREFIX_COLUMNS+c for c in self.y_cols],index=self.ts)

	def get_pw(self):
		return pd.DataFrame(self.pw,columns=[PORTFOLIO_WEIGHT_COLUMN],index=self.ts)

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
			
	def _verify_input_data(self):
		if self.ts is None: self.ts=pd.date_range('1950-01-01',freq='D',periods=self.x_test.shape[0])
		assert self.x_train.shape[0]==self.y_train.shape[0],"x_train and y_train must have the same number of observations"
		assert self.x_test.shape[0]==self.y_test.shape[0],"x_test and y_test must have the same number of observations"
		assert self.x_test.shape[0]==self.ts.size,"x_train and ts must have the same number of observations"		
		assert self.x_train.shape[1]==self.x_test.shape[1],"x_train and x_test must have the same number of variables"
		assert self.y_train.shape[1]==self.y_test.shape[1],"y_train and y_test must have the same number of variables"
		if self.x_cols is None: self.x_cols=['x_%s'%(i+1) for i in range(self.x_train.shape[1])]
		if self.y_cols is None: self.y_cols=['y_%s'%(i+1) for i in range(self.y_train.shape[1])]

	def set_model(self,model):
		# force a model
		self.model=copy.deepcopy(model)

	def _fit_transform(self):
		"""Fit and transform the training data using dedicated normalizers."""
		self.x_transform = copy.deepcopy(self.x_transform)
		self.y_transform = copy.deepcopy(self.y_transform)

		self.x_transform.fit(self.x_train)
		self.x_train = self.x_transform.transform(self.x_train)
		self.y_transform.fit(self.y_train)
		self.y_train = self.y_transform.transform(self.y_train)
		# can transform now the test x
		self.x_test = self.x_transform.transform(self.x_test)

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

# collection of Elements
class Elements:
	def __init__(self):
		self.element=[]

	@property
	def keys(self):
		return [e.key for e in self]
	
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

	def estimate(self,model,single_model=False,view_models=False):
		if single_model:
			model=copy.deepcopy(model)
			x=np.vstack([e.x_train for e in self])
			y=np.vstack([e.y_train for e in self])
			model.estimate(**{'x':x,'y':y})
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


class Dataset:

	def __init__(self, 
				datasets: Union[pd.DataFrame, Dict[str,pd.DataFrame]],  
				x_transform: BaseTransform = None, 
				y_transform: BaseTransform = None,
				target_prefix='y', 
				feature_prefix='x'
				):
		super().__init__()
		self.target_prefix = target_prefix
		self.feature_prefix = feature_prefix		
		self.folds_dates = []
		self.x_transform = x_transform if x_transform is not None else IdleTransform()  # Store the class, not an instance
		self.y_transform = y_transform if y_transform is not None else IdleTransform()  # Store the class, not an instance		

		self._verify_inputs(datasets)

	def _verify_inputs(self, datasets):
		# transform dict into lists
		if isinstance(datasets, dict):
			self.keys=[]
			self.datasets=[]
			for k,v in datasets.items():
				self.keys.append(k)
				self.datasets.append(v)
		else:
			self.datasets = [datasets] 
			self.keys=['Dataset']
		cols = self.datasets[0].columns.tolist()
		for df in self.datasets:
			if df.columns.tolist() != cols:
				raise ValueError("All DataFrames must have the same columns.")
		self.y_cols=[c for c in cols if c.startswith(self.target_prefix)]
		self.x_cols=[c for c in cols if c.startswith(self.feature_prefix)]
		
	def split_dates(self, k_folds=3):
		ts = pd.DatetimeIndex([])
		for df in self.datasets:
			ts = ts.union(df.index.unique())
		ts = ts.sort_values()
		idx_folds = np.array_split(ts, k_folds)
		self.folds_dates = [(fold[0], fold[-1]) for fold in idx_folds]
		return self

	def get_split_elements(self, test_fold: int, burn_fraction: float = 0.1, min_burn_points: int = 1, seq_path: bool = False)->Elements:
		
		elements = Elements()  # Store splits by name

		if seq_path and test_fold == 0:
			raise ValueError("Cannot start at fold 0 when path is sequential")
		if self.folds_dates is None:
			raise ValueError("Need to split before getting the split")
		
		for key, df in zip(self.keys, self.datasets):
			ts_lower, ts_upper = self.folds_dates[test_fold]
			df_pre_test = df[df.index < ts_lower]
			df_post_test = df[df.index > ts_upper] if not seq_path else pd.DataFrame(columns=df.columns)  # Empty if sequential
			
			x_train_pre, y_train_pre = self._slice_segment(df_pre_test, burn_fraction, min_burn_points)
			x_train_post, y_train_post = self._slice_segment(df_post_test, burn_fraction, min_burn_points)
			
			# Concatenate pre and post segments if non-sequential
			x_train = np.vstack([x_train_pre, x_train_post]) if x_train_pre.size and x_train_post.size else x_train_pre if x_train_pre.size else x_train_post
			y_train = np.vstack([y_train_pre, y_train_post]) if y_train_pre.size and y_train_post.size else y_train_pre if y_train_pre.size else y_train_post
			
			df_test = df[(df.index >= ts_lower) & (df.index <= ts_upper)]
			x_test, y_test = df_test.iloc[:, df.columns.str.startswith('x')].values, df_test.iloc[:, df.columns.str.startswith(self.target_prefix)].values

			if x_train.shape[0]!=0 and x_test.shape[0]!=0:			

				elements.add(
							Element(
									x_train, 
									y_train, 
									x_test, 
									y_test, 
									df_test.index, 
									self.x_transform, 
									self.y_transform, 
									key, 
									self.x_cols, 
									self.y_cols
									)
							)
		
		return elements

	def _slice_segment(self, df, burn_fraction, min_burn_points):
		if df.empty:
			return np.array([]), np.array([])		
		x = df.iloc[:, df.columns.str.startswith(self.feature_prefix)].values
		y = df.iloc[:, df.columns.str.startswith(self.target_prefix)].values
		idx = np.arange(x.shape[0])
		idx = self._random_subsequence(idx, burn_fraction, min_burn_points)
		return x[idx], y[idx]
	
	@staticmethod
	def _random_subsequence(ar, burn_fraction, min_burn_points):
		a, b = np.random.randint(max(min_burn_points, 1), max(int(ar.size * burn_fraction), min_burn_points + 1), size=2)
		return ar[a:-b]
	
	def view(self):
		for key, df in zip(self.keys, self.datasets):
			print(f'Dataset: {key}\n{df}\n')
		if self.folds_dates:
			print('Folds Dates:')
			for start, end in self.folds_dates:
				print(f'-> from {start} to {end}')

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
