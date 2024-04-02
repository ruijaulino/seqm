
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
import copy

try:
	from .models import ConditionalGaussian
except ImportError:
	from models import ConditionalGaussian

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

# base class that contains a data element in numpy array form
# models can be applied to it

class Element:
	def __init__(self, x_train, y_train, x_test, y_test, ts=None, normalizer_class=DummyNormalizer, name='', x_cols=None, y_cols=None):
		self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
		self.ts = pd.date_range('1950-01-01', freq='D', periods=self.x_test.shape[0]) if ts is None else ts
		self.name = name or 'Dataset 1'
		self.x_cols = x_cols or [f'x_{i+1}' for i in range(self.x_train.shape[1])]
		self.y_cols = y_cols or [f'y_{i+1}' for i in range(self.y_train.shape[1])]

		assert self.x_train.shape[0] == self.y_train.shape[0], "x_train and y_train must have the same number of observations"
		assert self.x_test.shape[1] == self.x_train.shape[1], "x_train and x_test must have the same number of variables"
		assert self.y_test.shape[1] == self.y_train.shape[1], "y_train and y_test must have the same number of variables"		

		self.normalizer_class = normalizer_class if normalizer_class is not None else DummyNormalizer  # Store the class, not an instance
		self.model = None  # Placeholder for a trained model
		# fit transform at instantiation
		self.x_normalizer,self.y_normalizer=None,None		
		self.s,self.w,self.pw=None,None,None
		
		self._verify_input_data()
		self._fit_transform()

	def get_s(self):
		return pd.DataFrame(self.s,columns=['s'],index=self.ts)

	def get_w(self):
		return pd.DataFrame(self.w,columns=[c+"_w" for c in self.y_cols],index=self.ts)

	def get_pw(self):
		return pd.DataFrame(self.pw,columns=['pw'],index=self.ts)

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
		self.x_normalizer = self.normalizer_class()
		self.y_normalizer = self.normalizer_class()
		self.x_normalizer.fit(self.x_train)
		self.x_train = self.x_normalizer.transform(self.x_train)
		self.y_normalizer.fit(self.y_train)
		self.y_train = self.y_normalizer.transform(self.y_train)

	def estimate(self, model, estimate_params=None):
		"""Train the model using the training data contained within this DataElement."""
		estimate_params = estimate_params or {}
		self.model=copy.deepcopy(model)  		
		self.model.train(x=self.x_train, y=self.y_train, **estimate_params)

	def evaluate(self):
		"""Evaluate the model using the test data and return performance metrics."""
		if self.model is None:
			raise Exception("Model has not been trained.")		
		n = self.y_test.shape[0]
		p = self.y_test.shape[1]
		self.s = np.zeros(n, dtype=np.float64)
		self.w = np.zeros((n, p), dtype=np.float64)
		self.pw = self.y_normalizer.p_scale*np.ones(n, dtype=np.float64)
		
		for i in range(n):
			# normalize y for input (make copy of y)
			y_normalized = self.y_normalizer.transform(np.array(self.y_test[:i]))
			w = self.model.get_weight(**{'y': y_normalized, 'x': self.x_test[:i], 'xq': self.x_test[i]})
			self.w[i] = w
			self.s[i] = np.dot(self.y_test[i], w)

class Elements:
	def __init__(self):
		self.element=[]

	@property
	def names(self):
		return [e.name for e in self]
	
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

	def estimate(self,model,single_model=False):
		if single_model:
			model=copy.deepcopy(model)
			x=np.vstack([e.x_train for e in self])
			y=np.vstack([e.y_train for e in self])
			model.estimate(**{'x':x,'y':y})
			# set same model for all element
			for e in self: e.set_model(model)
		else:
			for e in self: e.estimate(model)
		return self

	def evaluate(self):
		for e in self: e.evaluate()
		return self


class Data:

	def __init__(self, datasets: Union[pd.DataFrame, List[pd.DataFrame]], names: List[str] = None,  normalizer_class=DummyNormalizer, target_prefix='y', feature_prefix='x'):
		super().__init__()
		self.datasets = datasets if isinstance(datasets, list) else [datasets]
		self.target_prefix = target_prefix
		self.feature_prefix = feature_prefix		
		self.y_cols,self.x_cols=None,None
		self._verify_input_data(datasets)
		self.names = names if names is not None else [f'Dataset {i+1}' for i in range(len(self.datasets))]
		self.normalize = len(self.datasets) > 1
		self.folds_dates = None
		self.data_elements = {}  # Store splits by name
		self.normalizer_class = normalizer_class if normalizer_class is not None else DummyNormalizer  # Store the class, not an instance

	def _verify_input_data(self, datasets):
		if isinstance(datasets, list):
			cols = datasets[0].columns.tolist()
			for df in datasets:
				if df.columns.tolist() != cols:
					raise ValueError("All DataFrames must have the same columns.")
			self.y_cols=[c for c in cols if c.startswith(self.target_prefix)]
			self.x_cols=[c for c in cols if c.startswith(self.feature_prefix)]
	
	def split_dates(self, k_folds=3):
		self.data_elements = {}  # Store splits by name
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
		
		for name, df in zip(self.names, self.datasets):
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

				elements.add(Element(x_train, y_train, x_test, y_test, df_test.index, self.normalizer_class, name, self.x_cols, self.y_cols))
		
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
		for name, df in zip(self.names, self.datasets):
			print(f'Dataset: {name}\n{df}\n')
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
	def names(self):
		tmp=[]
		for es in self.elements:
			tmp+=es.names
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
		for name in self.names:
			self.s.update({name:[]})
			self.w.update({name:[]})
			self.pw.update({name:[]})
		# iterate elements
		for es in self:
			tmp={}
			# iterate series in element
			for e in es:
				if e.s is not None:
					# get performance dataframe
					self.s[e.name].append(e.get_s())
					# get weight dataframe
					self.w[e.name].append(e.get_w())
					# get portfolio weight
					self.pw[e.name].append(e.get_pw())
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
		for name in self.names:			
			self.results.update({name:pd.concat([self.s.get(name),self.w.get(name),self.pw.get(name)],axis=1)})
		if len(self.names)==1:
			self.results=self.results.get(self.names[0])
		return self.results

class ModelEvaluation:
	def __init__(self, data: Data):
		self.data = data

	def train(self):
		pass

	def test(self):
		pass

	def live(self):
		pass

	def cvbt(self, model, k_folds=4, seq_path=False, start_fold=0, n_paths=4, burn_fraction=0.1, min_burn_points=3, single_model=True):
		'''
		each path generates a dict like
		{dataset1:df1,dataset2,df2,..}
		where each df contains the columns [s,y1_w,y2_w,...,pw] at the same dates as the original dataset
		the output is a list of this dicts
		'''
		# Prepare the data
		self.data.split_dates(k_folds)		
		start_fold = max(1, start_fold) if seq_path else start_fold		
		paths = []
		for m in range(n_paths):
			path=Path()
			for fold_index in range(start_fold, k_folds):			
				elements = self.data.get_split_elements(
					test_fold=fold_index,
					burn_fraction=burn_fraction,
					min_burn_points=min_burn_points,
					seq_path=seq_path
				)
				elements.estimate(model,single_model=single_model).evaluate()
				path.add(elements)
			paths.append(path.get_results())
		return paths


if __name__=='__main__':
	# generate data
	def generate_lr(n=1000,a=0,b=0.1,start_date='2000-01-01'):
		x=np.random.normal(0,0.01,n)
		a=0
		b=0.1
		y=a+b*x+np.random.normal(0,0.01,n)
		dates=pd.date_range(start_date,periods=n,freq='B')
		data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
		return data
	data1=generate_lr(n=1000,a=0,b=0.1,start_date='2000-01-01')
	data2=generate_lr(n=700,a=0,b=0.1,start_date='2000-06-01')
	data3=generate_lr(n=1500,a=0,b=0.1,start_date='2001-01-01')
	model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=100)
	data=[data1,data2,data3]
	data=Data(data,normalizer_class=None)
	
	model_eval=ModelEvaluation(data)
	paths=model_eval.cvbt(model, k_folds=4, seq_path=False, start_fold=0, n_paths=4, burn_fraction=0.1, min_burn_points=3, single_model=True)
	print(len(paths))
	print(paths[0])
	# paths.post_process(seq_fees=False,pct_fee=0,sr_mult=1,n_boot=1000,name=None)
	# NOW IMPLEMENT POST PROCESS ON A LIST OF DICTS
