
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy

try:
	from .elements import Element,Elements,Path
	from .transform import BaseTransform,IdleTransform
	from .constants import *
except ImportError:
	from elements import Element,Elements,Path
	from transform import BaseTransform,IdleTransform
	from constants import *


# dict of Dataframes
class Dataset:
	def __init__(self, dataset = {}):
		self.dataset = dataset
		self.folds_dates = []

	# methods to behave like dict
	def add(self, key, item: pd.DataFrame):
		if not isinstance(item, ModelWrapper):
			raise TypeError("Item must be an instance of pd.DataFrame")
		self.dataset[key] = copy.deepcopy(item)

	def __getitem__(self, key):
		return self.dataset[key]

	def __setitem__(self, key, item: pd.DataFrame):
		if not isinstance(item, pd.DataFrame):
			raise TypeError("Item must be an instance of pd.DataFrame")
		self.dataset[key] = copy.deepcopy(item)

	def __len__(self):
		return len(self.dataset)

	def __iter__(self):
		return iter(self.dataset)

	def keys(self):
		return self.dataset.keys()

	def values(self):
		return self.dataset.values()

	def items(self):
		return self.dataset.items()

	def has_key(self,key:str):
		return key in self.keys()

	# specific methods

	# split dates into folds for posterior use
	def split_dates(self, k_folds=3):
		ts = pd.DatetimeIndex([])
		for k,df in self.dataset:
			ts = ts.union(df.index.unique())
		ts = ts.sort_values()
		idx_folds = np.array_split(ts, k_folds)
		self.folds_dates = [(fold[0], fold[-1]) for fold in idx_folds]
		return self

	def get_train(self) -> Elements:
		'''
		Get all data as training data
		'''
		elements = Elements()
		for key, df in self.items():
			# get arrays
			x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
			y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
			x_cols,y_cols=self.get_xy_cols(df)
			elements[key]=Element(
								x, 
								y, 
								x, 
								y, 
								df.index, 
								x_cols, 
								y_cols,
								key, 
								)							
		return elements


	def _slice_segment(self, df, burn_fraction, min_burn_points):
		if df.empty:
			return np.array([]), np.array([])		
		x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
		y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
		idx = np.arange(x.shape[0])
		idx = self._random_subsequence(idx, burn_fraction, min_burn_points)
		return x[idx], y[idx]

	@staticmethod
	def get_xy_cols(df):
		cols = df.columns.tolist()
		x_cols=[c for c in cols if c.startswith(FEATURE_PREFIX)]		
		y_cols=[c for c in cols if c.startswith(TARGET_PREFIX)]
		return x_cols, y_cols
	
	@staticmethod
	def _random_subsequence(ar, burn_fraction, min_burn_points):
		a, b = np.random.randint(max(min_burn_points, 1), max(int(ar.size * burn_fraction), min_burn_points + 1), size=2)
		return ar[a:-b]
	
	def get_train_test_elements(self, test_index: int, burn_fraction: float = 0.1, min_burn_points: int = 1, seq_path: bool = False) -> Elements:
		
		elements = Elements()  # Store splits by name

		if seq_path and test_index == 0:
			raise ValueError("Cannot start at fold 0 when path is sequential")
		if self.folds_dates is None:
			raise ValueError("Need to split before getting the split")
		
		for key, df in self.items():
			ts_lower, ts_upper = self.folds_dates[test_index]
			df_pre_test = df[df.index < ts_lower]
			df_post_test = df[df.index > ts_upper] if not seq_path else pd.DataFrame(columns=df.columns)  # Empty if sequential
			
			x_train_pre, y_train_pre = self._slice_segment(df_pre_test, burn_fraction, min_burn_points)
			x_train_post, y_train_post = self._slice_segment(df_post_test, burn_fraction, min_burn_points)
			
			# Concatenate pre and post segments if non-sequential
			x_train = np.vstack([x_train_pre, x_train_post]) if x_train_pre.size and x_train_post.size else x_train_pre if x_train_pre.size else x_train_post
			y_train = np.vstack([y_train_pre, y_train_post]) if y_train_pre.size and y_train_post.size else y_train_pre if y_train_pre.size else y_train_post
			
			df_test = df[(df.index >= ts_lower) & (df.index <= ts_upper)]
			x_test, y_test = df_test.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values, df_test.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values

			x_cols,y_cols=self.get_xy_cols(df)

			if x_train.shape[0]!=0 and x_test.shape[0]!=0:			

				elements[key]=Element(
									x_train, 
									y_train, 
									x_test, 
									y_test, 
									df_test.index, 
									x_cols, 
									y_cols,
									key, 
									)
		return elements









# handles a dict of dataframes with data to train/test models
class Dataset_:

	def __init__(self, datasets: Union[pd.DataFrame, Dict[str,pd.DataFrame]] ):
		super().__init__()
		self.folds_dates = []
		self._verify_inputs(datasets)

	@staticmethod
	def get_xy_cols(df):
		cols = df.columns.tolist()
		x_cols=[c for c in cols if c.startswith(FEATURE_PREFIX)]		
		y_cols=[c for c in cols if c.startswith(TARGET_PREFIX)]
		return x_cols, y_cols

	def _verify_inputs(self, datasets):
		if isinstance(datasets, dict):
			self.keys=[]
			self.datasets=[]
			for k,v in datasets.items():
				self.keys.append(k)
				self.datasets.append(v)
		else:
			self.datasets = [datasets] 
			self.keys=['Dataset']
		# cols = self.datasets[0].columns.tolist()
		# for df in self.datasets:
		#	if df.columns.tolist() != cols:
		#		raise ValueError("All DataFrames must have the same columns.")
		# self.y_cols=[c for c in cols if c.startswith(TARGET_PREFIX)]
		# self.x_cols=[c for c in cols if c.startswith(FEATURE_PREFIX)]
	
	# split dates into folds for posterior use
	def split_dates(self, k_folds=3):
		ts = pd.DatetimeIndex([])
		for df in self.datasets:
			ts = ts.union(df.index.unique())
		ts = ts.sort_values()
		idx_folds = np.array_split(ts, k_folds)
		self.folds_dates = [(fold[0], fold[-1]) for fold in idx_folds]
		return self

	def get_train(self) -> Elements:
		'''
		Get all data as training data
		'''
		elements = Elements()
		for key, df in zip(self.keys, self.datasets):
			# get arrays
			x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
			y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
			x_cols,y_cols=self.get_xy_cols(df)
			elements[key]=Element(
								x, 
								y, 
								x, 
								y, 
								df.index, 
								x_cols, 
								y_cols,
								key, 
								)							
		return elements

	def _slice_segment(self, df, burn_fraction, min_burn_points):
		if df.empty:
			return np.array([]), np.array([])		
		x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
		y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
		idx = np.arange(x.shape[0])
		idx = self._random_subsequence(idx, burn_fraction, min_burn_points)
		return x[idx], y[idx]
	
	@staticmethod
	def _random_subsequence(ar, burn_fraction, min_burn_points):
		a, b = np.random.randint(max(min_burn_points, 1), max(int(ar.size * burn_fraction), min_burn_points + 1), size=2)
		return ar[a:-b]
	
	def get_train_test_elements(self, test_index: int, burn_fraction: float = 0.1, min_burn_points: int = 1, seq_path: bool = False) -> Elements:
		
		elements = Elements()  # Store splits by name

		if seq_path and test_index == 0:
			raise ValueError("Cannot start at fold 0 when path is sequential")
		if self.folds_dates is None:
			raise ValueError("Need to split before getting the split")
		
		for key, df in zip(self.keys, self.datasets):
			ts_lower, ts_upper = self.folds_dates[test_index]
			df_pre_test = df[df.index < ts_lower]
			df_post_test = df[df.index > ts_upper] if not seq_path else pd.DataFrame(columns=df.columns)  # Empty if sequential
			
			x_train_pre, y_train_pre = self._slice_segment(df_pre_test, burn_fraction, min_burn_points)
			x_train_post, y_train_post = self._slice_segment(df_post_test, burn_fraction, min_burn_points)
			
			# Concatenate pre and post segments if non-sequential
			x_train = np.vstack([x_train_pre, x_train_post]) if x_train_pre.size and x_train_post.size else x_train_pre if x_train_pre.size else x_train_post
			y_train = np.vstack([y_train_pre, y_train_post]) if y_train_pre.size and y_train_post.size else y_train_pre if y_train_pre.size else y_train_post
			
			df_test = df[(df.index >= ts_lower) & (df.index <= ts_upper)]
			x_test, y_test = df_test.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values, df_test.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values

			x_cols,y_cols=self.get_xy_cols(df)

			if x_train.shape[0]!=0 and x_test.shape[0]!=0:			

				elements[key]=Element(
									x_train, 
									y_train, 
									x_test, 
									y_test, 
									df_test.index, 
									x_cols, 
									y_cols,
									key, 
									)
							

				# elements.add(
				# 			Element(
				# 					x_train, 
				# 					y_train, 
				# 					x_test, 
				# 					y_test, 
				# 					df_test.index, 
				# 					key, 
				# 					self.x_cols, 
				# 					self.y_cols
				# 					)
				# 			)
		
		return elements

	def cvbt(self):
		pass

	def train(self):
		'''
		Train model on this dataset
		'''
		pass

	def test(self,models):
		'''
		Test models on this dataset
		'''		
		pass

	def live(self,models):
		'''
		Get a weight based on this dataset
		'''
		pass


if __name__=='__main__':
	pass