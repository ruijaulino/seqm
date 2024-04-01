
import numpy as np
import pandas as pd
from typing import List, Union
import copy

class DummyNormalizer:
	def __init__(self):
		pass
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

class DataManager:

	# class to store a split of the data
	class Split:
		def __init__(self, x_train, y_train, x_test, y_test, ts, x_normalizer, y_normalizer, name=''):
			self.x_train = x_train
			self.y_train = y_train
			self.x_test = x_test
			self.y_test = y_test
			self.ts = ts
			self.x_normalizer = x_normalizer
			self.y_normalizer = y_normalizer
			self.name = name
			
		def view(self):
			print(f"Dataset: {self.name}")
			print("Train")
			print(self.x_train)
			print(self.y_train)
			print("Test")
			print(self.x_test)
			print(self.y_test)
			print(self.ts)
			print('')

	def __init__(self, data: Union[pd.DataFrame, List[pd.DataFrame]], names: List[str] = None, normalizer_class=DummyNormalizer):
		super().__init__()
		self._verify_input_data(data)
		self.data = data if isinstance(data, list) else [data]
		self.names = names if names is not None else [f'Dataset {i+1}' for i in range(len(self.data))]
		self.normalize = len(self.data) > 1
		self.folds_dates = None
		self.splits = {}  # Store splits by name
		self.normalizer_class = normalizer_class if normalizer_class is not None else DummyNormalizer  # Store the class, not an instance

	def join_splits_to_train(self):
		x=np.vstack([v.x_train for _,v in self.splits.items()])
		y=np.vstack([v.y_train for _,v in self.splits.items()])
		return {'x':x,'y':y}

	def _verify_input_data(self, data):
		if isinstance(data, list):
			cols = data[0].columns.tolist()
			for df in data:
				if df.columns.tolist() != cols:
					raise ValueError("All DataFrames must have the same columns.")
	
	def split(self, k_folds=3):
		ts = pd.DatetimeIndex([])
		for df in self.data:
			ts = ts.union(df.index.unique())
		ts = ts.sort_values()
		idx_folds = np.array_split(ts, k_folds)
		self.folds_dates = [(fold[0], fold[-1]) for fold in idx_folds]
		return self

	def get_split(self, test_fold: int, burn_fraction: float = 0.1, min_burn_points: int = 1, seq_path: bool = False):
		
		self.splits = {}  # Store splits by name

		if seq_path and test_fold == 0:
			raise ValueError("Cannot start at fold 0 when path is sequential")
		if self.folds_dates is None:
			raise ValueError("Need to split before getting the split")
		
		for name, df in zip(self.names, self.data):
			ts_lower, ts_upper = self.folds_dates[test_fold]
			df_pre_test = df[df.index < ts_lower]
			df_post_test = df[df.index > ts_upper] if not seq_path else pd.DataFrame(columns=df.columns)  # Empty if sequential
			
			x_train_pre, y_train_pre = self._process_segment(df_pre_test, burn_fraction, min_burn_points)
			x_train_post, y_train_post = self._process_segment(df_post_test, burn_fraction, min_burn_points)
			
			# Concatenate pre and post segments if non-sequential
			x_train = np.vstack([x_train_pre, x_train_post]) if x_train_pre.size and x_train_post.size else x_train_pre if x_train_pre.size else x_train_post
			y_train = np.vstack([y_train_pre, y_train_post]) if y_train_pre.size and y_train_post.size else y_train_pre if y_train_pre.size else y_train_post
			
			df_test = df[(df.index >= ts_lower) & (df.index <= ts_upper)]
			x_test, y_test = df_test.iloc[:, df.columns.str.startswith('x')].values, df_test.iloc[:, df.columns.str.startswith('y')].values

			if x_train.shape[0]!=0 and x_test.shape[0]!=0:			
				# Instantiate dedicated normalizers for this split
				x_normalizer = self.normalizer_class()
				y_normalizer = self.normalizer_class()
				# Fit and transform training data
				x_train, y_train = self._fit_transform(x_train, y_train, x_normalizer, y_normalizer)
				# Transform test data
				x_test, y_test = self._transform(x_test, x_normalizer, y_test, None)  # Assuming y_test does not need normalization or is not present

				self.splits[name] = self.Split(x_train, y_train, x_test, y_test, df_test.index, x_normalizer, y_normalizer, name)

		return self.splits

	def _fit_transform(self, x_train, y_train, x_normalizer, y_normalizer):
		"""Fit and transform the training data using dedicated normalizers."""
		x_normalizer.fit(x_train)
		x_train_transformed = x_normalizer.transform(x_train)
		y_normalizer.fit(y_train)
		y_train_transformed = y_normalizer.transform(y_train)

		return x_train_transformed, y_train_transformed

	def _transform(self, x_test, x_normalizer, y_test, y_normalizer):
		"""Transform the test data using the provided normalizers."""
		x_test_transformed = x_normalizer.transform(x_test)
		y_test_transformed = y_normalizer.transform(y_test) if y_normalizer is not None else y_test

		return x_test_transformed, y_test_transformed

	def _process_segment(self, df, burn_fraction, min_burn_points):
		if df.empty:
			return np.array([]), np.array([])		
		x = df.iloc[:, df.columns.str.startswith('x')].values
		y = df.iloc[:, df.columns.str.startswith('y')].values
		idx = np.arange(x.shape[0])
		idx = self._random_subsequence(idx, burn_fraction, min_burn_points)
		return x[idx], y[idx]
	
	@staticmethod
	def _random_subsequence(ar, burn_fraction, min_burn_points):
		a, b = np.random.randint(max(min_burn_points, 1), max(int(ar.size * burn_fraction), min_burn_points + 1), size=2)
		return ar[a:-b]
	
	def view(self):
		for name, df in zip(self.names, self.data):
			print(f'Dataset: {name}\n{df}\n')
		if self.folds_dates:
			print('Folds Dates:')
			for start, end in self.folds_dates:
				print(f'-> from {start} to {end}')

	# Example method to view all splits
	def view_splits(self):
		for name, split in self.splits.items():
			split.view()				

if __name__=='__main__':
	# create some random data for input
	n=20
	index=pd.date_range('2000-01-01',periods=n,freq='D')
	values=np.random.normal(0,1,(n,3))
	columns=['x1','x2','y1']
	df1=pd.DataFrame(values,index=index,columns=columns)

	index=pd.date_range('2000-01-10',periods=n,freq='D')
	values=np.random.normal(0,1,(n,3))
	columns=['x1','x2','y1']
	df2=pd.DataFrame(values,index=index,columns=columns)

	
	dm=DataManager([df1,df2],normalizer_class=Normalizer)
	dm.split(3)
	dm.view()
	test_fold=1
	splits=dm.get_split(test_fold, burn_fraction = 0.1, min_burn_points = 1, seq_path = False)
	for k,v in splits.items():
		v.view()
