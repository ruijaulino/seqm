# container for arrays
# generalize to more variables as needed, better to have a container for the arrays
import numpy as np
import pandas as pd
import copy

try:
	from .constants import *
except ImportError:	
	from constants import *

def get_df_cols(df):
	cols = df.columns.tolist()
	x_cols=[c for c in cols if c.startswith(FEATURE_PREFIX)]		
	y_cols=[c for c in cols if c.startswith(TARGET_PREFIX)]
	return x_cols, y_cols

class Arrays:
	def __init__(self, y = None, x = None, ts = None, x_cols = None, y_cols = None):
		self.y = y
		self.x = x
		self.ts = ts
		self.x_cols = x_cols
		self.y_cols = y_cols
		self.empty = False
		if self.y is None: self.empty = True

	@property
	def has_x(self):
		return self.x is not None

	def view(self):
		print('** Arrays **')
		if self.empty:
			print('empty')
		else:
			print('columns: ', self.x_cols, self.y_cols)
			print('y')
			print(self.y[:5])
			print('x')
			print(self.x[:5])

	def from_df(self, df, add_ts = False):
		if df.empty:
			self.empty = True
			return self
		self.empty = False
		self.x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
		if self.x.shape[1] == 0: self.x = None
		self.y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
		assert self.y.shape[1] != 0, "no target columns"
		if add_ts: self.ts = df.index
		self.x_cols, self.y_cols = get_df_cols(df)	
		return self

	def slice(self, burn_fraction, min_burn_points):
		if not self.empty:
			idx = np.arange(self.y.shape[0])
			idx = self._random_subsequence(idx, burn_fraction, min_burn_points)
			self.y = self.y[idx]
			if self.x is not None: self.x = self.x[idx]
		return self

	@staticmethod
	def _random_subsequence(ar, burn_fraction, min_burn_points):
		a, b = np.random.randint(max(min_burn_points, 1), max(int(ar.size * burn_fraction), min_burn_points + 1), size=2)
		return ar[a:-b]

	def _copy(self, arrays:'Arrays'):
		self.y = arrays.y
		self.x = arrays.x
		self.ts = arrays.ts
		self.x_cols = arrays.x_cols
		self.y_cols = arrays.y_cols
		self.empty = arrays.empty

	def stack(self, arrays:'Arrays'):
		# make sure both are not empty
		if self.empty and arrays.empty: raise Exception('both Arrays are empty. Cannot stack.')
		if self.empty and not arrays.empty:
			self._copy(arrays)
			return self
		if not self.empty and arrays.empty:
			return self
		# check if columns match
		self.check_cols(arrays)
		# ts is not stacked as this is used to concat training data
		self.y = np.vstack((self.y,arrays.y))		
		if self.x is not None:
			self.x = np.vstack((self.x,arrays.x))
		return self		

	def check_cols(self,arrays:'Arrays'):
		assert self.x_cols == arrays.x_cols,"x_cols are different"
		assert self.y_cols == arrays.y_cols,"x_cols are different"
		return self