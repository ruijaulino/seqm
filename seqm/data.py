import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy
import tqdm
try:
	from .constants import *
except ImportError:
	from constants import *
	

def add_unix_timestamp(df:pd.DataFrame):
	# create a column with unix timestamp
	# add dates are dealt as integers
	if 'ts' not in df.columns: df['ts'] = df.index.astype(np.int64) // 10 ** 9
	return df

def unix_timestamp_to_index(df:pd.DataFrame):
	if 'ts' in df.columns: df.index = pd.to_datetime( data['ts'] * 10 ** 9 )
	return df

# container to work with arrays only
# derived from a dataframe
class Data:
	def __init__(self, df:pd.DataFrame, copy_df = True):
		# set input
		self.df = copy.deepcopy(df) if copy_df else df
		# add unix timestamp
		self.add_unix_timestamp()
		# get x array		
		self.x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
		if self.x.shape[1] == 0: self.x = None
		# get z
		self.z = df.iloc[:, [e==STATE_COL for e in df.columns.tolist()]].values
		if self.z.shape[1] == 0: self.z = None
		# get y
		self.y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
		assert self.y.shape[1] != 0, "no target columns"		
		self.ts = df['ts'].values
		self.x_cols, self.y_cols = get_df_cols(df)			

	@classmethod
	def from_arrays(cls, diameter):
		return cls(radius=diameter / 2)

		
	def add_unix_timestamp(self):
		# create a column with unix timestamp
		# add dates are dealt as integers
		if 'ts' not in df.columns: self.df['ts'] = self.df.index.astype(np.int64) // 10 ** 9

	@property
	def has_x(self):
		return self.x is not None

	@property
	def has_z(self):
		return self.z is not None

	def slice(self, burn_fraction, min_burn_points):
		idx = np.arange(self.y.shape[0])
		idx = self._random_subsequence(idx, burn_fraction, min_burn_points)
		self.y = self.y[idx]
		if self.has_x: self.x = self.x[idx]
		if self.has_z: self.z = self.z[idx]
	return self

	@staticmethod
	def _random_subsequence(ar, burn_fraction, min_burn_points):
		a, b = np.random.randint(max(min_burn_points, 1), max(int(ar.size * burn_fraction), min_burn_points + 1), size=2)
		return ar[a:-b]