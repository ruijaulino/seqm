import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy
import tqdm
try:
	from .constants import *
	from .generators import linear
except ImportError:
	from constants import *
	from generators import linear
	
def add_unix_timestamp(df:pd.DataFrame):
	# create a column with unix timestamp
	# add dates are dealt as integers
	if 'ts' not in df.columns: df['ts'] = df.index.view(np.int64) // 10 ** 9
	return df

def unix_timestamp_to_index(df:pd.DataFrame):
	if 'ts' in df.columns: df.index = pd.to_datetime( data['ts'] * 10 ** 9 )
	return df

def get_df_cols(df):
	cols = df.columns.tolist()
	x_cols=[c for c in cols if c.startswith(FEATURE_PREFIX)]		
	y_cols=[c for c in cols if c.startswith(TARGET_PREFIX)]
	return x_cols, y_cols

# container to work with arrays only
# can derived from a dataframe
# provides a good analogy if the code is to be made in other language without dataframes
class Data:
	def __init__(self, y, x, z, idx, ts, x_cols, y_cols, safe_arrays = True):
		# add unix timestamp
		# self.empty = True if y is None else False
		self.empty = True if y is None else True if y.shape[0] == 0 else False
		self.y = None if y is None else np.copy(y) if safe_arrays else y
		self.x = None if x is None else np.copy(x) if safe_arrays else x
		self.z = None if z is None else np.copy(z) if safe_arrays else z
		self.idx = None if idx is None else np.copy(idx) if safe_arrays else idx
		self.ts = None if ts is None else np.copy(ts) if safe_arrays else ts
		self.x_cols,self.y_cols = x_cols,y_cols
		self.has_x = False if self.x is None else True
		self.has_z = False if self.z is None else True
		
		if self.idx is None: self.idx = np.zeros(self.y.shape[0],dtype=int)
		self.fix_idx()

	def converted_idx(self):
		'''
		idx: numpy (n,) array like 0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,...
			with the indication where the subsequences are
			creates an array like
			[
			[0,3],
			[3,7],
			...
			]
			with the indexes where the subsequences start and end
		'''
		aux=self.idx[1:]-self.idx[:-1]
		aux=np.where(aux!=0)[0]
		aux+=1
		aux_left=np.hstack(([0],aux))
		aux_right=np.hstack((aux,[self.idx.size]))
		out=np.hstack((aux_left[:,None],aux_right[:,None]))
		return out


	def fix_idx(self):
		# idx need to be an array like 000,1111,2222,33,444
		# but can be something like 111,000,11,00000,1111,22
		# i.e, the indexes of the sequence need to be in order for the
		# cvbt to work. Fix the array in case this is not verified
		# this is a bit of overhead but has to be this way
		idx=np.zeros(self.idx.size,dtype=int)
		aux=self.idx[1:]-self.idx[:-1]
		idx[np.where(aux!=0)[0]+1]=1
		self.idx=np.cumsum(idx)

	def get_ts_as_timestamp(self):
		return pd.to_datetime( self.ts * 10 ** 9 )

	def view(self,k=3):
		print('** Data **')
		if self.has_x:
			print(np.hstack((self.ts[:,None],self.y,self.x)))
		else:
			print(np.hstack((self.ts[:,None],self.y)))
		print('**********')

	@classmethod
	def from_df(cls, df: pd.DataFrame, safe_arrays = True):
 		# parse the dataframe into the inputs
		df = df.copy(deep = True)
		# add unix timestamp
		df = add_unix_timestamp(df)
		# get x array		
		x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
		if x.shape[1] == 0: x = None
		# get z
		z = df.iloc[:, [e==STATE_COL for e in df.columns.tolist()]].values
		if z.shape[1] == 0: z = None
		# get y
		y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
		assert y.shape[1] != 0, "no target columns"		
		ts = df['ts'].values
		idx = df.iloc[:, [e==MULTISEQ_IDX_COL for e in df.columns.tolist()]].values
		idx = None if idx.shape[1] == 0 else np.array(idx[:,0],dtype=int)
		x_cols, y_cols = get_df_cols(df)		
		return cls(y, x, z, idx, ts, x_cols, y_cols, safe_arrays)

	def _from_idx(self, idx:np.ndarray, create_new = False):		
		y_ = self.y[idx]
		ts_ = self.ts[idx]
		idx_ = self.idx[idx]
		x_ = self.x[idx] if self.has_x else None
		z_ = self.z[idx] if self.has_z else None 
		if create_new:
			return Data(y_, x_, z_, idx_, ts_, self.x_cols, self.y_cols, safe_arrays=True)
		else:
			self.y = y_
			self.x = x_
			self.ts = ts_
			self.z = z_
			self.idx = idx_
			return self		

	@staticmethod
	def _random_subsequence(ar, burn_fraction, min_burn_points):
		a, b = np.random.randint(max(min_burn_points, 1), max(int(ar.size * burn_fraction), min_burn_points + 1), size=2)
		return ar[a:-b]

	def random_segment(self, burn_fraction, min_burn_points, create_new = False):
		'''
		'''
		if self.empty: return self
		idx = np.arange(self.y.shape[0])
		idx = self._random_subsequence(idx, burn_fraction, min_burn_points)		
		return self._from_idx(idx, create_new)

	def after(self, ts:int, create_new = True):
		if self.empty: return self
		idx = np.where(self.ts > ts)[0]
		return self._from_idx(idx, create_new)

	def before(self, ts:int, create_new = True):
		if self.empty: return self
		idx = np.where(self.ts < ts)[0]
		return self._from_idx(idx, create_new)

	def between(self,ts_lower, ts_upper, create_new = True):
		if self.empty: return self
		idx = np.where( (self.ts<=ts_upper) & (self.ts>=ts_lower) )[0]
		return self._from_idx(idx, create_new)

	def _copy(self, data:'Data'):
		self.empty = data.empty
		self.y = data.y
		self.x = data.x
		self.z = data.z
		self.idx = data.idx
		self.ts = data.ts
		self.x_cols = data.x_cols
		self.y_cols = data.y_cols
		self.has_x = data.has_x
		self.has_z = data.has_z

	def stack(self, data:'Data', allow_both_empty = False):
		# make sure both are not empty
		if self.empty and data.empty:
			if allow_both_empty: 		
				return self
			else:
				raise Exception('both Data are empty. Cannot stack.')
		if self.empty and not data.empty:
			self._copy(data)
			return self
		if not self.empty and data.empty:
			return self
		# check if columns match
		self.check_cols(data)
		# ts is not stacked as this is used to concat training data
		self.y = np.vstack((self.y,data.y))		
		self.ts = np.hstack((self.ts,data.ts))
		if self.has_x: self.x = np.vstack((self.x,data.x))
		if self.has_z: self.z = np.vstack((self.z,data.z))
		self.idx = np.hstack((self.idx,data.idx+self.idx[-1]-data.idx[0]+1))
		return self		

	def check_cols(self, data:'Data'):
		assert self.x_cols == data.x_cols,"x_cols are different"
		assert self.y_cols == data.y_cols,"x_cols are different"
		assert self.has_z == data.has_z,"z is different"
		return self

	def build_train_inputs(self):
		'''
		build train inputs for models		
		'''
		return {'x' : self.x, 'y' : self.y, 'z' : self.z, 'idx':self.converted_idx()} 

if __name__ == '__main__':
	
	df = linear(n=10,a=0,b=0.1,start_date='2000-01-01')
	data = Data.from_df(df)
	
	ts = 947289600 # 947462400
	data.view()

	data.before(ts).view()

	tmp=data.after(947462400)# 
	print(tmp.y)
	print(tmp.empty)
	tmp.view()

	# data.view()
	# data.random_segment(0.1,3)
	# data.view()