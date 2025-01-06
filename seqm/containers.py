import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy
import tqdm
try:
	from .constants import *
	from .model_pipe import ModelPipe
except ImportError:
	from constants import *
	from model_pipe import ModelPipe
	
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
	x_cols = [c for c in cols if c.startswith(FEATURE_PREFIX)]		
	y_cols = [c for c in cols if c.startswith(TARGET_PREFIX)]
	t_cols = [c for c in cols if c.startswith(NON_TRADEABLE_TARGET_COL)]
	return x_cols, t_cols, y_cols

# container to work with arrays only
# can derived from a dataframe
# provides a good analogy if the code is to be made in other language without dataframes
class Data:
	def __init__(self, y, x, z, idx, t, ts, x_cols, t_cols, y_cols, safe_arrays = True):
		# add unix timestamp
		# self.empty = True if y is None else False
		self.empty = True if y is None else True if y.shape[0] == 0 else False
		self.y = None if y is None else np.copy(y) if safe_arrays else y
		self.x = None if x is None else np.copy(x) if safe_arrays else x
		self.z = None if z is None else np.copy(z) if safe_arrays else z
		self.idx = None if idx is None else np.copy(idx) if safe_arrays else idx
		self.t = None if t is None else np.copy(t) if safe_arrays else t
		self.ts = None if ts is None else np.copy(ts) if safe_arrays else ts
		self.x_cols, self.t_cols, self.y_cols = x_cols, t_cols, y_cols
		self.has_x = False if self.x is None else True
		self.has_t = False if self.t is None else True
		self.has_z = False if self.z is None else True

		
		if self.idx is None: self.idx = np.zeros(self.y.shape[0],dtype=int)
		self.fix_idx()
		assert self.y.shape[0] == self.idx.size, "y and idx have different sizes"

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

	def view(self):
		print('** Data **')
		df = pd.DataFrame(self.y,index = self.get_ts_as_timestamp())
		print(df)
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
		# get e array
		t = df.iloc[:, df.columns.str.startswith(NON_TRADEABLE_TARGET_COL)].values
		if t.shape[1] == 0: t = None
		# get z
		z = df.iloc[:, [e==STATE_COL for e in df.columns.tolist()]].values
		if z.shape[1] == 0: z = None
		# get y
		y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
		assert y.shape[1] != 0, "no target columns"		
		ts = df['ts'].values
		idx = df.iloc[:, [e==MULTISEQ_IDX_COL for e in df.columns.tolist()]].values
		idx = None if idx.shape[1] == 0 else np.array(idx[:,0],dtype=int)
		x_cols, t_cols, y_cols = get_df_cols(df)		
		return cls(y, x, z, idx, t, ts, x_cols, t_cols, y_cols, safe_arrays)

	def _from_idx(self, idx:np.ndarray, create_new = False):		
		y_ = self.y[idx]
		ts_ = self.ts[idx]
		idx_ = self.idx[idx]
		x_ = self.x[idx] if self.has_x else None
		t_ = self.t[idx] if self.has_t else None
		z_ = self.z[idx] if self.has_z else None 
		if create_new:
			return Data(y_, x_, z_, idx_, t_, ts_, self.x_cols, self.t_cols, self.y_cols, safe_arrays=True)
		else:
			self.empty = True if y_ is None else True if y_.shape[0] == 0 else False
			self.y = y_
			self.x = x_
			self.t = t_
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
		self.t = data.t
		self.idx = data.idx
		self.ts = data.ts
		self.x_cols = data.x_cols
		self.t_cols = data.t_cols
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
		if self.has_t: self.t = np.vstack((self.t,data.t))
		if self.has_z: self.z = np.vstack((self.z,data.z))
		self.idx = np.hstack((self.idx,data.idx+self.idx[-1]-data.idx[0]+1))
		return self		

	def check_cols(self, data:'Data'):
		assert self.x_cols == data.x_cols,"x_cols are different"
		assert self.y_cols == data.y_cols,"x_cols are different"
		assert self.t_cols == data.t_cols,"t_cols are different"
		assert self.has_z == data.has_z,"z is different"
		return self

	def build_train_inputs(self):
		'''
		build train inputs for models		
		'''
		return {'x' : self.x, 'y' : self.y, 'z' : self.z, 'idx':self.converted_idx(), 't': self.t} 


# dict of Data with properties
class Dataset:
	def __init__(self, dataset = {}):
		self.dataset = copy.deepcopy(dataset)
		# convert dict of DataFrames to Data is necessary
		for k,v in self.items():
			if isinstance(v,pd.DataFrame):
				self[k] = Data.from_df(v)
		self.folds_ts = []
		
	# methods to behave like dict
	def add(self, key, item: Union[pd.DataFrame,Data]):
		if isinstance(item, pd.DataFrame):
			item = Data.from_df(item)
		else:
			if not isinstance(item, Data):			
				raise TypeError("Item must be an instance of pd.DataFrame or Data")
		self.dataset[key] = item

	def __getitem__(self, key):
		return self.dataset[key]

	def __setitem__(self, key, item: Union[pd.DataFrame,Data]):
		if isinstance(item, pd.DataFrame):
			item = Data.from_df(item)
		else:
			if not isinstance(item, Data):			
				raise TypeError("Item must be an instance of pd.DataFrame or Data")		
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

	def split_ts(self, k_folds=3):
		# join all ts arrays
		ts = []
		for k,data in self.dataset.items():
			ts.append(data.ts)
		ts = np.hstack(ts)
		ts = np.unique(ts)
		idx_folds = np.array_split(ts, k_folds)
		self.folds_ts = [(fold[0], fold[-1]) for fold in idx_folds]
		
		return self

	def train_test_split(
						self, 
						test_fold_idx: int, 
						burn_fraction: float = 0.1, 
						min_burn_points: int = 1, 
						seq_path: bool = False) -> List:
		
		if seq_path and test_fold_idx == 0:
			raise ValueError("Cannot start at fold 0 when path is sequential")
		if len(self.folds_ts) is None:
			raise ValueError("Need to split before getting the split")
		# for each Data create a split
		train_test = []

		for key, data in self.items():

			ts_lower, ts_upper = self.folds_ts[test_fold_idx]			
			
			# create training data
			train_data = data.before(ts = ts_lower, create_new = True)
			train_data.random_segment(burn_fraction, min_burn_points)			
			# if path is non sequential add data after the test set
			if not seq_path:
				train_data_add = data.after(ts = ts_upper, create_new = True)
				train_data_add.random_segment(burn_fraction, min_burn_points)
				train_data.stack(train_data_add, allow_both_empty = True)
			
			# get test data
			test_data = data.between(ts_lower, ts_upper, create_new = True)
			
			train_test.append({'key':key, 'train_data': train_data, 'test_data':test_data})

		return train_test


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