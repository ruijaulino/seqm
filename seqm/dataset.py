
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy
import tqdm
try:
	from .models import ConditionalGaussian
	from .arrays import Arrays
	from .uts import add_unix_timestamp,unix_timestamp_to_index
	from .constants import *
	from .model_pipe import ModelPipe,ModelPipes,Path
	from .transform import RollPWScaleTransform
	from .post_process import post_process,portfolio_post_process
except ImportError:
	from models import ConditionalGaussian
	from arrays import Arrays
	from uts import add_unix_timestamp,unix_timestamp_to_index
	from constants import *
	from model_pipe import ModelPipe,ModelPipes,Path
	from transform import RollPWScaleTransform
	from post_process import post_process,portfolio_post_process

# dict of Dataframes with properties
class Dataset:
	def __init__(self, dataset = {}):
		self.dataset = dataset
		# add unix timestamp when data is inserted
		for k,v in self.dataset.items():v=add_unix_timestamp(v)
		self.folds_dates = []
		
	# methods to behave like dict
	def add(self, key, item: pd.DataFrame):
		if not isinstance(item, ModelWrapper):
			raise TypeError("Item must be an instance of pd.DataFrame")
		self.dataset[key] = copy.deepcopy(add_unix_timestamp(item))

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
		for k,df in self.dataset.items():
			ts = ts.union(df.index.unique())
		ts = ts.sort_values()
		idx_folds = np.array_split(ts, k_folds)
		self.folds_dates = [(fold[0], fold[-1]) for fold in idx_folds]
		return self

	def set_train_on_pipes(self, model_pipes: ModelPipes):
		'''
		Set the training data in the model_pipes
		'''
		assert self.keys()==model_pipes.keys(),"dataset and model_pipes must have the same keys"
		for key, df in self.items():
			# get arrays
			arrays = Arrays()
			arrays.from_df(df)
			# use the arrays class to pass by arrays
			#x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
			#y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
			#x_cols,y_cols=self.get_xy_cols(df)
			model_pipes[key].set_train_arrays(arrays) # set_cols(x_cols,y_cols).set_train_data(x_train=x, y_train=y, copy_array = True)
		return model_pipes

	def set_test_on_pipes(self, model_pipes: ModelPipes):
		# maybe this can be relaxed??
		assert self.keys()==model_pipes.keys(),"dataset and model_pipes must have the same keys"
		for key, df in self.items():
			# get arrays
			arrays = Arrays()
			arrays.from_df(df, add_ts = True)
			#x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
			#y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
			#x_cols,y_cols=self.get_xy_cols(df)
			model_pipes[key].set_test_arrays(arrays) #.check_cols(x_cols,y_cols).set_test_data(x_test=x, y_test=y, ts=df.index, copy_array = True)		
		return model_pipes

	# ------
	# REMOVE METHODS
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
	# ------

	@staticmethod
	def get_xy_cols(df):
		cols = df.columns.tolist()
		x_cols=[c for c in cols if c.startswith(FEATURE_PREFIX)]		
		y_cols=[c for c in cols if c.startswith(TARGET_PREFIX)]
		return x_cols, y_cols
	
	
	def set_train_test_fold(
							self, 
							model_pipes: ModelPipes, 
							test_index: int, 
							burn_fraction: float = 0.1, 
							min_burn_points: int = 1, 
							seq_path: bool = False) -> ModelPipes:
		

		if seq_path and test_index == 0:
			raise ValueError("Cannot start at fold 0 when path is sequential")
		if self.folds_dates is None:
			raise ValueError("Need to split before getting the split")
		assert self.keys()==model_pipes.keys(),"dataset and model_pipes must have the same keys"

		for key, df in self.items():

			ts_lower, ts_upper = self.folds_dates[test_index]
			
			arrays_train = Arrays()
			arrays_train.from_df(df[df.index < ts_lower]).slice(burn_fraction, min_burn_points)
			
			arrays_train_add = Arrays()
			df_train_add = df[df.index > ts_upper] if not seq_path else pd.DataFrame(columns=df.columns)  # Empty if sequential
			arrays_train_add.from_df(df_train_add).slice(burn_fraction, min_burn_points)

			# concat/stack
			arrays_train.stack(arrays_train_add)

			#x_train_pre, y_train_pre = self._slice_segment(df_pre_test, burn_fraction, min_burn_points)
			#x_train_post, y_train_post = self._slice_segment(df_post_test, burn_fraction, min_burn_points)
			
			# Concatenate pre and post segments if non-sequential
			#x_train = np.vstack([x_train_pre, x_train_post]) if x_train_pre.size and x_train_post.size else x_train_pre if x_train_pre.size else x_train_post
			#y_train = np.vstack([y_train_pre, y_train_post]) if y_train_pre.size and y_train_post.size else y_train_pre if y_train_pre.size else y_train_post
			
			df_test = df[(df.index >= ts_lower) & (df.index <= ts_upper)]
			arrays_test = Arrays()
			arrays_test.from_df(df_test, add_ts = True)

			# x_test, y_test = df_test.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values, df_test.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values

			# x_cols,y_cols=self.get_xy_cols(df)

			if arrays_train.empty and arrays_test.empty:
				model_pipes.remove(key)
			else:
				model_pipes[key].set_arrays(arrays_train,arrays_test)

			#if x_train.shape[0]!=0 and x_test.shape[0]!=0:			
			#	model_pipes[key].set_cols(x_cols, y_cols).set_data(x_train, y_train, x_test, y_test, df_test.index)
			#else:

		return model_pipes

	# main methods to make studies on the dataset
	def train(self, model_pipes:ModelPipes, share_model = True):
		# associate model pipes with the dataset
		model_pipes = self.set_train_on_pipes(model_pipes)
		# estimate models
		model_pipes.estimate()
		return model_pipes

	def test(self, model_pipes: ModelPipes):	
		path=Path()
		model_pipes = self.set_test_on_pipes(model_pipes)
		model_pipes.evaluate()
		path.add(model_pipes)
		path.join()
		paths=[path.get_results()]
		return paths
		
	def live(self, model_pipes: ModelPipes):
		out = {}
		for key, df in self.items():
			if model_pipes.has_keys(key):
				# note
				# for a live application it makes sense that the last observation of 
				# y has nan because it is not available yet!
				# assume here that the data has that format
				
				arrays = Arrays()
				arrays.from_df(df)

				#x = df.iloc[:, df.columns.str.startswith(FEATURE_PREFIX)].values
				#y = df.iloc[:, df.columns.str.startswith(TARGET_PREFIX)].values
				#x_cols,y_cols=self.get_xy_cols(df)
				
				# model_pipes[key].check_cols(x_cols,y_cols)
				
				xq_ = None
				x_ = None
				z_ = None
				if arrays.has_x:
					xq_ = arrays.x[-1]
					x_ = arrays.x[:-1]
				if arrays.has_z:
					z_ = arrays.z[-1]
				w = model_pipes[key].get_weight(
										xq = xq_, 
										x = x_, 
										y = arrays.y[:-1], 
										z = z_,
										apply_transform_x = True, 
										apply_transform_y = True
										)

				pw = model_pipes[key].get_pw(arrays.y[:-1])		
				out.update({key:{'w':w,'pw':pw}})		
		return out

	def cvbt(
			self, 
			model_pipes:ModelPipes, 
			k_folds=4, 
			seq_path=False, 
			start_fold=0, 
			n_paths=4, 
			burn_fraction=0.1, 
			min_burn_points=3, 
			view_models=False,
			**kwargs
			):
		'''
		each path generates a dict like
		{dataset1:df1,dataset2,df2,..}
		where each df contains the columns [s,y1_w,y2_w,...,pw] at the same dates as the original dataset
		the output is a list of this dicts
		'''
		# Prepare the data
		self.split_dates(k_folds)			
		start_fold = max(1, start_fold) if seq_path else start_fold		
		paths = []
		for m in tqdm.tqdm(range(n_paths)):
			path=Path()
			for fold_index in range(start_fold, k_folds):			
				# make a copy of the original model_pipes
				local_model_pipes = self.set_train_test_fold(
															model_pipes=copy.deepcopy(model_pipes),
															test_index=fold_index,
															burn_fraction=burn_fraction,
															min_burn_points=min_burn_points,
															seq_path=seq_path
															)			
				local_model_pipes.estimate()
				local_model_pipes.evaluate()
				path.add(local_model_pipes)
			path.join()
			paths.append(path.get_results())
		return paths


# generate data
def generate_lr(n=1000,a=0,b=0.1,start_date='2000-01-01'):
	x=np.random.normal(0,0.01,n)
	a=0
	b=0.1
	y=a+b*x+np.random.normal(0,0.01,n)
	dates=pd.date_range(start_date,periods=n,freq='D')
	data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
	return data

def run_train_test_live():
	data1=generate_lr(n=100,a=0,b=0.1,start_date='2000-01-01')
	data2=generate_lr(n=70,a=0,b=0.1,start_date='2000-06-01')
	data3=generate_lr(n=150,a=0,b=0.1,start_date='2001-01-01')
	
	# create dataset
	dataset=Dataset({'dataset 1':data1,'dataset 2':data2})

	# create models for dataset
	model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=100)
	model_pipes=ModelPipes(model)
	for key in dataset.keys():
		# model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=100)
		# did not set individual model on purpose
		# but can be done (need to test this feature!)
		model_pipe = ModelPipe(x_transform = RollPWScaleTransform(window=10),y_transform = RollPWScaleTransform(window=10))
		model_pipes[key] = model_pipe
	
	print('RUN TRAIN')
	# train model_pipes on the dataset
	model_pipes = dataset.train(model_pipes)
	print('RUN TEST')
	# generate new data for testing
	data1=generate_lr(n=100,a=0,b=0.1,start_date='2005-01-01')
	data2=generate_lr(n=70,a=0,b=0.1,start_date='2005-06-01')
	data3=generate_lr(n=150,a=0,b=0.1,start_date='2005-01-01')
	
	# create dataset
	dataset_test = Dataset({'dataset 1':data1,'dataset 2':data2})
	paths = dataset_test.test(model_pipes)
	portfolio_post_process(paths)

	print('RUN LIVE')
	# GET THE CURRENT WEIGHTS!
	out=dataset_test.live(model_pipes)
	print(out)

def run_cvbt():
	data1=generate_lr(n=1000,a=0,b=0.1,start_date='2000-01-01')
	data2=generate_lr(n=700,a=0,b=0.1,start_date='2000-06-01')
	data3=generate_lr(n=1500,a=0,b=0.1,start_date='2001-01-01')
	
	# create dataset
	dataset=Dataset({'dataset 1':data1})#,'dataset 2':data2})	
	model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=100)
	model_pipes=ModelPipes(model)
	for key in dataset.keys():
		# model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=100)
		# did not set individual model on purpose
		# but can be done (need to test this feature!)
		model_pipe = ModelPipe(x_transform = RollPWScaleTransform(window=10),y_transform = RollPWScaleTransform(window=10))
		model_pipes[key] = model_pipe

	paths=dataset.cvbt(
					model_pipes, 
					k_folds=4, 
					seq_path=False, 
					start_fold=0, 
					n_paths=4, 
					burn_fraction=0.1, 
					min_burn_points=3, 
					share_model=True, 
					view_models=False
					)
	portfolio_post_process(paths)

if __name__=='__main__':
	print('run cvbt')
	run_cvbt()
	# print('run train/test/live')
	# run_train_test_live()
