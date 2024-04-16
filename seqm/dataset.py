
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy
import tqdm
try:
	from .generators import linear,simulate_hmm
	from .data import Data
	from .models import ConditionalGaussian,GaussianHMM
	from .constants import *
	from .model_pipe import ModelPipe,ModelPipes,Path
	from .transform import RollPWScaleTransform
	from .post_process import post_process,portfolio_post_process
except ImportError:
	from generators import linear,simulate_hmm
	from data import Data
	from models import ConditionalGaussian,GaussianHMM
	from constants import *
	from model_pipe import ModelPipe,ModelPipes,Path
	from transform import RollPWScaleTransform
	from post_process import post_process,portfolio_post_process

# dict of Data with properties
class Dataset:
	def __init__(self, dataset = {}):
		self.dataset = dataset
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

	def set_train_test_fold(
							self, 
							model_pipes: ModelPipes, 
							test_index: int, 
							burn_fraction: float = 0.1, 
							min_burn_points: int = 1, 
							seq_path: bool = False) -> ModelPipes:
		

		if seq_path and test_index == 0:
			raise ValueError("Cannot start at fold 0 when path is sequential")
		if len(self.folds_ts) is None:
			raise ValueError("Need to split before getting the split")
		assert self.keys()==model_pipes.keys(),"dataset and model_pipes must have the same keys"

		for key, data in self.items():

			ts_lower, ts_upper = self.folds_ts[test_index]
			
			train_data = data.before(ts = ts_lower, create_new = True)
			train_data.random_segment(burn_fraction, min_burn_points)
			
			# if path is non sequential add data after the test set
			if not seq_path:
				train_data_add = data.after(ts = ts_upper, create_new = True)
				train_data_add.random_segment(burn_fraction, min_burn_points)
				train_data.stack(train_data_add)

			test_data = data.between(ts_lower, ts_upper, create_new = True)

			if train_data.empty and test_data.empty:
				model_pipes.remove(key)
			else:
				model_pipes[key].set_data(train_data, test_data)

		return model_pipes

	# main methods to make studies on the dataset
	def train(self, model_pipes:ModelPipes):
		# associate model pipes with the dataset
		assert self.keys()==model_pipes.keys(),"dataset and model_pipes must have the same keys"
		for key, data in self.items():
			model_pipes[key].set_train_data(data) 
		# estimate models
		model_pipes.estimate()
		return model_pipes

	def test(self, model_pipes: ModelPipes):	
		path=Path()
		assert self.keys()==model_pipes.keys(),"dataset and model_pipes must have the same keys"
		for key, data in self.items():
			model_pipes[key].set_test_data(data)
		model_pipes.evaluate()
		path.add(model_pipes)
		path.join()
		paths=[path.get_results()]
		return paths
		
	def live(self, model_pipes: ModelPipes):
		out = {}
		for key, data in self.items():
			if model_pipes.has_keys(key):
				# note
				# for a live application it makes sense that the last observation of 
				# y has nan because it is not available yet!
				# assume here that the data has that format				
				xq_ = None
				x_ = None
				z_ = None
				if data.has_x:
					xq_ = data.x[-1]
					x_ = data.x[:-1]
				if data.has_z:
					z_ = data.z[-1]
				w = model_pipes[key].get_weight(
										xq = xq_, 
										x = x_, 
										y = data.y[:-1], 
										z = z_,
										apply_transform_x = True, 
										apply_transform_y = True
										)
				pw = model_pipes[key].get_pw(data.y[:-1])		
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
			**kwargs
			):
		'''
		each path generates a dict like
		{dataset1:df1,dataset2,df2,..}
		where each df contains the columns [s,y1_w,y2_w,...,pw] at the same dates as the original dataset
		the output is a list of this dicts
		'''
		# Prepare the data
		self.split_ts(k_folds)					
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


def run_train_test_live():
	data1=linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
	data2=linear(n=700,a=0,b=0.1,start_date='2000-06-01')
	data3=linear(n=1500,a=0,b=0.1,start_date='2001-01-01')
	
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
	data1=linear(n=1000,a=0,b=0.1,start_date='2005-01-01')
	data2=linear(n=700,a=0,b=0.1,start_date='2005-06-01')
	data3=linear(n=1500,a=0,b=0.1,start_date='2005-01-01')
	
	# create dataset
	dataset_test = Dataset({'dataset 1':data1,'dataset 2':data2})
	paths = dataset_test.test(model_pipes)
	portfolio_post_process(paths)

	print('RUN LIVE')
	# GET THE CURRENT WEIGHTS!
	out=dataset_test.live(model_pipes)
	print(out)

def run_cvbt():
	
	data1=linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
	data2=linear(n=700,a=0,b=0.1,start_date='2000-06-01')
	data3=linear(n=1500,a=0,b=0.1,start_date='2001-01-01')
	
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


def multiseq():
	A=np.array([
				[0.9,0.1,0],
				[0,0.9,0.1],
				[0,0,1.]
			]) # state transition
	P=np.array([0.7,0.3,0]) # initial state distribution
	means=[
			np.array([-1]),
			np.array([0.]),
			np.array([1])
		] # let the means be different from zero 
	# list of covariance matrices (for each mixture)
	covs=[
		np.array([[0.25]]),
		np.array([[0.01]]),
		np.array([[0.25]])
	]
	k=100
	n_min=5
	n_max=15
	o=[]
	msidx=[]
	for i in range(k):
		n=np.random.randint(n_min,n_max)
		x_,z_=simulate_hmm(n,A,P,means,covs)
		o.append(x_)
		msidx.append(i*np.ones(x_.shape[0],dtype=int)[:,None])
		# plt.plot(x_)
	# plt.grid(True)
	# plt.show()
	y=np.vstack(o)  
	msidx=np.vstack(msidx)
	dates=pd.date_range('2000-01-01',periods=y.shape[0],freq='B')
	data1=pd.DataFrame(np.hstack((y,msidx)),columns=['y1','idx'],index=dates)
	
	k=100
	n_min=5
	n_max=15
	o=[]
	msidx=[]
	for i in range(k):
		n=np.random.randint(n_min,n_max)
		x_,z_=simulate_hmm(n,A,P,means,covs)
		o.append(x_)
		msidx.append(i*np.ones(x_.shape[0],dtype=int)[:,None])
		# plt.plot(x_)
	# plt.grid(True)
	# plt.show()
	y=np.vstack(o)  
	msidx=np.vstack(msidx)
	dates=pd.date_range('2000-01-01',periods=y.shape[0],freq='B')
	data2=pd.DataFrame(np.hstack((y,msidx)),columns=['y1','idx'],index=dates)
	
	print(data1)
	print(data2)

	dataset=Dataset({'dataset 1':data1,'dataset 2':data2})	


	A_zeros=[[0,2],[1,0],[2,0],[2,1]]
	A_groups=[]

	model=GaussianHMM(n_states=3,n_gibbs=250,A_zeros=A_zeros,A_groups=A_groups)

	model_pipes=ModelPipes(model)
	for key in dataset.keys():
		# model=GaussianHMM(n_states=3,n_gibbs=100,kelly_std=2,max_w=1)
		model_pipe_ = ModelPipe() 
		model_pipes[key] = model_pipe_

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
	
	# multiseq()

	print('run cvbt')
	run_cvbt()
	# print('run train/test/live')
	# run_train_test_live()
