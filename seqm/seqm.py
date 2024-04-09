
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
import copy
import tqdm

try:
	from .models import ConditionalGaussian
	from .loads import save_file
	from .transform import IdleTransform,BaseTransform,RollPWScaleTransform
	from .elements import Element,Elements
	from .dataset import Dataset
	from .model_pipe import ModelPipe,ModelPipes,Path
	from .post_process import post_process,portfolio_post_process
except ImportError:
	from models import ConditionalGaussian
	from loads import save_file
	from transform import IdleTransform,BaseTransform,RollPWScaleTransform
	from elements import Element,Elements
	from dataset import Dataset
	from model_pipe import ModelPipe,ModelPipes,Path
	from post_process import post_process,portfolio_post_process


def check_keys():
	pass


def train(dataset: Dataset, model_pipes:ModelPipes, share_model = True):
	# associate model pipes with the dataset
	model_pipes = dataset.set_train(model_pipes)
	# estimate models
	model_pipes.estimate(share_model = share_model)

	return model_pipes

def test(dataset: Dataset, model_pipes: ModelPipes):	
	path=Path()
	model_pipes = dataset.set_test(model_pipes)
	model_pipes.evaluate()
	path.add(model_pipes)
	path.join()
	paths=[path.get_results()]
	return paths
	
def live(dataset: Dataset, model_pipes: ModelPipes):
	# just evaluate last point
	
	return dataset.live(model_pipes)

def cvbt(
		dataset:Dataset, 
		model_pipes:ModelPipes, 
		k_folds=4, 
		seq_path=False, 
		start_fold=0, 
		n_paths=4, 
		burn_fraction=0.1, 
		min_burn_points=3, 
		share_model=True, 
		view_models=False
		):
	'''
	each path generates a dict like
	{dataset1:df1,dataset2,df2,..}
	where each df contains the columns [s,y1_w,y2_w,...,pw] at the same dates as the original dataset
	the output is a list of this dicts
	'''
	# Prepare the data
	dataset.split_dates(k_folds)			
	start_fold = max(1, start_fold) if seq_path else start_fold		
	paths = []
	for m in range(n_paths):#tqdm.tqdm(range(n_paths)):
		path=Path()
		for fold_index in range(start_fold, k_folds):			
			# make a copy of the original model_pipes
			local_model_pipes = dataset.set_train_test_fold(
														model_pipes=copy.deepcopy(model_pipes),
														test_index=fold_index,
														burn_fraction=burn_fraction,
														min_burn_points=min_burn_points,
														seq_path=seq_path
														)			
			local_model_pipes.estimate(share_model = share_model)
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
	dates=pd.date_range(start_date,periods=n,freq='B')
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
	print('single train')
	# single train
	model_pipes = train(dataset, model_pipes, share_model = True)
	for k,e in model_pipes.items():
		e.view()
	print('----------')
	print('RUN TEST')
	data1=generate_lr(n=100,a=0,b=0.1,start_date='2005-01-01')
	data2=generate_lr(n=70,a=0,b=0.1,start_date='2005-06-01')
	data3=generate_lr(n=150,a=0,b=0.1,start_date='2005-01-01')
	
	# create dataset
	dataset=Dataset({'dataset 1':data1,'dataset 2':data2})
	paths=test(dataset, copy.deepcopy(model_pipes))
	# portfolio_post_process(paths)

	print('----------')
	print('RUN LIVE')
	# GET THE CURRENT WEIGHTS!
	out=live(dataset, model_pipes)
	print(out)

	




def run_cvbt():
	data1=generate_lr(n=1000,a=0,b=0.1,start_date='2000-01-01')
	data2=generate_lr(n=700,a=0,b=0.1,start_date='2000-06-01')
	data3=generate_lr(n=1500,a=0,b=0.1,start_date='2001-01-01')
	
	# create dataset
	dataset=Dataset({'dataset 1':data1,'dataset 2':data2})	
	model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=100)
	model_pipes=ModelPipes(model)
	for key in dataset.keys():
		# model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=100)
		# did not set individual model on purpose
		# but can be done (need to test this feature!)
		model_pipe = ModelPipe(x_transform = RollPWScaleTransform(window=10),y_transform = RollPWScaleTransform(window=10))
		model_pipes[key] = model_pipe

	paths=cvbt(
		dataset, 
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

	# run_cvbt()

	run_train_test_live()



