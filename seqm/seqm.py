
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
	from .elements import Element,Elements,Path
	from .dataset import Dataset
	from .model_pipe import ModelPipe,ModelPipes
	from .post_process import post_process,portfolio_post_process
except ImportError:
	from models import ConditionalGaussian
	from loads import save_file
	from transform import IdleTransform,BaseTransform,RollPWScaleTransform
	from elements import Element,Elements,Path
	from dataset import Dataset
	from model_pipe import ModelPipe,ModelPipes
	from post_process import post_process,portfolio_post_process


def check_keys():
	pass


def train(dataset: Dataset, model_pipes:ModelPipes, share_training_data = True):
	
	# check keys
	assert dataset.keys()==model_pipes.keys(),"dataset and model_pipes must have the same keys"

	
	
	# parse dataset into train elements
	elements = dataset.get_train() 
	elements.view()


	
	print('all_cols_equal: ', elements.all_cols_equal)
	print('ola!')
	print(sdfsdf)				
	elements.estimate(model, single_model = single_model)
	return elements.get_model() if return_model else elements   

	# return trained model pipelines
	return model_pipes

def test(dataset: Dataset):	
	pass


def live(dataset: Dataset):
	# just evaluate last point
	pass


def cvbt(dataset:Dataset, model_wrapper:ModelPipes, k_folds=4, seq_path=False, start_fold=0, n_paths=4, burn_fraction=0.1, min_burn_points=3, share_training_data=True, view_models=False):
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
	for m in tqdm.tqdm(range(n_paths)):
		path=Path()
		for fold_index in range(start_fold, k_folds):			
			elements = dataset.get_train_test_elements(
														test_index=fold_index,
														burn_fraction=burn_fraction,
														min_burn_points=min_burn_points,
														seq_path=seq_path
														)
			
			elements.view()




			print('ola!')
			print(sdfsdf)

			elements.set_model_wrapper(model_wrapper)
			
			elements.estimate(single_model=single_model,view_models=view_models)
			elements.evaluate()
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
	data1=generate_lr(n=100,a=0,b=0.1,start_date='2000-01-01')
	data2=generate_lr(n=70,a=0,b=0.1,start_date='2000-06-01')
	data3=generate_lr(n=150,a=0,b=0.1,start_date='2001-01-01')
	
	# create dataset
	dataset=Dataset({'dataset 1':data1,'dataset 2':data2})

	# create models for dataset
	model_pipes=ModelPipes()
	for key in dataset.keys():
		model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=100)
		model_pipe = ModelPipe(model, x_transform = RollPWScaleTransform(window=10),y_transform = RollPWScaleTransform(window=10))
		model_pipes[key] = model_pipe

	train(dataset, model_pipes, share_training_data = True)

	#elements = dataset.get_train()
	#elements.view()

	# CVBT to evaluate model
	# paths=cvbt(dataset, model_wrapper, k_folds=4, seq_path=False, start_fold=0, n_paths=4, burn_fraction=0.1, min_burn_points=3, single_model=True)
	
	# train(dataset, model_wrapper)

	# portfolio_post_process(paths,pct_fee=0.,seq_fees=False,sr_mult=1,n_boot=1000,view_weights=True,use_pw=True)

	# TRAIN/TEST
	# model=train(dataset, model, single_model = False, return_model = True)
	# for k,m in model.items(): m.view()
	
	# LIVE

