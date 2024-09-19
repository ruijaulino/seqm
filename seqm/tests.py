

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy
import tqdm
try:
	from .generators import linear,simulate_hmm
	from .data import Data
	from .dataset import Dataset
	from .models import ConditionalGaussian,GaussianHMM
	from .constants import *
	from .model_pipe import ModelPipe,ModelPipes,Path
	from .transform import RollPWScaleTransform
	from .post_process import post_process,portfolio_post_process
except ImportError:
	from generators import linear,simulate_hmm
	from data import Data
	from dataset import Dataset
	from models import ConditionalGaussian,GaussianHMM
	from constants import *
	from model_pipe import ModelPipe,ModelPipes,Path
	from transform import RollPWScaleTransform
	from post_process import post_process,portfolio_post_process



if __name__=='__main__':
	data1=linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
	data2=linear(n=700,a=0,b=0.1,start_date='2000-06-01')
	data3=linear(n=1500,a=0,b=0.1,start_date='2001-01-01')
	
	# create dataset
	dataset=Dataset({'dataset 1':data1,'dataset 2':data2, 'dataset 3':data3})	
	
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