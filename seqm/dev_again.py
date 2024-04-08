
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
	from .model_wrapper import ModelWrapper
	from .post_process import post_process,portfolio_post_process
	from .constants import *
except ImportError:
	from models import ConditionalGaussian
	from loads import save_file
	from transform import IdleTransform,BaseTransform,RollPWScaleTransform
	from elements import Element,Elements,Path
	from dataset import Dataset
	from model_wrapper import ModelWrapper
	from post_process import post_process,portfolio_post_process
	from constants import *


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
	
	datasets_dict={'data1':data1,'data2':data2}

	# convert the dict
	if isinstance(datasets_dict, dict):
		keys=[]
		datasets=[]
		for k,v in datasets_dict.items():
			keys.append(k)
			datasets.append(v)
	else:
		datasets = datasets_dict 
		keys=['Dataset']
	cols = datasets[0].columns.tolist()
	for df in datasets:
		if df.columns.tolist() != cols:
			raise ValueError("All DataFrames must have the same columns.")
	y_cols=[c for c in cols if c.startswith(TARGET_PREFIX)]
	x_cols=[c for c in cols if c.startswith(FEATURE_PREFIX)]	


	k_folds=3
	# split dates for cvbt
	ts = pd.DatetimeIndex([])
	for df in datasets:
		ts = ts.union(df.index.unique())
	ts = ts.sort_values()
	idx_folds = np.array_split(ts, k_folds)
	folds_dates = [(fold[0], fold[-1]) for fold in idx_folds]	

	print(folds_dates)



