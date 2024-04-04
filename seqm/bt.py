
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
import copy
import tqdm

try:
	from .models import ConditionalGaussian
	from .loads import save_file
	from .transform import IdleTransform,BaseTransform
	from .data import Element,Elements,Dataset,Path
except ImportError:
	from models import ConditionalGaussian
	from loads import save_file
	from transform import IdleTransform,BaseTransform
	from data import Element,Elements,Dataset,Path

def train(self):
	pass

def test(self):
	pass

def live(self):
	pass

def cvbt(dataset:Dataset, model, k_folds=4, seq_path=False, start_fold=0, n_paths=4, burn_fraction=0.1, min_burn_points=3, single_model=True, view_models=False):
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
			elements = dataset.get_split_elements(
				test_fold=fold_index,
				burn_fraction=burn_fraction,
				min_burn_points=min_burn_points,
				seq_path=seq_path
			)
			elements.estimate(model,single_model=single_model,view_models=view_models).evaluate()
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
	data1=generate_lr(n=1000,a=0,b=0.1,start_date='2000-01-01')
	data2=generate_lr(n=700,a=0,b=0.1,start_date='2000-06-01')
	data3=generate_lr(n=1500,a=0,b=0.1,start_date='2001-01-01')
	
	model=ConditionalGaussian(n_gibbs=None,kelly_std=3,max_w=100)
	data={'Dataset 1':data1,'Dataset 2':data2,'Dataset 3':data3}
	dataset=Dataset(data,x_transform_class=IdleTransform)
	
	paths=cvbt(dataset,model, k_folds=4, seq_path=False, start_fold=0, n_paths=4, burn_fraction=0.1, min_burn_points=3, single_model=True)

	# save_file(paths,'paths_dev.pkl')

	#print(len(paths))

	#print(paths[0])
	


	# paths.post_process(seq_fees=False,pct_fee=0,sr_mult=1,n_boot=1000,name=None)
	# NOW IMPLEMENT POST PROCESS ON A LIST OF DICTS
