# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import copy
import pickle
import os
import sys
import importlib  
from typing import List, Dict,Union
try:
	from .data_manager import Normalizer,DataManager
	from .post_proc import PathOutput,CVBTOut
	from .models import ConditionalGaussian
except ImportError:
	from data_manager import Normalizer,DataManager
	from post_proc import PathOutput,CVBTOut
	from models import ConditionalGaussian

# EVALUATE THE MODEL
# def evaluate(parameters,model,y,x,idx=None,z=None):
def evaluate(model,y,x,y_norm=None,idx=None,**kwargs):	
	# note: fees can be calculated after this is done!
	if x is not None:
		assert y.shape[0]==x.shape[0],"x and y must have the same number of observations"
	n=y.shape[0]
	p=y.shape[1]
	if y_norm is None:
		y_norm=np.ones(p) 
	if idx is None:
		idx=np.array([[0,n]],dtype=int)
	n_seq=idx.shape[0]
	s=np.zeros(n,dtype=np.float64)
	weights=np.zeros((n,p),dtype=np.float64)
	w_prev=np.zeros(p,dtype=np.float64)
	w=np.zeros(p,dtype=np.float64)
	for l in range(n_seq): 
		for i in range(idx[l][0],idx[l][1]):
			# build inputs for model
			y_=np.array(y[idx[l][0]:i])
			y_/=y_norm		  
			model_inputs={'y':y_}
			if x is not None:
				model_inputs.update({'x':x[idx[l][0]:i]})
				model_inputs.update({'xq':x[i]})
			w=model.get_weight(**model_inputs)
			weights[i]=w
			s[i]=np.dot(y[i],w)
	# compute fees here	
	return s,weights	

class Inference:
	def __init__(self,k_folds=4,seq_path=False,start_fold=0,n_paths=4,burn_fraction=0.1,min_burn_points=3,single_model=True):
		self.k_folds = k_folds
		self.seq_path = seq_path
		self.start_fold = start_fold
		self.n_paths = n_paths
		self.burn_fraction = burn_fraction
		self.min_burn_points = min_burn_points
		self.single_model = single_model
	
	def cvbt(self,data_manager:DataManager,model):

		#n_paths=5
		#burn_f=0.1
		#min_burn_points=20	
		# weight_method='inv_vol'
		
		# create splits by dates
		data_manager.split(self.k_folds)	
		
		start_fold=max(1,start_fold) if self.seq_path else self.start_fold	  
		
		paths_output=[]
		
		# for m in tqdm.tqdm(range(n_paths)):	
		# path_output=PathOutput(data.names)

		# Simulate a backtest path
		for i in range(start_fold,self.k_folds):					
			splits=data_manager.get_split(
										test_fold=i,
										burn_fraction = self.burn_fraction, 
										min_burn_points = self.min_burn_points, 
										seq_path = self.seq_path
										)
			local_model=None
			if self.single_model:
				# build a single model based on all datasets
				# create copy of model
				local_model=copy.deepcopy(model)
				# join training data
				local_model.estimate(**data_manager.join_splits_to_train())

			for k,split in splits.items():
				# train a specific model for each 
				if not self.single_model:
					print('not yet')
					print(Sdfsdf)
					# local_model=copy.deepcopy(model)					
				# evaluate	 
				split.view()
				print(Sdfsd)
				s,w=evaluate(local_model,**{'x':split.})
				# create dataframes with the results
				df_s=pd.DataFrame(s_,columns=data.y_cols,index=test_data_local.get('ts'))
				df_w=pd.DataFrame(w_,columns=data.y_cols,index=test_data_local.get('ts'))
				# inverse vol to strategy on asset
				df_pw=pd.DataFrame(np.ones_like(s_)*(1/np.mean(test_data_local['y_norm'])),columns=data.y_cols,index=test_data_local.get('ts'))
				# add it to path results
				path_output.add(test_data_local.get('name'),df_s,df_w,df_pw)			
		# join the information 
		path_output.join()


		# paths_output.append(path_output)



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
	data=[data1,data2,data3]
	data_mngr=DataManager(data,normalizer_class=None)
	# data.view()
	bt=Inference()
	bt.cvbt(data_mngr,model)
	#data=seqm.bt3.Data(data_lst)		