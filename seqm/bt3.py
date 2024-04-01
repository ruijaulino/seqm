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
	from .data import Data
	from .post_proc import PathOutput,CVBTOut
except ImportError:
	from data import Data
	from post_proc import PathOutput,CVBTOut


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

	
def cvbt(data:'Data',model,k_folds=4,seq_path=False,start_fold=0,n_paths=5,burn_f=0.1,min_burn_points=10)->'CVBTOut':
	'''
	model: instance of the model class
	'''
	#k_folds=4
	#seq_path=False
	start_fold=0
	#n_paths=5
	#burn_f=0.1
	#min_burn_points=20	
	# weight_method='inv_vol'
	# create splits by dates
	data.split(k_folds)	
	
	if seq_path:
		start_fold=max(1,start_fold)	  
	
	paths_output=[]
	
	for m in tqdm.tqdm(range(n_paths)):	
		path_output=PathOutput(data.names)
		for i in range(start_fold,k_folds):		
			# create copy of model
			local_model=copy.deepcopy(model)
			# train
			train_data,test_data=data.get_train_test_set(
														test_fold_idx=i,
														seq_path=seq_path,
														burn_f=burn_f,
														min_burn_points=min_burn_points,
														iid=False)

			local_model.estimate(**train_data)

			for test_data_local in test_data:
				# evaluate	 
				s_,w_=evaluate(
									local_model,
								   **test_data_local
								   )
				# create dataframes with the results
				df_s=pd.DataFrame(s_,columns=data.y_cols,index=test_data_local.get('ts'))
				df_w=pd.DataFrame(w_,columns=data.y_cols,index=test_data_local.get('ts'))
				# inverse vol to strategy on asset
				df_pw=pd.DataFrame(np.ones_like(s_)*(1/np.mean(test_data_local['y_norm'])),columns=data.y_cols,index=test_data_local.get('ts'))
				# add it to path results
				path_output.add(test_data_local.get('name'),df_s,df_w,df_pw)			
		# join the information 
		path_output.join()
		paths_output.append(path_output)
	return CVBTOut(paths_output) 


if __name__=='__main__':
	pass