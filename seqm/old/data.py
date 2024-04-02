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

def random_subsequence(ar,burn_f=0.1,min_burn_points=1):
	'''
	Generates a random subsequence of an array
	ar: numpy (n,) array 
	burn_f: float between 0 and 1 with the percentage of data to burn at each side
	min_burn_points: integer with the minimum number of points to burn

	return ar[a:-b] where a and b are random indexes to build a subarray from ar
	'''
	min_burn_points=max(min_burn_points,1)
	a,b=np.random.randint(min_burn_points,max(int(ar.size*burn_f),min_burn_points+1),size=2)
	return ar[a:-b]		
				
def get_slice(df,x_cols,y_cols,burn_f=0.1,min_burn_points=1):
	# get a piece of a dataframe separated by x and y
	x=df[x_cols].values
	y=df[y_cols].values
	# get a random index	
	random_idx=random_subsequence(
								np.arange(x.shape[0],dtype=int),
								burn_f,
								min_burn_points)	
	# TO DO
	# implement here 
	# - burning of points
	# - random subsampling if iid
	if random_idx.size!=0:
		return x[random_idx],y[random_idx]
	else:
		return None,None

class TrainTest(object):
	def __init__(self):
	    self.datasets = {}  # Stores datasets by name

	def add(self, name, x_train, y_train, x_test, y_test, ts=None, x_std=None, y_std=None):
		"""Adds a new dataset, ensuring unique names."""
		if name in self.datasets:
			raise ValueError(f"Dataset with name {name} already exists.")
		self.datasets[name] = {
			"x_train": x_train,
			"y_train": y_train,
			"x_test": x_test,
			"y_test": y_test,
			"ts": ts,
			"x_std": x_std,
			"y_std": y_std,
		}

	def view(self, name):
		"""Prints the details of a specified dataset."""
		if name not in self.datasets:
			print(f"No dataset found with name {name}.")
			return
		dataset = self.datasets[name]
		print(f'Dataset: {name}')
		for key, value in dataset.items():
			print(f"{key}: {value}")
		print()

	def view_all(self):
		"""Prints the details of all datasets."""
		for name in self.datasets:
			self.view(name)

	def join_training_data(self):
		"""Joins training data from all datasets."""
		if not self.datasets:
			raise ValueError("No datasets to join.")
		x_train = np.vstack([dataset["x_train"] for dataset in self.datasets.values()])
		y_train = np.vstack([dataset["y_train"] for dataset in self.datasets.values()])
		return x_train, y_train




class Data(object):
	# class to handle input dataframes and split them
	def __init__(self,data: Union[pd.DataFrame,List[pd.DataFrame]],names: List[str]=None):
		data=copy.deepcopy(data)		
		if isinstance(data,pd.DataFrame):
			data=[data]		 
		self.n_data=len(data)
		self.input_data=data
		self.normalize=False
		if self.n_data!=1:
			self.normalize=True
		self.names=names
		if self.names is not None:
			if len(self.names)!=self.n_data:
				self.names=None
		if self.names is None:
			self.names=['Dataset %s'%(i+1) for i in range(self.n_data)]		
		self._check_input()
		self.folds_dates=None

	def view(self):
		for i,name in enumerate(self.names):
			print(name)
			print(self.input_data[i])
		self.view_folds_dates()

	def _check_input(self):
		'''
		verify if the input has a correct format
		'''
		self.cols=self.input_data[0].columns.tolist()
		self.y_cols=[e for e in self.cols if e[0]=='y']	
		self.x_cols=[e for e in self.cols if e[0]=='x']	
		assert len(self.y_cols)!=0,"data must have columns like y1,y2,..."	
		# all dataframes should have the same columns
		for e in self.input_data:
			assert e.columns.tolist()==self.cols	
	
	def split(self,k_folds=3):
		# must split by indexes/dates
		# join all dates
		# assume that all dates are comparable
		# first join together all indexes
		self.ts=self.input_data[0].index
		for i in range(1,self.n_data):
			self.ts=self.ts.append(self.input_data[i].index)
		# then get the unique values and sort
		self.ts=self.ts.unique().sort_values()
		# the folds are defined as divisions of ts
		idx=np.arange(self.ts.size,dtype=int)
		idx_folds=np.array_split(idx,k_folds)
		self.folds_dates=[]
		for i in range(k_folds):
			self.folds_dates.append([self.ts[idx_folds[i][0]],self.ts[idx_folds[i][-1]]])
		return self
	
	def view_folds_dates(self):
		if self.folds_dates is not None:
			print('Folds dates')
			for e in self.folds_dates:
				print('-> from '+ str(e[0]) + ' to ' + str(e[1]) )
				
	def get_traintest_split(self,test_fold_idx:int,seq_path:bool=False,burn_f:float=0.1,min_burn_points:int=1,iid:bool=False,single_model:bool=False)->'TrainTest':
		'''
		build sets for train and test for a given fold
		test_fold_idx: int with the fold dates that define the test data
		seq_path: bool with the indication that the path is sequential
		f_burn: fraction of points to burn at both ends
		single_model: i
		min_burn_points
		'''
		if seq_path and test_fold_idx==0:
			raise Exception('Cannot start at fold 0 when path is sequential')
		assert self.folds_dates is not None,"need to split before get the split"

		out=TrainTest()
		# for each input dataframe
		for j,df in enumerate(self.input_data):
						
			# lower and upper bound on time indexes that define the folder 
			ts_lower=self.folds_dates[test_fold_idx][0]
			ts_upper=self.folds_dates[test_fold_idx][1]
			
			x_train=[]
			y_train=[]
			
			# ------------------------
			# TRAIN DATA		
			# before test period
			df_before=df[df.index<ts_lower].copy()
			if not df_before.empty:				
				x_,y_=get_slice(df_before,self.x_cols,self.y_cols,burn_f,min_burn_points)
				if x_ is not None:
					x_train.append(x_)
					y_train.append(y_)			
			
			# after test period (only if path is not sequential)
			if not seq_path:
				df_after=df[df.index>ts_upper].copy()
				if not df_after.empty:
					x_,y_=get_slice(df_after,self.x_cols,self.y_cols,burn_f,min_burn_points)
					if x_ is not None:
						x_train.append(x_)
						y_train.append(y_)	

			# ------------------------
			# If there is training data, merge it and build the test data
			if len(x_train)!=0:				
				x_train=np.vstack(x_train)
				y_train=np.vstack(y_train)
				# ------------------------
				# NORMALIZE
				x_train_std=np.ones(x_train.shape[1])
				y_train_std=np.ones(y_train.shape[1])
				if self.normalize:
					x_train_std=np.std(x_train,axis=0)
					y_train_std=np.std(y_train,axis=0)							
				# ------------------------
				# ADD TO TRAINING DATA
				x_train/=x_train_std
				y_train/=y_train_std
				# ------------------------
				# TEST DATA 
				df_test=df[(df.index>=ts_lower) & (df.index<=ts_upper)].copy()
				if not df_test.empty:
					x_test=df_test[self.x_cols].values
					y_test=df_test[self.y_cols].values
					# normalize by training data
					x_test/=x_train_std
					# add to test data
					# Build train test object
					out.add(name=self.names[j],
							x_train=x_train,
							y_train=y_train,
							x_test=x_test,
							y_test=y_test,
							ts=df_test.index,
							x_std=x_train_std,
							y_std=y_train_std
							)									
		return out

if __name__=='__main__':

	# create some random data for input
	n=20
	index=pd.date_range('2000-01-01',periods=n,freq='D')
	values=np.random.normal(0,1,(n,3))
	columns=['x1','x2','y1']
	df=pd.DataFrame(values,index=index,columns=columns)
	

	data=Data(df)
	data.split(k_folds=3)
	data.view()

	print('-----------------------')
	test_fold_idx=1
	out=data.get_traintest_split(test_fold_idx,seq_path=False,burn_f=0.1,min_burn_points=1,iid=False)
	out.view_all()
	print('-----')
