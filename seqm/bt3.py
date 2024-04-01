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

class Data(object):
	def __init__(self,data: Union[pd.DataFrame,List[pd.DataFrame]],names=None):
		# when it is a list is is assumed that we are in a mode of
		# applying the same model to multiple series and aggregate them
		# properly
		# otherwise it's just the original behaviour
		# TO DO
		# implement states, multiseq, etc
		# should be simple to generalize
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
	
	def view_folds_dates(self):
		for e in self.folds_dates:
			print(e)

	def random_subsequence(self,ar,burn_f=0.1,min_burn_points=1):
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
			
	def get_train_test_set(self,test_fold_idx,seq_path=False,burn_f=0.1,min_burn_points=1,iid=False):
		'''
		build sets for train and test for a given fold
		test_fold_idx: int with the fold dates that define the test data
		seq_path: bool with the indication that the path is sequential
		f_burn: fraction of points to burn at both ends
		min_burn_points
		'''
		if seq_path and test_fold_idx==0:
			raise Exception('Cannot start at fold 0 when path is sequential')
		# train sets
		x_train=[]		
		y_train=[]		
		test_data=[]
		train_data={}
		for j,df in enumerate(self.input_data):
			test_data_local={}
			
			ts_lower=self.folds_dates[test_fold_idx][0]
			ts_upper=self.folds_dates[test_fold_idx][1]
			
			x_train_local=[]
			y_train_local=[]
			
			# ------------------------
			# TRAIN DATA		
			# before test period
			df_before=df[df.index<ts_lower].copy()
			if not df_before.empty:
				tmp_x=df_before[self.x_cols].values
				tmp_y=df_before[self.y_cols].values
				
				random_idx=self.random_subsequence(
											np.arange(tmp_x.shape[0],dtype=int),
											burn_f,
											min_burn_points)
				
				# TO DO
				# implement here 
				# - burning of points
				# - random subsampling if iid
				x_train_local.append(tmp_x[random_idx])
				y_train_local.append(tmp_y[random_idx])			
			
			# after test period (only if path is not sequential)
			if not seq_path:
				df_after=df[df.index>ts_upper].copy()
				if not df_after.empty:
					tmp_x=df_after[self.x_cols].values
					tmp_y=df_after[self.y_cols].values
					random_idx=self.random_subsequence(
												np.arange(tmp_x.shape[0],dtype=int),
												burn_f,
												min_burn_points)

					# TO DO
					# implement here 
					# - burning of points
					# - random subsampling if iid
					x_train_local.append(tmp_x[random_idx])
					y_train_local.append(tmp_y[random_idx])			
			# ------------------------
			# MERGE TRAINING SETS
			if len(x_train_local)!=0:
				x_train_local=np.vstack(x_train_local)
				y_train_local=np.vstack(y_train_local)
				# ------------------------
				# NORMALIZE
				x_train_std=np.ones(x_train_local.shape[1])
				y_train_std=np.ones(y_train_local.shape[1])
				if self.normalize:
					x_train_std=np.std(x_train_local,axis=0)
					y_train_std=np.std(y_train_local,axis=0)							
				# ------------------------
				# ADD TO TRAINING DATA
				x_train.append(x_train_local/x_train_std)
				y_train.append(y_train_local/y_train_std)
				# ------------------------
				# TEST DATA - if there is no train data we cannot normalize for testing!
				df_test=df[(df.index>=ts_lower) & (df.index<=ts_upper)].copy()
				if not df_test.empty:
					tmp_x=df_test[self.x_cols].values
					tmp_y=df_test[self.y_cols].values
					# normalize
					tmp_x/=x_train_std
					# add to test data
					test_data_local.update({'x':tmp_x})
					test_data_local.update({'y':tmp_y})			
					test_data_local.update({'y_norm':y_train_std})
					test_data_local.update({'ts':df_test.index})
					test_data_local.update({'name':self.names[j]})
					test_data.append(test_data_local) 
		x_train=np.vstack(x_train)
		y_train=np.vstack(y_train)
		train_data={'x':x_train,'y':y_train}
		return train_data,test_data

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

class PathOutput(object):
	def __init__(self,names):
		self.names=names
		self.results={}
		for name in names:
			self.results.update({name:{'s':[],'w':[],'pw':[]}})
		
	def add(self,name,df_s,df_w,df_pw=None):
		'''
		s: strategy
		w: weight
		pw: portfolio weight to multiply s
		'''
		assert name in self.names,"should not happen"
		self.results.get(name)['s'].append(df_s)
		self.results.get(name)['w'].append(df_w)
		if df_pw is None:
			df_pw=pd.DataFrame(np.ones_like(df_s.values),index=df_s.index,columns=df_s.columns)
		self.results.get(name)['pw'].append(df_pw)
		
	def join(self,weight_method=None):
		# join s
		portfolio_s=[]
		portfolio_w=[]
		for name in self.names:
			tmp=self.results.get(name)
			s=pd.concat(tmp.get('s'))
			w=pd.concat(tmp.get('w'))
			pw=pd.concat(tmp.get('pw'))
			s=s.sort_index()
			w=w.sort_index()
			pw=pw.sort_index()
			self.results.get(name)['s']=s
			self.results.get(name)['w']=w
			self.results.get(name)['pw']=pw
			# change columns of s
			s.columns=[name+'_'+e for e in s.columns.tolist()]
			pw.columns=[name+'_'+e for e in pw.columns.tolist()]			
			portfolio_s.append(s)
			portfolio_w.append(pw)
		portfolio_s=pd.concat(portfolio_s,axis=1)
		portfolio_w=pd.concat(portfolio_w,axis=1)
		# fill na with zero
		portfolio_s=portfolio_s.fillna(0)
		portfolio_w=portfolio_w.fillna(0)
		# this should always be a absolute value
		# not to override the strategy
		portfolio_w/=np.sum(np.abs(portfolio_w),axis=1).values[:,None]
		self.portfolio_s=pd.DataFrame(np.sum(portfolio_s*portfolio_w,axis=1),columns=['s'])

		
class CVBTOut(object):
	def __init__(self,paths_output:List[PathOutput]):
		self.paths_output=paths_output
		self.n_paths=len(self.paths_output)	   
	
	def view_names(self):
		print(self.paths_output[0].names)
	
	def _convert_pct_fees(self,pct_fees=[0]):
		if isinstance(pct_fees,float):
			pct_fees=np.array([pct_fees])
		elif isinstance(pct_fees,int):
			pct_fees=np.array([pct_fees])
		else:
			pct_fees=np.array(pct_fees)   
		return pct_fees
		
	def calculate_fees(self,s,weights,seq_fees,pct_fees):
		if seq_fees:
			dw=np.abs(weights[1:]-weights[:-1])
			dw=np.vstack(([np.zeros_like(dw[0])],dw))
			dw=np.sum(dw,axis=1)
		else:
			dw=np.sum(np.abs(weights),axis=1)
		s_fees=np.zeros((pct_fees.size,s.shape[0],s.shape[1]))
		for i in range(pct_fees.size):
			s_fees[i]=s-pct_fees[i]*dw
		return s_fees

	def bootstrap_sharpe(self,s,n_boot=1000):
		l=s.size
		idx=np.arange(l,dtype=int)
		idx_=np.random.choice(idx,(l,n_boot),replace=True)
		s_=s[idx_]
		boot_samples=np.mean(s_,axis=0)/np.std(s_,axis=0)
		return boot_samples
	
	def post_process(self,seq_fees=False,pct_fees=[0],sr_mult=1,n_boot=1000,name=None):
		pct_fees=self._convert_pct_fees(pct_fees)
		# by default use the results for the first dataframe used as input
		# this will work ny default because in general there is only one
		if name is None:
			name=self.paths_output[0].names[0]
		# get and joint results for name
		# after that, just use the existent function
		s=[]
		w=[]
		for i,path in enumerate(self.paths_output):
			s.append(path.results.get(name)['s'].values)
			w.append(path.results.get(name)['w'].values)
		s=np.hstack(s)
		w=np.stack(w,axis=2)
		ts=self.paths_output[0].results.get(name)['s'].index
		
		s_fees=self.calculate_fees(s,w,seq_fees,pct_fees)

		# make plots!
		paths_sr=sr_mult*np.mean(s,axis=0)/np.std(s,axis=0)
		idx_lowest_sr=np.argmin(paths_sr)

		b_samples=self.bootstrap_sharpe(s[:,idx_lowest_sr],n_boot=n_boot)
		b_samples*=sr_mult
		valid=False
		if np.sum(b_samples<0)==0:
			valid=True
		if valid:
			print('-> ACCEPT STRATEGY')
		else:
			print('-> REJECT STRATEGY')
		print()
		print('** Performance summary **')
		print()
		print('Return: ', np.power(sr_mult,2)*np.mean(s))
		print('Standard deviation: ', sr_mult*np.std(s))
		print('Sharpe: ', sr_mult*np.mean(s)/np.std(s))
		print()
		for i in range(pct_fees.size):
			print('Return fee=%s: '%pct_fees[i], 
				  np.power(sr_mult,2)*np.mean(s_fees[i]))
			print('Standard deviation fee=%s: '%pct_fees[i], 
				  sr_mult*np.std(s_fees[i]))
			print('Sharpe fee=%s: '%pct_fees[i], 
				  sr_mult*np.mean(s_fees[i])/np.std(s_fees[i]))
			print()
		print('**')

		# bootstrap estimate of sharpe
		if self.n_paths!=1:
			plt.title('Distribution of paths SR [no fees]')
			plt.hist(paths_sr,density=True)
			plt.show()
			for i in range(pct_fees.size):
				tmp=sr_mult*np.mean(s_fees[i],axis=0)/np.std(s_fees[i],axis=0)
				plt.title('Distribution of paths SR [fee=%s]'%pct_fees[i])
				plt.hist(tmp,density=True)
				plt.show()

		c=['r','y','m','b']
		aux=pd.DataFrame(np.cumsum(s,axis=0),index=ts)
		aux.plot(color='g',title='Equity curves no fees',legend=False)
		plt.grid(True)
		plt.show()

		ax=aux.plot(color='g',title='Equity curves w/ fees',legend=False)
		for i in range(min(pct_fees.size,len(c))):
			aux=pd.DataFrame(np.cumsum(s_fees[i],axis=0),index=ts)
			ax=aux.plot(ax=ax,color=c[i],legend=False)
		plt.grid(True)
		plt.show()

		plt.title('(Worst path) SR bootstrap distribution')
		plt.hist(b_samples,density=True)
		plt.grid(True)
		plt.show() 

		plt.title('Strategy returns distribution')
		plt.hist(s.ravel(),bins=50,density=True)
		plt.grid(True)
		plt.show()		

		aux=pd.DataFrame(np.sum(w,axis=1),index=ts)
		aux.plot(title='Weights sum',legend=False)
		plt.grid(True)
		plt.show()

		aux=pd.DataFrame(np.sum(np.abs(w),axis=1),index=ts)
		aux.plot(title='Total Leverage',legend=False)
		plt.grid(True)
		plt.show()

		p=w.shape[1]

		for i in range(p):
			aux=pd.DataFrame(w[:,i,:],index=ts)
			aux.plot(title='Weights for asset %s'%(i+1),legend=False)
			plt.grid(True)
			plt.show()		

	def portfolio_post_process(self,seq_fees=False,pct_fees=0,sr_mult=1,n_boot=1000):
		
		pct_fees=self._convert_pct_fees(pct_fees)
		if pct_fees.size!=1:
			print('Warning: only first entry of pct_fees is being considered!')
		# get and joint results for name
		# after that, just use the existent function

		# for each path, for each asset calculate performance with fees
		# and join in a portfolio with the weights
		names=self.paths_output[0].names
		s=[]
		for i,path in enumerate(self.paths_output):
			path_s=[]
			path_pw=[]
			for name in names:
				s_=path.results.get(name)['s']#.values
				w_=path.results.get(name)['w']#.values
				pw_=path.results.get(name)['pw']#.values
				# take always the first entry
				s_fees=self.calculate_fees(s_.values,w_.values[:,:,None],seq_fees,pct_fees)[0]
				path_s.append(pd.DataFrame(s_fees,columns=s_.columns,index=s_.index))
				path_pw.append(pw_)
			path_s=pd.concat(path_s,axis=1)
			path_pw=pd.concat(path_pw,axis=1)
			# fill na with zero
			path_s=path_s.fillna(0)
			path_pw=path_pw.fillna(0)
			# this should always be a absolute value
			# not to override the strategy
			path_pw/=np.sum(np.abs(path_pw),axis=1).values[:,None]
			path_s=pd.DataFrame(np.sum(path_s*path_pw,axis=1),columns=['s'])					  
			s.append(path_s)
		s=pd.concat(s,axis=1)
		
		ts=s.index
		s=s.values
		
		# make plots!
		paths_sr=sr_mult*np.mean(s,axis=0)/np.std(s,axis=0)
		idx_lowest_sr=np.argmin(paths_sr)

		b_samples=self.bootstrap_sharpe(s[:,idx_lowest_sr],n_boot=n_boot)
		b_samples*=sr_mult
		valid=False
		if np.sum(b_samples<0)==0:
			valid=True
		if valid:
			print('-> ACCEPT STRATEGY')
		else:
			print('-> REJECT STRATEGY')
		print()
		print('** Performance summary **')
		print()
		print('Return: ', np.power(sr_mult,2)*np.mean(s))
		print('Standard deviation: ', sr_mult*np.std(s))
		print('Sharpe: ', sr_mult*np.mean(s)/np.std(s))

		# bootstrap estimate of sharpe
		if self.n_paths!=1:
			plt.title('Distribution of paths SR')
			plt.hist(paths_sr,density=True)
			plt.show()

		c=['r','y','m','b']
		aux=pd.DataFrame(np.cumsum(s,axis=0),index=ts)
		aux.plot(color='g',title='Equity curves',legend=False)
		plt.grid(True)
		plt.show()

		plt.title('(Worst path) SR bootstrap distribution')
		plt.hist(b_samples,density=True)
		plt.grid(True)
		plt.show() 

		plt.title('Strategy returns distribution')
		plt.hist(s.ravel(),bins=50,density=True)
		plt.grid(True)
		plt.show()
		
		
def cvbt(data:'Data',model,k_folds=4,seq_path=False,start_fold=0,n_paths=5,burn_f=0.1,min_burn_points=10):
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