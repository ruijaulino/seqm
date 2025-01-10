from typing import List,Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
	from .loads import load_file
	from .constants import * 
except ImportError:
	from loads import load_file
	from constants import *

# stats functions
def calculate_fees(s,weights,seq_fees,pct_fee):
	'''
	s: numpy (n,k) array with several return sequences
	seq_fees: bool with the indication that the fees are sequential
	pct_fees: scalar or array with the fees to be considered
	returns
	s_fees: numpy (n_fees,n,k) with the returns sequences for different fees
	'''	
	if seq_fees:
		dw=np.abs(weights[1:]-weights[:-1])
		dw=np.vstack(([np.zeros_like(dw[0])],dw))
		dw=np.sum(dw,axis=1)
	else:
		dw=np.sum(np.abs(weights),axis=1)
	return s-pct_fee*dw

def bootstrap_sharpe(s,n_boot=1000):
	'''
	bootstrat samples of sharpe ratio for an array of returns s ~ (n,)
	'''
	l=s.size
	idx=np.arange(l,dtype=int)
	idx_=np.random.choice(idx,(l,n_boot),replace=True)
	s_=s[idx_]
	boot_samples=np.mean(s_,axis=0)/np.std(s_,axis=0)
	return boot_samples


def valid_strategy(s,n_boot,sr_mult,pct_fee=0, view = True):
	'''
	check if the paths represent a strategy with a positive
	sharpe ratio via bootstrap from the worst path
	s: numpy (n,k) array with strategy returns
	'''
	if isinstance(pct_fee, dict): pct_fee = list(pct_fee.values())[0]
	paths_sr=sr_mult*np.mean(s,axis=0)/np.std(s,axis=0)
	idx_lowest_sr=np.argmin(paths_sr)
	b_samples=bootstrap_sharpe(s[:,idx_lowest_sr],n_boot=n_boot)
	b_samples*=sr_mult
	valid=False
	if np.sum(b_samples<0)==0:
		valid=True
	if valid:
		txt='** ACCEPT STRATEGY **' if pct_fee == 0 else '** ACCEPT STRATEGY at FEE=%s **'%pct_fee
		if view: print(txt)		
	else:
		txt='** REJECT STRATEGY **' if pct_fee == 0 else '** REJECT STRATEGY at FEE=%s **'%pct_fee
		if view: print(txt)		
	if s.shape[1]!=1:
		txt='Distribution of paths SHARPE' if pct_fee == 0 else 'Distribution of paths SHARPE at FEE=%s'%pct_fee
		if view:
			plt.title(txt)
			plt.hist(paths_sr,density=True)
			plt.grid(True)
			plt.show()
	if view:
		plt.title('(Worst path) SR bootstrap distribution')
		plt.hist(b_samples,density=True)
		plt.grid(True)
		plt.show() 
	return valid

def performance_summary(s,sr_mult,pct_fee=0):
	if isinstance(pct_fee, dict): pct_fee = list(pct_fee.values())[0]
	print()
	txt='** PERFORMANCE SUMMARY **' if pct_fee == 0 else '** PERFORMANCE SUMMARY at FEE=%s **'%pct_fee
	print(txt)
	print()
	print('Return: ', np.power(sr_mult,2)*np.mean(s))
	print('Standard deviation: ', sr_mult*np.std(s))
	print('Sharpe: ', sr_mult*np.mean(s)/np.std(s))
	print()

def equity_curve(s,ts,color='g',pct_fee=0, title:str = 'Equity curve'):
	if isinstance(pct_fee, dict): pct_fee = list(pct_fee.values())[0]
	title=title if pct_fee == 0 else title+' FEE=%s'%pct_fee
	s_df=pd.DataFrame(np.cumsum(s,axis=0),index=ts)
	s_df.plot(color=color,title=title,legend=False)	
	plt.grid(True)
	plt.show()

def returns_distribution(s,pct_fee=0,bins=50):
	if isinstance(pct_fee, dict): pct_fee = list(pct_fee.values())[0]
	title='Strategy returns distribution' if pct_fee == 0 else 'Strategy returns distribution FEE=%s'%pct_fee
	plt.title(title)
	plt.hist(s.ravel(),bins=bins,density=True)
	plt.grid(True)
	plt.show()			

def visualize_weights(w,ts,cols=None):
	aux=pd.DataFrame(np.sum(w,axis=1),index=ts)
	aux.plot(title='Weights sum',legend=False)
	plt.grid(True)
	plt.show()

	aux=pd.DataFrame(np.sum(np.abs(w),axis=1),index=ts)
	aux.plot(title='Total Leverage',legend=False)
	plt.grid(True)
	plt.show()

	p=w.shape[1]
	if p>1:
		for i in range(p):
			title='Weight for asset %s'%(i+1) if cols is None else 'Weight for '+cols[i]
			aux=pd.DataFrame(w[:,i,:],index=ts)
			aux.plot(title=title,legend=False)
			plt.grid(True)
			plt.show()	


def filter_paths(paths:List[Dict[str,pd.DataFrame]],start_date:str = '',end_date:str = ''):
	# make a copy
	if start_date == '' and end_date == '': return paths
	f_paths=[]
	for elem in paths:
		tmp={}
		for k,df in elem.items():
			f_df = df
			if start_date != '':
				f_df = f_df[f_df.index > start_date]
			if end_date != '':
				f_df = f_df[f_df.index < end_date]
			tmp.update({k:f_df})
		f_paths.append(tmp)
	return f_paths


def check_valid(paths:List[Dict[str,pd.DataFrame]],pct_fee=0.,seq_fees=False,sr_mult=1,n_boot=1000,key=None,start_date='',end_date='', output_paths:bool = True,*args,**kwargs):

	paths = filter_paths(paths,start_date,end_date)

	if len(paths) == 0:
		print('No paths to process!')
		return
	keys = [k for k in paths[0]]

	# transform into dict
	if not isinstance(pct_fee, dict):
		tmp = {}
		for k in keys: tmp.update({k:pct_fee})
		pct_fee = tmp

	# by default use the results for the first dataframe used as input
	# this will work by default because, in general, there is only one
	key = key if key is not None else keys[0]
	# get and joint results for key
	s=[]
	w=[]	

	for path in paths:
		df=path.get(key)
		s.append(df[[STRATEGY_COLUMN]].values)
		w.append(df.iloc[:, df.columns.str.startswith(WEIGHT_PREFIX_COLUMNS)].values)
	if len(s)==0:
		return False
	ts=paths[0].get(key).index

	# stack arrays
	s=np.hstack(s)
	w=np.stack(w,axis=2)
	s=calculate_fees(s,w,seq_fees,pct_fee.get(key, 0))

	return valid_strategy(s,n_boot,sr_mult,pct_fee=pct_fee, view = False)


def post_process(paths:List[Dict[str,pd.DataFrame]],pct_fee=0.,seq_fees=False,sr_mult=1,n_boot=1000,key=None,start_date='',end_date='', output_paths:bool = True,*args,**kwargs):
	'''
	paths: list of dict like [{'dataset 1':df,'dataset 2':df},{'dataset 1':df,'dataset 2':df},...]
		each element of the paths list is the result for a given path
		for each path (element of the list) we have the results for the different datasets
	'''
	
	paths = filter_paths(paths,start_date,end_date)

	if len(paths) == 0:
		print('No paths to process!')
		return
	keys = [k for k in paths[0]]

	# transform into dict
	if not isinstance(pct_fee, dict):
		tmp = {}
		for k in keys: tmp.update({k:pct_fee})
		pct_fee = tmp

	# by default use the results for the first dataframe used as input
	# this will work by default because, in general, there is only one
	key = key if key is not None else keys[0]
	# get and joint results for key
	s=[]
	w=[]	

	for path in paths:
		df=path.get(key)
		s.append(df[[STRATEGY_COLUMN]].values)
		w.append(df.iloc[:, df.columns.str.startswith(WEIGHT_PREFIX_COLUMNS)].values)
	if len(s)==0:
		print('No results to process!')
		return
	ts=paths[0].get(key).index

	# stack arrays
	s=np.hstack(s)
	w=np.stack(w,axis=2)
	s=calculate_fees(s,w,seq_fees,pct_fee.get(key, 0))


	# post processing
	
	equity_curve(s,ts,color='g',pct_fee=pct_fee)	
	
	returns_distribution(s,pct_fee=pct_fee,bins=50)
	
	visualize_weights(w,ts)

	valid_strategy(s,n_boot,sr_mult,pct_fee=pct_fee)

	performance_summary(s,sr_mult,pct_fee=pct_fee)
	
	if output_paths: return pd.DataFrame(s, index = ts, columns = [f'path_{i+1}' for i in range(s.shape[1])])

def portfolio_post_process(paths:List[Dict[str,pd.DataFrame]],pct_fee=0.,seq_fees=False,sr_mult=1,n_boot=1000,view_weights=True,use_pw=True,multiplier=1,start_date='',end_date='', n_boot_datasets:int = None, output_paths:bool = True, *args, **kwargs):
 
	paths = filter_paths(paths,start_date,end_date)

	keys = [k for k in paths[0]]

	# transform into dict
	if not isinstance(pct_fee, dict):
		tmp = {}
		for k in keys: tmp.update({k:pct_fee})
		pct_fee = tmp

	if not n_boot_datasets: n_boot_datasets = 1

	paths_s=[]
	paths_pw=[]
	paths_leverage=[]
	paths_net_leverage = []
	paths_n_datasets=[]

	paths_s_datasets_boot = []
	for path in paths:
		for k in range(n_boot_datasets):

			# at least the first is always with the full dataset
			if k == 0:
				keys_sample = keys
			else:
				keys_sample = list(set(np.random.choice(keys, len(keys))))

			path_s=[]
			path_pw=[]
			path_w_abs_sum=[] # to build leverage
			path_w_sum = []
			for key in keys_sample:
				df=path.get(key)
				s=df[[STRATEGY_COLUMN]]
				w=df.iloc[:, df.columns.str.startswith(WEIGHT_PREFIX_COLUMNS)]
				pw=df[[PORTFOLIO_WEIGHT_COLUMN]]
				if not use_pw:
					pw=pd.DataFrame(np.ones_like(pw.values),columns=pw.columns,index=pw.index)			
				s=pd.DataFrame(calculate_fees(s.values,w.values[:,:,None],seq_fees,pct_fee.get(key, 0)),columns=s.columns,index=s.index)
				path_s.append(s)
				path_pw.append(pw)
				path_w_abs_sum.append(pd.DataFrame(w.abs().sum(axis=1),columns=[key]))
				path_w_sum.append(pd.DataFrame(w.abs().sum(axis=1),columns=[key]))

			path_s=pd.concat(path_s,axis=1)
			path_pw=pd.concat(path_pw,axis=1)
			path_w_abs_sum=pd.concat(path_w_abs_sum,axis=1)
			path_w_sum=pd.concat(path_w_sum,axis=1)
			path_s.columns=keys_sample
			path_pw.columns=keys_sample
			path_w_abs_sum.columns=keys_sample
			path_w_sum.columns = keys_sample
			# fill na with zero
			path_s=path_s.fillna(0)
			path_pw=path_pw.fillna(method = 'ffill')
			path_w_abs_sum=path_w_abs_sum.fillna(0)
			path_w_sum = path_w_sum.fillna(0)

			path_pw/=np.sum(np.abs(path_pw),axis=1).values[:,None]
			path_pw*=multiplier
			non_zero_counts = path_pw.apply(lambda row: (row != 0).sum(), axis=1)
		
			path_s=pd.DataFrame(np.sum(path_s*path_pw,axis=1),columns=['s'])

			paths_s_datasets_boot.append(path_s)

			if k == 0:
				paths_s.append(path_s)
				paths_pw.append(path_pw)
				paths_net_leverage.append(pd.DataFrame(np.sum(path_w_sum*path_pw,axis=1),columns=['s']))
				paths_leverage.append(pd.DataFrame(np.sum(path_w_abs_sum*path_pw,axis=1),columns=['s']))
				paths_n_datasets.append(pd.DataFrame(non_zero_counts,columns=['n']))
				
			

	s=pd.concat(paths_s,axis=1)

	w=np.stack([e.values for e in paths_pw],axis=2)
	lev=pd.concat(paths_leverage,axis=1)
	net_lev = pd.concat(paths_net_leverage, axis = 1)
	n_datasets=pd.concat(paths_n_datasets,axis=1)

	

	out = s.copy(deep = True)
	out.columns = [f'path_{i+1}' for i in range(len(out.columns))]

	ts=s.index
	s=s.values



	equity_curve(s,ts,color='g',pct_fee=pct_fee)

	if n_boot_datasets > 1:
		s_dataset_boot = pd.concat(paths_s_datasets_boot,axis=1)
		equity_curve(s_dataset_boot,s_dataset_boot.index,color='g',pct_fee=pct_fee, title = 'Equity curve dataset bootstrap')
	
	returns_distribution(s,pct_fee=pct_fee,bins=50)
	
	if view_weights: visualize_weights(w,ts,keys)
	lev.plot(legend=False,title='Paths Leverage')
	plt.grid(True)
	plt.show()
	
	net_lev.plot(legend=False,title='Paths Net Leverage')
	plt.grid(True)
	plt.show()




	n_datasets.plot(legend=False,title='Number of datasets')
	plt.grid(True)
	plt.show()	

	valid_strategy(s,n_boot,sr_mult,pct_fee=pct_fee)

	performance_summary(s,sr_mult,pct_fee=pct_fee)

	if output_paths: return out

if __name__=='__main__':
	paths=load_file('paths_dev.pkl')
	portfolio_post_process(paths,pct_fee=0.,seq_fees=False,sr_mult=1,n_boot=1000,view_weights=True,use_pw=True)
	# post_process(paths=paths,pct_fee=0,seq_fees=False,sr_mult=1,n_boot=1000,key=None)
	




