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


def valid_strategy(s,n_boot,sr_mult,pct_fee=0):
	'''
	check if the paths represent a strategy with a positive
	sharpe ratio via bootstrap from the worst path
	s: numpy (n,k) array with strategy returns
	'''
	paths_sr=sr_mult*np.mean(s,axis=0)/np.std(s,axis=0)
	idx_lowest_sr=np.argmin(paths_sr)
	b_samples=bootstrap_sharpe(s[:,idx_lowest_sr],n_boot=n_boot)
	b_samples*=sr_mult
	valid=False
	if np.sum(b_samples<0)==0:
		valid=True
	print()
	if valid:
		txt='** ACCEPT STRATEGY **' if pct_fee == 0 else '** ACCEPT STRATEGY at FEE=%s **'%pct_fee
		print(txt)		
	else:
		txt='** REJECT STRATEGY **' if pct_fee == 0 else '** REJECT STRATEGY at FEE=%s **'%pct_fee
		print(txt)		
	if s.shape[1]!=1:
		txt='Distribution of paths SHARPE' if pct_fee == 0 else 'Distribution of paths SHARPE at FEE=%s'%pct_fee
		plt.title(txt)
		plt.hist(paths_sr,density=True)
		plt.grid(True)
		plt.show()
	plt.title('(Worst path) SR bootstrap distribution')
	plt.hist(b_samples,density=True)
	plt.grid(True)
	plt.show() 

def performance_summary(s,sr_mult,pct_fee=0):
	print()
	txt='** PERFORMANCE SUMMARY **' if pct_fee == 0 else '** PERFORMANCE SUMMARY at FEE=%s **'%pct_fee
	print(txt)
	print()
	print('Return: ', np.power(sr_mult,2)*np.mean(s))
	print('Standard deviation: ', sr_mult*np.std(s))
	print('Sharpe: ', sr_mult*np.mean(s)/np.std(s))
	print()

def equity_curve(s,ts,color='g',pct_fee=0):
	title='Equity curves' if pct_fee == 0 else 'Equity curves FEE=%s'%pct_fee
	s_df=pd.DataFrame(np.cumsum(s,axis=0),index=ts)
	s_df.plot(color=color,title=title,legend=False)	
	plt.grid(True)
	plt.show()

def returns_distribution(s,pct_fee=0,bins=50):
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
			title='Weight for asset %s'%(i+1) if cols is not None else cols[i]
			aux=pd.DataFrame(w[:,i,:],index=ts)
			aux.plot(title=title,legend=False)
			plt.grid(True)
			plt.show()		

def post_process(paths:List[Dict[str,pd.DataFrame]],pct_fee=0.,seq_fees=False,sr_mult=1,n_boot=1000,key=None):
	'''
	paths: list of dict like [{'dataset 1':df,'dataset 2':df},{'dataset 1':df,'dataset 2':df},...]
		each element of the paths list is the result for a given path
		for each path (element of the list) we have the results for the different datasets

	'''
	if len(paths) == 0:
		print('No paths to process!')
		return
	keys = [k for k in paths[0]]
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
	s=calculate_fees(s,w,seq_fees,pct_fee)

	# post processing
	
	equity_curve(s,ts,color='g',pct_fee=pct_fee)	
	
	returns_distribution(s,pct_fee=pct_fee,bins=50)
	
	visualize_weights(w,ts)

	valid_strategy(s,n_boot,sr_mult,pct_fee=pct_fee)

	performance_summary(s,sr_mult,pct_fee=pct_fee)
	

def portfolio_post_process(paths:List[Dict[str,pd.DataFrame]],pct_fee=0.,seq_fees=False,sr_mult=1,n_boot=1000,use_portfolio_weight=True):

	keys = [k for k in paths[0]]

	paths_s=[]
	paths_pw=[]
	for path in paths:
		path_s=[]
		path_pw=[]
		for key in keys:
			df=path.get(key)
			s=df[[STRATEGY_COLUMN]]
			w=df.iloc[:, df.columns.str.startswith(WEIGHT_PREFIX_COLUMNS)]
			pw=df[[PORTFOLIO_WEIGHT_COLUMN]]
			s=pd.DataFrame(calculate_fees(s.values,w.values[:,:,None],seq_fees,pct_fee),columns=s.columns,index=s.index)
			path_s.append(s)
			path_pw.append(pw)
		path_s=pd.concat(path_s,axis=1)
		path_pw=pd.concat(path_pw,axis=1)
		path_s.columns=keys
		path_pw.columns=keys
		# fill na with zero
		path_s=path_s.fillna(0)
		path_pw=path_pw.fillna(0)
		path_pw/=np.sum(np.abs(path_pw),axis=1).values[:,None]
		
		path_s=pd.DataFrame(np.sum(path_s*path_pw,axis=1),columns=['s'])
		paths_s.append(path_s)

		paths_pw.append(path_pw)
	s=pd.concat(paths_s,axis=1)
	w=np.stack([e.values for e in paths_pw],axis=2)
	ts=s.index
	s=s.values

	equity_curve(s,ts,color='g',pct_fee=pct_fee)	
	
	returns_distribution(s,pct_fee=pct_fee,bins=50)
	
	visualize_weights(w,ts,keys)

	valid_strategy(s,n_boot,sr_mult,pct_fee=pct_fee)

	performance_summary(s,sr_mult,pct_fee=pct_fee)

if __name__=='__main__':
	paths=load_file('paths_dev.pkl')
	portfolio_post_process(paths,pct_fee=0.,seq_fees=False,sr_mult=1,n_boot=1000,use_portfolio_weight=True)
	# post_process(paths=paths,pct_fee=0,seq_fees=False,sr_mult=1,n_boot=1000,key=None)
	




