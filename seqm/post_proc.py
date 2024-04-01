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

if __name__=='__main__':
	pass


