# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import copy
import tqdm
import pickle
from typing import List, Dict



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

class Datum(object):
	
	def __init__(self,data):
		'''
		data: pandas DataFrame of list of DataFrames
		'''
		if isinstance(data,pd.DataFrame):
			data=[data]		
		self.check_cols(data)
		
		self.y=[]
		self.x=[]
		self.ts=None		
				
		for d in data:
			y,x,ts=self.parse_data(d)
			if self.ts is None:
				self.ts=ts
			else:
				self.ts=self.ts.append(ts)

			self.y.append(y)
			if x is not None:
				self.x.append(x)
			# self.ts+=ts

		self.y=np.vstack(self.y)
		if len(self.x)!=0:
			self.x=np.vstack(self.x)
		else:
			self.x=None
		self.n=self.y.shape[0]
		self.p=self.y.shape[1]
		self.aux_idx=np.arange(self.n,dtype=int)

	def view(self):
		print('** Datum **')
		print(
			pd.DataFrame(
						np.hstack((self.y,self.x)),
						columns=self.y_cols+self.x_cols,
						index=self.ts)
			)
	
	def get_model_train_input(self,burn_f=0.1,min_burn_points=1):
		train_idx=random_subsequence(self.aux_idx,burn_f,min_burn_points)
		model_inputs={'y':self.y[train_idx],'x':self.x[train_idx]}
		return model_inputs

	def get_model_test_input(self):
		return {'y':self.y,'x':self.x}	
	
	def check_cols(self,data):
		'''
		data: list of DataFrames
		'''
		self.cols=data[0].columns.tolist()
		self.y_cols=[e for e in self.cols if e[0]=='y']	
		self.x_cols=[e for e in self.cols if e[0]=='x']			
		assert len(self.y_cols)!=0,"data must have columns like y1,y2,..."		
		for e in data:
			assert e.columns.tolist()==self.cols
			
	def parse_data(self,data):
		'''
		data: pandas DataFrame with columns y1,y2,..,<x1>,<x2>,..,<idx>,<z>
		'''
		y=None
		x=None
		ts=None

		y=data[self.y_cols].values
		ts=data.index
		# get x (features)
		if len(self.x_cols)!=0:
				x=data[self.x_cols].values	
		return y,x,ts 
	
	def build_train_test(self,k_folds=4,seq_path=False,start_fold=0)->List['TrainTestDatum']:
		'''
		Build train and test sets from data
		outputs:
		list of dict of TrainTestDatum
		# TO DO
		# ADD SUPPORT FOR MULTISEQUENCE AND THE OTHER FEATURES IN CVBT
		'''

		if seq_path:
			start_fold=max(1,start_fold)		
		#print('seq_path: ', seq_path)
		#print('start_fold: ',start_fold)
		
		train_test_datum_lst=[]
		n=self.y.shape[0]
		idx=np.arange(n,dtype=int)
		idx_folds=np.array_split(idx,k_folds)
		idx_folds_idx=np.arange(k_folds,dtype=int)


		for i in range(start_fold,k_folds):
			# only passing x and y for now...
			df_test=pd.DataFrame(
							np.hstack((self.y[idx_folds[i]],self.x[idx_folds[i]])),
							columns=self.y_cols+self.x_cols,
							index=self.ts[idx_folds[i]]
							)
			df_train=[]
			# indexes before
			tmp_idx=idx_folds_idx[idx_folds_idx<i]
			if tmp_idx.size!=0:
				tmp_idx_folds=np.hstack([idx_folds[e] for e in tmp_idx])
				df_train.append(
								pd.DataFrame(
									np.hstack((self.y[tmp_idx_folds],self.x[tmp_idx_folds])),
									columns=self.y_cols+self.x_cols,
									index=self.ts[tmp_idx_folds]
									)
							   )
			# indexes after if path is not sequential
			if not seq_path:
				tmp_idx=idx_folds_idx[idx_folds_idx>i]
				if tmp_idx.size!=0:
					# print(idx_folds)
					tmp_idx_folds=np.hstack([idx_folds[e] for e in tmp_idx])
					df_train.append(
									pd.DataFrame(
										np.hstack((self.y[tmp_idx_folds],self.x[tmp_idx_folds])),
										columns=self.y_cols+self.x_cols,
										index=self.ts[tmp_idx_folds]
										)
								   )
			train_test_datum_lst.append(TrainTestDatum(train=[Datum(e) for e in df_train],test=Datum(df_test)))#{'test':Datum(df_test),'train':[Datum(e) for e in df_train]})
		return train_test_datum_lst

class TrainTestDatum(object):
	def __init__(self,train:List['Datum']=None,test:'Datum'=None):
	# def __init__(self,train:List[Datum]=None,test:Datum=None):	
		self.train=train
		if isinstance(self.train,Datum):
			self.train=[self.train]
		self.test=test
		self.n_train_datum=len(self.train)
		self.n_test=self.test.n
		self.p=self.test.p
		self.ts=self.test.ts
	
	def get_model_train_input(self,burn_f=0.1,min_burn_points=1):
		
		# choose one train Datum at random with probability
		# weighted by the number of observations
		# in this class this method is just a selector of training Datum
		p=np.array([self.train[i].n for i in range(self.n_train_datum)],dtype=float)
		p/=np.sum(p)

		idx_train_datum=np.random.choice(np.arange(self.n_train_datum),p=p)	   
		return self.train[idx_train_datum].get_model_train_input(burn_f,min_burn_points)
	
	def get_model_test_input(self):
		return self.test.get_model_test_input()
	
	def view(self):
		print('**** TrainTestDatum ****')
		print('*** Train ***')		
		for i in range(self.n_train_datum):
			self.train[i].view()
		print('*** Test ***')
		self.test.view()


# EVALUATE THE MODEL
# def evaluate(parameters,model,y,x,idx=None,z=None):
def evaluate(model,y,x,idx=None,z=None):	
	# note: fees can be calculated after this is done!
	if x is not None:
		assert y.shape[0]==x.shape[0],"x and y must have the same number of observations"
	n=y.shape[0]
	p=y.shape[1]
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
			model_inputs={'y':y[idx[l][0]:i]}
			if x is not None:
				model_inputs.update({'x':x[idx[l][0]:i]})
				model_inputs.update({'xq':x[i]})
			if z is not None:
				model_inputs.update({'z':z[i]})
			w=model.get_weight(**model_inputs)
			weights[i]=w
			s[i]=np.dot(y[i],w)
	# compute fees here	
	return s,weights


class CVBTOut(object):
	def __init__(self,s,weights,ts):
		self.s=s
		self.weights=weights		
		self.ts=ts		
		self.s_fees=None
		#self.seq_fees=seq_fees
		#self.pct_fees=pct_fees
		#self.sr_mult=sr_mult
		#self.n_boot=n_boot
		# self.n_paths=n_paths
		self.n_paths=self.s.shape[1]		
		# seq_fees=None,pct_fees=None,sr_mult=1,n_boot=100,n_paths=1
		
	def calculate_fees(self,seq_fees,pct_fees):
		if seq_fees:
			dw=np.abs(self.weights[1:]-self.weights[:-1])
			dw=np.vstack(([np.zeros_like(dw[0])],dw))
			dw=np.sum(dw,axis=1)
		else:
			dw=np.sum(np.abs(self.weights),axis=1)
		s_fees=np.zeros((pct_fees.size,self.s.shape[0],self.s.shape[1]))
		for i in range(pct_fees.size):
			s_fees[i]=self.s-pct_fees[i]*dw
		return s_fees

	def bootstrap_sharpe(self,s,n_boot=1000):
		l=s.size
		idx=np.arange(l,dtype=int)
		idx_=np.random.choice(idx,(l,n_boot),replace=True)
		s_=s[idx_]
		boot_samples=np.mean(s_,axis=0)/np.std(s_,axis=0)
		return boot_samples

	def post_process(self,seq_fees=False,pct_fees=[0],sr_mult=1,n_boot=1000):

		if isinstance(pct_fees,float):
			pct_fees=np.array([pct_fees])
		elif isinstance(pct_fees,int):
			pct_fees=np.array([pct_fees])
		else:
			pct_fees=np.array(pct_fees)   

		s_fees=self.calculate_fees(seq_fees,pct_fees)

		# make plots!
		paths_sr=sr_mult*np.mean(self.s,axis=0)/np.std(self.s,axis=0)
		idx_lowest_sr=np.argmin(paths_sr)

		b_samples=self.bootstrap_sharpe(self.s[:,idx_lowest_sr],n_boot=n_boot)
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
		print('Return: ', np.power(sr_mult,2)*np.mean(self.s))
		print('Standard deviation: ', sr_mult*np.std(self.s))
		print('Sharpe: ', sr_mult*np.mean(self.s)/np.std(self.s))
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
		aux=pd.DataFrame(np.cumsum(self.s,axis=0),index=self.ts)
		aux.plot(color='g',title='Equity curves no fees',legend=False)
		plt.grid(True)
		plt.show()

		ax=aux.plot(color='g',title='Equity curves w/ fees',legend=False)
		for i in range(min(pct_fees.size,len(c))):
			aux=pd.DataFrame(np.cumsum(s_fees[i],axis=0),index=self.ts)
			ax=aux.plot(ax=ax,color=c[i],legend=False)
		plt.grid(True)
		plt.show()

		plt.title('(Worst path) SR bootstrap distribution')
		plt.hist(b_samples,density=True)
		plt.grid(True)
		plt.show() 

		plt.title('Strategy returns distribution')
		plt.hist(self.s.ravel(),bins=50,density=True)
		plt.grid(True)
		plt.show()		

		aux=pd.DataFrame(np.sum(self.weights,axis=1),index=self.ts)
		aux.plot(title='Weights sum',legend=False)
		plt.grid(True)
		plt.show()

		aux=pd.DataFrame(np.sum(np.abs(self.weights),axis=1),index=self.ts)
		aux.plot(title='Total Leverage',legend=False)
		plt.grid(True)
		plt.show()

		p=self.weights.shape[1]

		for i in range(p):
			aux=pd.DataFrame(self.weights[:,i,:],index=self.ts)
			aux.plot(title='Weights for asset %s'%(i+1),legend=False)
			plt.grid(True)
			plt.show()



def cvbt(train_test_datum_lst:List['TrainTestDatum'],model,):
	'''
	train_test_datum: list of dict of TrainTestDatum
	model: instance of the model class
	'''
	
	# parameters to receive as inputs
	burn_f=0.1
	min_burn_points=1	
	n_paths=20
	
	# number of train/test sets
	n_train_test=len(train_test_datum_lst)
	# total number of test observations
	# build test idx to store performance and weights
	# store test index (datetimes, etc)
	n_test=0
	test_idx=[]
	ts=[]
	aux_max=0
	for i in range(n_train_test):
		n_test+=train_test_datum_lst[i].n_test
		test_idx.append(np.arange(train_test_datum_lst[i].n_test))
		test_idx[-1]+=aux_max
		aux_max=test_idx[-1][-1]+1
		ts+=train_test_datum_lst[i].ts
	# number of variables
	p=train_test_datum_lst[0].p
	# store performance
	s=np.zeros((n_test,n_paths),dtype=np.float64) 
	# store weights
	weights=np.zeros((n_test,p,n_paths),dtype=np.float64)		
	# run this for all paths
	for m in tqdm.tqdm(range(n_paths)):	
		# CVBT routine
		# select random set of training data
		# evaluate model
		# store results
		for i in range(n_train_test):		
			# create copy of model
			local_model=copy.deepcopy(model)
			# train
			model_train_input=train_test_datum_lst[i].get_model_train_input(burn_f,min_burn_points)
			local_model.estimate(**model_train_input)
			# evaluate	 
			s[test_idx[i],m],weights[test_idx[i],:,m]=evaluate(
													local_model,
													**train_test_datum_lst[i].get_model_test_input()
													)
	out=CVBTOut(s,weights,ts)
	return out			




if __name__=='__main__':
	print('ola')
