# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import tqdm
import pickle

# UTILITIES
def parse_data(data):
	'''
	data: pandas DataFrame with columns y1,y2,..,<x1>,<x2>,..,<idx>,<z>
	'''	
	y=None
	x=None
	z=None
	idx=None	
	
	cols=data.columns	
	# get x
	y_cols=[e for e in cols if e[0]=='y']	
	assert len(y_cols)!=0,"data must have columns like y1,y2,..."
	y=data[y_cols].values	
	# get x (features)
	x_cols=[e for e in cols if e[0]=='x']
	if len(x_cols)!=0:
		x=data[x_cols].values		
	# get idx (subsequence index)
	if 'idx' in cols:
		tmp=data['idx'].values  
		# idx need to be an array like 000,1111,2222,33,444
		# but can be something like 111,000,11,00000,1111,22
		# i.e, the indexes of the sequence need to be in order for the
		# cvbt to work. Fix the array in case this is not verified
		# this is a bit of overhead but has to be this way
		idx=np.zeros(tmp.size,dtype=int)
		aux=tmp[1:]-tmp[:-1]
		idx[np.where(aux!=0)[0]+1]=1
		idx=np.cumsum(idx)
		
	# get z (observed state sequence; for example, weekday)
	if 'z' in cols:
		z=data['z'].values
	return y,x,z,idx


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



def build_train_folds(index_exclude,k_folds,seq_path=False):
	# select one of the sides with probability 
	# proportional to the number of observations
	if seq_path:
		assert index_exclude>0,"seq_path cannot be created from the first fold"
		folds_idx_before=np.arange(0,index_exclude,dtype=int)
		return folds_idx_before

	p=np.zeros(2)	
	folds_idx_before=np.arange(0,index_exclude,dtype=int)
	folds_idx_after=np.arange(index_exclude+1,k_folds,dtype=int)
	p[0]=folds_idx_before.size
	p[1]=folds_idx_after.size
	p/=np.sum(p)
	if np.random.choice([0,1],p=p)==0:
		return folds_idx_before
	else:
		return folds_idx_after


def convert_msidx(msidx):
	'''
	msidx: numpy (n,) array like 0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,...
		with the indication where the subsequences are
		creates an array like
		[
		[0,3],
		[3,7],
		...
		]
		with the indexes where the subsequences start and end
	'''
	# previous implementation
	#out=[]
	#c=0
	#for i in range(1,msidx.size):
	#	 if msidx[i]!=msidx[i-1]:
	#			out.append([c,i])
	#			c=i
	#out.append([c,msidx.size])
	#out=np.array(out,dtype=int)
	aux=msidx[1:]-msidx[:-1]
	aux=np.where(aux!=0)[0]
	aux+=1
	aux_left=np.hstack(([0],aux))
	aux_right=np.hstack((aux,[msidx.size]))
	out=np.hstack((aux_left[:,None],aux_right[:,None]))
	return out



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


def calculate_fees(s,weights,seq_fees,pct_fees):
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


def bootstrap_sharpe(s,n_boot=1000):
	l=s.size
	idx=np.arange(l,dtype=int)
	idx_=np.random.choice(idx,(l,n_boot),replace=True)
	s_=s[idx_]
	boot_samples=np.mean(s_,axis=0)/np.std(s_,axis=0)
	return boot_samples		




class Parameters(object):	
	def __init__(
				self,
				k_folds=5,
				burn_f=0.1,
				min_burn_points=1,
				n_paths=1,   
				seq_path=False,
				start_fold=0,
				pct_fees=0,
				seq_fees=True,
				sr_mult=np.sqrt(250),
				n_boot=1000				
				):
		'''
		k_folds: number of folds for CV
		burn_f: fraction of data wrt to train size of
			data near test to be burned
		n_paths: number of bt paths to generate		
		'''
		self.k_folds=k_folds
		self.burn_f=burn_f
		self.n_paths=n_paths
		self.min_burn_points=min_burn_points
		self.seq_path=seq_path
		self.start_fold=start_fold
		self.pct_fees=pct_fees
		self.seq_fees=seq_fees
		self.sr_mult=sr_mult
		self.n_boot=n_boot
		# 
		if isinstance(self.pct_fees,float):
			self.pct_fees=np.array([self.pct_fees])
		elif isinstance(self.pct_fees,int):
			self.pct_fees=np.array([self.pct_fees])			
		else:
			self.pct_fees=np.array(self.pct_fees)		


def train(data,model,view=True,plot_hist=True):
	'''	
	'''
	# parse data into arrays
	y,x,z,msidx=parse_data(data)

	has_msidx=True
	if msidx is None:
		has_msidx=False

	# create copy of model
	local_model=copy.deepcopy(model)		

	# estimate model
	model_inputs={'y':y,'idx':None}
	if x is not None:
		model_inputs.update({'x':x})   
	if has_msidx:			
		model_inputs.update({'idx':convert_msidx(msidx)}) 
	if z is not None:
		model_inputs.update({'z':z})

	local_model.estimate(**model_inputs)
	if view:
		local_model.view(plot_hist)

	return local_model


def cvbt(data,model,parameters):
	'''
	Cross-validation backtest
	model must have methods .estimate and .get_weight
	data: pandas DataFrame like 
					y1,y2,...,yp,x1,x2,...,xq,z,idx
		timestamp 1
		timestamp 2
		...
		timestamp T
		y<.> are the columns with the returns
		x<.> are the columns with features (optional)
		z is the column with a observed state condition the model (optional)
		idx is the column with multisequence index (optional)
		
		Note:
		idx can represent the behavious over a period like a week of hour
		z can be used to make specific model for each weekday for example		
	'''
	# parse data into arrays
	y,x,z,msidx=parse_data(data)
	# build folds on multisequence indexes 
	# this creates a bit of overhead but allows for a clean
	# implementation of the cross validation with multiple sequences
	
	has_msidx=True
	if msidx is None:
		has_msidx=False
		msidx=np.arange(y.shape[0],dtype=int)
		nu=y.shape[0]
	else:
		nu=np.unique(msidx).size
	
	assert nu>3*parameters.k_folds,"Not enought points to run cvbt"
	
	n=y.shape[0]
	p=y.shape[1]
	
	idx=np.arange(nu,dtype=int)
	
	idx_folds=np.array_split(idx,parameters.k_folds)
	
	s=np.zeros((n,parameters.n_paths),dtype=np.float64) 
	weights=np.zeros((n,p,parameters.n_paths),dtype=np.float64)
		
	
	start_fold=parameters.start_fold
	if parameters.seq_path:
		start_fold=max(1,parameters.start_fold)
	
	# run this for all paths
	for m in tqdm.tqdm(range(parameters.n_paths)):
		# CVBT ROUTINE
		for i in range(start_fold,parameters.k_folds):

			# get folds to be used for training (select folds to the left or to the right)
			train_folds=build_train_folds(i,parameters.k_folds,parameters.seq_path)

			# build the actual indexes on data for training
			train_idx=np.hstack([idx_folds[e] for e in train_folds])

			# burn indexes/get random subsequence
			train_idx=random_subsequence(train_idx,parameters.burn_f,parameters.min_burn_points)		

			# map (possible) subsequence indexes to true indexes in x (and f)
			train_idx=np.where(np.in1d(msidx,train_idx))[0]
			test_idx=np.where(np.in1d(msidx,idx_folds[i]))[0]

			# create copy of model
			local_model=copy.deepcopy(model)		

			# estimate model
			model_inputs={'y':y[train_idx],'idx':None}
			if x is not None:
				model_inputs.update({'x':x[train_idx]})   
			if has_msidx:			
				model_inputs.update({'idx':convert_msidx(msidx[train_idx])}) 
			if z is not None:
				model_inputs.update({'z':z[train_idx]})

			local_model.estimate(**model_inputs)

			y_test=y[test_idx]
			x_test=None
			idx_test=None
			z_test=None
			if x is not None:
				x_test=x[test_idx]
			if has_msidx:
				idx_test=convert_msidx(msidx[test_idx])
			if z is not None:
				z_test=z[test_idx]

			# evaluate model
			s[test_idx,m],weights[test_idx,:,m]=evaluate(local_model,y_test,x_test,idx_test,z_test)
	
	out=Results(s,weights,data.index,parameters.seq_fees,parameters.pct_fees,parameters.sr_mult,parameters.n_boot,parameters.n_paths)	
	return out


class Results(object):
	def __init__(self,s,weights,ts,seq_fees=None,pct_fees=None,sr_mult=1,n_boot=100,n_paths=1):
		self.s=s
		self.weights=weights
		self.ts=ts
		self.s_fees=None

		self.seq_fees=seq_fees
		self.pct_fees=pct_fees
		self.sr_mult=sr_mult
		self.n_boot=n_boot
		self.n_paths=n_paths		
		
	def post_process(self,seq_fees=None,pct_fees=None):
		if seq_fees is None:
			seq_fees=self.seq_fees
		if pct_fees is None:
			pct_fees=self.pct_fees
		
		if isinstance(pct_fees,float):
			pct_fees=np.array([pct_fees])
		elif isinstance(pct_fees,int):
			pct_fees=np.array([pct_fees])
		else:
			pct_fees=np.array(pct_fees)   
		
		self.s_fees=calculate_fees(self.s,self.weights,seq_fees,pct_fees)
		
		# make plots!
		
		paths_sr=self.sr_mult*np.mean(self.s,axis=0)/np.std(self.s,axis=0)
		idx_lowest_sr=np.argmin(paths_sr)
		
		b_samples=bootstrap_sharpe(self.s[:,idx_lowest_sr],n_boot=self.n_boot)
		b_samples*=self.sr_mult
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
		print('Return: ', np.power(self.sr_mult,2)*np.mean(self.s))
		print('Standard deviation: ', self.sr_mult*np.std(self.s))
		print('Sharpe: ', self.sr_mult*np.mean(self.s)/np.std(self.s))
		print()
		for i in range(pct_fees.size):
			print('Return fee=%s: '%pct_fees[i], 
				  np.power(self.sr_mult,2)*np.mean(self.s_fees[i]))
			print('Standard deviation fee=%s: '%pct_fees[i], 
				  self.sr_mult*np.std(self.s_fees[i]))
			print('Sharpe fee=%s: '%pct_fees[i], 
				  self.sr_mult*np.mean(self.s_fees[i])/np.std(self.s_fees[i]))
			print()
		print('**')		

		# bootstrap estimate of sharpe
		if self.n_paths!=1:
			plt.title('Distribution of paths SR [no fees]')
			plt.hist(paths_sr,density=True)
			plt.show()
			for i in range(pct_fees.size):				
				tmp=self.sr_mult*np.mean(self.s_fees[i],axis=0)/np.std(self.s_fees[i],axis=0)			
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
			aux=pd.DataFrame(np.cumsum(self.s_fees[i],axis=0),index=self.ts)
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


class Inference(object):
	def __init__(
				self,
				k_folds=5,
				burn_f=0.1,
				min_burn_points=1,
				n_paths=1,   
				seq_path=False,
				start_fold=0,
				pct_fees=0,
				seq_fees=True,
				sr_mult=np.sqrt(250),
				n_boot=1000				
				):
		'''
		k_folds: number of folds for CV
		burn_f: fraction of data wrt to train size of
			data near test to be burned
		n_paths: number of bt paths to generate		
		'''
		self.k_folds=k_folds
		self.burn_f=burn_f
		self.n_paths=n_paths
		self.min_burn_points=min_burn_points
		self.seq_path=seq_path
		self.start_fold=start_fold
		self.pct_fees=pct_fees
		self.seq_fees=seq_fees
		self.sr_mult=sr_mult
		self.n_boot=n_boot
		# 
		if isinstance(self.pct_fees,float):
			self.pct_fees=np.array([self.pct_fees])
		elif isinstance(self.pct_fees,int):
			self.pct_fees=np.array([self.pct_fees])			
		else:
			self.pct_fees=np.array(self.pct_fees)		

	def cvbt(self,data,model):
		'''
		Cross-validation backtest
		model must have methods .estimate and .get_weight
		data: pandas DataFrame like 
						y1,y2,...,yp,x1,x2,...,xq,z,idx
			timestamp 1
			timestamp 2
			...
			timestamp T
			y<.> are the columns with the returns
			x<.> are the columns with features (optional)
			z is the column with a observed state condition the model (optional)
			idx is the column with multisequence index (optional)
			
			Note:
			idx can represent the behavious over a period like a week of hour
			z can be used to make specific model for each weekday for example		
		'''
		# parse data into arrays
		y,x,z,msidx=parse_data(data)
		# build folds on multisequence indexes 
		# this creates a bit of overhead but allows for a clean
		# implementation of the cross validation with multiple sequences		
		has_msidx=True
		if msidx is None:
			has_msidx=False
			msidx=np.arange(y.shape[0],dtype=int)
			nu=y.shape[0]
		else:
			nu=np.unique(msidx).size		
		assert nu>=self.k_folds,"Not enought points to run cvbt"		
		n=y.shape[0]
		p=y.shape[1]		
		idx=np.arange(nu,dtype=int)		
		idx_folds=np.array_split(idx,self.k_folds)		
		s=np.zeros((n,self.n_paths),dtype=np.float64) 
		weights=np.zeros((n,p,self.n_paths),dtype=np.float64)		
		start_fold=self.start_fold
		if self.seq_path:
			start_fold=max(1,self.start_fold)		
		# run this for all paths
		for m in tqdm.tqdm(range(self.n_paths)):
			# CVBT ROUTINE
			for i in range(start_fold,self.k_folds):
				# get folds to be used for training (select folds to the left or to the right)
				train_folds=build_train_folds(i,self.k_folds,self.seq_path)
				# build the actual indexes on data for training
				train_idx=np.hstack([idx_folds[e] for e in train_folds])
				# burn indexes/get random subsequence
				train_idx=random_subsequence(train_idx,self.burn_f,self.min_burn_points)		
				# map (possible) subsequence indexes to true indexes in x (and f)
				train_idx=np.where(np.in1d(msidx,train_idx))[0]
				test_idx=np.where(np.in1d(msidx,idx_folds[i]))[0]
				# create copy of model
				local_model=copy.deepcopy(model)		
				# estimate model
				model_inputs={'y':y[train_idx],'idx':None}
				if x is not None:
					model_inputs.update({'x':x[train_idx]})   
				if has_msidx:			
					model_inputs.update({'idx':convert_msidx(msidx[train_idx])}) 
				if z is not None:
					model_inputs.update({'z':z[train_idx]})
				local_model.estimate(**model_inputs)
				y_test=y[test_idx]
				x_test=None
				idx_test=None
				z_test=None
				if x is not None:
					x_test=x[test_idx]
				if has_msidx:
					idx_test=convert_msidx(msidx[test_idx])
				if z is not None:
					z_test=z[test_idx]
				# evaluate model
				s[test_idx,m],weights[test_idx,:,m]=evaluate(local_model,y_test,x_test,idx_test,z_test)		
		out=Results(s,weights,data.index,self.seq_fees,self.pct_fees,self.sr_mult,self.n_boot,self.n_paths)
		return out
	
	def test(self,data,model):
		'''
		Test on new data an already trained model
		model must have methods and .get_weight (already trained usign .train(.))
		data: pandas DataFrame like 
						y1,y2,...,yp,x1,x2,...,xq,z,idx
			timestamp 1
			timestamp 2
			...
			timestamp T
			y<.> are the columns with the returns
			x<.> are the columns with features (optional)
			z is the column with a observed state condition the model (optional)
			idx is the column with multisequence index (optional)
			
			Note:
			idx can represent the behavious over a period like a week of hour
			z can be used to make specific model for each weekday for example		
		'''
		# parse data into arrays
		y,x,z,msidx=parse_data(data)
		# build folds on multisequence indexes 
		# this creates a bit of overhead but allows for a clean
		# implementation of the cross validation with multiple sequences		
		has_msidx=True
		if msidx is None:
			has_msidx=False
			msidx=np.arange(y.shape[0],dtype=int)
			nu=y.shape[0]
		else:
			nu=np.unique(msidx).size		
		n=y.shape[0]
		p=y.shape[1]		
		idx=np.arange(nu,dtype=int)		
		idx_folds=np.array_split(idx,self.k_folds)		
		
		s=np.zeros((n,1),dtype=np.float64) 
		weights=np.zeros((n,p,1),dtype=np.float64)		
						
		y_test=y
		x_test=None
		idx_test=None
		z_test=None
		if x is not None:
			x_test=x
		if has_msidx:
			idx_test=convert_msidx(msidx)
		if z is not None:
			z_test=z
		# create copy of model
		local_model=copy.deepcopy(model)		
		# evaluate model
		s[:,0],weights[:,:,0]=evaluate(local_model,y_test,x_test,idx_test,z_test)		
		out=Results(s,weights,data.index,self.seq_fees,self.pct_fees,self.sr_mult,self.n_boot,1)
		return out

	def train(self,data,model,view=True,plot_hist=True):
		'''	
		'''
		# parse data into arrays
		y,x,z,msidx=parse_data(data)
		has_msidx=True
		if msidx is None:
			has_msidx=False
		# create copy of model
		local_model=copy.deepcopy(model)		
		# estimate model
		model_inputs={'y':y,'idx':None}
		if x is not None:
			model_inputs.update({'x':x})   
		if has_msidx:			
			model_inputs.update({'idx':convert_msidx(msidx)}) 
		if z is not None:
			model_inputs.update({'z':z})
		local_model.estimate(**model_inputs)
		if view:
			local_model.view(plot_hist)
		return local_model

def live(data,model):

	# y1 y2 ... x1 x2 ... z idx [COLUMNS]

	# v1 v2 ... v3 v4 ... v5 i1
	# ...
	# ...
	# Na Na ... v6 v7 ... v8 i1
	#
	# so we can use the last row to predict the data
	# if features are present
	
	y,x,z,msidx=parse_data(data)
	
	has_msidx=True
	if msidx is not None:
		assert msidx[-1]==msidx[-2],"Last two observations of idx should match"
		# keep only the last sequence
		idx=np.where(msidx==msidx[-1])[0]
		x=x[idx]
		y=y[idx]
		z=z[idx]
					
	if x is not None:
		# remove last observation because it is a NaN
		model_inputs={'y':y[:-1]}
		model_inputs.update({'x':x[:-1]})	  
		model_inputs.update({'xq':x[-1]})
		if z is not None:
			model_inputs.update({'z':z[-1]})
	else:
		model_inputs={'y':y}
		if z is not None:
			model_inputs.update({'z':z[-1]})
	w=model.get_weight(**model_inputs)			
	return w

def save_model(model,filepath):
	with open(filepath,'wb') as out:
		pickle.dump(model,out,pickle.HIGHEST_PROTOCOL)

def load_model(filepath):
	with open(filepath, 'rb') as inp:
		model=pickle.load(inp)
	return model

if __name__=='__main__':
	pass
