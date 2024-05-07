import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import invwishart
from numpy.lib.stride_tricks import sliding_window_view
import tqdm
from numba import jit
from abc import ABC, abstractmethod


# TEMPLATE


class BaseModel(ABC):
	
	@abstractmethod
	def estimate(self,y: np.ndarray, x: np.ndarray):
		"""Subclasses must implement this method"""
		pass

	@abstractmethod
	def get_weight(self,xq: np.ndarray) -> np.ndarray:
		"""Subclasses must implement this method"""
		pass
	
	@abstractmethod
	def inverse_transform(self,arr: np.ndarray) -> np.ndarray:
		"""Subclasses must implement this method"""
		pass

	@abstractmethod
	def pw(self, arr: np.ndarray) -> float:
		"""Subclasses must implement this method"""
		pass




class TemplateModel(object):	
	
	def __init__(self,*args,**kwargs):
		self.p=1
		self.w=None
	
	def view(self):
		print('Template Model')
		
	def estimate(self,y,x=None,idx=None,z=None,**kwargs):
		self.p=y.shape[1]
		self.w=np.ones(self.p)
		self.w/=np.sum(self.w)
		
	def get_weight(self,xq=None,y=None,x=None,*args,**kwargs):
		return self.w


# SIMULATE

def simulate_mvgmm(n,phi,means,covs):
	# simulate states
	z=np.random.choice(np.arange(phi.size,dtype=int),p=phi,size=n)
	x=np.zeros((n,means[0].size))
	for i in range(n):
		x[i]=np.random.multivariate_normal(means[z[i]],covs[z[i]])
	return x,z


def simulate_hmm(n,A,P,means,covs):
	'''
	n: integer with the number of points to generate
	A: numpy (n_states,n_states) array with transition prob
	P: numpy (n_states,) array with init state prob
	means: list with len=n_states of numpy (n_states,) array with the means for each
		variable
	covs:list with len=n_states of numpy (n_states,n_states) array with the covariances
		for each variable
	'''	
	states=np.arange(A.shape[0],dtype=int)
	z=np.zeros(n,dtype=int)
	x=np.zeros((n,means[0].size))
	z[0]=np.random.choice(states,p=P)
	x[0]=np.random.multivariate_normal(means[z[0]],covs[z[0]])
	for i in range(1,n):
		z[i]=np.random.choice(states,p=A[z[i-1]])
		x[i]=np.random.multivariate_normal(means[z[i]],covs[z[i]])
	return x,z



# UTILITIES

def mvgauss_prob(x,mean,cov_inv,cov_det):
	'''
	x: numpy (n,p) array 
		each row is a joint observation
	mean: numpy (p,) array with the location parameter
	cov_inv: numpy(p,p) array with the inverse covariance 
			can be computed with np.linalg.inv(cov)
	cov_det: scalar with the determinant of the covariance matrix 
			can be computed with np.linalg.det(cov)
	returns:
		out: numpy(n,) array where each value out[i] is the 
				probability of observation x[i] 
	'''
	k=mean.size # number of variables
	x_c=x-mean # center x
	# vectorized computation
	return np.exp(-0.5*np.sum(x_c*np.dot(x_c,cov_inv),axis=1))/np.sqrt(np.power(2*np.pi,k)*cov_det)


def create_q(n_states,z_lags):
	'''
	create q array with the combinations of states and lags
	** these are the new states to solve the first order HMM problem **
	'''
	if z_lags==1:
		q=np.arange(n_states)[:,None]
		return q	
	def aux(n_states,obj):
		q=[]
		for m in range(n_states):
			if len(obj)==0:
				q.append(m)
			for e in obj:
				if not isinstance(e,list):
					q.append([e,m])
				else:
					q.append(e+[m])			
		return q
	q=[]
	for lag in range(z_lags):
		q=aux(n_states,q)
	q=np.array(q,dtype=int)
	q=q[:,::-1]
	return q


# this can be compiled
@jit(nopython=True)
def forward(prob,A,P):
	'''
	Forward algorithm for the HMM
	prob: numpy (n,n_states) array with
		the probability of each observation
		for each state
	A: numpy (n_states,n_states) array with the state
		transition matrix
	P: numpy (n_states,) array with the initial
		state probability
	returns:
		alpha: numpy (n,n_states) array meaning
			p(state=i|obs <= i)
		c: numpy (n,) array with the normalization
			constants
	'''
	n_obs=prob.shape[0]
	n_states=prob.shape[1]
	alpha=np.zeros((n_obs,n_states),dtype=np.float64)
	c=np.zeros(n_obs,dtype=np.float64)
	alpha[0]=P*prob[0]
	c[0]=1/np.sum(alpha[0])
	alpha[0]*=c[0]
	for i in range(1,n_obs):
		alpha[i]=np.dot(A.T,alpha[i-1])*prob[i] 
		c[i]=1/np.sum(alpha[i])
		alpha[i]*=c[i]
	return alpha,c

# this can be compiled
@jit(nopython=True)
def backward_sample(A,alpha,q,transition_counter,init_state_counter):
	'''
	Backward sample from the state transition matrix and state sequence
	A: numpy (n_states,n_states) array with the state
		transition matrix
	alpha: numpy (n,n_states) array meaning
		p(state=i|obs <= i)		
	q: numpy (n,) to store the sample of state sequence
	transition_counter: numpy (n_states,n_states) array to store 
		transition counts to be used to sample a state transition 
		matrix
	init_state_counter: numpy (n_states,) array to store the
		number of times state i is the initial one
	returns:
		none (q and transition_counter are changed inside this function)
	'''	
	# backward walk to sample from the state sequence
	n=q.size
	# sample the last hidden state with probability alpha[-1]
	q[n-1]=np.searchsorted(np.cumsum(alpha[-1]),np.random.random(),side="right")#]
	# aux variable
	p=np.zeros(A.shape[0],dtype=np.float64)
	# iterate backwards
	for j in range(n-2,-1,-1):
		# from formula
		p=A[:,q[j+1]]*alpha[j] 
		# normalize (from formula)
		p/=np.sum(p) 
		# sample hidden state with probability p
		q[j]=np.searchsorted(np.cumsum(p),np.random.random(),side="right")#]
		# increment transition counter (we can do this calculation incrementally)
		transition_counter[q[j],q[j+1]]+=1 
	# increment initial state counter
	init_state_counter[q[0]]+=1





class MovingAverage(object):

	def __init__(self, windows = 20, quantile = 0.9, vary_weights = True, reverse_signal = False, **kwargs):
		self.windows = windows
		if isinstance(self.windows,int): self.windows = [self.windows]
		self.windows = np.array(self.windows, dtype = int)
		self.quantile = quantile		
		self.vary_weights = vary_weights
		self.reverse_signal = reverse_signal
		self.w_norm = np.ones(self.windows.size, dtype = float)
		self.aux_w = 1
		self.p = 1

	def view(self, plot_hist=True):
		pass

	def estimate(self, y, **kwargs): 
		'''
		The estimate is just to calculate the bounds
		that the weights can have		
		'''
		# calculate the weight bounds
		if y.ndim == 1: y = y[:,None]
		self.p = y.shape[1]
		if self.vary_weights:
			# check if we have enough points to estimate weights bounds
			if y.shape[0] < 2*np.max(self.windows):
				self.aux_w = 0 # to multiply weights by zero on get_weight			
			else:
				for i in range(self.windows.size):
					m_mean = np.mean(sliding_window_view(y, window_shape = self.windows[i], axis = 0 ), axis=-1)
					m_var = np.var(sliding_window_view(y, window_shape = self.windows[i], axis = 0 ), axis=-1)
					m_mean = np.nan_to_num(m_mean, copy = True, nan = 0.0)
					m_var = np.nan_to_num(m_var, copy = True, nan = 0.0)
					m_var[m_var == 0] = np.inf
					w = m_mean / m_var
					if self.reverse_signal: w *= -1
					w = np.sum(np.abs(w),axis = 1)
					w = w[w!=0]
					w.sort()
					w_norm = w[int(self.quantile*w.size)]
					if w_norm == 0: w_norm = np.inf
					self.w_norm[i] = w_norm

	def predict(self, y, **kwargs):
		pass

	def get_weight(self, y, **kwargs):
		'''
		simple moving average prediction
		'''
		if y.shape[0] > np.max(self.windows):
			ws = np.zeros((self.windows.size, self.p))
			for i in range(self.windows.size):
				if self.vary_weights:
					mean = np.mean(y[-self.windows[i]:],axis=0)
					var = np.var(y[-self.windows[i]:],axis=0)
					mean = np.nan_to_num(mean, copy = True, nan = 0.0)
					var = np.nan_to_num(var, copy = True, nan = 0.0)
					var[var == 0] = np.inf					
					w = mean / var
					w /= self.w_norm[i]
				else:
					w = np.sign(np.mean(y[-self.windows[i]:],axis=0))
				l = np.sum(np.abs(w))
				if l > 1: 
					w /= l
				ws[i] = w
			w = np.mean(ws, axis = 0)
			w *= self.aux_w # in case we could not estimate the weight bounds
			if self.reverse_signal: w *= -1
			return w
		else:
			return np.zeros(self.p)



# MODELS
class Gaussian(object):
	def __init__(self,n_gibbs=None,f_burn=0.1,min_k=0.25,max_k=0.25,names=None):
		self.f_burn=f_burn
		self.n_gibbs=n_gibbs
		self.no_gibbs=False
		if self.n_gibbs is None:
			self.no_gibbs=True
			self.n_gibbs=0

		self.min_k=min_k
		self.max_k=max_k
		self.names=names
		# real number of samples to simulate
		self.n_gibbs_sim=int(self.n_gibbs*(1+self.f_burn))
		self.p=1
		# to be calculated!
		self.gibbs_cov=None
		self.gibbs_mean=None
		self.mean=None
		self.cov=None
		self.cov_inv=None
		self.w=None
		self.w_norm=1

		assert self.max_k<=1 and self.max_k>=0,"max_k must be between 0 and 1"
		assert self.min_k<=1 and self.min_k>=0,"min_k must be between 0 and 1"
		assert self.max_k>=self.min_k,"max_k must be larger or equal than min_k"
		
	
	def view(self,plot_hist=True):
		if self.names is None:
			self.names=["x%s"%(i+1) for i in range(self.p)]
		if len(self.names)!=self.p:
			self.names=["x%s"%(i+1) for i in range(self.p)]					
		print('** Gaussian **')
		print('Mean')
		print(self.mean)
		print('Covariance')
		print(self.cov)
		if self.gibbs_mean is not None:
			if plot_hist:
				for i in range(self.p):
					plt.hist(self.gibbs_mean[:,i],density=True,alpha=0.5,label='Mean %s'%(self.names[i]))
				plt.legend()
				plt.grid(True)
				plt.show()
		if self.gibbs_cov is not None:
			if plot_hist:
				for i in range(self.p):
					for j in range(i,self.p):
						plt.hist(self.gibbs_cov[:,i,j],density=True,alpha=0.5,label='Cov(%s,%s)'%(self.names[i],self.names[j]))
				plt.legend()
				plt.grid(True)
				plt.show()			
			
	def estimate(self,y,**kwargs):		
		# Gibbs sampler
		assert y.ndim==2,"y must be a matrix"
		
		if self.no_gibbs:
			self.mean=np.mean(y,axis=0)
			self.cov=np.cov(y.T)
			if self.cov.ndim==0:
				self.cov=np.array([[self.cov]])				
			# regularize
			self.cov=self.max_k*np.diag(np.diag(self.cov))+(1-self.max_k)*self.cov	
			self.cov_inv=np.linalg.inv(self.cov)
			self.w=np.dot(self.cov_inv,self.mean)		
			self.w_norm=np.sum(np.abs(self.w))		
		else:
				
			n=y.shape[0]
			self.p=y.shape[1]		
			# compute data covariance
			c=np.cov(y.T)
			if c.ndim==0:
				c=np.array([[c]])		
			c_diag=np.diag(np.diag(c))
			# prior parameters
			m0=np.zeros(self.p) # mean prior center		
			V0=c_diag.copy() # mean prior covariance		
			S0aux=c_diag.copy() # covariance prior scale (to be multiplied later)
			# precalc
			y_mean=np.mean(y,axis=0)
			invV0=np.linalg.inv(V0)
			invV0m0=np.dot(invV0,m0)
			# initialize containers
			self.gibbs_cov=np.zeros((self.n_gibbs_sim,self.p,self.p))
			self.gibbs_mean=np.zeros((self.n_gibbs_sim,self.p))
			# initialize cov
			self.gibbs_cov[0]=c		
			# sample
			for i in range(1,self.n_gibbs_sim):
				# sample for mean
				invC=np.linalg.inv(self.gibbs_cov[i-1])
				Vn=np.linalg.inv(invV0+n*invC)
				mn=np.dot(Vn,invV0m0+n*np.dot(invC,y_mean))
				self.gibbs_mean[i]=np.random.multivariate_normal(mn,Vn)
				# sample from cov
				# get random k value (shrinkage value)
				k=np.random.uniform(self.min_k,self.max_k)
				n0=k*n/(1-k)
				S0=n0*S0aux
				v0=n0+self.p+1			
				vn=v0+n
				St=np.dot((y-self.gibbs_mean[i]).T,(y-self.gibbs_mean[i]))
				Sn=S0+St
				self.gibbs_cov[i]=invwishart.rvs(df=vn,scale=Sn) 
			self.gibbs_mean=self.gibbs_mean[-self.n_gibbs:]
			self.gibbs_cov=self.gibbs_cov[-self.n_gibbs:]
			self.mean=np.mean(self.gibbs_mean,axis=0)
			self.cov=np.mean(self.gibbs_cov,axis=0)
			self.cov_inv=np.linalg.inv(self.cov)
			self.w=np.dot(self.cov_inv,self.mean)
			self.w_norm=np.sum(np.abs(self.w))

	def get_weight(self,normalize=True,**kwargs):
		if normalize:
			return self.w/self.w_norm
		else:
			return self.w


class StateGaussian(object):
	def __init__(self,n_gibbs=None,f_burn=0.1,min_k=0.25,max_k=0.25,min_points=10,set_state_to_zero=[]):
		self.n_gibbs=n_gibbs
		self.no_gibbs=False
		if self.n_gibbs is None:
			self.no_gibbs=True
		self.f_burn=f_burn
		self.max_k=max_k
		self.min_k=min_k
		self.min_points=min_points
		self.gaussians={}
		self.default_w=0
		self.p=1
		self.max_w_norm=1
		self.set_state_to_zero=set_state_to_zero

	def view(self,plot_hist=True):
		print('StateGaussian')
		print('max w norm: ', self.max_w_norm)
		for k,v in self.gaussians.items():
			print('State z=%s'%k)
			v.view(plot_hist)
			print()
			print()

	def estimate(self,y,z,**kwargs):

		self.max_w_norm=0
		self.p=y.shape[1]
		n=y.shape[0]
		
		if z.ndim!=1:
			z=z[:,None]

		#if z is None:
		#	z=np.zeros(n,dtype=int)
		# assert z.ndim==1,"z must be a vector"
		z=np.array(z,dtype=int)
		uz=np.unique(z)
		for e in uz:
			g=Gaussian(self.n_gibbs,self.f_burn,self.min_k,self.max_k)
			i=np.where(z==e)[0]

			if i.size>self.min_points:
				g.estimate(y[i])
				self.gaussians.update({e:g})
				self.max_w_norm=max(self.max_w_norm,g.w_norm)

	def get_weight(self,z,**kwargs):
		
		if isinstance(z,np.ndarray):
			z=z[-1]
		
		if z in self.set_state_to_zero:
			return np.zeros(self.p)

		g=self.gaussians.get(z)
		if g is None:
			return self.default_w*np.ones(self.p)
		else:
			return g.get_weight(normalize=False)/self.max_w_norm




class ConditionalGaussian(object):
	def __init__(self,n_gibbs=None,f_burn=0.1,min_k=0.25,max_k=0.25,kelly_std=2,max_w=1):
		self.n_gibbs=n_gibbs
		self.no_gibbs=False
		if self.n_gibbs is None:
			self.no_gibbs=True
		self.f_burn=f_burn
		self.max_k=max_k
		self.kelly_std=kelly_std
		self.max_w=max_w
		self.min_k=min_k
		self.g=None
		# to calculate after Gaussian estimate
		self.my=None
		self.mx=None
		self.Cyy=None
		self.Cxx=None
		self.Cyx=None
		self.invCxx=None
		self.pred_gain=None
		self.cov_reduct=None
		self.pred_cov=None 
		self.prev_cov_inv=None
		self.w_norm=1

	
	def view(self,plot_hist=True):
		if self.g is not None:
			self.g.view(plot_hist=plot_hist)
	
	def estimate(self,y,x,**kwargs): 
		x=x.copy()
		y=y.copy()		
		if x.ndim==1:
			x=x[:,None]
		if y.ndim==1:
			y=y[:,None]		
		p=y.shape[1]
		q=x.shape[1]		
		z=np.hstack((y,x))
		names=[]
		for i in range(p):
			names.append("y%s"%(i+1))
		for i in range(q):
			names.append("x%s"%(i+1))
		self.g=Gaussian(self.n_gibbs,self.f_burn,self.min_k,self.max_k,names=names)
		self.g.estimate(z)
		# extract distribution of y|x from the estimated covariance
		y_idx=np.arange(p)
		x_idx=np.arange(p,p+q)		
		self.my=self.g.mean[y_idx]
		self.mx=self.g.mean[x_idx]
		self.Cyy=self.g.cov[y_idx][:,y_idx]
		self.Cxx=self.g.cov[x_idx][:,x_idx]
		self.Cyx=self.g.cov[y_idx][:,x_idx]
		self.invCxx=np.linalg.inv(self.Cxx)
		self.pred_gain=np.dot(self.Cyx,self.invCxx)
		self.cov_reduct=np.dot(self.pred_gain,self.Cyx.T)
		self.pred_cov=self.Cyy-self.cov_reduct
		self.pred_cov_inv=np.linalg.inv(self.pred_cov)
		# compute normalization
		x_move=np.sqrt(np.diag(self.Cxx))*self.kelly_std
		self.w_norm=np.sum(np.abs(np.dot( self.pred_cov_inv , self.my + np.dot(np.abs(self.pred_gain),x_move+self.mx) )))

	def predict(self,xq):
		return self.my+np.dot(self.pred_gain,xq-self.mx)
	
	def expected_value(self,xq):
		return predict(xq)
	
	def covariance(self,xq):
		return self.pred_cov
	
	def get_weight(self,xq,normalize=True,**kwargs):
		if normalize:
			w=np.dot(self.pred_cov_inv,self.predict(xq))/self.w_norm
			d=np.sum(np.abs(w))
			if d>self.max_w:
				w/=d
				w*=self.max_w
			return w			
		else:
			return np.dot(self.pred_cov_inv,self.predict(xq))



class IndividualConditionalGaussian(object):
	def __init__(self,n_gibbs=None,f_burn=0.1,min_k=0.25,max_k=0.25,kelly_std=2,max_w=1):
		self.n_gibbs=n_gibbs
		self.no_gibbs=False
		if self.n_gibbs is None:
			self.no_gibbs=True
		self.f_burn=f_burn
		self.min_k=min_k
		self.max_k=max_k
		self.kelly_std=kelly_std
		self.max_w=max_w
		self.p=0
		self.q=0
		self.cond_gaussians=[]
	
	def view(self,plot_hist=True):
		if self.g is not None:
			self.g.view(plot_hist=plot_hist)
	
	def estimate(self,y,x,**kwargs): 
		x=x.copy()
		y=y.copy()		
		if x.ndim==1:
			x=x[:,None]
		if y.ndim==1:
			y=y[:,None]		
		self.p=y.shape[1]
		self.q=x.shape[1]		
		for i in range(self.q):
			tmp=ConditionalGaussian(n_gibbs=self.n_gibbs,f_burn=self.f_burn,min_k=self.min_k,max_k=self.max_k,kelly_std=self.kelly_std,max_w=self.max_w)
			tmp.estimate(y=y,x=x[:,[i]])
			self.cond_gaussians.append(tmp)

	
	def get_weight(self,xq,normalize=True,**kwargs):
		
		w=np.zeros((self.q,self.p))
		for i in range(self.q):
			w[i]=self.cond_gaussians[i].get_weight(xq[[i]],normalize)
		w=np.mean(w,axis=0)
		return w


class SameConditionalGaussian(object):
	def __init__(self,n_gibbs=None,f_burn=0.1,min_k=0.25,max_k=0.25,kelly_std=2,max_w=1):
		self.n_gibbs=n_gibbs
		self.no_gibbs=False
		if self.n_gibbs is None:
			self.no_gibbs=True
		self.f_burn=f_burn
		self.min_k=min_k
		self.max_k=max_k
		self.kelly_std=kelly_std
		self.max_w=max_w
		self.p=0
		self.q=0
		self.cond_gaussian=None
	
	def view(self,plot_hist=True):
		if self.g is not None:
			self.g.view(plot_hist=plot_hist)
	
	def estimate(self,y,x,**kwargs): 
		# all features are the same feature or at least behave in the same way
		# train model with all data
		# predict individually
		x=x.copy()
		y=y.copy()		
		if x.ndim==1:
			x=x[:,None]
		if y.ndim==1:
			y=y[:,None]		
		
		self.p=y.shape[1]
		self.q=x.shape[1]	

		x_all=[]
		y_all=[]

		for i in range(self.q):
			x_all.append(x[:,[i]])
			y_all.append(y)
		x_all=np.vstack(x_all)
		y_all=np.vstack(y_all)
		self.cond_gaussian=ConditionalGaussian(n_gibbs=self.n_gibbs,f_burn=self.f_burn,min_k=self.min_k,max_k=self.max_k,kelly_std=self.kelly_std,max_w=self.max_w)
		self.cond_gaussian.estimate(y=y_all,x=x_all)

	def get_weight(self,xq,normalize=True,**kwargs):		
		w=np.zeros((self.q,self.p))
		for i in range(self.q):
			w[i]=self.cond_gaussian.get_weight(xq[[i]],normalize)
		w=np.mean(w,axis=0)
		return w



class StateConditionalGaussian(object):
	def __init__(self,n_gibbs=None,f_burn=0.1,min_k=0.25,max_k=0.25,min_points=10,kelly_std=2,max_w=1):
		self.n_gibbs=n_gibbs
		self.no_gibbs=False
		self.kelly_std=kelly_std
		self.max_w=max_w
		if self.n_gibbs is None:
			self.no_gibbs=True
		self.f_burn=f_burn
		self.max_k=max_k
		self.min_k=min_k
		self.min_points=min_points
		self.cond_gaussians={}
		self.default_w=0
		self.p=1
		self.max_w_norm=1

	def view(self,plot_hist=True):
		print('StateGaussian')
		for k,v in self.cond_gaussians.items():
			print('State z=%s'%k)
			v.view(plot_hist)
			print()
			print()

	def estimate(self,y,x,z=None,**kwargs):

		self.max_w_norm=0
		self.p=y.shape[1]
		n=y.shape[0]
		if z is None:
			z=np.zeros(n,dtype=int)
		# assert z.ndim==1,"z must be a vector"
		if z.ndim!=1:
			z=z[:,None]		
		z=np.array(z,dtype=int)
		uz=np.unique(z)
		for e in uz:
			g=ConditionalGaussian(self.n_gibbs,self.f_burn,self.min_k,self.max_k)
			i=np.where(z==e)[0]
			if i.size>self.min_points:
				g.estimate(y[i],x[i])
				self.cond_gaussians.update({e:g})
				self.max_w_norm=max(self.max_w_norm,g.w_norm)

	def get_weight(self,xq,z,**kwargs):
		if isinstance(z,np.ndarray):
			z=z[-1]
		g=self.cond_gaussians.get(z)
		if g is None:
			return self.default_w*np.ones(self.p)
		else:
			w=g.get_weight(xq,normalize=False)/self.max_w_norm
			d=np.sum(np.abs(w))
			if d>self.max_w:
				w/=d
				w*=self.max_w
			return w
			
# --------------------------------------------------------------------

class GaussianShrink(object):
	def __init__(self,pca_components=1,names=None):
		self.names=names
		self.pca_components=pca_components
		# real number of samples to simulate
		self.p=1
		# to be calculated!
		self.sigma2=None
		self.W=None
		self.mean=None
		self.cov=None
		self.cov_inv=None
		self.w=None
		self.w_norm=1
	
	def view(self,plot_hist=True):
		if self.names is None:
			self.names=["x%s"%(i+1) for i in range(self.p)]
		if len(self.names)!=self.p:
			self.names=["x%s"%(i+1) for i in range(self.p)]					
		print('** Gaussian **')
		print('Mean')
		print(self.mean)
		print('Covariance')
		print(self.cov)
		print('sigma2')
		print(self.sigma2)
		print('W')
		print(self.W)

	def estimate(self,y,**kwargs):		
		# Gibbs sampler
		assert y.ndim==2,"y must be a matrix"
		
		self.p=y.shape[1]
		self.pca_components=min(self.pca_components,self.p)

		self.mean=np.mean(y,axis=0)
		self.cov=np.cov(y.T)
		if self.cov.ndim==0:
			self.cov=np.array([[self.cov]])				

		vals,vec=np.linalg.eig(self.cov)
		idx=np.argsort(vals)[::-1]
		vals=vals[idx]
		vec=vec[:,idx]
		self.sigma2=np.mean(vals[self.pca_components:])
		L=np.power(np.diag(vals[:self.pca_components])-self.sigma2*np.eye(self.pca_components),0.5)
		self.W=vec[:,:self.pca_components]
		self.W=np.dot(self.W,L)
		# self.sigma2=self.gamma/2		

		# regularize
		self.cov=np.eye(self.p)*self.sigma2+np.dot(self.W,self.W.T)
		self.cov_inv=np.linalg.inv(self.cov)
		self.w=np.dot(self.cov_inv,self.mean)		
		self.w_norm=np.sum(np.abs(self.w))		

	def get_weight(self,normalize=True,**kwargs):
		if normalize:
			return self.w/self.w_norm
		else:
			return self.w




class ConditionalGaussianShrink(object):
	def __init__(self,pca_components=1,kelly_std=2,max_w=1):
		self.kelly_std=kelly_std
		self.max_w=max_w
		self.pca_components=pca_components
		self.g=None
		# to calculate after Gaussian estimate
		self.my=None
		self.mx=None
		self.Cyy=None
		self.Cxx=None
		self.Cyx=None
		self.invCxx=None
		self.pred_gain=None
		self.cov_reduct=None
		self.pred_cov=None 
		self.prev_cov_inv=None
		self.w_norm=1
	
	def view(self,plot_hist=True):
		if self.g is not None:
			self.g.view(plot_hist=plot_hist)
	
	def estimate(self,y,x,**kwargs): 
		x=x.copy()
		y=y.copy()		
		if x.ndim==1:
			x=x[:,None]
		if y.ndim==1:
			y=y[:,None]		
		p=y.shape[1]
		q=x.shape[1]		
		z=np.hstack((y,x))
		names=[]
		for i in range(p):
			names.append("y%s"%(i+1))
		for i in range(q):
			names.append("x%s"%(i+1))
		self.g=GaussianShrink(self.pca_components)
		self.g.estimate(z)
		# extract distribution of y|x from the estimated covariance
		y_idx=np.arange(p)
		x_idx=np.arange(p,p+q)		
		self.my=self.g.mean[y_idx]
		self.mx=self.g.mean[x_idx]
		self.Cyy=self.g.cov[y_idx][:,y_idx]
		self.Cxx=self.g.cov[x_idx][:,x_idx]
		self.Cyx=self.g.cov[y_idx][:,x_idx]
		self.invCxx=np.linalg.inv(self.Cxx)
		self.pred_gain=np.dot(self.Cyx,self.invCxx)
		self.cov_reduct=np.dot(self.pred_gain,self.Cyx.T)
		self.pred_cov=self.Cyy-self.cov_reduct
		self.pred_cov_inv=np.linalg.inv(self.pred_cov)
		# compute normalization
		x_move=np.sqrt(np.diag(self.Cxx))*self.kelly_std
		self.w_norm=np.sum(np.abs(np.dot( self.pred_cov_inv , self.my + np.dot(np.abs(self.pred_gain),x_move+self.mx) )))

	def predict(self,xq):
		return self.my+np.dot(self.pred_gain,xq-self.mx)
	
	def expected_value(self,xq):
		return predict(xq)
	
	def covariance(self,xq):
		return self.pred_cov
	
	def get_weight(self,xq,normalize=True,**kwargs):
		if normalize:
			w=np.dot(self.pred_cov_inv,self.predict(xq))/self.w_norm
			d=np.sum(np.abs(w))
			if d>self.max_w:
				w/=d
				w*=self.max_w
			return w			
		else:
			return np.dot(self.pred_cov_inv,self.predict(xq))

# --------------------------------------------------------------------

class GaussianMixture(object):

	def __init__(self,n_states=2,n_gibbs=1000,f_burn=0.1,min_k=0.25,max_k=0.25,names=None):

		self.n_states=n_states
		self.f_burn=f_burn
		self.n_gibbs=n_gibbs
		self.max_k=max_k
		self.min_k=min_k
		self.names=names
		# real number of samples to simulate
		self.n_gibbs_sim=int(self.n_gibbs*(1+self.f_burn))
		self.p=1
		self.gibbs_cov=None
		self.gibbs_mean=None
		self.w_norm=1
		assert self.max_k<=1 and self.max_k>=0,"max_k must be between 0 and 1"
		assert self.min_k<=1 and self.min_k>=0,"min_k must be between 0 and 1"
		assert self.max_k>=self.min_k,"max_k must be larger or equal than min_k"
		
	def view(self,plot_hist=True,round_to=5):
		print('** Gaussian Mixture **')
		if self.names is None:
			self.names=['x%s'%(i+1) for i in range(self.p)]
		for j in range(self.n_states):
			print('State %s'%(j+1))
			print('Prob: ', self.states_pi[j])
			print('Mean')
			print(np.round(self.states_mean[j],round_to))
			print('Covariance')
			print(np.round(self.states_cov[j],round_to))
			print()		
			if plot_hist:
				if self.gibbs_mean is not None:
					for i in range(self.p):
						plt.hist(self.gibbs_mean[j,:,i],density=True,alpha=0.5,label='Mean %s'%(self.names[i]))
					plt.legend()
					plt.grid(True)
					plt.show()
				if self.gibbs_cov is not None:
					for i in range(self.p):
						for q in range(i,self.p):
							plt.hist(self.gibbs_cov[j,:,i,q],density=True,alpha=0.5,label='Cov(%s,%s)'%(self.names[i],self.names[q]))
					plt.legend()
					plt.grid(True)
					plt.show()			

	def estimate(self,x,**kwargs):		
		# Gibbs sampler
		assert x.ndim==2,"x must be a matrix"
		n=x.shape[0]
		self.p=x.shape[1]		
		# compute data covariance
		c=np.cov(x.T)# np.dot(x.T,x)/(n-1)		
		if c.ndim==0:
			c=np.array([[c]])		
		c_diag=np.diag(np.diag(c))
		# prior parameters
		m0=np.zeros(self.p) # mean prior center		
		V0=c_diag.copy() # mean prior covariance		
		S0aux=c_diag.copy() # covariance prior scale (to be multiplied later)
		alpha0=self.n_states
		# precalc
		x_mean=np.mean(x,axis=0)
		invV0=np.linalg.inv(V0)
		invV0m0=np.dot(invV0,m0)
		# initialize containers
		self.gibbs_pi=np.zeros((self.n_states,self.n_gibbs_sim))
		self.gibbs_cov=np.zeros((self.n_states,self.n_gibbs_sim,self.p,self.p))
		self.gibbs_mean=np.zeros((self.n_states,self.n_gibbs_sim,self.p))
		aux=np.zeros((n,self.n_states))
		
		# initialize cov
		for i in range(self.n_states):
			self.gibbs_cov[i,0]=c		
		
		# others
		possible_states=np.arange(self.n_states)
		c=np.random.choice(possible_states,n)
		n_count=np.zeros(self.n_states)		
		
		# previous used parameters to sample when there are not observations
		# on that state
		prev_mn=np.zeros((self.n_states,self.p))
		prev_Vn=np.zeros((self.n_states,self.p,self.p))
		prev_vn=np.zeros(self.n_states)
		prev_Sn=np.zeros((self.n_states,self.p,self.p))
		for j in range(self.n_states):
			prev_mn[j]=m0
			prev_Vn[j]=V0
			prev_vn[j]=self.p+1+1
			prev_Sn[j]=S0aux
		
		# sample
		for i in range(1,self.n_gibbs_sim):
			for j in range(self.n_states):
				# basically, this is the code to sample from a multivariate
				# gaussian but constrained to observations where state=j
				idx=np.where(c==j)[0]		
				# just sample from the prior!
				if idx.size==0:
					self.gibbs_mean[j,i]=np.random.multivariate_normal(prev_mn[j],prev_Vn[j])
					self.gibbs_cov[j,i]=invwishart.rvs(df=prev_vn[j],scale=prev_Sn[j])			   
				else:
					n_count[j]=idx.size
					x_=x[idx]
					x_mean=np.mean(x_,axis=0)
					# ---------------------
					# sample for mean
					invC=np.linalg.inv(self.gibbs_cov[j,i-1])
					invV0=np.diag(1/np.diag(self.gibbs_cov[j,i-1]))
					invV0m0=np.dot(invV0,m0)

					Vn=np.linalg.inv(invV0+n_count[j]*invC)
					mn=np.dot(Vn,invV0m0+n_count[j]*np.dot(invC,x_mean))
					prev_mn[j]=mn
					prev_Vn[j]=Vn
					self.gibbs_mean[j,i]=np.random.multivariate_normal(mn,Vn)
					
					St=np.dot((x_-self.gibbs_mean[j,i]).T,(x_-self.gibbs_mean[j,i]))
					
					# sample from cov
					# get random k value (shrinkage value)
					k=np.random.uniform(self.min_k,self.max_k)
					
					n0=n_count[j]*k/(1-k)
					
					# choice here between the fixed prior or a changing one...
					S0=n0*np.diag(np.diag(St))/n_count[j]# 
					# S0=n0*S0aux # change this to use the diagonal of St
					
					v0=n0+self.p+1	# make sure the values make sense 		
					vn=v0+n_count[j]

					Sn=S0+St
					prev_vn[j]=vn
					prev_Sn[j]=Sn
					self.gibbs_cov[j,i]=invwishart.rvs(df=vn,scale=Sn)				 
				# ----------------------
			# sample pi
			self.gibbs_pi[:,i]=np.random.dirichlet(n_count+alpha0/self.n_states)
			for j in range(self.n_states):
				cov_inv=np.linalg.inv(self.gibbs_cov[j,i])
				cov_det=np.linalg.det(self.gibbs_cov[j,i])
				aux[:,j]=self.gibbs_pi[j,i]*mvgauss_prob(x,self.gibbs_mean[j,i],cov_inv,cov_det)   
			# this is a hack to sample fast from a multinomial with different probabilities!	  
			aux/=np.sum(aux,axis=1)[:,None]
			uni=np.random.uniform(0, 1,size=n)
			aux=np.cumsum(aux,axis=1)
			wrows,wcols=np.where(aux>uni[:,None])
			un,un_idx=np.unique(wrows,return_index=True)	
			c=wcols[un_idx]
				   
		self.gibbs_mean=self.gibbs_mean[:,-self.n_gibbs:,:]
		self.gibbs_cov=self.gibbs_cov[:,-self.n_gibbs:,:,:]
		self.gibbs_pi=self.gibbs_pi[:,-self.n_gibbs:]
		
		self.states_mean=np.mean(self.gibbs_mean,axis=1)
		self.states_cov=np.mean(self.gibbs_cov,axis=1)		
		self.states_pi=np.mean(self.gibbs_pi,axis=1)

		# compute w norm
		self.states_cov_inv=np.zeros_like(self.states_cov)
		self.w_norm=0
		for i in range(self.n_states):
			self.states_cov_inv[i]=np.linalg.inv(self.states_cov[i])
			self.w_norm=max(self.w_norm, np.sum(np.abs(np.dot(self.states_cov_inv[i],self.states_mean[i])))  )

	def expected_value(self):
		ev=np.zeros(self.p)
		for i in range(self.n_states):
			ev+=self.states_pi[i]*self.states_mean[i]
		return ev

	def predict(self,xq):
		return self.expected_value(xq)
	
	def covariance(self):
		cov=np.zeros((self.p,self.p))
		mu=np.zeros(self.p)
		for i in range(self.n_states):			
			cov+=self.states_pi[i]*self.states_cov[i]
			mu+=self.states_pi[i]*self.states_mean[i]
		cov-=np.dot(mu[:,None],mu[None,:])
		return cov
	
	def get_weight(self,normalize=True,**kwargs):
		mu=self.expected_value()
		cov=self.covariance()
		w=np.dot(np.linalg.inv(cov),mu)
		if normalize:
			return w/self.w_norm
		else:
			return w
	
	def sample(self,n_samples):
		samples=np.zeros((n_samples,self.p))
		states=np.arange(self.n_states,dtype=int)
		for i in range(n_samples):
			# sample a state
			zi=np.random.choice(states,p=self.states_pi)
			samples[i]=np.random.multivariate_normal(self.states_mean[zi],self.states_cov[zi])		
		return samples

class ConditionalGaussianMixture(object):
	def __init__(self,n_states,n_gibbs=1000,f_burn=0.1,min_k=0.25,max_k=0.25,kelly_std=2,max_w=1):
		self.n_states=n_states
		self.n_gibbs=n_gibbs
		self.f_burn=f_burn
		self.kelly_std=kelly_std
		self.max_w=max_w
		self.max_k=max_k
		self.min_k=min_k
		self.gm=None
		self.p=1
		self.q=1

	def view(self,plot_hist=True):
		if self.gm is not None:
			self.gm.view(plot_hist=plot_hist)
	
	def estimate(self,y,x,**kwargs): 
		x=x.copy()
		y=y.copy()		
		if x.ndim==1:
			x=x[:,None]
		if y.ndim==1:
			y=y[:,None]		
		self.p=y.shape[1]
		self.q=x.shape[1]		
		z=np.hstack((y,x))
		names=[]
		for i in range(self.p):
			names.append("y%s"%(i+1))
		for i in range(self.q):
			names.append("x%s"%(i+1))

		self.gm=GaussianMixture(self.n_states,self.n_gibbs,self.f_burn,self.min_k,self.max_k,names)
		self.gm.estimate(z)
		# extract distribution of y|x from the estimated covariance
		y_idx=np.arange(self.p)
		x_idx=np.arange(self.p,self.p+self.q)	
		
		self.states_pi=self.gm.states_pi		
		self.states_my=np.zeros((self.n_states,self.p))
		self.states_mx=np.zeros((self.n_states,self.q))
		self.states_Cyy=np.zeros((self.n_states,self.p,self.p))
		self.states_Cxx=np.zeros((self.n_states,self.q,self.q))
		self.states_Cxx_inv=np.zeros((self.n_states,self.q,self.q))
		self.states_Cxx_det=np.zeros((self.n_states))
		self.states_Cyx=np.zeros((self.n_states,self.p,self.q))
		self.states_invCxx=np.zeros((self.n_states,self.q,self.q))
		self.states_pred_gain=np.zeros((self.n_states,self.p,self.q))
		self.states_cov_reduct=np.zeros((self.n_states,self.p,self.p))
		self.states_pred_cov=np.zeros((self.n_states,self.p,self.p))
		self.states_pred_cov_inv=np.zeros((self.n_states,self.p,self.p))
		
		self.w_norm=0

		for i in range(self.n_states):
			self.states_my[i]=self.gm.states_mean[i][y_idx]
			self.states_mx[i]=self.gm.states_mean[i][x_idx]
			self.states_Cyy[i]=self.gm.states_cov[i][y_idx][:,y_idx]
			self.states_Cxx[i]=self.gm.states_cov[i][x_idx][:,x_idx]
			self.states_Cxx_inv[i]=np.linalg.inv(self.states_Cxx[i])
			self.states_Cxx_det[i]=np.linalg.det(self.states_Cxx[i])
			self.states_Cyx[i]=self.gm.states_cov[i][y_idx][:,x_idx]
			self.states_invCxx[i]=np.linalg.inv(self.states_Cxx[i])
			self.states_pred_gain[i]=np.dot(self.states_Cyx[i],self.states_invCxx[i])
			self.states_cov_reduct[i]=np.dot(self.states_pred_gain[i],self.states_Cyx[i].T)
			self.states_pred_cov[i]=self.states_Cyy[i]-self.states_cov_reduct[i]
			self.states_pred_cov_inv[i]=np.linalg.inv(self.states_pred_cov[i])		

			# compute normalization
			x_move=np.sqrt(np.diag(self.states_Cxx[i]))*self.kelly_std
			self.w_norm=max(self.w_norm,np.sum(np.abs(np.dot( self.states_pred_cov_inv[i] , self.states_my[i] + np.dot(np.abs(self.states_pred_gain[i]),x_move+self.states_mx[i]) ))))

	def expected_value(self,xq):
		if isinstance(xq,float):
			xq=np.array([xq])
		assert xq.ndim==1,"xq must be a vector"
		pis=np.zeros(self.n_states)
		for i in range(self.n_states):
			pis[i]=self.states_pi[i]*mvgauss_prob(np.array([xq]),self.states_mx[i],self.states_Cxx_inv[i],self.states_Cxx_det[i])[0]
		pis/=np.sum(pis)
		ev=np.zeros(self.states_my.shape[1])
		for i in range(self.n_states):		
			ev+=pis[i]*(self.states_my[i]+np.dot(self.states_pred_gain[i],xq-self.states_mx[i]))
		return ev

	def predict(self,xq):
		return self.expected_value(xq)
	
	def covariance(self,xq):
		if isinstance(xq,float):
			xq=np.array([xq])
		assert xq.ndim==1,"xq must be a vector"		
		pis=np.zeros(self.n_states)
		for i in range(self.n_states):
			 pis[i]=self.states_pi[i]*mvgauss_prob(np.array([xq]),self.states_mx[i],self.states_Cxx_inv[i],self.states_Cxx_det[i])[0]
		pis/=np.sum(pis)
		p=self.states_my.shape[1]
		cov=np.zeros((p,p))
		mu=np.zeros(p)
		for i in range(self.n_states):			
			cov+=pis[i]*(self.states_Cyy[i]+np.dot(self.states_my[i][:,None],self.states_my[i][None,:]))
			mu+=pis[i]*self.states_my[i]
		cov-=np.dot(mu[:,None],mu[None,:])
		return cov
	
	def get_weight(self,xq,normalize=True,**kwargs):

		w=np.dot(np.linalg.inv(self.covariance(xq)),self.expected_value(xq))   

		if normalize:
			w/=self.w_norm
			d=np.sum(np.abs(w))
			if d>self.max_w:
				w/=d
				w*=self.max_w
		return w

	def sample(self,n_samples):
		samples=self.gm.sample(n_samples)
		y_idx=np.arange(self.p)
		x_idx=np.arange(self.p,self.p+self.q)			
		x=samples[:,x_idx]
		y=samples[:,y_idx]
		return y,x

# ConditionalGaussianMixture(n_states,n_gibbs=1000,f_burn=0.1,min_k=0.25,max_k=0.25,kelly_std=2,max_w=1)
class SameConditionalGaussianMixture(object):
	def __init__(self,n_states,n_gibbs=1000,f_burn=0.1,min_k=0.25,max_k=0.25,kelly_std=2,max_w=1):
		self.n_states=n_states
		self.n_gibbs=n_gibbs
		self.f_burn=f_burn
		self.min_k=min_k
		self.max_k=max_k
		self.kelly_std=kelly_std
		self.max_w=max_w
		self.p=0
		self.q=0
		self.cond_gaussian_mixture=None
	
	def view(self,plot_hist=True):
		if self.g is not None:
			self.g.view(plot_hist=plot_hist)
	
	def estimate(self,y,x,**kwargs): 
		# all features are the same feature or at least behave in the same way
		# train model with all data
		# predict individually
		x=x.copy()
		y=y.copy()		
		if x.ndim==1:
			x=x[:,None]
		if y.ndim==1:
			y=y[:,None]		
		
		self.p=y.shape[1]
		self.q=x.shape[1]	

		x_all=[]
		y_all=[]

		for i in range(self.q):
			x_all.append(x[:,[i]])
			y_all.append(y)
		x_all=np.vstack(x_all)
		y_all=np.vstack(y_all)
		self.cond_gaussian_mixture=ConditionalGaussianMixture(self.n_states,n_gibbs=self.n_gibbs,f_burn=self.f_burn,min_k=self.min_k,max_k=self.max_k,kelly_std=self.kelly_std,max_w=self.max_w)
		self.cond_gaussian_mixture.estimate(y=y_all,x=x_all)

	def get_weight(self,xq,normalize=True,**kwargs):		
		w=np.zeros((self.q,self.p))
		for i in range(self.q):
			w[i]=self.cond_gaussian_mixture.get_weight(xq[[i]],normalize)
		w=np.mean(w,axis=0)
		return w


# n_states,n_gibbs=1000,f_burn=0.1,max_k=0.25,kelly_std=2,max_w=1
class StateConditionalGaussianMixture(object):
	def __init__(self,n_states,n_gibbs=None,f_burn=0.1,min_k=0.25,max_k=0.25,min_points=10,kelly_std=2,max_w=1):
		self.n_states=n_states
		self.n_gibbs=n_gibbs
		self.no_gibbs=False
		self.kelly_std=kelly_std
		self.min_k = min_k
		self.max_w=max_w
		if self.n_gibbs is None:
			self.no_gibbs=True
		self.f_burn=f_burn
		self.max_k=max_k
		self.min_points=min_points
		self.cond_gaussians_mix={}
		self.default_w=0
		self.p=1
		self.max_w_norm=1

	def view(self,plot_hist=True):
		print('StateGaussianMixture')
		for k,v in self.cond_gaussians_mix.items():
			print('State z=%s'%k)
			v.view(plot_hist)
			print()
			print()

	def estimate(self,y,x,z=None,**kwargs):
		self.max_w_norm=0
		self.p=y.shape[1]
		n=y.shape[0]
		if z is None:
			z=np.zeros(n,dtype=int)
		# assert z.ndim==1,"z must be a vector"
		if z.ndim!=1:
			z=z[:,None]		
		z=np.array(z,dtype=int)
		uz=np.unique(z)
		for e in uz:
			g=ConditionalGaussianMixture(self.n_states,self.n_gibbs,self.f_burn,self.min_k,self.max_k,self.kelly_std,self.max_w)

			i=np.where(z==e)[0]
			if i.size>self.min_points:
				g.estimate(y[i],x[i])
				self.cond_gaussians_mix.update({e:g})
				self.max_w_norm=max(self.max_w_norm,g.w_norm)

	def get_weight(self,xq,z,**kwargs):
		if isinstance(z,np.ndarray):
			z=z[-1]
		g=self.cond_gaussians_mix.get(z)
		if g is None:
			return self.default_w*np.ones(self.p)
		else:
			w=g.get_weight(xq,normalize=False)/self.max_w_norm
			d=np.sum(np.abs(w))
			if d>self.max_w:
				w/=d
				w*=self.max_w
			return w

# ---------------------------------------------------------







# Gaussian HMM
class GaussianHMM(object):
	def __init__(self,n_states=2,n_gibbs=1000,A_zeros=[],A_groups=[],f_burn=0.1,max_k=0.25,pred_l=None,allowed_sides='all',**kwargs):
		'''
		n_states: integer with the number of states
		n_gibbs: integer with the number of gibbs iterations
		A_zeros: list of list like [[0,0],[0,1],[3,1]]
			with the entries of the transition matrix that are 
			set to zero
		A_groups: list of lists like [[0,1],[2,3]] of disjoint elements
			where each sublist is the set of states that have the same 
			emissions, i.e, they are the same state
		f_burn: float in (0,1) with the fraction of points to burn at
			the beginning of the samples
		max_k: covariance shrinkage parameter
		'''
		self.n_states=n_states
		self.f_burn=f_burn
		self.n_gibbs=n_gibbs
		self.A_zeros=A_zeros
		self.A_groups=A_groups
		self.pred_l=pred_l
		self.allowed_sides=allowed_sides
		if len(self.A_groups)==0:
			self.A_groups=[[e] for e in range(self.n_states)]   
		self.eff_n_states=len(self.A_groups)
		self.max_k=max_k
		# real number of samples to simulate
		self.n_gibbs_sim=int(self.n_gibbs*(1+self.f_burn))
		self.p=1
		self.P=None
		self.gibbs_P=None
		self.gibbs_A=None		
		self.gibbs_mean=None
		self.gibbs_cov=None
		self.A=None
		self.states_mean=None
		self.states_cov=None 
		self.states_cov_inv=None
		self.states_cov_det=None
		self.w_norm=1

	def view(self,plot_hist=False):
		'''
		plot_hist: if true, plot histograms, otherwise just show the parameters
		'''
		print('** Gaussian HMM **')
		print('Groups')
		for e in self.A_groups:
			print('States %s have the same emission'%','.join([str(a) for a in e]))
		print('Initial state probability')
		print(self.P)
		if plot_hist:
			for i in range(self.n_states):
				plt.hist(self.gibbs_P[:,i],density=True,alpha=0.5,label='P[%s]'%(i))
			plt.legend()
			plt.grid(True)
			plt.show()		
		print('State transition')
		print(np.round(self.A,3))
		print()
		if plot_hist:
			for i in range(self.n_states):
				for j in range(self.n_states):
					if [i,j] not in self.A_zeros:
						plt.hist(self.gibbs_A[:,i,j],density=True,alpha=0.5,label='A[%s->%s]'%(i,j))
			plt.legend()
			plt.grid(True)
			plt.show()
		for j in range(self.eff_n_states):
			print('State %s'%(j+1))
			print('Mean')
			print(self.states_mean[j])
			print('Covariance')
			print(self.states_cov[j])
			print()
			if plot_hist:
				if self.gibbs_mean is not None:
					for i in range(self.p):
						plt.hist(self.gibbs_mean[j,:,i],density=True,alpha=0.5,label='Mean x%s'%(i+1))
					plt.legend()
					plt.grid(True)
					plt.show()
				if self.gibbs_cov is not None:
					for i in range(self.p):
						for q in range(i,self.p):
							plt.hist(self.gibbs_cov[j,:,i,q],density=True,alpha=0.5,label='Cov(x%s,x%s)'%(i+1,q+1))
					plt.legend()
					plt.grid(True)
					plt.show()

	def next_state_prob(self,y,l=None):
		'''
		computes a vector with the next state probability
		given a input sequence
		xyq: numpy (n,self.p) array with observation
		l: integer to filter recent data in y -> y=y[-l:]
		'''
		assert y.ndim==2,"y must be a matrix"
		# just return the initial state probability 
		if y.shape[0]==0:
			return self.P

		assert y.shape[1]==self.p,"y must have the same number of variables as the training data"
		if l is not None:
			y=y[-l:]
		if self.states_cov_inv is None:
			self.states_cov_inv=np.zeros((self.eff_n_states,self.p,self.p))			
			self.states_cov_det=np.zeros(self.eff_n_states)
			for s in range(self.eff_n_states):
				self.states_cov_inv[s]=np.linalg.inv(self.states_cov[s])
				self.states_cov_det[s]=np.linalg.det(self.states_cov[s])
		n=y.shape[0]
		# declare arrays		
		# probability of observations given state
		prob=np.zeros((n,self.n_states),dtype=np.float64) 
		# probability of observations given state		
		eff_prob=np.zeros((n,self.eff_n_states),dtype=np.float64) 
		for s in range(self.eff_n_states):
			# use vectorized function
			eff_prob[:,s]=mvgauss_prob(
										y,
										self.states_mean[s],
										self.states_cov_inv[s],
										self.states_cov_det[s]
										)
			prob[:,self.A_groups[s]]=eff_prob[:,[s]]  
		alpha,_=forward(prob,self.A,self.P)
		next_state_prob=np.dot(self.A.T,alpha[-1])  
		return next_state_prob
	
	def get_weight(self,y,normalize=True,**kwargs):
		'''
		compute betting weight given an input sequence
		y: numpy (n,p) array with a sequence
			each point is a joint observations of the variables
		l: integer to filter recent data in y -> y=y[-l:]
		returns:
			w: numpy (p,) array with weights to allocate to each asset
			in y
		'''
		next_state_prob=self.next_state_prob(y,self.pred_l)		
		# group next state prob
		tmp=np.zeros(self.eff_n_states)
		for i,e in enumerate(self.A_groups):
			tmp[i]=np.sum(next_state_prob[e])
		next_state_prob=tmp		
		# compute expected value		
		mu=np.sum(self.states_mean*next_state_prob[:,None],axis=0)
		# compute second central moment of the mixture distribution
		cov=np.zeros((self.p,self.p))
		for s in range(self.eff_n_states):
			cov+=(next_state_prob[s]*self.states_cov[s])
			cov+=(next_state_prob[s]*self.states_mean[s]*self.states_mean[s][:,None])
		cov-=(mu*mu[:,None])
		w=np.dot(np.linalg.inv(cov),mu)
		if self.allowed_sides=='long':
			w[np.where(w<0)[0]]=0
		if self.allowed_sides=='short':
			w[np.where(w>0)[0]]=0						
		if normalize:
			w/=self.w_norm
		return w

	def estimate(self,y,idx=None,**kwargs):	 
		'''
		Estimate the HMM parameters with Gibbs sampling
		y: numpy (n,p) array
			each row is a joint observation of the variables
		idx: None or array with the indexes that define subsequences
			for example, idx=[[0,5],[5,12],[12,30]] means that subsequence 1 is y[0:5],
			subsequence 2 is y[5:12], subsequence 3 is y[12:30], ...				   
		'''
		assert y.ndim==2,"y must be a matrix"
		
		if idx is None:
			idx=np.array([[0,y.shape[0]]],dtype=int)		 
		
		# just form safety
		idx=np.array(idx,dtype=int)

		n_seqs=idx.shape[0]
		
		self.states_cov_inv=None
		self.states_cov_det=None

		n=y.shape[0]
		self.p=y.shape[1]

		# initial state probabilities will not be sampled
		# a equal probability will be assumed
		self.P=np.ones(self.n_states)
		self.P/=np.sum(self.P)
		

		# generate variable with the possible states
		states=np.arange(self.n_states,dtype=np.int32)

		# compute data covariance
		c=np.cov(y.T)
		# fix when y has only one column
		if c.ndim==0:
			c=np.array([[c]])
		c_diag=np.diag(np.diag(c)) # diagonal matrix with the covariances

		# Prior distribution parameters
		# these parameters make sense for the type of problems
		# we are trying to solve - assuming zero correlation makes sense
		# as a prior and zero means as well due to the low 
		# values of financial returns
		m0=np.zeros(self.p) # mean: prior location (just put it at zero...)
		V0=c_diag.copy() # mean: prior covariance
		S0aux=c_diag.copy() # covariance prior scale (to be multiplied later)
		alpha0=self.n_states
		alpha=1 # multinomial prior (dirichelet alpha)
		zero_alpha=0.001 # multinomial prior (dirichelet alpha) when there is no transition
		
		alpha_p=0.05 # multinomial prior (dirichelet alpha) for init state distribution
				
		# Precalculations
		
		# the prior alphas need to be calculated before
		# because there may be zero entries in the A matrix
		alphas=[]
		for s in range(self.n_states):
			tmp=alpha*np.ones(self.n_states)
			for e in self.A_zeros:
				if e[0]==s:
					tmp[e[1]]=zero_alpha		
			alphas.append(tmp)
				
		# y_mean=np.mean(y,axis=0)
		invV0=np.linalg.inv(V0)
		invV0m0=np.dot(invV0,m0)
		
		self.eff_n_states=len(self.A_groups)
		
		# initialize containers
		transition_counter=np.zeros((self.n_states,self.n_states)) # counter for state transitions
		init_state_counter=np.zeros(self.n_states) # counter for initial state observations
		
		eff_prob=np.zeros((n,self.eff_n_states)) # probability of observations given state
		prob=np.zeros((n,self.n_states),dtype=np.float64) # probability of observations given state
		
		forward_alpha=np.zeros((n,self.n_states),dtype=np.float64)
		forward_c=np.zeros(n,dtype=np.float64)		
						
		
		self.gibbs_cov=np.zeros((self.eff_n_states,self.n_gibbs_sim,self.p,self.p)) # store sampled covariances
		self.gibbs_mean=np.zeros((self.eff_n_states,self.n_gibbs_sim,self.p)) # store sampled means
		self.gibbs_A=np.zeros((self.n_gibbs_sim,self.n_states,self.n_states)) # store sampled transition matricess
		self.gibbs_P=np.zeros((self.n_gibbs_sim,self.n_states))
		
		# initialize covariances and means
		for s in range(self.eff_n_states):
			self.gibbs_mean[s,0]=m0
			self.gibbs_cov[s,0]=c   
		# initialize state transition
		# assume some persistency of state as a initial parameter
		# this makes sense because if this is not the case then this is
		# not very usefull
		if len(self.A_zeros)==0:
			init_mass=0.9
			tmp=init_mass*np.eye(self.n_states)
			remaining_mass=(1-init_mass)/(self.n_states-1)
			tmp[tmp==0]=remaining_mass		
			self.gibbs_A[0]=tmp
		else:
			# initialize in a different way!
			tmp=np.ones((self.n_states,self.n_states))
			for e in self.A_zeros:
				tmp[e[0],e[1]]=0
			tmp/=np.sum(tmp,axis=1)[:,None]
			self.gibbs_A[0]=tmp
		
		self.gibbs_P[0]=np.ones(self.n_states)
		self.gibbs_P[0]/=np.sum(self.gibbs_P[0])		
		
		# create and initialize variable with
		# the states associated with each variable
		# assume equal probability in states
		q=np.random.choice(states,size=n)

		# previous used parameters to sample when there are not observations
		# on that state
		prev_mn=np.zeros((self.n_states,self.p))
		prev_Vn=np.zeros((self.n_states,self.p,self.p))
		prev_vn=np.zeros(self.n_states)
		prev_Sn=np.zeros((self.n_states,self.p,self.p))
		for j in range(self.n_states):
			prev_mn[j]=m0
			prev_Vn[j]=V0
			prev_vn[j]=self.p+1+1
			prev_Sn[j]=S0aux

		# Gibbs sampler
		for i in range(1,self.n_gibbs_sim):

			transition_counter*=0 # set this to zero
			init_state_counter*=0 # set this to zero
			# evaluate the probability of each
			# observation in y under the previously 
			# sampled parameters
			for s in range(self.eff_n_states):
				# compute inverse and determinant
				cov_inv=np.linalg.inv(self.gibbs_cov[s,i-1])
				cov_det=np.linalg.det(self.gibbs_cov[s,i-1])
				# use vectorized function
				eff_prob[:,s]=mvgauss_prob(y,self.gibbs_mean[s,i-1],cov_inv,cov_det)  
				prob[:,self.A_groups[s]]=eff_prob[:,[s]]			
			
			# use multiple sequences

			for l in range(n_seqs):		 
				# compute alpha variable
				forward_alpha,_=forward(prob[idx[l][0]:idx[l][1]],self.gibbs_A[i-1],self.gibbs_P[i-1])#,forward_alpha,forward_c)			
				# backward walk to sample from the state sequence			
				backward_sample(
								self.gibbs_A[i-1],
								forward_alpha,
								q[idx[l][0]:idx[l][1]],
								transition_counter,
								init_state_counter)

			# now, with a sample from the states (in q variable)
			# it is all quite similar to a gaussian mixture!
			for j in range(self.n_states):
				# sample from transition matrix										
				self.gibbs_A[i,j]=np.random.dirichlet(alphas[j]+transition_counter[j])				
			# make sure that the entries are zero!
			for e in self.A_zeros:
				self.gibbs_A[i,e[0],e[1]]=0.
			self.gibbs_A[i]/=np.sum(self.gibbs_A[i],axis=1)[:,None]
			
			# sample from initial state distribution
			self.gibbs_P[i]=np.random.dirichlet(alpha_p+init_state_counter)   
			
			for j in range(self.eff_n_states):
				# basically, this is the code to sample from a multivariate
				# gaussian but constrained to observations where state=j		   
				idx_states=np.where(np.in1d(q,self.A_groups[j]))[0]
				# just sample from the prior!
				if idx_states.size==0:
					self.gibbs_mean[j,i]=np.random.multivariate_normal(prev_mn[j],prev_Vn[j])
					self.gibbs_cov[j,i]=invwishart.rvs(df=prev_vn[j],scale=prev_Sn[j])			   
				else:
					n_count=idx_states.size
					x_=y[idx_states]
					y_mean_=np.mean(x_,axis=0)
					# ---------------------
					# sample for mean
					invC=np.linalg.inv(self.gibbs_cov[j,i-1])
					Vn=np.linalg.inv(invV0+n_count*invC)
					mn=np.dot(Vn,invV0m0+n_count*np.dot(invC,y_mean_))
					prev_mn[j]=mn
					prev_Vn[j]=Vn
					self.gibbs_mean[j,i]=np.random.multivariate_normal(mn,Vn)
					# sample from cov
					# get random k value (shrinkage value)
					k=np.random.uniform(0,self.max_k)
					n0=k*n_count
					S0=n0*S0aux
					v0=n0+self.p+1
					vn=v0+n_count
					St=np.dot((x_-self.gibbs_mean[j,i]).T,(x_-self.gibbs_mean[j,i]))
					Sn=S0+St
					prev_vn[j]=vn
					prev_Sn[j]=Sn
					self.gibbs_cov[j,i]=invwishart.rvs(df=vn,scale=Sn)				 

		# burn observations
		self.gibbs_A=self.gibbs_A[-self.n_gibbs:]
		self.gibbs_P=self.gibbs_P[-self.n_gibbs:]
		self.gibbs_mean=self.gibbs_mean[:,-self.n_gibbs:,:]
		self.gibbs_cov=self.gibbs_cov[:,-self.n_gibbs:,:,:]

		self.A=np.mean(self.gibbs_A,axis=0)
		self.P=np.mean(self.gibbs_P,axis=0)
		self.states_mean=np.mean(self.gibbs_mean,axis=1)
		self.states_cov=np.mean(self.gibbs_cov,axis=1)		

		
		# compute w norm
		self.states_cov_inv=np.zeros_like(self.states_cov)
		self.states_cov_det=np.zeros(self.eff_n_states)
		

		self.w_norm=0
		for i in range(self.eff_n_states):
			self.states_cov_inv[i]=np.linalg.inv(self.states_cov[i])
			self.states_cov_det[i]=np.linalg.det(self.states_cov[i])
			self.w_norm=max(self.w_norm,np.sum(np.abs(np.dot(self.states_cov_inv[i],self.states_mean[i])))  )


class ConditionalGaussianHMM(object):
	def __init__(self,n_states=2,n_gibbs=1000,A_zeros=[],A_groups=[],f_burn=0.1,max_k=0.25,kelly_std=2,max_w=1,pred_l=None):
		self.n_states=n_states
		self.n_gibbs=n_gibbs
		self.A_zeros=A_zeros
		self.A_groups=A_groups
		self.pred_l=pred_l
		self.kelly_std=kelly_std
		self.max_w=max_w
		self.f_burn=f_burn
		self.max_k=max_k
		self.ghmm=None
		self.w_norm=1

	def view(self,plot_hist=True):
		if self.ghmm is not None:
			self.ghmm.view(plot_hist=plot_hist)
	
	def estimate(self,y,x,idx=None,**kwargs): 
		x=x.copy()
		y=y.copy()		
		if x.ndim==1:
			x=x[:,None]
		if y.ndim==1:
			y=y[:,None]		
		p=y.shape[1]
		q=x.shape[1]		
		z=np.hstack((y,x))

		self.ghmm=GaussianHMM(
							n_states=self.n_states,
							n_gibbs=self.n_gibbs,
							A_zeros=self.A_zeros,
							A_groups=self.A_groups,
							f_burn=self.f_burn,
							max_k=self.max_k,
							pred_l=self.pred_l)
		self.ghmm.estimate(z,idx=idx)
		
		# extract distribution of y|x from the estimated covariance
		y_idx=np.arange(p)
		x_idx=np.arange(p,p+q)	
				
		self.states_my=np.zeros((self.n_states,p))
		self.states_mx=np.zeros((self.n_states,q))
		self.states_Cyy=np.zeros((self.n_states,p,p))
		self.states_Cxx=np.zeros((self.n_states,q,q))
		self.states_Cxx_inv=np.zeros((self.n_states,q,q))
		self.states_Cxx_det=np.zeros((self.n_states))
		self.states_Cyx=np.zeros((self.n_states,p,q))
		self.states_invCxx=np.zeros((self.n_states,q,q))
		self.states_pred_gain=np.zeros((self.n_states,p,q))
		self.states_cov_reduct=np.zeros((self.n_states,p,p))
		self.states_pred_cov=np.zeros((self.n_states,p,p))
		self.states_pred_cov_inv=np.zeros((self.n_states,p,p))
		self.w_norm=0

		for i in range(self.n_states):
			self.states_my[i]=self.ghmm.states_mean[i][y_idx]
			self.states_mx[i]=self.ghmm.states_mean[i][x_idx]
			self.states_Cyy[i]=self.ghmm.states_cov[i][y_idx][:,y_idx]
			self.states_Cxx[i]=self.ghmm.states_cov[i][x_idx][:,x_idx]
			self.states_Cxx_inv[i]=np.linalg.inv(self.states_Cxx[i])
			self.states_Cxx_det[i]=np.linalg.det(self.states_Cxx[i])
			self.states_Cyx[i]=self.ghmm.states_cov[i][y_idx][:,x_idx]
			self.states_invCxx[i]=np.linalg.inv(self.states_Cxx[i])
			self.states_pred_gain[i]=np.dot(self.states_Cyx[i],self.states_invCxx[i])
			self.states_cov_reduct[i]=np.dot(self.states_pred_gain[i],self.states_Cyx[i].T)
			self.states_pred_cov[i]=self.states_Cyy[i]-self.states_cov_reduct[i]
			self.states_pred_cov_inv[i]=np.linalg.inv(self.states_pred_cov[i])			

			# compute normalization
			x_move=np.sqrt(np.diag(self.states_Cxx[i]))*self.kelly_std
			self.w_norm=max(self.w_norm,np.sum(np.abs(np.dot( self.states_pred_cov_inv[i] , self.states_my[i] + np.dot(np.abs(self.states_pred_gain[i]),x_move+self.states_mx[i]) ))))


	def get_weight(self,xq,y,x,normalize=True,**kwargs):		
		# calculate next state probabilities, then it is the same
		# as a mixture model
		if isinstance(xq,float):
			xq=np.array([xq])
		if x.ndim==1:
			x=x[:,None]
		if y.ndim==1:
			y=y[:,None]
		# if self.pred_l is not None:
		#	x=x[-self.pred_l:]
		#	y=y[-self.pred_l:]

		z=np.hstack((y,x))		
		nsp=self.ghmm.next_state_prob(z,self.pred_l)	 
		# this is the same as for a gaussian mixture
		# but the mixing weights come from the hidden
		# states probability
		pis=np.zeros(self.n_states)
		for s in range(self.n_states):
			 pis[s]=nsp[s]*mvgauss_prob(np.array([xq]),self.states_mx[s],self.states_Cxx_inv[s],self.states_Cxx_det[s])[-1]
		pis/=np.sum(pis)						
		p=self.states_my.shape[1]
		mu=np.zeros(p)
		cov=np.zeros((p,p))
		for s in range(self.n_states):		
			tmp_mu=self.states_my[s]+np.dot(self.states_pred_gain[s],xq-self.states_mx[s])
			mu+=pis[s]*tmp_mu
			cov+=pis[s]*(self.states_pred_cov[s]+tmp_mu*tmp_mu[:,None])
		cov-=mu*mu[:,None]			
		w=np.dot(np.linalg.inv(cov),mu)
		if normalize:
			w/=self.w_norm
			d=np.sum(np.abs(w))
			if d>self.max_w:
				w/=d
				w*=self.max_w
		return w	


class GaussianTrack(object):	
	def __init__(self,phi,phi_m=None,min_l=10,max_l=100,diag_cov=True,reg_cov=False,
						eq_sharpe=False,min_points_stats=10,quantile=0.9,reg_l=0.1,
						 leverage=1,use_x_data=False,allowed_sides='all',**kwargs):		
		self.phi=phi
		self.phi_m=phi_m
		if self.phi_m is None:
			 self.phi_m=self.phi
		self.min_l=min_l 
		self.max_l=max_l
		assert self.min_l<=self.max_l,"min_l must be lower than max_l"
		self.diag_cov=diag_cov
		self.reg_cov=reg_cov
		self.reg_l=reg_l
		self.eq_sharpe=eq_sharpe
		self.min_points_stats=min_points_stats
		self.max_w=[]
		self.quantile=quantile
		self.leverage=leverage
		self.use_x_data=use_x_data
		self.allowed_sides=allowed_sides
		self.p=None
		self.w_norm=None

	def view(self,plot_hist=True):
		print()
		print('GaussianTrack')
		print('-> nothing to see')
		print()

	def estimate(self,y,x=None,**kwargs): 
		'''
		??
		'''
		z=y
		if self.use_x_data:
			z=x
		self.p=z.shape[1]
		# to estimate the model just run it for the data 
		for i in range(1,z.shape[0]):
			_=self.get_weight(y=z[:i],x=z[:i],is_estimating=True)

	def predict(self,y,**kwargs):
		assert y.ndim==2,"y must be a matrix"
		if y.shape[0]>self.min_l:
			# parameters
			y=y[-self.max_l:]
			v=self.p+1+1/(1+self.phi)
			k=1/(1+self.phi)			
			# weighting scheme for observations
			t=np.arange(y.shape[0])[::-1]
			# weights for observations
			wmean=np.power(self.phi_m,t)*(1-self.phi_m)
			wcov=np.power(self.phi,t+1)*(1-self.phi)
			# should sum to one
			wmean/=np.sum(wmean)
			wcov/=np.sum(wcov)
			# compute mean with weighted observations
			mean=np.sum(y*wmean[:,None],axis=0)
			# compute covariance (here is more the second non central moment 
			# should not make much difference)
			cov=y[:,None,:]*y[:,:,None]
			cov*=wcov[:,None,None]
			cov=np.sum(cov,axis=0)	 
			if self.diag_cov:
				cov=np.diag(np.diag(cov))	   
			if self.reg_cov:
				pass
			
			if self.diag_cov:
				cov=np.diag(np.diag(cov))

			if self.reg_cov:
				# use regularization				
				scale=np.sqrt(np.diag(cov))
				tmp=np.diag(1/scale)
				R=np.dot(np.dot(tmp,cov),tmp)
				R=R-np.eye(R.shape[0])
				R*=(self.reg_l/(R.shape[0]-1))
				R+=np.eye(R.shape[0])
				tmp=np.diag(scale)
				cov=np.dot(np.dot(tmp,R),tmp)
		else:
			cov=None # np.diag(np.ones(self.p,dtype=float))
			mean=np.zeros(self.p,dtype=float)
		return mean,cov		

	def get_weight(self,y,x=None,is_estimating=False,**kwargs):
		z=y
		if self.use_x_data:
			z=x
		mean,cov=self.predict(z)
		if cov is not None:						
			scale=np.sqrt(np.diag(cov))
			D=np.diag(1/scale)
			R=np.dot(np.dot(D,cov),D)
			sr=mean/scale						
			if self.eq_sharpe:
				sr=np.sign(sr)
			if self.reg_cov:				
				#Rinv=R-np.eye(R.shape[0])
				#Rinv=np.eye(R.shape[0])-Rinv
				Rinv=2*np.eye(R.shape[0])-R
			else:
				Rinv=np.linalg.inv(R)

			w=np.dot(D,np.dot(Rinv,sr))
					
			# else:				
			# 	w=np.dot(np.linalg.inv(cov),mean)
			
			if self.allowed_sides=='long':
				w[np.where(w<0)[0]]=0
			if self.allowed_sides=='short':
				w[np.where(w>0)[0]]=0

			# acomulate stats only if we are estimating the model!
			if is_estimating:
				self.max_w.append(np.sum(np.abs(w)))
		else:
			w=0*mean

		if len(self.max_w)>self.min_points_stats:
			# just repeat this calculations...
			aux=np.array(self.max_w)
			aux.sort()
			norm=aux[int(self.quantile*aux.size)]
			w/=norm			
			d=np.sum(np.abs(w))
			if d>1:
				w/=d
			# control concentration ?
			w*=self.leverage
			# round			
			w=np.round(np.abs(w),2)*np.sign(w)
			return w
		else:
			return 0*w 


# PROB CCR
# FINISH THIS!!


def firsteig(A,init_vec=None,tol=1e-8,max_iter=100,info=True):
	p=A.shape[0]
	if init_vec is None:
		u=np.ones(p)
	else:
		u=init_vec
	u/=np.sqrt(np.sum(np.power(u,2)))	
	y=np.dot(A,u)
	l_=0
	for i in range(max_iter):
		u=y/np.sqrt(np.sum(np.power(y,2)))
		y=np.dot(A,u)
		l=np.dot(u,y)
		if np.abs(l-l_)<tol*np.abs(l):
			return l,u
		l_=l
	if info:
		print('Eigenvalue did not converge')
	return l,u

class ProbCCR(object):

	def __init__(self,n_gibbs=None,f_burn=0.1,max_k=0.25,kelly_std=2,max_w=1):
		self.n_gibbs=n_gibbs
		self.no_gibbs=False
		if self.n_gibbs is None:
			self.no_gibbs=True
		self.f_burn=f_burn
		self.max_k=max_k
		self.kelly_std=kelly_std
		self.max_w=max_w
		self.g=None
		# to calculate after Gaussian estimate
		self.my=None
		self.mx=None
		self.Cyy=None
		self.Cxx=None
		self.Cyx=None
		self.invCxx=None
		self.pred_gain=None
		self.cov_reduct=None
		self.pred_cov=None 
		self.prev_cov_inv=None
		self.w_norm=1

	def view(self,plot_hist=True):
		if self.g is not None:
			self.g.view(plot_hist=plot_hist)
	
	def estimate(self,y,x,**kwargs): 
		x=x.copy()
		y=y.copy()		
		if x.ndim==1:
			x=x[:,None]
		if y.ndim==1:
			y=y[:,None]		
		p=y.shape[1]
		q=x.shape[1]	

		z=np.hstack((y,x))
		names=[]
		for i in range(p):
			names.append("y%s"%(i+1))
		for i in range(q):
			names.append("x%s"%(i+1))
		self.g=Gaussian(self.n_gibbs,self.f_burn,self.max_k,names=names)
		self.g.estimate(z)
		# extract distribution of y|x from the estimated covariance
		y_idx=np.arange(p)
		x_idx=np.arange(p,p+q)		
		self.my=self.g.mean[y_idx]
		self.mx=self.g.mean[x_idx]
		self.Cyy=self.g.cov[y_idx][:,y_idx]
		self.Cxx=self.g.cov[x_idx][:,x_idx]
		self.Cyx=self.g.cov[y_idx][:,x_idx]
		self.invCxx=np.linalg.inv(self.Cxx)
		self.pred_gain=np.dot(self.Cyx,self.invCxx)
		self.cov_reduct=np.dot(self.pred_gain,self.Cyx.T)
		self.pred_cov=self.Cyy-self.cov_reduct
		self.pred_cov_inv=np.linalg.inv(self.pred_cov)
		# compute normalization
		x_move=np.sqrt(np.diag(self.Cxx))*self.kelly_std
		self.w_norm=np.sum(np.abs(np.dot( self.pred_cov_inv , self.my + np.dot(np.abs(self.pred_gain),x_move+self.mx) )))

	def predict(self,xq):
		return self.my+np.dot(self.pred_gain,xq-self.mx)
	
	def expected_value(self,xq):
		return predict(xq)
	
	def covariance(self,xq):
		return self.pred_cov
	
	def get_weight(self,xq,normalize=True,**kwargs):
		if normalize:
			w=np.dot(self.pred_cov_inv,self.predict(xq))/self.w_norm
			d=np.sum(np.abs(w))
			if d>self.max_w:
				w/=d
				w*=self.max_w
			return w			
		else:
			return np.dot(self.pred_cov_inv,self.predict(xq))






class Emission(ABC):
	
	@abstractmethod
	def view(self, plot_hist:bool = False):
		"""Subclasses must implement this method"""
		pass		
		
	@abstractmethod
	def set_gibbs_parameters(self, n_gibbs:int, f_burn:float, n_gibbs_sim:int = None):
		"""Subclasses must implement this method"""
		pass
	
	@abstractmethod
	def gibbs_initialize(self, y:np.ndarray, x:np.ndarray = None, **kwargs):
		"""Subclasses must implement this method"""
		pass
	
	@abstractmethod		
	def gibbs_posterior_sample(self, y:np.ndarray, x:np.ndarray, iteration:int, **kwargs):
		"""Subclasses must implement this method"""
		pass
	
	@abstractmethod
	def gibbs_burn_and_mean(self):
		"""Subclasses must implement this method"""
		pass
	
	@abstractmethod
	def gibbs_prob(self, y:np.ndarray, x:np.ndarray, iteration:int, **kwargs):
		"""Subclasses must implement this method"""
		pass
	
	@abstractmethod
	def prob(self, y:np.ndarray, x:np.ndarray, **kwargs):
		"""Subclasses must implement this method"""
		pass

	@abstractmethod
	def predict_moments(self, xq:np.ndarray, **kwargs):
		"""Subclasses must implement this method"""
		pass	
	@abstractmethod
	def predict_pi(self, xq:np.ndarray, **kwargs):
		"""Subclasses must implement this method"""
		pass	



	
class GaussianEmission(Emission):
	
	def __init__(self, n_gibbs:int = 100, f_burn:float = 0.1, mean = None, max_k = 0.1):
		self.n_gibbs = n_gibbs
		self.f_burn = f_burn
		self.mean_value = mean
		self.max_k = max_k
		self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))		
		self.p = 1
		self.w_norm = 1
		self.gibbs_cov = None
		self.gibbs_mean = None
		self.cov = None
		self.mean = None
		self.cov_inv = None
		self.cov_det = None 
		# auxiliar parameters
		self.S0aux = None
		self.invV0, self.invV0m0 = None, None
		self.prev_mn, self.prev_Vn, self.prev_vn, self.prev_Sn = None, None, None, None 

	def view(self, plot_hist = False):
		print('** Gaussian Emission **')
		print('Mean')
		print(self.mean)
		print('Covariance')
		print(self.cov)
		print()
		if plot_hist:
			if self.gibbs_mean is not None:
				for i in range(self.p):
					plt.hist(
							self.gibbs_mean[:,i],
							density=True,
							alpha=0.5,
							label='Mean x%s'%(i+1)
							)
				plt.legend()
				plt.grid(True)
				plt.show()
			if self.gibbs_cov is not None:
				for i in range(self.p):
					for j in range(i,self.p):
						plt.hist(
								self.gibbs_cov[:,i,j],
								density=True,
								alpha=0.5,
								label='Cov(x%s,x%s)'%(i+1,j+1)
								)
				plt.legend()
				plt.grid(True)
				plt.show()

	def set_gibbs_parameters(self, n_gibbs, f_burn, n_gibbs_sim = None):
		self.n_gibbs = n_gibbs
		self.f_burn = f_burn
		aux = int(self.n_gibbs*(1+self.f_burn))
		self.n_gibbs_sim = aux if n_gibbs_sim is None else n_gibbs_sim
			
	def gibbs_initialize(self, y, **kwargs):
		assert y.ndim == 2, "y must be a matrix"
		self.p = y.shape[1]
		# this will be updated later
		self.cov = np.eye(self.p)
		self.mean = np.zeros(self.p)
		# apply setted value
		if self.mean_value is not None: 
			self.mean = self.mean_value*np.ones(self.p)
		# Covariance samples
		self.gibbs_cov = np.zeros((self.n_gibbs_sim,self.p,self.p)) 
		# Mean samples
		self.gibbs_mean = np.zeros((self.n_gibbs_sim,self.p))  

		# compute data covariance
		c=np.cov(y.T)
		# fix when y has only one column
		if c.ndim==0:
			c=np.array([[c]])
		# diagonal matrix with the covariances
		c_diag=np.diag(np.diag(c))		 
		# Prior distribution parameters
		# these parameters make sense for the type of problems
		# we are trying to solve - assuming zero correlation makes sense
		# as a prior and zero means as well due to the low 
		# values of financial returns
		m0 = np.mean(y, axis = 0) # mean: prior location (just put it at zero...)
		V0 = c_diag.copy() # mean: prior covariance
		self.S0aux = c_diag.copy() # covariance prior scale (to be multiplied later)		
		self.invV0 = np.linalg.inv(V0)
		self.invV0m0 = np.dot(self.invV0, m0)	
		
		# initialize
		self.gibbs_mean[0] = m0
		self.gibbs_cov[0] = c		  
		
		# store parameters
		self.prev_mn = m0
		self.prev_Vn = V0
		self.prev_vn = self.p+1+1
		self.prev_Sn = self.S0aux

	def gibbs_posterior_sample(self, y:np.ndarray, iteration:int, **kwargs):
		'''
		y: current set of observations
		'''
		assert 0<iteration<self.n_gibbs_sim, "iteration is larger than the number of iterations"
		if y.shape[0] == 0:
			self.gibbs_mean[iteration] = np.random.multivariate_normal(self.prev_mn, self.prev_Vn)
			self.gibbs_cov[iteration] = invwishart.rvs(df = self.prev_vn, scale = self.prev_Sn)  
		else:
			n = y.shape[0]
			y_mean_ = np.mean(y, axis=0)
			# Sample from mean
			invC = np.linalg.inv(self.gibbs_cov[iteration-1])
			Vn = np.linalg.inv(self.invV0 + n*invC)
			mn = np.dot(Vn, self.invV0m0 + n*np.dot(invC,y_mean_))
			self.prev_mn = mn
			self.prev_Vn = Vn
			if self.mean_value is None:
				self.gibbs_mean[iteration] = np.random.multivariate_normal(mn, Vn)
			else:
				self.gibbs_mean[iteration] = self.mean_value*np.ones(self.p)
			# Sample from cov
			# Get random k value (shrinkage value)
			k = np.random.uniform(0, self.max_k)
			n0 = k*n
			S0 = n0*self.S0aux
			v0 = n0 + self.p + 1
			vn = v0 + n
			St = np.dot((y-self.gibbs_mean[iteration]).T,(y-self.gibbs_mean[iteration]))
			Sn = S0 + St
			self.prev_vn = vn
			self.prev_Sn = Sn
			self.gibbs_cov[iteration] = invwishart.rvs(df = vn, scale = Sn)			
	
	def gibbs_burn_and_mean(self):
		self.gibbs_mean = self.gibbs_mean[-self.n_gibbs:,:]
		self.gibbs_cov = self.gibbs_cov[-self.n_gibbs:,:,:]		
		self.mean = np.mean(self.gibbs_mean,axis=0)
		self.cov = np.mean(self.gibbs_cov,axis=0)	
		self.cov_inv = np.linalg.inv(self.cov)
		self.cov_det = np.linalg.det(self.cov)
		self.w_norm = np.sum(np.abs(np.dot(self.cov_inv,self.mean)))		
	
	def estimate(self, y:np.ndarray, **kwargs):
		self.gibbs_initialize(y)
		for i in range(1, self.n_gibbs_sim):
			self.gibbs_posterior_sample(y, i)
		self.gibbs_burn_and_mean()		
		
	def gibbs_prob(self, y:np.ndarray, iteration:int, **kwargs):
		assert 0<iteration<self.n_gibbs_sim, "iteration is larger than the number of iterations"
		cov_inv = np.linalg.inv(self.gibbs_cov[iteration-1])
		cov_det = np.linalg.det(self.gibbs_cov[iteration-1])
		# use vectorized function
		return mvgauss_prob(y, self.gibbs_mean[iteration-1], cov_inv, cov_det)  

	def prob(self, y:np.ndarray,  **kwargs):		
		# use vectorized function
		return mvgauss_prob(y, self.mean, self.cov_inv, self.cov_det)  
	
	def predict_moments(self, **kwargs):
		return self.mean, self.cov
	
	def predict_pi(self, **kwargs):
		return 1



class ConditionalGaussianEmission(Emission):
	
	def __init__(self, n_gibbs:int = 100, f_burn:float = 0.1, mean = None, max_k = 0.1, kelly_std = 2):
		self.n_gibbs = n_gibbs
		self.f_burn = f_burn
		self.mean_value = mean
		self.max_k = max_k
		self.kelly_std = kelly_std
		self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))		
		self.p = 1
		self.q = 1
		self.gibbs_cov = None
		self.gibbs_mean = None
		self.cov = None
		self.mean = None		
		self.gaussian_emission = GaussianEmission(
												n_gibbs = n_gibbs,
												f_burn = f_burn,
												mean = mean,
												max_k = max_k
												)
		# auxiliar parameters
		self.S0aux = None
		self.invV0, self.invV0m0 = None, None
		self.prev_mn, self.prev_Vn, self.prev_vn, self.prev_Sn = None, None, None, None 

	def view(self, plot_hist = False):
		print('** Conditional Gaussian Emission **')
		self.gaussian_emission.view(plot_hist = plot_hist)
		
	def set_gibbs_parameters(self, n_gibbs, f_burn, n_gibbs_sim = None):
		self.gaussian_emission.set_gibbs_parameters(
													n_gibbs = n_gibbs, 
													f_burn = f_burn, 
													n_gibbs_sim = n_gibbs_sim
													)
		
	def gibbs_initialize(self, y, x, **kwargs):
		assert y.ndim == 2, "y must be a matrix"
		assert x.ndim == 2, "x must be a matrix"
				
		x=x.copy()
		y=y.copy()
		
		self.p = y.shape[1]
		self.q = x.shape[1]
		
		self.y_idx = np.arange(self.p)
		self.x_idx = np.arange(self.p,self.p+self.q)		
		
		z=np.hstack((y,x)) 
		
		self.gaussian_emission.gibbs_initialize(y = z)
		
		return z
		
	def gibbs_posterior_sample(
								self, 
								y:np.ndarray, 
								x:np.ndarray, 
								iteration:int, 
								z:np.ndarray = None, 
								**kwargs
								):
		'''
		y: current set of observations
		x: current set of observations
		'''
		if z is None: z = np.hstack((y,x))		 
		self.gaussian_emission.gibbs_posterior_sample(y = z, iteration = iteration)

	def gibbs_burn_and_mean(self):
		self.gaussian_emission.gibbs_burn_and_mean()
		# compute fields for conditonal distribution
		self.my = self.gaussian_emission.mean[self.y_idx]
		self.mx = self.gaussian_emission.mean[self.x_idx]
		self.Cyy = self.gaussian_emission.cov[self.y_idx][:,self.y_idx]
		self.Cxx = self.gaussian_emission.cov[self.x_idx][:,self.x_idx]
		self.Cxx_inv = np.linalg.inv(self.Cxx)
		self.Cxx_det = np.linalg.det(self.Cxx)
		self.Cyx = self.gaussian_emission.cov[self.y_idx][:,self.x_idx]
		self.invCxx = np.linalg.inv(self.Cxx)
		self.pred_gain = np.dot(self.Cyx, self.invCxx)
		self.cov_reduct = np.dot(self.pred_gain, self.Cyx.T)
		self.pred_cov = self.Cyy - self.cov_reduct
		self.pred_cov_inv = np.linalg.inv(self.pred_cov)			
		# compute normalization
		x_move = np.sqrt(np.diag(self.Cxx))*self.kelly_std
		self.w_norm = np.sum(np.abs(np.dot(self.pred_cov_inv,self.my+np.dot(np.abs(self.pred_gain),x_move + self.mx))))

	def estimate(self, y:np.ndarray, x:np.ndarray, **kwargs):
		z = self.gibbs_initialize(y = y, x = x)
		for i in range(1, self.n_gibbs_sim):
			self.gibbs_posterior_sample(y = y, x = x, iteration = i, z = z)
		self.gibbs_burn_and_mean()	 
		
	def gibbs_prob(self, y:np.ndarray, x:np.ndarray, iteration:int, **kwargs):
		z = np.hstack((y,x)) 
		return self.gaussian_emission.gibbs_prob(y = z, iteration = iteration)

	def prob(self, y:np.ndarray, x:np.ndarray,  **kwargs):		
		# use vectorized function
		z = np.hstack((y,x)) 
		return self.gaussian_emission.prob(y = z)
	
	def predict_moments(self, xq:np.ndarray, **kwargs):
		mean = self.my + np.dot(self.pred_gain, xq - self.mx)
		cov = self.pred_cov + mean*mean[:,None]			
		return mean, cov

	def predict_pi(self, xq:np.ndarray, **kwargs):
		assert xq.ndim == 1, "xq must be a vector"
		return mvgauss_prob(
					np.array([xq]),
					self.mx,
					self.Cxx_inv,
					self.Cxx_det
					)[-1]		



class HMM(object):
	
	def __init__(
				self,
				emissions:List[Emission] = [],				
				n_gibbs = 1000,
				A_zeros = [],
				emissions_indexes = [],
				f_burn = 0.1,
				pred_l = None,
				**kwargs
				):
		'''
		emissions: list of instances derived from Emission
				one for each state
		n_gibbs: number of gibbs samples
		A_zeros: list of lists with entries of A to be set to zero
		emissions_indexes: list of lists
			example
					emissions_indexes = [[0,2],[1,3]]
				means 
					emission[0] is applied to states 0 and 2
					emission[1] is applied to states 1 and 3
				also, this implies that the number of states is 4!
		f_burn: fraction to burn  
		'''
		self.emissions = emissions   
			   
		self.n_gibbs = n_gibbs
		self.f_burn = f_burn
		
		self.A_zeros = A_zeros
		
		self.emissions_indexes = emissions_indexes 
		if len(self.emissions_indexes) == 0:
			self.n_states = len(self.emissions)
			self.emissions_indexes = [[e] for e in range(self.n_states)]   
		
		self.n_states = 0 
		for e in self.emissions_indexes: self.n_states += len(e)
		self.eff_n_states = len(self.emissions_indexes)	
		assert self.eff_n_states == len(self.emissions), "emissions do not match groups"
		# real number of samples to simulate
		self.n_gibbs_sim = int(self.n_gibbs*(1+self.f_burn))
		self.pred_l = None
		self.p = 1
		self.P = None
		self.gibbs_P = None
		self.A = None
		self.gibbs_A = None
		self.w_norm = 1
		# **
		for emission in self.emissions:
			emission.set_gibbs_parameters(self.n_gibbs, self.f_burn, self.n_gibbs_sim)
		# **
		# Dirichelet prior parameters
		self.ALPHA = 1
		self.ZERO_ALPHA = 0.001
		self.ALPHA_P = 0.05 
		# A init
		self.INIT_MASS = 0.9

	def view(self, plot_hist = False):
		'''
		plot_hist: if true, plot histograms, otherwise just show the parameters
		'''
		print('** Gaussian HMM **')
		print('Groups')
		for e in self.emissions_indexes:
			print('States %s have the same emission'%','.join([str(a) for a in e]))
		print('Initial state probability')
		print(self.P)
		if plot_hist:
			for i in range(self.n_states):
				plt.hist(self.gibbs_P[:,i], density=True, alpha=0.5, label='P[%s]'%(i))
			plt.legend()
			plt.grid(True)
			plt.show()
		print('State transition matrix')
		print(np.round(self.A,3))
		print()
		if plot_hist:
			for i in range(self.n_states):
				for j in range(self.n_states):
					if [i,j] not in self.A_zeros:
						plt.hist(
								self.gibbs_A[:,i,j],
								density=True,
								alpha=0.5,
								label='A[%s->%s]'%(i,j)
								)
			plt.legend()
			plt.grid(True)
			plt.show()
		for emission in self.emissions:
			emission.view(plot_hist = plot_hist)
			 
	def dirichlet_priors(self):
		alphas = []
		for s in range(self.n_states):
			tmp = self.ALPHA*np.ones(self.n_states)
			for e in self.A_zeros:
				if e[0] == s:
					tmp[e[1]] = self.ZERO_ALPHA
			alphas.append(tmp)
		return alphas
	
	def estimate(self, y, x = None, idx = None, **kwargs):	 
		assert y.ndim==2,"y must be a matrix"
		if x is not None: 
			assert x.ndim==2,"x must be a matrix"
			assert y.shape[0] == x.shape[0], "x and y must have the same number of osbervations"		
		# **
		# number of observations
		n = y.shape[0]
		# idx for multisequence
		if idx is None:
			idx = np.array([[0,n]], dtype = int)		
		# convert to integer to make sure this is well defined
		idx = np.array(idx, dtype = int)
		# number of sequences
		n_seqs = idx.shape[0]
		# **
		
		# **
		# Dirichlet prior
		alphas = self.dirichlet_priors()
		# **
		
		# **
		# Containers
		# counter for state transitions
		transition_counter=np.zeros((self.n_states,self.n_states)) 
		# counter for initial state observations
		init_state_counter=np.zeros(self.n_states) 
		# forward alpha
		forward_alpha=np.zeros((n,self.n_states),dtype=np.float64)
		# forward normalization variable
		forward_c=np.zeros(n,dtype=np.float64)
		# transition matrix samples
		self.gibbs_A=np.zeros((self.n_gibbs_sim,self.n_states,self.n_states)) 
		# initial state probability samples
		self.gibbs_P=np.zeros((self.n_gibbs_sim,self.n_states))
		# ** 
		
		# **
		# Initialize
		# assume some persistency of state as a initial parameter		
		tmp = self.INIT_MASS*np.eye(self.n_states)
		remaining_mass = (1-self.INIT_MASS)/(self.n_states-1)
		tmp[tmp == 0] = remaining_mass	
		# set zeros
		for e in self.A_zeros:
			tmp[e[0],e[1]] = 0			
		tmp /= np.sum(tmp,axis=1)[:,None]
		self.gibbs_A[0] = tmp

		self.gibbs_P[0] = np.ones(self.n_states)
		self.gibbs_P[0] /= np.sum(self.gibbs_P[0])		
		# initialize emissions
		for emission in self.emissions:
			emission.gibbs_initialize(y = y, x = x)
		# **

		# **
		# create and initialize variable with
		# the states associated with each variable
		# assume equal probability in states
		q=np.random.choice(np.arange(self.n_states,dtype = int),size=n)

		# **
		prob=np.zeros((n,self.n_states), dtype=np.float64) 
		# **
		# Gibbs sampler
		for i in range(1,self.n_gibbs_sim):
			# **
			# set counters to zero
			transition_counter*=0 
			init_state_counter*=0 
			# **
			# evaluate the probability of each observation
			for s in range(self.eff_n_states):
				prob[:,self.emissions_indexes[s]] = self.emissions[s].gibbs_prob(
																				y = y, 
																				x = x, 
																				iteration = i
																				)[:,None]
				
			# **
			# sample form hidden state variable
			for l in range(n_seqs):
				# compute alpha variable
				forward_alpha,_=forward(
									prob[idx[l][0]:idx[l][1]],
									self.gibbs_A[i-1],
									self.gibbs_P[i-1]
									)
				# backward walk to sample from the state sequence
				backward_sample(
								self.gibbs_A[i-1],
								forward_alpha,
								q[idx[l][0]:idx[l][1]],
								transition_counter,
								init_state_counter
								)

			# **
			# sample from transition matrix
			for j in range(self.n_states):				
				self.gibbs_A[i,j] = np.random.dirichlet(alphas[j]+transition_counter[j])
			# make sure that the entries are zero!
			for e in self.A_zeros:
				self.gibbs_A[i,e[0],e[1]] = 0.
			self.gibbs_A[i] /= np.sum(self.gibbs_A[i],axis=1)[:,None]
			# **
			# sample from initial state distribution
			
			
			self.gibbs_P[i] = np.random.dirichlet(self.ALPHA_P + init_state_counter)   
			# perform the gibbs step with the state sequence sample q
			for j in range(self.eff_n_states):
				idx_states=np.where(np.in1d(q,self.emissions_indexes[j]))[0]
				self.emissions[j].gibbs_posterior_sample(
														y = y[idx_states],
														x = None if x is None else x[idx_states],
														iteration = i
														)
				
		# burn observations
		self.gibbs_A=self.gibbs_A[-self.n_gibbs:]
		self.gibbs_P=self.gibbs_P[-self.n_gibbs:]
		
		for emission in self.emissions:
			emission.gibbs_burn_and_mean()
		
		self.A=np.mean(self.gibbs_A,axis=0)
		self.P=np.mean(self.gibbs_P,axis=0)
		
			
		self.w_norm = 0
		for j in range(self.eff_n_states):
			self.w_norm = max(self.w_norm, self.emissions[j].w_norm)
		# compute the emissions parameters with the samples
		# self.emissions.gibbs_mean()
		# self.w_norm = self.emissions.compute_w_norm()

	def next_state_prob(self, y:np.ndarray, x:np.ndarray = None, l:int = None):
		'''
		computes a vector with the next state probability
		given a input sequence
		xyq: numpy (n,self.p) array with observation
		l: integer to filter recent data in y -> y=y[-l:]
		'''
		assert y.ndim == 2, "y must be a matrix"
		if x is not None: 
			assert x.ndim == 2, "x must be a matrix"
			assert y.shape[0] == x.shape[0], "x and y must have the same number of osbervations"   
		# just return the initial state probability 
		if y.shape[0]==0:
			return self.P
		assert y.shape[1]==self.p,"y must have the same number of variables as the training data"
		if l is not None:
			y = y[-l:]
			if x is not None: x = x[-l:]
		
		n = y.shape[0]
		prob=np.zeros((n,self.n_states), dtype=np.float64) 
		# evaluate the probability of each observation
		for s in range(self.eff_n_states):
			prob[:,self.emissions_indexes[s]] = self.emissions[s].prob(
																		y = y, 
																		x = x,																		 
																		)[:,None] 
		alpha, _ = forward(prob, self.A, self.P)
		next_state_prob = np.dot(self.A.T, alpha[-1])  
		return next_state_prob	

	def get_weight(
				self, 
				y:np.ndarray, 
				x:np.ndarray = None, 
				xq:np.ndarray = None, 
				normalize = True, 
				**kwargs
				):
		'''
		compute betting weight given an input sequence
		y: numpy (n,p) array with a sequence
			each point is a joint observations of the variables
		l: integer to filter recent data in y -> y=y[-l:]
		returns:
			w: numpy (p,) array with weights to allocate to each asset
			in y
		'''
		
		# next state probability
		next_state_prob = self.next_state_prob(y, x, self.pred_l)		
		
		# group next state prob
		tmp = np.zeros(self.eff_n_states)
		for i,e in enumerate(self.emissions_indexes):
			tmp[i] = np.sum(next_state_prob[e])
		next_state_prob = tmp
		
		# correct weighting of predicitve mixture in case
		# the hidden states have predictors
		for i in range(self.eff_n_states):
			 next_state_prob[i] *= self.emissions[i].predict_pi(xq = xq)
		next_state_prob /= np.sum(next_state_prob)
		
		# build the mixture
		mixture_mean = np.zeros(self.p)
		mixture_cov = np.zeros((self.p, self.p))
		for i in range(self.eff_n_states):
			mean, cov = self.emissions[i].predict_moments(xq = xq)
			mixture_mean += next_state_prob[i]*mean
			mixture_cov += next_state_prob[i]*cov
			mixture_cov += next_state_prob[i]*mean*mean[:,None]
		mixture_cov -= mixture_mean*mixture_mean[:,None]
		
		w = np.dot(np.linalg.inv(mixture_cov), mixture_mean)

		if normalize:
			w /= self.w_norm
		
		return w		



if __name__=='__main__':
	x=np.random.normal(0,1,100)
	y=np.random.normal(0,1,100)

	model=ConditionalGaussianMixture(n_states=2,n_gibbs=1000,f_burn=0.1,max_k=0.25)
	model.estimate(y=y,x=x)
	model.view()
	samples_y,samples_x=model.sample(1000)

	plt.hist(samples_x[:,0])
	plt.show()

	#plt.plot(x,y,'.')
	#plt.show()
	#print('ola')

