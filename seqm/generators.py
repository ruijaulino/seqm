import numpy as np
import pandas as pd


def linear(n=1000,a=0,b=0.1,start_date='2000-01-01'):
	x=np.random.normal(0,0.01,n)
	y=a+b*x+np.random.normal(0,0.01,n)
	dates=pd.date_range(start_date,periods=n,freq='D')
	data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
	return data


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



if __name__=='__main__':
	pass