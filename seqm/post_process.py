

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
	s_fees=s-pct_fee*dw
	return s_fees

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

class Paths:
	def __init__(self):
		self.path=[]

	def add(self,item:Path):
		if isinstance(item, Path):
			self.path.append(item)
		else:
			raise TypeError("Item must be an instance of Path")

	def __getitem__(self, index):
		return self.path[index]

	def __len__(self):
		return len(self.path)

	def __iter__(self):
		return iter(self.path)

	def post_process(self,seq_fees=False,pct_fee=0,sr_mult=1,n_boot=1000,name=None):
		'''
		visualize results for a given dataset in paths
		'''
		if len(self)==0:
			print('No results to process!')
			return
		# by default use the results for the first dataframe used as input
		# this will work ny default because in general there is only one
		if name is None:
			name=self[0].names[0]
		# get and joint results for name
		# after that, just use the existent function
		s=[]
		w=[]
		for path in self:
			if path.joined:
				s.append(path.s.get(name).values)
				w.append(path.w.get(name).values)
		if len(s)==0:
			print('No results to process!')
			return
		s=np.hstack(s)
		w=np.stack(w,axis=2)
		ts=self[0].s.get(name).index
		
		s_fees=calculate_fees(s,w,seq_fees,pct_fee)

		# make plots!
		paths_sr=sr_mult*np.mean(s,axis=0)/np.std(s,axis=0)
		idx_lowest_sr=np.argmin(paths_sr)

		b_samples=bootstrap_sharpe(s[:,idx_lowest_sr],n_boot=n_boot)
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
		print('Return fee=%s: '%pct_fee, 
			  np.power(sr_mult,2)*np.mean(s_fees))
		print('Standard deviation fee=%s: '%pct_fee, 
			  sr_mult*np.std(s_fees))
		print('Sharpe fee=%s: '%pct_fee, 
			  sr_mult*np.mean(s_fees)/np.std(s_fees))
		print()
		print('**')

		# bootstrap estimate of sharpe
		if len(self)!=1:
			plt.title('Distribution of paths SR [no fees]')
			plt.hist(paths_sr,density=True)
			plt.show()

			tmp=sr_mult*np.mean(s_fees,axis=0)/np.std(s_fees,axis=0)
			plt.title('Distribution of paths SR [fee=%s]'%pct_fee)
			plt.hist(tmp,density=True)
			plt.show()

		c=['r','y','m','b']
		aux=pd.DataFrame(np.cumsum(s,axis=0),index=ts)
		aux.plot(color='g',title='Equity curves no fees',legend=False)
		plt.grid(True)
		plt.show()

		ax=aux.plot(color='g',title='Equity curves w/ fees',legend=False)
		aux=pd.DataFrame(np.cumsum(s_fees,axis=0),index=ts)
		ax=aux.plot(ax=ax,color=c[0],legend=False)
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
