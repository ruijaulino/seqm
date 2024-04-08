



# class with a model and a transformer
class ModelWrapper:
	
	def __init__(
				self,
				model = None, 
				x_transform:BaseTransform = None, 
				y_transform:BaseTransform = None,
				key: str = ''
				):
		self.model=copy.deepcopy(model)
		self.x_transform = copy.deepcopy(x_transform) if x_transform is not None else IdleTransform()  # Store the class, not an instance
		self.y_transform = copy.deepcopy(y_transform) if y_transform is not None else IdleTransform()  # Store the class, not an instance
		self.key = key or 'Dataset'

	# setters

	def set_model(self, model):
		self.model=copy.deepcopy(model)

	# fit transform
	def fit_x_transform(self,x_train):
		self.x_transform.fit(x_train)
		return self

	def fit_y_transform(self,y_train):
		self.y_transform.fit(y_train)
		return self

	def fit_transforms(self,x_train,y_train):
		return self.fit_x_transform(x_train).fit_y_transform(y_train)
	
	# apply transform
	def transform_x(self, x, copy_array = True):
		if copy_array: x = np.array(x)
		return self.x_transform.transform(x)
	
	def transform_y(self, y, copy_array = True):
		if copy_array: y = np.array(y)
		return self.y_transform.transform(y)

	# estimate model
	def estimate(self, x, y, **kwargs):
		"""Train the model using the training data contained within this DataElement."""
		self.model.estimate(x=x, y=y)
		return self

	# get weight
	def get_weight(self, xq, x, y, apply_transform_x = True, apply_transform_y = True):
		# process inputs
		if apply_transform_y: y = self.transform_y(y,True)
		if x is not None:
			if apply_transform_x: x = self.transform_x(x,True)
		if xq is not None:
			if apply_transform_x: xq = self.transform_x(xq,True)
		return self.model.get_weight(**{'y': y, 'x': x, 'xq': xq})





class Element:
	def __init__(
			self, 
			x_train: np.ndarray, 
			y_train: np.ndarray, 
			x_test: np.ndarray, 
			y_test: np.ndarray, 
			ts=None, 
			x_transform: BaseTransform = None, 
			y_transform: BaseTransform = None, 
			key: str = '', 
			x_cols = None, 
			y_cols = None
			):
		self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
		self.ts = pd.date_range('1950-01-01', freq='D', periods=self.x_test.shape[0]) if ts is None else ts
		self.key = key or 'Dataset'
		self.x_cols = x_cols or [f'x_{i+1}' for i in range(self.x_train.shape[1])]
		self.y_cols = y_cols or [f'y_{i+1}' for i in range(self.y_train.shape[1])]

		assert self.x_train.shape[0] == self.y_train.shape[0], "x_train and y_train must have the same number of observations"
		assert self.x_test.shape[1] == self.x_train.shape[1], "x_train and x_test must have the same number of variables"
		assert self.y_test.shape[1] == self.y_train.shape[1], "y_train and y_test must have the same number of variables"		

		self.x_transform = x_transform if x_transform is not None else IdleTransform()  # Store the class, not an instance
		self.y_transform = y_transform if y_transform is not None else IdleTransform()  # Store the class, not an instance

		self.model = None  # Placeholder for a trained model
		self.s,self.w,self.pw=None,None,None
		
		self._verify_input_data()
		self._fit_transform()

	def get_s(self):
		return pd.DataFrame(self.s,columns=[STRATEGY_COLUMN],index=self.ts)

	def get_w(self):
		return pd.DataFrame(self.w,columns=[WEIGHT_PREFIX_COLUMNS+c for c in self.y_cols],index=self.ts)

	def get_pw(self):
		return pd.DataFrame(self.pw,columns=[PORTFOLIO_WEIGHT_COLUMN],index=self.ts)

	def view(self):
		print('** Element **')
		print('- Train')
		print(pd.DataFrame(np.hstack((self.x_train,self.y_train)),columns=self.x_cols+self.y_cols))
		print('- Test')
		print(pd.DataFrame(np.hstack((self.x_test,self.y_test)),columns=self.x_cols+self.y_cols,index=self.ts))
		if self.s is not None:
			print('- Strategy')
			print(self.get_s())
			print('- Weight')
			print(self.get_pw())
			
	def _verify_input_data(self):
		if self.ts is None: self.ts=pd.date_range('1950-01-01',freq='D',periods=self.x_test.shape[0])
		assert self.x_train.shape[0]==self.y_train.shape[0],"x_train and y_train must have the same number of observations"
		assert self.x_test.shape[0]==self.y_test.shape[0],"x_test and y_test must have the same number of observations"
		assert self.x_test.shape[0]==self.ts.size,"x_train and ts must have the same number of observations"		
		assert self.x_train.shape[1]==self.x_test.shape[1],"x_train and x_test must have the same number of variables"
		assert self.y_train.shape[1]==self.y_test.shape[1],"y_train and y_test must have the same number of variables"
		if self.x_cols is None: self.x_cols=['x_%s'%(i+1) for i in range(self.x_train.shape[1])]
		if self.y_cols is None: self.y_cols=['y_%s'%(i+1) for i in range(self.y_train.shape[1])]

	def set_model(self,model):
		# force a model
		self.model=copy.deepcopy(model)

	def _fit_transform(self):
		"""Fit and transform the training data using dedicated normalizers."""
		self.x_transform = copy.deepcopy(self.x_transform)
		self.y_transform = copy.deepcopy(self.y_transform)

		self.x_transform.fit(self.x_train)
		self.x_train = self.x_transform.transform(self.x_train)
		self.y_transform.fit(self.y_train)
		self.y_train = self.y_transform.transform(self.y_train)
		# can transform now the test x
		self.x_test = self.x_transform.transform(self.x_test)

	def estimate(self, model, **kwargs):
		"""Train the model using the training data contained within this DataElement."""
		self.model=copy.deepcopy(model)  		
		self.model.estimate(x=self.x_train, y=self.y_train)

	def evaluate(self):
		"""Evaluate the model using the test data and return performance metrics."""
		if self.model is None:
			raise Exception("Model has not been estimated.")		
		n = self.y_test.shape[0]
		p = self.y_test.shape[1]
		self.s = np.zeros(n, dtype=np.float64)
		self.w = np.zeros((n, p), dtype=np.float64)
		self.pw = np.ones(n, dtype=np.float64)
		# self.pw = self.y_transform.p_scale*np.ones(n, dtype=np.float64)
		
		for i in range(n):
			# normalize y for input (make copy of y)
			# make sure this array does not get modified
			y_test_copy = np.array(self.y_test[:i])
			y_hat = self.y_transform.transform(y_test_copy)
			# the x input is already normalized!
			w = self.model.get_weight(**{'y': y_hat, 'x': self.x_test[:i], 'xq': self.x_test[i]})
			self.w[i] = w
			self.s[i] = np.dot(self.y_test[i], w)
			self.pw[i] = self.y_transform.pw(y_test_copy)


if __name__=='__main__':
	pass