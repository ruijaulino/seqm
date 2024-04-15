import pandas as pd
import numpy as np

try:
	from .generators import linear
except ImportError:
	from generators import linear

def add_unix_timestamp(df:pd.DataFrame):
	# create a column with unix timestamp
	# add dates are dealt as integers
	if 'ts' not in df.columns: df['ts'] = df.index.astype(np.int64) // 10 ** 9
	return df

def unix_timestamp_to_index(df:pd.DataFrame):
	if 'ts' in df.columns: df.index = pd.to_datetime( data['ts'] * 10 ** 9 )
	return df

if __name__=='__main__':
	
	n=10 
	data = linear(n=1000,a=0,b=0.1,start_date='2000-01-01')	
	data = add_unix_timestamp(data)
	data.reset_index(inplace=True,drop=True)
	print(data)

	data = unix_timestamp_to_index(data)
	print(data)
