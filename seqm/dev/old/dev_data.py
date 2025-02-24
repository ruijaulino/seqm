import numpy as np
import pandas as pd
from typing import List, Union, Dict
import copy

try:
    from .generators import *
    from .constants import *
    from .models import *
except ImportError:
    from generators import *
    from constants import *
    from models import *

# it is faster to work with arrays

class Array:
    def __init__(self, values: np.ndarray, cols, ts:np.ndarray = None, copy:bool = True):
        self.values = np.copy(values) if copy else values
        self.cols = cols
        self.ts = None
        if isinstance(self.cols, str): 
            self.cols = [self.cols]
        self.p = 1
        if self.values.ndim == 1: 
            self.n = self.values.size
        else:            
            self.n, self.p = self.values.shape
            if self.p == 1:
                self.values = self.values[:, 0]                
        assert len(self.cols) == self.p, "cols with wrong size"
        if ts:
            self.ts = np.copy(ts) if copy else ts
            assert self.ts.ndim == 1, "ts must be a vector or not defined"
            assert self.ts.size == self.n, "ts must have the same number of observations as values"

    @property
    def empty(self):
        return self.n == 0
    
    def __getitem__(self, idx):
        """ Access a slice of the matrix along axis 0 (rows) """
        if self.ts:
            return Array(self.values[idx], self.cols, self.ts[idx], True)             
        else:
            return Array(self.values[idx], self.cols, None, True) 
    
    def __setitem__(self, idx, value):
        
        """ Set values for specific rows in the matrix """
        self.values[idx] = value  # key is a slice for rows and value is the new data
    
    def __repr__(self):
        """ Show the matrix as a string """
        if self.ts:
            return pd.DataFrame(self.values, columns = self.cols, index = ts).__repr__()
        else:
            return pd.DataFrame(self.values, columns = self.cols).__repr__()

    def stack(self, array):
        # stronger condition than just comparing p
        assert self.cols == array.cols, "trying to stack arrays with different cols"
        if self.p == 1:
            self.values = np.hstack((self.values, array.values))
            self.n = self.values.size
        else:
            self.values = np.vstack((self.values, array.values))
            self.n = self.values.shape[0]

# wrapper for ms idx
class MSIDX(Array):
    def __init__(self, msidx:np.ndarray):
        super().__init__(np.array(msidx, dtype=int), 'msidx')
        # fix
        assert self.p == 1, "MSIDX must be a vector"
        msidx = np.zeros(self.n, dtype = int)
        aux = self.values[1:] - self.values[:-1]
        msidx[np.where(aux!=0)[0] + 1] = 1
        self.values = np.cumsum(msidx)

    def convert(self):
        '''
            msidx array like 0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,...
            with the indication where the subsequences are
            
            this method creates an array like
            [
            [0,3],
            [3,7],
            ...
            ]
            with the indexes where the subsequences start and end
        '''
        aux = self.values[1:] - self.values[:-1]
        aux = np.where(aux != 0)[0]
        aux += 1
        aux_left = np.hstack(([0],aux))
        aux_right = np.hstack((aux,[self.n]))
        out = np.hstack((aux_left[:,None], aux_right[:,None]))
        return out

    # override stack method
    def stack(self, array):
        self.values = np.hstack((self.values, array.values+self.values[-1]-array.values[0]+1))
        # set stacked msidx already fixed
        self.n = self.values.size

class TS(Array):
    def __init__(self, ts:np.ndarray):
        super().__init__(ts, 'ts')
    
    @classmethod
    def from_index(cls, index: pd.DatetimeIndex):
        # add unix timestamp
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("index must be a DatetimeIndex")        
        ts = df.index.view(np.int64) // 10 ** 9
        return cls(ts = df.index.view(np.int64) // 10 ** 9)

    def as_timestamp(self):
        return pd.to_datetime( self.values * 10 ** 9 )

    def __lt__(self, value):
        return self.values < value

    def __gt__(self, value):
        return self.values > value

    def __le__(self, value):
        return self.values <= value

    def __ge__(self, value):
        return self.values >= value



# integer variable
class State(Array):
    def __init__(self, z:np.ndarray, name:str):
        super().__init__(np.array(z, dtype=int), name)


# add others of necessary
    
TARGET_PREFIX = 'y' # target variable (returns)
FEATURE_PREFIX = 'x' # continuous feature
STATE_COL = 'z' # discrete state
NON_TRADEABLE_TARGET_COL = 't' # non tradeable target 
MULTISEQ_IDX_COL = 'msidx'
STRATEGY_COL = 'strategy'
PORTFOLIO_WEIGHT_COL = 'pw'
WEIGHT_PREFIX = 'weight'


# container for data based on arrays
class Data:
    def __init__(self, arrays:Dict[str, Array], safe:bool = True):
        # mapping of variables
        assert 'ts' in arrays.keys(), "arrays must have ts"
        assert 'y' in arrays.keys(), "arrays must have y"
        self.arrays = copy.deepcopy(arrays) if safe else arrays
        self.n = self.ts.n
        self._checks()

    def __getitem__(self, idx):
        """ Enables NumPy-like slicing and indexing """
        return Data({k:v[idx] for k, v in self.arrays.items()}, safe = True)

    def __repr__(self):
        return self.arrays.__repr__()

    def _checks(self):
        assert all(e.n == self.n for e in self.arrays.values()), "all arrays must have the same number of observations"

    @property
    def empty(self):
        return self.n == 0

    def copy(self):
        """ Returns a deep copy of the Array object """
        return Data(self.arrays, safe = True)
        
    @classmethod
    def from_df(cls, df: pd.DataFrame):
        
        cols = df.columns.tolist()

        arrays = {'ts': TS.from_index(df.index)}
        
        # get y
        y_cols = [c for c in cols if c.startswith(TARGET_PREFIX)]
        if y_cols:
            arrays.update({'y': Array(df[y_cols].values, y_cols)})
        else:
            raise Exception("no target columns")     
            
        # get x array       
        x_cols = [c for c in cols if c.startswith(FEATURE_PREFIX)]      
        if x_cols:
            arrays.update({'x': Array(df[x_cols].values, x_cols)})

        # get t array       
        t_cols = [c for c in cols if c.startswith(NON_TRADEABLE_TARGET_COL)]      
        if t_cols:
            arrays.update({'t': Array(df[t_cols].values, t_cols)})

        if STATE_COL in cols:
            arrays.update({'t': State(df[STATE_COL].values)})

        if MULTISEQ_IDX_COL in cols:
            arrays.update({'msidx': MSIDX(df[MULTISEQ_IDX_COL].values)})
        else:
            arrays.update({'msidx': MSIDX(np.zeros(len(df),dtype=int))})
        return cls(arrays)

    def __getattr__(self, name):
        # Get column index(es) based on prefix
        array = self.arrays.get(name, None)
        if array:
            return array
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Handle setting values for dynamic columns like 'y', 'x', etc.
        if isinstance(value, Array):
            self.arrays.update({k, copy.deepcopy(value)})
            self._checks()
        else:
            # regular working
            super().__setattr__(name, value)
        return None

    def after(self, ts:int, create_new = True):
        if self.empty: return self
        return self[self.ts > ts]

    def before(self, ts:int, create_new = True):
        if self.empty: return self
        return self[self.ts < ts]

    def between(self,ts_lower, ts_upper, create_new = True):
        if self.empty: return self
        return self[ (self.ts<=ts_upper) & (self.ts>=ts_lower)]

    @staticmethod
    def _random_subsequence(ar, burn_fraction, min_burn_points):
        a, b = np.random.randint(max(min_burn_points, 1), max(int(ar.size * burn_fraction), min_burn_points + 1), size=2)
        return ar[a:-b]

    def random_segment(self, burn_fraction, min_burn_points, create_new = False):
        '''
        '''
        if self.empty: return self
        idx = np.arange(self.n)
        idx = self._random_subsequence(idx, burn_fraction, min_burn_points)     
        return self[idx]

    def stack(self, data:'Data', allow_both_empty = False):
        # make sure both are not empty
        if self.empty and data.empty:
            if allow_both_empty:        
                return self
            else:
                raise Exception('both Data are empty. Cannot stack.')
        if self.empty and not data.empty:
            self.arrays = copy.deepcopy(data.arrays)
            self.n = data.n            
            return self

        if not self.empty and data.empty:
            return self

        # check if columns match
        assert self.arrays.keys() == data.arrays.keys(), "trying to merge with data with different fields"
        # stacking procedures already programmed in array
        for k, v in self.arrays.items():
            v.stack(getattr(data, k))
        return self     

    def converted_idx(self):
        '''
            msidx array like 0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,...
            with the indication where the subsequences are
            
            this method creates an array like
            [
            [0,3],
            [3,7],
            ...
            ]
            with the indexes where the subsequences start and end
        '''
        return self.msidx.convert()

    def as_dict(self):
        return self.arrays

if __name__ == '__main__':    
    df = linear(n=10,a=0,b=0.1,start_date='2000-01-01')
    data = Data.from_df(df)

    tmp = data.copy()

    print(data.ts)
    # print(data.cols.index('y'))
    ts = 947289600 # 947462400

    print(data.before(ts))

    print(data.after(947462400))
    print('------')
    df1 = linear(n=10,a=0,b=0.1,start_date='2000-01-01')
    data1 = Data.from_df(df1)
    df2 = linear(n=10,a=0,b=0.1,start_date='2000-01-01')
    data2 = Data.from_df(df2)
    data1.stack(data2)
    print(data1)        
