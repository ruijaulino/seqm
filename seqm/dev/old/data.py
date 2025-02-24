import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import copy
import tqdm
try:
    from .generators import *
    from .constants import *
    from .model_pipe import ModelPipe
except ImportError:
    from generators import *
    from constants import *
    from model_pipe import ModelPipe
    
def add_unix_timestamp(df:pd.DataFrame):
    # create a column with unix timestamp
    # add dates are dealt as integers
    if 'ts' not in df.columns: df['ts'] = df.index.view(np.int64) // 10 ** 9
    return df

def unix_timestamp_to_index(df:pd.DataFrame):
    if 'ts' in df.columns: df.index = pd.to_datetime( data['ts'] * 10 ** 9 )
    return df

def get_df_cols(df):
    cols = df.columns.tolist()
    x_cols = [c for c in cols if c.startswith(FEATURE_PREFIX)]      
    y_cols = [c for c in cols if c.startswith(TARGET_PREFIX)]
    t_cols = [c for c in cols if c.startswith(NON_TRADEABLE_TARGET_COL)]
    return x_cols, t_cols, y_cols

def check_empty_list(l):
    if len(l) == 0:
        return None
    else:
        return l

# wrapper over a dict to allow keys to be accessed like a method
class DictWrapper:
    def __init__(self, data=None):
        self._data = data if data is not None else {}

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'DictWrapper' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __iter__(self):
        """Allows iteration over keys in the dictionary."""
        return iter(self._data)

    def keys(self):
        """Returns the keys of the dictionary."""
        return self._data.keys()

    def __repr__(self):
        return repr(self._data)
    
    def method(self):
        print('ola ', self.name)
        for k in self:
            print('ole: ', k)

# base container for an array
class Array:
    def __init__(self, arr:np.ndarray, cols):
        self.arr = np.copy(arr)
        self.cols = cols if isinstance(cols, list) else [cols]
        self.p = 1
        if self.arr.ndim == 2:
            self.n, self.p = self.arr.shape
        else:
            self.n = self.arr.size
        assert len(cols) == self.p, "arr and cols must have the same number of variables"
    
    @property
    def empty(self):
        return self.n == 0

    def at(self, idx:np.ndarray):
        # filter array on indexes
        assert idx.ndim == 1, "idx must be a vector"
        return Array(self.arr[idx], self.cols)


# time index, derived from Array
class TS(Array):
    def __init__(self, ts):
        super().__init__(ts, 'ts')  # Call the parent constructor

    def before(self):
        pass

    def after(self):
        pass

    def between(self):
        pass

# multiseq indexes, derived from Array
class MSIDX(Array):
    def __init__(self, msidx):
        super().__init__(msidx, 'msidx')  # Call the parent constructor

    def before(self):
        pass

    def after(self):
        pass

    def between(self):
        pass        




class Data:
    def __init__(self, data = None):
        self._data = data if data is not None else {}

    def __getattr__(self, key):
        try:
            return self._data.get(key, None)
        except KeyError:
            raise AttributeError(f"'DictWrapper' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            if isinstance(value, np.ndarray):
                # always make copies of arrays
                self._data[key] = np.copy(value)
            else:
                self._data[key] = value
        # run checks to make sure the data makes sense
        self._checks()

    def __iter__(self):
        """Allows iteration over keys in the dictionary."""
        return iter(self._data)

    def keys(self):
        """Returns the keys of the dictionary."""
        return self._data.keys()

    def values(self):
        """Return all values stored in the object."""
        return self._data.values()

    def __repr__(self):
        return repr(self._data)
    
    def _checks(self):
        print('in _checks')
        # ts needs to be defined
        assert self.ts, "ts needs to be defined"
        assert self.y, "y needs to be defined"                
        # check number of observations
        shapes = [
            v.size if v.ndim == 1 else v.shape[0]
            for v in self.values() if isinstance(v, np.ndarray)
        ]
        assert len(set(shapes)) == 1 if shapes else True, "arrays must have the same number of observations"
    
    @property
    def empty(self):
        return len(self._data) == 0 or all(v.shape[0] == 0 for v in self.values() if isinstance(v, np.ndarray))

    def view(self):
        print('** Data **')
        
        print(pd.DataFrame(self.data, columns = self.cols, index = self.get_ts_as_timestamp()))
        print('**********')
        print()

    @classmethod
    def from_df(cls, df: pd.DataFrame):    
        pass


    def method(self):
        print('ola ', self.name)
        for k in self:
            print('ole: ', k)




# Data structure
class Data:
    def __init__(self, arrays:Array):
        # add unix timestamp
        # self.empty = True if y is None else False        
        self.arrays = copy.deepcopy(arrays)
        assert self.arrays.ts.ndim == 1, "ts must be a vector"
        assert self.data.ndim == 2, "data must be a matrix"
        assert self.ts.size == self.data.shape[0], "ts and data must have the same number of observations"
        assert len(self.cols) == self.data.shape[1], "cols and data must have the same number of variables"                
        self.fix_msidx()

    @property
    def empty(self):
        return len(self.data) == 0 or all(value.shape[0] == 0 for value in self.data.values())
    
    def fix_msidx(self):
        # idx need to be an array like 000,1111,2222,33,444
        # but can be something like 111,000,11,00000,1111,22
        # i.e, the indexes of the sequence need to be in order for the
        # cvbt to work. Fix the array in case this is not verified
        # this is a bit of overhead but has to be this way
        msidx = np.zeros(self.data.get('msidx').size,dtype=int)
        aux = self.data.get('msidx')[1:] - self.data.get('msidx')[:-1]
        msidx[np.where(aux!=0)[0]+1] = 1
        self.data['msidx'] = np.cumsum(msidx)

    def converted_msidx(self):
        '''
        idx: numpy (n,) array like 0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,...
            with the indication where the subsequences are
            creates an array like
            [
            [0,3],
            [3,7],
            ...
            ]
            with the indexes where the subsequences start and end
        '''
        aux=self.idx[1:]-self.idx[:-1]
        aux=np.where(aux!=0)[0]
        aux+=1
        aux_left=np.hstack(([0],aux))
        aux_right=np.hstack((aux,[self.idx.size]))
        out=np.hstack((aux_left[:,None],aux_right[:,None]))
        return out

    def get_ts_as_timestamp(self):
        return pd.to_datetime( self.ts * 10 ** 9 )

    def view(self):
        print('** Data **')
        print(pd.DataFrame(self.data, columns = self.cols, index = self.get_ts_as_timestamp()))
        print('**********')
        print()

    @classmethod
    def from_df(cls, df: pd.DataFrame, safe_arrays = True):
        # parse the dataframe into the inputs
        df = df.copy(deep = True)
        # add unix timestamp
        df = add_unix_timestamp(df)        
        ts = df['ts'].values
        data = df.values
        cols = list(df.columns)
        return cls(ts = ts, data = data, cols = cols)

    def _from_idx(self, idx:np.ndarray, create_new = False):        
        ts_ = self.ts[idx]
        data_ = self.data[idx]                
        if create_new:
            return Data(ts = ts_, data = data_, cols = self.cols)
        else:
            # I think this is never used.. (I think we need to fix the msidx here as well)
            self.ts = ts_
            self.data = data
            return self     

    @staticmethod
    def _random_subsequence(ar, burn_fraction, min_burn_points):
        a, b = np.random.randint(max(min_burn_points, 1), max(int(ar.size * burn_fraction), min_burn_points + 1), size=2)
        return ar[a:-b]

    def random_segment(self, burn_fraction, min_burn_points, create_new = False):
        '''
        '''
        if self.empty: return self
        idx = np.arange(self.y.shape[0])
        idx = self._random_subsequence(idx, burn_fraction, min_burn_points)     
        return self._from_idx(idx, create_new)

    def after(self, ts:int, create_new = True):
        if self.empty: return self
        idx = np.where(self.ts > ts)[0]
        return self._from_idx(idx, create_new)

    def before(self, ts:int, create_new = True):
        if self.empty: return self
        idx = np.where(self.ts < ts)[0]
        return self._from_idx(idx, create_new)

    def between(self,ts_lower, ts_upper, create_new = True):
        if self.empty: return self
        idx = np.where( (self.ts<=ts_upper) & (self.ts>=ts_lower) )[0]
        return self._from_idx(idx, create_new)

    def _copy(self, data:'Data'):
        self.ts = data.ts
        self.data = data.data

    def stack(self, data:'Data', allow_both_empty:bool = False):
        # make sure both are not empty
        if self.empty and data.empty:
            if allow_both_empty:        
                return self
            else:
                raise Exception('both Data are empty. Cannot stack.')
        if self.empty and not data.empty:
            self._copy(data)
            return self
        if not self.empty and data.empty:
            return self
        # check if columns match
        assert self.cols == data.cols,"cols are different"
        
        # must fix msidx from data


        self.data = np.vstack((self.data, data.data))
        self.ts = np.hstack((self.ts, data.ts))
        self.idx = np.hstack((self.idx,data.idx+self.idx[-1]-data.idx[0]+1))
        return self     


    def build_train_inputs(self):
        '''
        build train inputs for models       
        '''
        return {'x' : self.x, 'y' : self.y, 'z' : self.z, 'oos_s':self.oos_s, 'idx':self.converted_msidx(), 't': self.t} 


if __name__ == '__main__':
    
    df = linear(n=10,a=0,b=0.1,start_date='2000-01-01')
    print(df)
    print(sdfsd)
    data = Data.from_df(df)
    
    ts = 947289600 # 947462400
    data.view()

    data.before(ts).view()

    tmp=data.after(947462400)# 
    print(tmp.y)
    print(tmp.empty)
    tmp.view()

    # data.view()
    # data.random_segment(0.1,3)
    # data.view()