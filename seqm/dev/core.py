import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict
import copy
import tqdm


# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
# ABSTRACT CLASSES

# Model class
class Model(ABC):
    
    @abstractmethod
    def estimate(self, data:'Data', **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def get_weight(self, data:'Data', **kwargs):
        """Subclasses must implement this method"""
        pass

class PortfolioModel(ABC):
    
    @abstractmethod
    def estimate(self, dataset:'Dataset', model_pipe:Union['ModelPipeContainer','ModelPipe']):
        """Subclasses must implement this method"""
        cvbt_path()
        pass

# Transform class
class Transform(ABC):
    
    @abstractmethod
    def view(self):
        pass

    @abstractmethod
    def estimate(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass

    @abstractmethod
    def transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass
    
    @abstractmethod
    def inverse_transform(self, arr:np.ndarray, **kwargs):
        """Subclasses must implement this method"""
        pass


# Transforms class: a dict of transforms
class Transforms(dict):

    def add(self, var:str, transform:Transform):
        self[var] = transform

    def estimate(self, data:'Data'):
        for k, v in self.items():
            v.estimate(getattr(data, k))

    #def transform(self, data:'Data'):
    #    for k, v in self.items():
    #        v.estimate(getattr(data, k))
    # also implement inverse_transform


# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
# DATA CLASSES

# -----------------------
# Constants & Configurations
# -----------------------
TARGET_PREFIX = 'y'               # Target variable (returns)
FEATURE_PREFIX = 'x'              # Continuous feature
STATE_COL = 'z'                   # Discrete state
NON_TRADEABLE_TARGET_COL = 't'    # Non-tradeable target
MULTISEQ_IDX_COL = 'msidx'
STRATEGY_COL = 'strategy'
PORTFOLIO_WEIGHT_COL = 'pw'
WEIGHT_PREFIX = 'weight'


# -----------------------
# Array and Derived Classes
# Numpy array wrapper
# by default, does not make any copies!
# -----------------------
class Array:
    def __init__(self, values: np.ndarray, cols: Union[str, List[str]], ts: np.ndarray = None):
        # No extra copying is performed here – similar to NumPy’s behavior.
        self.values = values
        self.cols = [cols] if isinstance(cols, str) else cols
        self.ts = ts
        self._update_shape()

    def _update_shape(self):
        if self.values.ndim == 1:
            self.n = self.values.size
            self.p = 1
        else:
            self.n, self.p = self.values.shape
        if len(self.cols) != self.p:
            raise ValueError("Number of columns does not match the data shape.")
        if self.ts is not None:
            if self.ts.ndim != 1 or self.ts.size != self.n:
                raise ValueError("Timestamp array must be 1D and match the number of observations.")

    def set_values(self, values: np.ndarray):
        self.values = values
        self._update_shape()

    def __array__(self):
        return self.values

    def shape(self):
        return self.values.shape

    @property
    def empty(self):
        return self.n == 0

    def __getitem__(self, idx):
        ts_slice = self.ts[idx] if self.ts is not None else None
        # Return a view (using NumPy’s slicing, which is typically a view)
        return self.__class__(self.values[idx], self.cols, ts_slice)

    def __setitem__(self, idx, value):
        self.values[idx] = value

    def __repr__(self):
        index = self.ts if self.ts is not None else None
        return pd.DataFrame(self.values, columns=self.cols, index=index).__repr__()

    def stack(self, array: 'Array'):
        if self.cols != array.cols:
            raise ValueError("Cannot stack arrays with different columns.")
        # For multivariate arrays, stack vertically; for univariate, stack horizontally.
        if self.values.ndim == 1:            
            self.values = np.hstack((self.values, array.values))
            self.n = self.values.size
        else:
            self.values = np.vstack((self.values, array.values))
            self.n = self.values.shape[0]
        if self.ts is not None and array.ts is not None:
            self.ts = np.concatenate((self.ts, array.ts))

class MSIDX(Array):
    def __init__(self, msidx: np.ndarray, *args, **kwargs):
        super().__init__(np.array(msidx.ravel(), dtype=int), 'msidx')
        if self.p != 1:
            raise ValueError("MSIDX must be a 1D vector.")
        # Compute differences to mark subsequence boundaries.
        diff = np.diff(self.values, prepend=self.values[0])
        change = (diff != 0).astype(int)
        self.values = np.cumsum(change)
        self._compute_start()

    def _compute_start(self):
        mask = np.r_[True, self.values[1:] != self.values[:-1]]
        self.start = np.where(mask)[0]

    def _compute_idx_limits(self):
        change_idx = np.where(np.diff(self.values) != 0)[0] + 1
        start_indices = np.hstack(([0], change_idx))
        end_indices = np.hstack((change_idx, [self.n]))
        self.idx_limits = np.column_stack((start_indices, end_indices))

    def stack(self, array: 'MSIDX'):
        offset = self.values[-1] - array.values[0] + 1
        adjusted_values = array.values + offset
        self.values = np.hstack((self.values, adjusted_values))
        self.n = self.values.size
        self._compute_start()

class TS(Array):
    def __init__(self, ts: np.ndarray, *args, **kwargs):
        super().__init__(ts.ravel(), 'ts')

    @classmethod
    def from_index(cls, index: pd.DatetimeIndex):
        # Convert DatetimeIndex to Unix timestamps.
        ts = index.view(np.int64) // 10 ** 9
        return cls(ts)

    def as_datetime(self):
        return pd.to_datetime(self.values * 10 ** 9)

    def __lt__(self, value):
        return self.values < value

    def __gt__(self, value):
        return self.values > value

    def __le__(self, value):
        return self.values <= value

    def __ge__(self, value):
        return self.values >= value

class State(Array):
    def __init__(self, z: np.ndarray, name: str = STATE_COL):
        super().__init__(np.array(z.ravel(), dtype=int), name)

# -----------------------
# Data Container Class
# behaves like a dict of Arrays
# -----------------------
class Data:
    def __init__(self, arrays: Dict[str, Array]):
        self.arrays = arrays
        if 'ts' not in self.arrays or 'y' not in self.arrays:
            raise ValueError("Data must contain at least 'ts' and 'y' arrays.")
        self._checks()

    @property
    def n(self):
        return self.arrays['ts'].n

    @property
    def p(self):
        return self.arrays['y'].p

    def __getitem__(self, idx):
        # Support tuple indexing: if a tuple is provided, use its first element for row slicing.
        if isinstance(idx, tuple):
            idx = idx[0]
        return Data({k: v[idx] for k, v in self.arrays.items()})

    def __repr__(self):
        # create dataframe with the columns
        # this is not supposed to be used many times
        # so can be slower...
        cols = []
        values = []
        for k, v in self.arrays.items():
            cols += v.cols
            values.append(np.atleast_2d(v.values.T).T)
        values = np.hstack(values)
        print(self.ts)
        return pd.DataFrame(values, columns = cols, index = self.ts.as_datetime()).__repr__()

    def _checks(self):
        # Ensure all arrays have the same number of observations.
        n_obs = self.n
        for arr in self.arrays.values():
            if arr.n != n_obs:
                raise ValueError("All arrays must have the same number of observations.")
        # delete this later....
        # Create default strategy and weight arrays if missing.
        if STRATEGY_COL not in self.arrays:
            self.arrays[STRATEGY_COL] = Array(np.zeros(n_obs), STRATEGY_COL)
        if 'w' not in self.arrays:
            weight_cols = [f"{WEIGHT_PREFIX}_{col}" for col in self.arrays['y'].cols]
            self.arrays['w'] = Array(np.zeros((n_obs, self.p)), weight_cols)

    def copy(self):
        """
        Return a deep copy of the Data instance.
        Use this method when you need an independent copy of the data.
        """
        return Data(copy.deepcopy(self.arrays))

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        arrays = {'ts': TS.from_index(df.index)}
        # Target variable(s)
        y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
        if y_cols:
            arrays['y'] = Array(df[y_cols].values, y_cols)
        else:
            raise ValueError("No target columns found (prefix 'y').")
        # Feature variable(s)
        x_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
        if x_cols:
            arrays['x'] = Array(df[x_cols].values, x_cols)
        # Non-tradeable targets
        t_cols = [c for c in df.columns if c.startswith(NON_TRADEABLE_TARGET_COL)]
        if t_cols:
            arrays['t'] = Array(df[t_cols].values, t_cols)
        # State variable
        if STATE_COL in df.columns:
            arrays[STATE_COL] = State(df[STATE_COL].values, STATE_COL)
        # Multi-sequence index
        if MULTISEQ_IDX_COL in df.columns:
            arrays[MULTISEQ_IDX_COL] = MSIDX(df[MULTISEQ_IDX_COL].values)
        else:
            arrays[MULTISEQ_IDX_COL] = MSIDX(np.zeros(len(df), dtype=int))
        return cls(arrays)

    def __getattr__(self, name):
        if name in self.arrays:
            return self.arrays[name]
        raise AttributeError(f"'Data' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == 'arrays':
            super().__setattr__(name, value)
        elif isinstance(value, Array):
            self.arrays[name] = value
            self._checks()
        elif isinstance(value, np.ndarray):
            if name in self.arrays:
                self.arrays[name].set_values(value)
                self._checks()
            else:
                raise ValueError(f"Array {name} must be created first.")
        else:
            super().__setattr__(name, value)

    def after(self, ts: int):
        if self.n == 0:
            return self
        return self[self.ts > ts]

    def before(self, ts: int):
        if self.n == 0:
            return self
        return self[self.ts < ts]

    def between(self, ts_lower: int, ts_upper: int):
        if self.n == 0:
            return self
        condition = (self.ts.values >= ts_lower) & (self.ts.values <= ts_upper)
        return self[condition]

    @staticmethod
    def _random_subsequence(arr: np.ndarray, burn_fraction: float, min_burn_points: int):
        start = np.random.randint(max(min_burn_points, 1), max(int(arr.size * burn_fraction), min_burn_points + 1))
        end = np.random.randint(max(min_burn_points, 1), max(int(arr.size * burn_fraction), min_burn_points + 1))
        return arr[start:-end]

    def random_segment(self, burn_fraction: float, min_burn_points: int):
        if self.n == 0:
            return self
        idx = np.arange(self.n)
        idx_segment = self._random_subsequence(idx, burn_fraction, min_burn_points)
        return self[idx_segment]

    def stack(self, data: 'Data', allow_both_empty: bool = False):
        if self.n == 0 and data.n == 0:
            if allow_both_empty:
                return self
            raise ValueError("Both Data objects are empty. Cannot stack.")
        if self.n == 0:
            self.arrays = data.arrays
            return self
        if data.n == 0:
            return self
        if set(self.arrays.keys()) != set(data.arrays.keys()):
            raise ValueError("Data objects have different fields and cannot be stacked.")
        for key, arr in self.arrays.items():
            arr.stack(data.arrays[key])
        return self

    def as_dict(self):
        return {k: v.values for k, v in self.arrays.items()}

    def model_input(self, idx: int = None):
        """
        Return the most recent subsequence for model input.
        If idx is None, use the last observation.
        Note: This uses the new tuple-indexing feature so you can write:
              return self[start, idx+1]
        """
        if idx is None:
            idx = self.n - 1
        if not hasattr(self.msidx, 'start'):
            raise ValueError("msidx does not have computed start indices.")
        valid_starts = self.msidx.start[self.msidx.start <= idx]
        start = valid_starts[-1] if valid_starts.size > 0 else 0
        return self[start: idx + 1]


# dict of Data with properties
class Dataset(dict):
    
    def __init__(self):
        self.folds_ts = None
        # methods to behave like dict
    
    def add(self, key:str, item: Union[pd.DataFrame,Data]):
        if isinstance(item, pd.DataFrame):
            item = Data.from_df(item)
        else:
            if not isinstance(item, Data):            
                raise TypeError("Item must be an instance of pd.DataFrame or Data")
        self[key] = item

    def split_ts(self, k_folds = 3):
        # join all ts arrays, compute unique values
        # and split array
        ts = []
        for k, data in self.items():
            ts.append(data.ts.values)
        ts = np.hstack(ts)
        ts = np.unique(ts)
        idx_folds = np.array_split(ts, k_folds)
        self.folds_ts = [(fold[0], fold[-1]) for fold in idx_folds]
        return self

    def split(
            self, 
            test_fold_idx: int, 
            burn_fraction: float = 0.1, 
            min_burn_points: int = 1, 
            seq_path: bool = False
            ):
        
        if seq_path and test_fold_idx == 0:
            raise ValueError("Cannot start at fold 0 when path is sequential")
        if len(self.folds_ts) is None:
            raise ValueError("Need to split before getting the split")
        
        train_dataset = Dataset()
        test_dataset = Dataset()

        for key, data in self.items():

            ts_lower, ts_upper = self.folds_ts[test_fold_idx]            
            
            # create training data
            train_data = data.before(ts = ts_lower)
            train_data.random_segment(burn_fraction, min_burn_points)            
            # if path is non sequential add data after the test set
            if not seq_path:
                train_data_add = data.after(ts = ts_upper)
                train_data_add.random_segment(burn_fraction, min_burn_points)
                train_data.stack(train_data_add, allow_both_empty = True)
            
            # get test data
            test_data = data.between(ts_lower, ts_upper)
            
            train_dataset.add(key, train_data)
            test_dataset.add(key, test_data)    

        return train_dataset, test_dataset


# ----------------------------------------------
# MODEL PIPELINE CLASSES

# ModelPipeUnit
# Handles a Model and Transforms applied to a Data object
class ModelPipeUnit:
    def __init__(self, model:Model = None, transforms:Transforms = None):
        self.model = model
        self.transforms = transforms
    
    def estimate(self, data:Data):
        # apply transforms
        # estimate model
        pass
    
    def evaluate(self, data:Data):
        # transform dataset
        # evaluate model
        # change dataset in place
        return dataset

# List of ModelPipeUnits
# Objective here is to train a stack of model and then average predictions
# Works on top of Data
class ModelPipeStack(list):
    
    def add(self, model:Model = None, transforms:Transforms = None):
        self.append(ModelPipeUnit(model, transforms))
    
    def estimate(self, data:Data):
        # pipes cannot be empty!
        for unit in self: unit.estimate(data)
    
    def evaluate(self, data:Data):
        res = []
        for pipe in self:
            tmp = pipe.evaluate(dataset.copy())
            res.append(tmp)
        # copied datasets were changed in place
        # take mean of weights to compute performance
        # dataset.w = mean of w in res
        return dataset


# Dict of ModelPipeStack
# Objective here is to handle for several data in a dataset where
# each one has a ModelPipeStack associated

class ModelPipeContainer(dict):    
    
    def add(self, key:str, model:Model, transforms:Transforms = None):
        if key not in self:
            self[key] = ModelPipeStack()
        self[key].add(model, transforms)
    
    def estimate(self, dataset:Dataset):
        for k, data in dataset:
            self[k].estimate(data)
    
    def evaluate(self, dataset:Dataset):
        for k, data in dataset.items():
            self[k].evaluate(data)   
        return dataset

# Full Model Pipe for several datasets
# also includes a portfolio model that decides how allocation
# works along datasets    

class ModelPipe():
    def __init__(self, portfolio_model:PortfolioModel = None):
        self.portfolio_model = portfolio_model
        self.model_pipe_container = ModelPipeContainer()
    
    def add(self, key, model, transforms = None):
        self.model_pipe_container.add(key, model, transforms)
    
    def estimate(self, datasets):        
        # dataset_dict is a dict of dataset        
        if self.portfolio_model:
            self.portfolio_model.estimate(datasets, self.model_pipe_container)
        self.model_pipe_container.estimate(datasets)

    def evaluate(self, dataset):
        # dataset_dict is a dict of dataset
        for k, dataset in dataset_dict.items():
            self.model_pipes[k].evaluate(dataset)
        # correct predictions/weights with portfolio model!        
        return dataset
    

# changes dataset in place
def cvbt_path(
            dataset:Dataset, 
            model_pipe:Union[ModelPipeContainer,ModelPipe],

            ):
    

    return dataset



def linear(n=1000,a=0,b=0.1,start_date='2000-01-01'):
    x=np.random.normal(0,0.01,n)
    y=a+b*x+np.random.normal(0,0.01,n)
    dates=pd.date_range(start_date,periods=n,freq='D')
    data=pd.DataFrame(np.hstack((y[:,None],x[:,None])),columns=['y1','x1'],index=dates)
    return data


def test_data():
    df = linear(n=1000,a=0,b=0.1,start_date='2000-01-01')
    data = Data.from_df(df)
    df1 = linear(n=10,a=0,b=0.1,start_date='2000-01-01')
    data1 = Data.from_df(df1)
    df2 = linear(n=10,a=0,b=0.1,start_date='2000-01-01')
    data2 = Data.from_df(df2)
    data1.stack(data2)
    print(data1)        
    print(data1.as_dict())
    print('----------')
    print(data1.model_input(2))


def test_dataset():
    df1 = linear(n=10,a=0,b=0.1,start_date='2000-01-01')
    df2 = linear(n=10,a=0,b=0.1,start_date='2000-01-01')

    dataset = Dataset()
    dataset.add('a', df1)
    dataset.add('b', df2)
    dataset.split_ts(2)
    train, test = dataset.split(0)
    print(dataset)
    print('=======================')
    print(train)
    print('=======================')
    print(test)

if __name__ == '__main__':
    test_dataset()