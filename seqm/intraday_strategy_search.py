# autoregressive type linear regression strategy search

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import datetime
import time
from numba import jit
from numba.typed import List
plt.style.use('dark_background')
    


def linear_model(x, y, calc_s:bool = False, use_qr:bool = True):
    assert x.ndim == 1, "x must be a vector"
    assert y.ndim == 1, "y must be a vector"
    assert x.size == y.size, "x and y must have the same length"
    n = x.size
    X = np.column_stack((np.ones(n), x))
    if calc_s:
        if use_qr:
            # Compute b using QR decomposition
            Q, R = np.linalg.qr(X)
            b = np.linalg.solve(R, Q.T @ y)
            h = np.sum(Q**2, axis=1)
            w = (X @ b - y*h)/(1-h)
            s = y*w
        else:
            tmp = np.linalg.pinv(X.T @ X) @ X.T
            b = tmp @ y
            h = np.diag(X @ tmp)
            w = (X @ b - y*h)/(1-h)
            s = y*w
    else:
        b = np.linalg.pinv(X.T @ X) @ X.T @ y
        s = None
        w = None
    return b, s, w




# Function to generate adjacent subsets (contiguous blocks of variables)
@jit(nopython = True)
def generate_adjacent_subsets(p, max_size = 1):
    # subsets = []
    subsets_start = []
    subsets_end = []
    max_size += 1  # This is fine now that max_size is initialized as an integer    
    max_size = min(max_size, p+1)
    # Generate subsets of each size where the variables are adjacent
    for size in range(2, max_size):
        for start in range(p - size + 1):
            subsets_start.append(start)
            subsets_end.append(start+size-1)
    return subsets_start, subsets_end

# function to generate tasks
# tasks are a list with two integers
# task = [2,5] means we are using subset 2 to predict subset 5
# subsets are generated with generate_adjacent_subsets
@jit(nopython=True)
def generate_tasks(subsets_start, subsets_end, min_idx_target=0):
    num_subsets = len(subsets_start)
    tasks = List()  # Initialize numba-typed list
    for i in range(num_subsets):
        for j in range(num_subsets):
            if subsets_start[j] >= subsets_end[i] and subsets_start[j] >= min_idx_target:
                tasks.append((i, j))
    return tasks

# Function to compute correlation for a pair of subsets
def compute_strategy(px_matrix, feature_start, feature_end, target_start, target_end, i, j, valid_fraction = 0.5):
    '''
    px_matrix: numpy (n,p) array with prices
        the idea is that they are prices over day periods and
        each row represents a single day
    '''
    min_points = 50
    n, p = px_matrix.shape
    x = px_matrix[:,feature_end]/px_matrix[:,feature_start] - 1
    y = px_matrix[:,target_end]/px_matrix[:,target_start] - 1
    idx = ~np.isnan(x) & ~np.isnan(y)
    x = x[idx]
    y = y[idx]
    # keep most points
    if x.size/n > valid_fraction:
        if x.size > min_points:        
            _, s, _ = linear_model(x, y, calc_s = True, use_qr = True)
            sr = np.mean(s)/np.std(s)
            if sr>0:
                return i, j, sr
            else:
                return None
        else:
            return None
    else:
        return None


def model_search_base(px_matrix, max_window:int = None, n_jobs:int = 1, min_idx_target:int = 0, valid_fraction:float = 0.5):
    p = px_matrix.shape[1]

    print('Generate subsets..')
    if max_window is None:
        max_window = p  # or some default integer value        
    subsets_start, subsets_end = generate_adjacent_subsets(p, max_size=max_window)

    # convert into numpy array
    subsets_start = np.array(subsets_start, dtype = int)
    subsets_end = np.array(subsets_end, dtype = int)
    print('Generate tasks..')
    tasks = generate_tasks(
                subsets_start, 
                subsets_end, 
                min_idx_target)

    # Parallel computation with progress bar
    results = Parallel(n_jobs = n_jobs)(
        delayed(compute_strategy)(
                    px_matrix,
                    subsets_start[t[0]],
                    subsets_end[t[0]],
                    subsets_start[t[1]],
                    subsets_end[t[1]],
                    t[0],
                    t[1],
                    valid_fraction
                    ) for t in tqdm(tasks, desc="Computing Sharpes")
    )
        
    n_results = len(tasks)
    features = np.empty(n_results, dtype=int)
    targets = np.empty(n_results, dtype=int)
    sharpe = np.empty(n_results, dtype=float)
    count = 0

    for idx, res in enumerate(results):
        if res:
            i, j, sr = res
            features[count] = i
            targets[count] = j
            sharpe[count] = sr
            count += 1

    # Truncate arrays to actual size
    features = features[:count]
    targets = targets[:count]
    sharpe = sharpe[:count]

    return subsets_start, subsets_end, features, targets, sharpe


def build_data(data:pd.DataFrame, add_prev_day:bool = True):
    assert len(data.columns) == 1, "data must have a single column with prices"
    data = data.copy(deep = True)
    data.columns = ['PX']
    # Add a column for date and time
    data['Date'] = data.index.date
    data['Time'] = data.index.time
    # Pivot the dataframe to create a matrix where each row is a day, and columns are time intervals
    data.drop_duplicates(['Date','Time'], inplace = True)
    data = data.pivot(index='Date', columns='Time', values='PX')
    data.columns = [str(e) for e in data.columns] # make columns string
    if add_prev_day:
        data_shift = data.shift(1)
        data_shift.columns = [f'prev_{e}' for e in data_shift.columns]
        data = pd.concat((data_shift, data), axis = 1)
    return data


def intraday_linear_models_search(data:pd.DataFrame, max_window:int = None, add_prev_day:bool = True, quantile:float = 0.95, n_jobs:int = 10, valid_fraction:float = 0.5, filename = None):
    start_time = time.time()

    data = build_data(data, add_prev_day = add_prev_day)
    min_idx_target = 0
    if add_prev_day: min_idx_target = len(data.columns) // 2 # since we added the previous day, constrain the search for models not to repeat calculations

    cols = data.columns

    px_matrix = data.values

    # search models
    subsets_start, subsets_end, features, targets, sharpe = model_search_base(
                                                                        px_matrix, 
                                                                        max_window = max_window, 
                                                                        n_jobs = n_jobs, 
                                                                        min_idx_target = min_idx_target,
                                                                        valid_fraction = valid_fraction
                                                                        )

    # Compute threshold
    if sharpe.size:
        sr_th = np.quantile(sharpe, quantile)
        print(f"Sharpe Threshold: {sr_th:.4f}")
        
        if np.where(sharpe>sr_th)[0].size<10000:
            sr_th = -1
        
        # Prepare results
        res = []
        for i in range(len(sharpe)):
            if sharpe[i] >= sr_th:
                res.append([
                    cols[subsets_start[features[i]]],
                    cols[subsets_end[features[i]]],
                    cols[subsets_start[targets[i]]],
                    cols[subsets_end[targets[i]]],
                    sharpe[i]
                ])
        res = pd.DataFrame(res, columns=['Feature Start', 'Feature End', 'Target Start', 'Target End', 'Sharpe'])
        res = res.sort_values('Sharpe')
        # Save results
        if filename is None:
            filename = f'strategy_search_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        res.to_csv(filename, index=False)
        print(f"Saved as {filename}")
    else:
        print('Did not found any viable strategy..')
        res = pd.DataFrame(columns=['Feature Start', 'Feature End', 'Target Start', 'Target End', 'Sharpe'])
    print(f"Execution Time: {(time.time() - start_time)/60:.2f} minutes")
    return res


def test_subsets():
    p = 5
    max_size = p
    subsets_start, subsets_end = generate_adjacent_subsets(p, max_size)
    for i in range(len(subsets_start)):
        print(f"Subset {i}: from index {subsets_start[i]} to {subsets_end[i]}")

if __name__ == '__main__':
     # Example call
    n = 100
    data = pd.DataFrame(10 + np.cumsum(np.random.normal(0,0.0001,n)), index = pd.date_range('2000-01-01', periods = n, freq = '15T'))
    print(data)
    out = intraday_linear_models_search(data, max_window=5, quantile=0.99, n_jobs=1, filename=None)
    print(out)


