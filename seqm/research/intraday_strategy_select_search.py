import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time
import copy
# path to add utils file
import sys
from numba import jit
from joblib import Parallel, delayed
from tqdm import tqdm



def approx_nested_loocv_linear_model_select_old(X:np.ndarray, y:np.ndarray):
    '''
    Given many features and a target return y
    this function approximates the computation
    of estimation of out of sample sharpe
    There are p competitive models and we need
    to choose one. The estimated sharpe is deplected
    to account for feature selection
    X: n, p array with features
    y: n, vector with target returns    
    '''
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)
    n, p = X.shape   
    sxy = np.sum(X*y[:,None], axis = 0)
    sx = np.sum(X, axis = 0)
    sy = np.sum(y)
    sxx = np.sum(X*X, axis = 0)
    bi = ((n-1)*(sxy-X*y[:,None]) - (sx-X)*(sy-y)[:,None]) / ((n-1)*(sxx-X*X) - (sx-X)*(sx-X))
    ai = ((sy-y)[:,None] - bi*(sx-X))/(n-1) 
    oos_strategy = y[:,None]*(ai+bi*X)
    # compute best model index
    oos_sharpes = np.mean(oos_strategy, axis = 0) / np.std(oos_strategy, axis = 0)
    best_feature = np.argmax(oos_sharpes)
    
    # approximate nested cv
    # Compute the mean and std excluding the i-th row
    total_sum = np.sum(oos_strategy, axis=0)  # Total sum of all rows
    total_sq_sum = np.sum(oos_strategy**2, axis=0)  # Total squared sum for std calculation
    mean_excl = (total_sum - oos_strategy) / (n - 1)  # Mean excluding each row
    var_excl = (total_sq_sum - oos_strategy**2) / (n - 1) - mean_excl**2  # Variance excluding each row
    std_excl = np.sqrt(var_excl)  # Standard deviation
    # Calculate Sharpe ratios excluding each row
    sr_excl = mean_excl / std_excl
    # Find the index of the maximum Sharpe ratio for each row
    idx_max = np.argmax(sr_excl, axis=1)
    # Get sharpe at indexes
    rn = np.arange(n)
    nested_oos_strategy = oos_strategy[rn, idx_max]     
    nested_oos_sharpe = np.mean(nested_oos_strategy) / np.std(nested_oos_strategy)
    oos_sharpe = oos_sharpes[best_feature]
    return best_feature, nested_oos_sharpe, oos_sharpe

def approx_nested_loocv_linear_model_select(X:np.ndarray, y:np.ndarray, valid_points_frac = 0.25):
    '''
    Approximates out-of-sample Sharpe ratio while penalizing based on sample size,
    including during the nested LOOCV approximation. Handles fully missing features.

    X: n, p array with features (can contain NaNs)
    y: n, vector with target returns (can contain NaNs)
    '''
    
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)

    n, p = X.shape

    k = int(n*valid_points_frac)

    # Mask NaNs in X and y
    X_masked = np.ma.masked_invalid(X)
    y_masked = np.ma.masked_invalid(y)
    valid_mask = (~X_masked.mask) & (~y_masked.mask[:, None])

    # Replace NaNs with 0 for valid computations
    X_filled = X_masked.filled(0)
    y_filled = y_masked.filled(0)

    # Precompute terms for all features
    sxy = np.sum(X_filled * y_filled[:, None] * valid_mask, axis=0)
    sx = np.sum(X_filled * valid_mask, axis=0)
    sy = np.sum(y_filled[:, None] * valid_mask, axis=0)
    sxx = np.sum(X_filled**2 * valid_mask, axis=0)
    n_valid = np.sum(valid_mask, axis=0)

    # Identify features with at least k valid points
    valid_features = n_valid > k
    if not np.any(valid_features):
        return -1, -1, -1, -1

    # Filter out invalid features
    sxy = sxy[valid_features]
    sx = sx[valid_features]
    sy = sy[valid_features]
    sxx = sxx[valid_features]
    X_filled = X_filled[:, valid_features]
    valid_mask = valid_mask[:, valid_features]
    n_valid = n_valid[valid_features]

    # Compute slopes (b) and intercepts (a) using LOOCV formulas
    numerator_bi = (n_valid - 1) * (sxy - X_filled * y_filled[:, None]) - (sx - X_filled) * (sy - y_filled[:, None])
    denominator_bi = (n_valid - 1) * (sxx - X_filled**2) - (sx - X_filled)**2
    bi = numerator_bi / denominator_bi
    ai = (sy - y_filled[:, None] - bi * (sx - X_filled)) / (n_valid - 1)

    # Compute out-of-sample strategies
    oos_strategy = y_filled[:, None] * (ai + bi * X_filled) * valid_mask

    # Sharpe ratios for each feature
    mean_oos = np.ma.sum(oos_strategy, axis=0) / n_valid
    std_oos = np.sqrt(np.ma.sum(oos_strategy**2, axis=0) / n_valid - mean_oos**2)
    oos_sharpes = mean_oos / std_oos

    # Penalize Sharpe ratios based on sample size
    max_n_valid = np.max(n_valid)
    sharpe_penalized = oos_sharpes * np.sqrt(n_valid / max_n_valid)

    # Find the best feature among valid ones
    best_feature_index = np.argmax(sharpe_penalized)
    best_feature = np.flatnonzero(valid_features)[best_feature_index]  # Map back to original index

    # --- Nested LOOCV Sharpe with Penalization ---
    total_sum = np.sum(oos_strategy, axis=0)
    total_sq_sum = np.sum(oos_strategy**2, axis=0)

    # Compute leave-one-out mean and variance for each row and feature
    mean_excl = (total_sum - oos_strategy) / (n_valid - 1)
    var_excl = (total_sq_sum - oos_strategy**2) / (n_valid - 1) - mean_excl**2
    std_excl = np.sqrt(var_excl)
    sr_excl = mean_excl / std_excl  # Leave-one-out Sharpe ratios

    # Penalize Sharpe ratios in nested CV
    n_valid_excl = n_valid - 1  # Number of valid points excluding row i
    sr_excl_penalized = sr_excl * np.sqrt(n_valid_excl / max_n_valid)

    # Find the best feature for each row based on penalized Sharpe
    idx_max = np.argmax(sr_excl_penalized, axis=1)
    rn = np.arange(n)
    nested_oos_strategy = oos_strategy[rn, idx_max]

    # get out results without number of samples penalization
    # it is only used for model selection
    nested_oos_sharpe = np.mean(nested_oos_strategy) / np.std(nested_oos_strategy)
    oos_sharpe = oos_sharpes[best_feature_index]

    return best_feature, nested_oos_sharpe, oos_sharpe    




@jit(nopython=True)
def build_features(px_matrix:np.ndarray, target_start:int):
    n, p = px_matrix.shape
    px_matrix = np.ascontiguousarray(px_matrix)
    n_features = target_start * (target_start + 1) // 2  # Triangular number formula
    X = np.zeros((n, n_features))
    features_desc = np.zeros((n_features, 2), dtype=np.int32)
    
    # Precompute all start-end combinations (features_desc)
    c = 0
    for size in range(2, target_start + 2):  # Size of the window
        for start in range(target_start - size + 2):  # Starting index
            features_desc[c, 0] = start
            features_desc[c, 1] = start + size - 1
            c += 1
    
    # Compute all features in vectorized form
    for i in range(n_features):
        start = features_desc[i, 0]
        end = features_desc[i, 1]
        X[:, i] = px_matrix[:, end] / px_matrix[:, start] - 1
    
    return X, features_desc


def intraday_linear_model_select_search_base(
        px_matrix:np.ndarray, 
        target_start:int, 
        target_end:int, 
        valid_points_frac:float = 0.25
        ):
    try:
        if target_start < 2:
            return -1, -1, -1
        if target_end <= target_start:
            return -1, -1, -1, -1
        px_matrix = np.ascontiguousarray(px_matrix)
        n, p = px_matrix.shape
        y = px_matrix[:,target_end]/px_matrix[:,target_start] - 1
        X, features_desc = build_features(px_matrix, target_start)
        # check data for valid points
        feature, nested_sr, sr = approx_nested_loocv_linear_model_select(X, y, valid_points_frac)
        # get definition of best features
        feature_start, feature_end = features_desc[feature]
        return feature_start, feature_end, nested_sr, sr
    except:
        return -1, -1, -1, -1


def build_data(data:pd.DataFrame, add_prev_day:bool = True, valid_periods = None):
    assert len(data.columns) == 1, "data must have a single column with prices"
    data = data.copy(deep = True)
    data.columns = ['PX']
    # Add a column for date and time
    data['Date'] = data.index.date
    data['Time'] = data.index.strftime('%H:%M')
    # Pivot the dataframe to create a matrix where each row is a day, and columns are time intervals
    data.drop_duplicates(['Date','Time'], inplace = True)
    data = data.pivot(index='Date', columns='Time', values='PX')
    data.columns = [str(e) for e in data.columns] # make columns string
    if valid_periods:
        keep = []
        for c in data.columns:
            if c in valid_periods:
                keep.append(c)
        data = data[keep]
    if add_prev_day:
        data_shift = data.shift(1)
        data_shift.columns = [f'prev_{e}' for e in data_shift.columns]
        data = pd.concat((data_shift, data), axis = 1)
    return data

@jit(nopython=True)
def build_tasks(p, min_p = 0):
    n_tasks = p * (p + 1) // 2  # Triangular number formula
    tasks = np.zeros((n_tasks, 2), dtype=np.int32)    
    # Precompute all start-end combinations (features_desc)
    c = 0
    for size in range(2, p + 2):  # Size of the window
        for start in range(p - size + 2):  # Starting index
            tasks[c, 0] = start
            tasks[c, 1] = start + size - 1
            c += 1
    tasks = tasks[tasks[:,0] >= min_p]
    return tasks


def intraday_linear_model_select_search(
                        data:pd.DataFrame, 
                        valid_points_frac = 0.25, 
                        n_jobs = 20, 
                        valid_periods = None,
                        filename = None
                        ):
    start_time = time.time()
    data = pd.DataFrame(data)
    data = build_data(data, True, valid_periods)    
    # build tasks
    tasks = build_tasks(len(data.columns)-1, min_p = len(data.columns) // 2)
    # tasks = tasks[:50]
    cols = [str(e) for e in data.columns]
    px_matrix = data.values
    
    print('PX Matrix size: ', px_matrix.shape)
    
    out = Parallel(n_jobs = n_jobs)(
        delayed(intraday_linear_model_select_search_base)(
                        px_matrix, 
                        task[0], 
                        task[1], 
                        valid_points_frac
                    ) for task in tqdm(tasks, desc="Computing Sharpes")
        )
            
    #out = []
    #for task in tasks:        
    #    feature_start, feature_end, sr = intraday_linear_model_select_search_base(
    #        px_matrix, 
    #        task[0], 
    #        task[1], 
    #        nan_fraction_th
    #        )
    #    out.append([feature_start, feature_end, sr])

    # Compute threshold
    v = []
    for i in range(len(out)):
        v.append([
            cols[out[i][0]],
            cols[out[i][1]],
            cols[tasks[i][0]],
            cols[tasks[i][1]],
            out[i][2],
            out[i][3]
        ])        
    results = pd.DataFrame(v, columns = ['Feature Start', 
                                         'Feature End', 
                                         'Target Start', 
                                         'Target End', 
                                         'Nested Sharpe',
                                         'Sharpe'
                                        ])
    results = results.sort_values('Sharpe')
    # Save results
    if not filename:
        filename = f'strategy_search_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results.to_csv(filename, index=False)
    print(f"Execution Time: {(time.time() - start_time)/60:.2f} minutes")
    return results            



if __name__ == '__main__':

    data = pd.read_csv('dev_search_select\\NAS100_USD_M15.csv', index_col = 'TIMESTAMP')
    data.index = pd.to_datetime(data.index)
    data.index = data.index.tz_localize('utc')
    data.index = data.index.tz_convert('US/Eastern')
    data = data[['MID_OPEN']]


    print('Run with all..')
    intraday_linear_model_select_search(data, n_jobs = 1)


    data = build_data(data, add_prev_day = True)
    cols = list(data.columns)
    target_start = cols.index('15:45:00')
    target_end = target_start + 1
    print('Calc..')
    px_matrix = data.values
    feature_start, feature_end, nested_sr, sr = intraday_linear_model_select_search_base(px_matrix, target_start, target_end)
    print(f'Model: {cols[feature_start]} to {cols[feature_end]} ---> {cols[target_start]} to {cols[target_end]} | {nested_sr} {sr}')



    pass