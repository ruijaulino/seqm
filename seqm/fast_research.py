import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm


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


def intraday_linear_models_search(px:pd.DataFrame, pct_fee:float = 0):
    '''
    px: pandas DataFrame with a single column with prices    
    '''
    px = px.copy(deep = True)
    px = px[[px.columns[0]]]
    px.columns = ['px']
    # add day and time columns
    px['day'] = px.index.date
    px['time'] = px.index.time
    # pivot the DataFrame
    day_px = px.pivot(index='day', columns='time', values='px')    
    # give column names
    day_px.columns = [str(e.hour*100+e.minute) for e in day_px.columns]
    
    # add data for the previous day
    prev_day_px = day_px.shift(1).copy(deep = True)
    prev_day_px.columns = ['prev_'+e for e in prev_day_px.columns]
    px_2_days = pd.concat((prev_day_px, day_px), axis = 1)
    
    # iterate all combinations
    cols_day = list(day_px.columns)
    cols_prev_day = list(prev_day_px.columns)
    cols = list(px_2_days.columns)

    sharpe = []
    col_feature = []
    col_start = []
    col_end = []


    for i in tqdm.tqdm(range(1, len(cols))):
        for j in range(i+1, min(i+int(len(cols)/2),len(cols))):
            for k in range(i):
                
                # feature
                f = px_2_days[cols[i]]/px_2_days[cols[k]]-1
                # target
                x = px_2_days[cols[j]]/px_2_days[cols[i]]-1
                
                # feature, target, df
                df = pd.concat((px_2_days[cols[i]]/px_2_days[cols[k]]-1, px_2_days[cols[j]]/px_2_days[cols[i]]-1), axis = 1)
                df.columns = ['f', 'x']
                df = df.dropna()
                df = df[df.abs().sum(axis = 1)!=0]
                # just a simple filter
                if len(df)>100:
                    try:
                        b, s, w = linear_model(df['f'].values, df['x'].values, calc_s = True)
                        s -= pct_fee*np.abs(w)
                        sharpe.append(np.mean(s)/np.std(s))
                        col_feature.append(cols[k])
                        col_start.append(cols[i])
                        col_end.append(cols[j])
                    except:
                        pass

    results = pd.DataFrame()
    results['sharpe'] = sharpe
    results['col_feature'] = col_feature
    results['col_start'] = col_start
    results['col_end'] = col_end
    results = results.dropna()
    results = results.sort_values('sharpe')
    return results


if __name__ == '__main__':

    n = 10000
    # just generate random data
    px = np.cumsum(np.random.normal(0, 1, n))
    px -= np.min(px)
    px += np.max(px)

    px = pd.DataFrame(px, columns = ['px'], index = pd.date_range(start = '2000-01-01', freq = '4H', periods = n))

    res = intraday_linear_models_search(px, pct_fee = 0)
    print(res)





