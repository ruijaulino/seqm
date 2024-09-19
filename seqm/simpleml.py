import numpy as np
import matplotlib.pyplot as plt

def np_ffill(arr):
    assert arr.ndim == 1, "arr must be a vector"
    # fill with the value 0 the array of indexes
    # where we have nan
    idx = np.where(~np.isnan(arr), np.arange(arr.size), 0)
    # forward fill with the maximum value
    # this index means that the value is not
    # nan at that index
    np.maximum.accumulate(idx, out=idx)
    # evaluate array at that index
    out = arr[idx]
    return out

def simple_data_explore(x, y, k = None, view:bool = True):
    assert x.ndim == 1, "x must be a vector"
    assert y.ndim == 1, "y must be a vector"
    assert x.size == y.size, "x and y must have the same number of observations"
    n = x.size
    if not k: k = int(np.sqrt(n))
    buckets = np.linspace(np.min(x), np.max(x), k)
    y_est = np.zeros(k-1)
    scale_est = np.zeros(k-1)
    for i in range(k-1):
        tmp = y[np.logical_and(x>buckets[i], x<buckets[i+1])]
        if tmp.size:
            y_est[i] = np.mean(tmp)
            scale_est[i] = np.std(tmp)
        else:
            y_est[i] = np.nan
            scale_est[i] = np.nan
    # fix nans with forward fills
    y_est = np_ffill(np_ffill(y_est)[::-1])[::-1]
    scale_est = np_ffill(np_ffill(scale_est)[::-1])[::-1]
    x_est = (buckets[1:]+buckets[1:])/2
    if view:
        plt.plot(x_est, y_est, '.')
        plt.plot(x_est, y_est + scale_est, '.')
        plt.plot(x_est, y_est - scale_est, '.')
        plt.grid(True)
        plt.show()
    return x_est, y_est, scale_est



