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

def simple_data_explore(x, y, k = None, view:bool = True, view_scale:bool = True, ffill:bool = False, by_points:bool = True):
    assert x.ndim == 1, "x must be a vector"
    assert y.ndim == 1, "y must be a vector"
    assert x.size == y.size, "x and y must have the same number of observations"
    n = x.size
    if not k: k = int(np.sqrt(n))
    if by_points:
        xs = np.sort(x)
        xss = np.array_split(xs, k)
        buckets = np.zeros(k)
        for i in range(k-1):
            buckets[i] = xss[i][0]
        buckets[k-1] = xss[k-1][-1]
    else:
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
    if ffill:
        y_est = np_ffill(np_ffill(y_est)[::-1])[::-1]
        scale_est = np_ffill(np_ffill(scale_est)[::-1])[::-1]
    idx = np.logical_and(~np.isnan(y_est), ~np.isnan(scale_est))
    x_est = (buckets[1:]+buckets[:-1])/2
    y_est = y_est[idx]
    x_est = x_est[idx]
    scale_est = scale_est[idx]

    if view:
        plt.title('Data Exploration')
        plt.plot(x_est, y_est, '.', label = 'mean')
        if view_scale:
            plt.plot(x_est, y_est + scale_est, '.', label = 'upper band')
            plt.plot(x_est, y_est - scale_est, '.', label = 'lower band')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True)
        plt.show()
    return x_est, y_est, scale_est


if __name__ == '__main__':
    n = 1000
    a = 0.5
    x = np.random.normal(0,1,n)
    y = a*x + np.random.normal(0,1,n)
    simple_data_explore(x, y)




