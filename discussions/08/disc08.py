from glob import glob
import pandas as pd
import numpy as np

from scipy.stats import linregress


def rmse(datasets):
    """
    Return the RMSE of each of the datasets.

    >>> datasets = {k:pd.read_csv('data/dataset_%d.csv' % k) for k in range(7)}    
    >>> out = rmse(datasets)
    >>> len(out) == 7
    True
    >>> isinstance(out, pd.Series)
    True
    """

    arr, ind = [], []
    for k, v in datasets.items():
        lm = linregress(v.X, v.Y) # Fit model
        pred = lambda x: lm.slope * x + lm.intercept # Function
        pred_y = pred(v.X) # Predicted Y
        rmse = np.sqrt(np.mean((pred_y - v.Y)**2)) # RMSE
        arr.append(rmse)
        ind.append(k)
    return pd.Series(arr, index=ind)


def heteroskedasticity(datasets):
    """
    Return a boolean series giving whether a dataset is
    likely heteroskedastic.

    >>> datasets = {k:pd.read_csv('data/dataset_%d.csv' % k) for k in range(7)}    
    >>> out = heteroskedasticity(datasets)
    >>> len(out) == 7
    True
    >>> isinstance(out, pd.Series)
    True
    """
    arr, ind = [], []
    for k, v in datasets.items():    
        lm = linregress(v.X, v.Y) # Fit model
        pred = lambda x: lm.slope * x + lm.intercept # Function
        pred_y = pred(v.X) # Predicted Y
        res_sq = (pred_y - v.Y)**2
        lms = linregress(v.X, res_sq) # Residuals regression
        p_val = lms.pvalue

        if p_val < 0.05:
            arr.append(True)
        else: 
            arr.append(False)

        ind.append(k)
        # print(p_val)
    return pd.Series(arr, index=ind)
