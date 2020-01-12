
import numpy as np
import os

def data2array(filepath):
    """
    data2array takes in the filepath of a 
    data file like `restaurant.csv` in 
    data directory, and returns a 1d array
    of data.

    :Example:
    >>> fp = os.path.join('data', 'restaurant.csv')
    >>> arr = data2array(fp)
    >>> isinstance(arr, np.ndarray)
    True
    >>> arr.dtype == np.dtype('float64')
    True
    >>> arr.shape[0]
    100000
    """
    file = open(filepath) # Use open instead of pandas
    file.readline() # Get rid of 'Bill'
    return np.array([float(elem) for elem in file]) # Convert to array


def ends_in_9(arr):
    """
    ends_in_9 takes in an array of dollar amounts 
    and returns the proprtion of values that end 
    in 9 in the hundredths place.

    :Example:
    >>> arr = np.array([23.04, 45.00, 0.50, 0.09])
    >>> out = ends_in_9(arr)
    >>> 0 <= out <= 1
    True
    """
    
    arr_int = np.round(arr * 100).astype(int) # Round to int
    arr_mod = arr_int % 10 # Take remainder
    return np.sum(arr_mod == 9) / len(arr) # Get proportion
