
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 0
# ---------------------------------------------------------------------

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.

    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0: # Empty list
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1: # Consecutive differece is 1
            return True

    return False


# ---------------------------------------------------------------------
# Question # 1 
# ---------------------------------------------------------------------

def median(nums):
    """
    median takes a non-empty list of numbers,
    returning the median element of the list.
    If the list has even length, it should return
    the mean of the two elements in the middle.

    :param nums: a non-empty list of numbers.
    :returns: the median of the list.
    
    :Example:
    >>> median([6, 5, 4, 3, 2]) == 4
    True
    >>> median([50, 20, 15, 40]) == 30
    True
    >>> median([1, 2, 3, 4]) == 2.5
    True
    >>> median([0, -1, 1, 100]) == 0.5
    True
    """
    
    nums.sort()
    if len(nums) % 2 == 0: # Even length
        ind = len(nums) // 2
        return (nums[ind] + nums[ind - 1]) / 2 # Average
    else: # Odd length
        ind = len(nums) // 2
        return nums[ind] # List element


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance
    as integers is also i.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    elements as described above.

    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    >>> same_diff_ints([])
    False
    >>> same_diff_ints([9,4,1,1,-4,-2])
    True
    """
    
    if len(ints) == 0: # Empty list
        return False
    
    for i in range(len(ints) - 1):
        for j in range(i + 1, len(ints)): # Following elements
            if np.abs(i - j) == np.abs(ints[i] - ints[j]): # Diff == Apart
                # print(ints[i], ints[j])
                return True
    return False


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def prefixes(s):
    """
    prefixes returns a string of every 
    consecutive prefix of the input string.

    :param s: a string.
    :returns: a string of every consecutive prefix of s.

    :Example:
    >>> prefixes('Data!')
    'DDaDatDataData!'
    >>> prefixes('Marina')
    'MMaMarMariMarinMarina'
    >>> prefixes('aaron')
    'aaaaaraaroaaron'
    """
    
    if len(s) == 0: # Empty string
        return s
    
    prefixes = ''
    for i in range(1, len(s)+1): # End indices
        prefixes += s[:i] # Append prefixes
    return prefixes


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def evens_reversed(N):
    """
    evens_reversed returns a string containing 
    all even integers from  1  to  N  (inclusive)
    in reversed order, separated by spaces. 
    Each integer is zero padded.

    :param N: a non-negative integer.
    :returns: a string containing all even integers 
    from 1 to N reversed, formatted as decsribed above.

    :Example:
    >>> evens_reversed(7)
    '6 4 2'
    >>> evens_reversed(10)
    '10 08 06 04 02'
    >>> evens_reversed(0)
    ''
    >>> evens_reversed(1)
    ''
    """
    
    evens = []
    for num in range(N, 1, -1): # Reversed order
        if num % 2 == 0: # Even number
            evens.append(str(num).zfill(len(str(N))))
    return ' '.join(evens) # Joined by space


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of the line.

    :param fh: a file object to read from.
    :returns: a string of last characters from fh

    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """
    
    lasts = ''
    for line in fh: # Read lines as string
        line = line.rstrip() # Strip new lines
        if len(line) != 0: # Not empty lines
            lasts += line[-1]
    return lasts


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """

    return A * A


def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is divisble by 16.

    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.

    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 33]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """

    return A % 16 == 0


def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """
    
    return np.round(np.diff(A) / A[:-1], 2) # Round 2 decimal


def arr_4(A):
    """
    Create a function arr_4 that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: an integer of the total number of shares.

    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """

    leftover = np.cumsum(20 % A) >= A # Cumulative sum >= A
    return np.where(leftover)[0][0] # First True 


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def movie_stats(movies):
    """
    movies_stats returns a series as specified in the notebook.

    :param movies: a dataframe of summaries of
    movies per year as found in `movies_by_year.csv`
    :return: a series with index specified in the notebook.

    :Example:
    >>> movie_fp = os.path.join('data', 'movies_by_year.csv')
    >>> movies = pd.read_csv(movie_fp)
    >>> out = movie_stats(movies)
    >>> isinstance(out, pd.Series)
    True
    >>> 'num_years' in out.index
    True
    >>> isinstance(out.loc['second_lowest'], str)
    True
    """
    dic = {}
    
    try: # Total number of years
        num_years = len(movies['Year'])
        dic.update({'num_years': num_years})
    except:
        pass

    try: # Total number of movies
        tot_movies = np.sum(movies['Number of Movies'])
        dic.update({'tot_movies': tot_movies})
    except:
        pass

    try: # Earliest year with fewest movies
        yr_fewest_movies = np.min(movies[movies['Number of Movies'] == np.min(movies['Number of Movies'])]['Year'])
        dic.update({'yr_fewest_movies': yr_fewest_movies})
    except:
        pass

    try: # Average gross
        avg_gross = np.mean(movies['Total Gross'])
        dic.update({'avg_gross': avg_gross})
    except:
        pass

    try: # Year with highest gross per movie
        highest_per_movie = movies.loc[(movies['Total Gross'] / movies['Number of Movies']).idxmax()]['Year']
        dic.update({'highest_per_movie': highest_per_movie})
    except:
        pass

    try: # Name of top movie during the second-lowest total gross year
        second_lowest = movies[movies['Total Gross'] == movies['Total Gross'].nsmallest().iloc[1]]['#1 Movie'].iloc[0]
        dic.update({'second_lowest': second_lowest})
    except:
        pass

    try: # Average number of movies made the year after a Harry Potter movie was the top movie
        avg_after_harry = np.mean(movies[movies['Year'].isin(movies[movies['#1 Movie'].str.contains(pat='Harry Potter')]['Year'].values + 1)]['Number of Movies'])
        dic.update({'avg_after_harry': avg_after_harry})
    except:
        pass

    return pd.Series(dic)
    

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a 
    properly formatted dataframe (as described in 
    the question).

    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data, 
    as specificed in the question statement.

    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """

    df = pd.DataFrame(columns=['first', 'last', 'weight', 'height', 'geo'])
    with open(fp) as file:
        file.readline() # Get rid of title
        for i, line in enumerate(file):
            content = line.rstrip().split(',') # Strip new lines
            content = list(filter(None, content)) # Remove empty strings
            content[-2:] = [','.join(content[-2:])] # Combine geo elements
            content = list(map(lambda it: it.strip('\"'), content)) # Strip elements of ""
            content[2] = float(content[2]) # Cast weight as float
            content[3] = float(content[3]) # Cast height as float
            df = df.append(pd.Series(content, index=['first', 'last', 'weight', 'height', 'geo']), ignore_index=True)
    return df


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q00': ['consecutive_ints'],
    'q01': ['median'],
    'q02': ['same_diff_ints'],
    'q03': ['prefixes'],
    'q04': ['evens_reversed'],
    'q05': ['last_chars'],
    'q06': ['arr_%d' % d for d in range(1, 5)],
    'q07': ['movie_stats'],
    'q08': ['parse_malformed']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
