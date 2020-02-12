import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def first_round():
    """
    :return: list with two values
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    """
    return [0.093, 'NR']


def second_round():
    """
    :return: list with three values
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    >>> out[2] is "ND" or out[2] is "D"
    True
    """
    return [0.039, 'R', 'D']


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def verify_child(heights):
    """
    Returns a series of p-values assessing the missingness
    of child-height columns on father height.

    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    dic = {}
    children = heights.columns[heights.columns.str.contains('^child_')] # Get children columns
    for child in children: # Loop through child_X
        dic.update({child:ks_permutation(heights, child, 'father')})
    return dic


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.

    :Example:
    >>> set(missing_data_amounts()) <= set(range(1,6))
    True
    """

    return [2]

# Helper function to calculate ks-statistic
def ks_permutation(df, test, depend, N=100):
    """
    Given testing column and dependent column, caculate the p-value
    using ks-statistic for N permutation round.
    
    :param df: dataframe containing data
    :param test: testing column
    :param depend: dependent column
    :param N: N simulations
    """
    df_miss = df.assign(test_is_null=df[test].isnull()) # Convert to True/False
    
    gpA = df.loc[df_miss['test_is_null'], depend]
    gpB = df.loc[~df_miss['test_is_null'], depend]
    obs_ks, p_val = ks_2samp(gpA, gpB) # Get the ks observed value
    
    kslist = []
    for _ in range(N):

        # shuffle the dependent column
        shuffled_dep = (
            df_miss[depend]
            .sample(replace=False, frac=1)
            .reset_index(drop=True)
        )

        # 
        shuffled = (
            df_miss
            .assign(**{'Shuffled ' + depend: shuffled_dep})
        )

        ks, _ = ks_2samp(
            shuffled.loc[shuffled['test_is_null'], 'Shuffled ' + depend],
            shuffled.loc[~shuffled['test_is_null'], 'Shuffled ' + depend]
        )

        # add it to the list of results
        kslist.append(ks)
    
    # pd.Series(kslist).plot(kind='hist', density=True, alpha=0.8)
    # plt.scatter(obs_ks, 0, color='red', s=40);
    
    # return np.min([np.count_nonzero(kslist >= obs_ks) / len(kslist), np.count_nonzero(kslist <= obs_ks) / len(kslist)])
    return np.count_nonzero(kslist >= obs_ks) / len(kslist) #, np.count_nonzero(kslist <= obs_ks) / len(kslist)])


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a dataframe with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> df['child'] = df['child_50']
    >>> out = cond_single_imputation(df)
    >>> out.isnull().sum() == 0
    True
    >>> (df.child.std() - out.std()) > 0.5
    True
    """
    quartile_heights = new_heights.assign(quartile=pd.qcut(new_heights['father'], 4)) # Find the quartile for each height of father
    child_mean = quartile_heights.groupby('quartile')['child'].transform('mean') # A series of child mean heights
    imputed = new_heights['child'].fillna(pd.Series(child_mean))
    return imputed

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    """  
    binned = np.histogram(child.dropna(), bins=10) # Value counts (first line)
    prop = binned[0] / binned[0].sum() # proportion of bins
    bins = binned[1] # N + 1 bin boundaries
    # print(prop)
    indices = np.random.choice(a=10, p=prop, size=N)
    # print(np.sum(indices == 8) / 10000)
    samples = []
    for ind in indices: # Loop through samples
        lower, upper = bins[ind], bins[ind+1] # Upper and lower bounds of bin
        samples.append(np.random.uniform(low=lower, high=upper))
    return np.array(samples)


def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isnull().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    """
    child_dist = quantitative_distribution(child, len(child))
    imputed = child.fillna(pd.Series(child_dist)) # Fill NaN with distribution values
    return imputed


# ---------------------------------------------------------------------
# Question # X
# ---------------------------------------------------------------------

def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> list1, list2 = answers()
    >>> len(list1)
    4
    >>> len(list2)
    6
    """
    answer = (['https://soundcloud.com/', # 1. soundcloud, some disallow
               'https://cfmriweb.ucsd.edu/', # 1. wiki, some disallow
               'https://www.thesaurus.com/', # 1. thesaurus, some disallow
               'https://ucsd.sona-systems.com/', # 2. SONA, disallow completely
               'https://www.linkedin.com/', # 2. LinkedIn, disallow completely
               'https://facebook.com/'])  # 2. Facebook, disallow completely
    return answer

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['first_round', 'second_round'],
    'q02': ['verify_child', 'missing_data_amounts'],
    'q03': ['cond_single_imputation'],
    'q04': ['quantitative_distribution', 'impute_height_quant'],
    'q05': ['answers']
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
