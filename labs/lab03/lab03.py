
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1,11))
    True
    """
    return [3, 6, 7]


def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1,11))
    True
    """
    return [1, 4, 8]


def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1,5))
    True
    """
    return [2, 4]


def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in [1,2,3,4,5]
    True
    """
    return 5


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def clean_apps(df):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> len(cleaned) == len(df)
    True
    >>> cleaned.Reviews.dtype == int
    True
    '''
    
    cleaned = df.copy() # Deep copy of dataframe
    cleaned['Reviews'] = cleaned['Reviews'].astype(int) # Cast Reviews to int
    cleaned['Size'] = cleaned['Size'].apply(size_to_kilo).astype(float) # Convert size to kilobyte
    cleaned['Installs'] = cleaned['Installs'].str.replace(',', '').str.replace('+', '').astype(int) # Strip , +
    cleaned['Type'] = cleaned['Type'].apply(lambda tp: 1 if tp == 'Free' else 0).astype(int) # Binary format of Type
    cleaned['Price'] = cleaned['Price'].str.replace('$', '').astype(float) # Strip $ and convert to float
    cleaned['Last Updated'] = cleaned['Last Updated'].str[-4:].astype(int) # Strip everthing but the year
    return cleaned


def store_info(cleaned):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> info = store_info(cleaned)
    >>> len(info)
    4
    >>> info[2] in cleaned.Category.unique()
    True
    '''
    
    # Q2.1
    df_install = cleaned.groupby('Last Updated').aggregate({'App': 'count', 'Installs': 'median'}) # Aggregate count over App, median over Installs
    year = df_install[df_install['App'] >= 100]['Installs'].idxmax() # Year with App >= 100, Installs median max

    # Q2.2
    df_rating = cleaned.groupby('Content Rating')['Rating'].min() # Group by Content Rating to find min Rating
    cont_rate = df_rating.idxmax() # Content Rating with max min ratings

    # Q2.3 # ASK!!!!!!!!!!!!!!!!!!!!! include free/$0 app?
    df_categh = cleaned.groupby('Category')['Price'].mean() # Group by Category to find average Price
    # df_categ = cleaned[cleaned['Type'] == 0].groupby('Category')['Price'].mean() # Not include Free App
    categ_h = df_categh.idxmax()

    # Q2.4
    df_categl = cleaned[cleaned['Reviews'] >= 1000].groupby('Category')['Rating'].mean()
    categ_l = df_categl.idxmin()

    return [year, cont_rate, categ_h, categ_l]

# Helper function to convert mega to kilo
def size_to_kilo(size):
    """
    Convert Megabyte and Kilobyte to Kilobytes,
    and strip string off M or K.
    
    :param size: string to convert
    :return: float, the converted number
    """
    
    if size == np.nan: # No size info
        return 0
    
    prev, last = size[:-1], size[-1]
    if last == 'M': # Megabyte
        return float(prev) * 1000
    else: # Kilobyte
        return float(prev)


# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def std_reviews_by_app_cat(cleaned):
    """
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> play = pd.read_csv(fp)
    >>> clean_play = clean_apps(play)
    >>> out = std_reviews_by_app_cat(play)
    >>> set(out.columns) == set(['Category', 'Reviews'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """
    
    review_by_categ = cleaned[['Category', 'Reviews']].copy() # Deep copy of cleaned 'Category' & 'Reviews'
    review_by_categ['Reviews'] = review_by_categ.groupby('Category')['Reviews'].transform(standard_units) # Transform into standard units
    return review_by_categ


def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0].lower() in ['medical', 'family', 'equal']
    True
    >>> out[1] in ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',\
       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',\
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',\
       'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',\
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',\
       'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',\
       'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',\
       'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']
    True
    """
    return ['equal', 'GAME']

# Helper function to calculate standard units
def standard_units(nums):
    """
    Convert any array/Series of numbers to standard units.
    
    :param nums: an array of number
    :return: standardized array/Series of nums
    """
    
    return (nums - np.mean(nums))/np.std(nums)


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def read_survey(dirname):
    """
    read_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    
    surveys = pd.DataFrame(columns=['first name', 'last name', 'current company', 'job title', 'email', 'university'])
    for fp in np.array(os.listdir(dirname))[pd.Series(os.listdir(dirname)).str.contains('^survey[0-9]+.csv$')]: # Read through file names
        df = pd.read_csv(os.path.join(dirname, fp)) # Read from directory path
        df.columns = df.columns.str.lower().str.replace('_', ' ') # Standardize column names
        surveys = pd.concat([surveys, df], ignore_index=True, sort=False) # Append new df to surveys
    return surveys


def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a list containing the most common first name, job held, 
    university attended, and current company
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> all([isinstance(x, str) for x in out])
    True
    """
    
    df_c = df.copy().fillna('') # Deep copy, fill NaN with empty string
    name = df_c[~(df_c['first name'] == '')]['first name'].value_counts() # Johannah
    name_ind = name[name == name.max()].sort_index(ascending=False).idxmax()

    job = df_c[~(df_c['job title'] == '')]['job title'].value_counts() # Chemical Engineer
    job_ind = job[job == job.max()].sort_index(ascending=False).idxmax()

    univer = df_c[~(df_c['university'] == '')]['university'].value_counts() # Southwest University
    univer_ind = univer[univer == univer.max()].sort_index(ascending=False).idxmax()

    comp = df_c[(df_c['email'].str.contains('.com$')) & ~(df_c['current company'] == '')]['current company'].value_counts() # Tillman LLC
    comp_ind = comp[comp == comp.max()].sort_index(ascending=False).idxmax()
    
    return [name_ind, job_ind, univer_ind, comp_ind]


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def combine_surveys(dirname):
    """
    combine_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 0 - 1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = combine_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> combine_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """

    dataframes = [] # List to hold all dataframes
    for fp in np.array(os.listdir(dirname))[pd.Series(os.listdir(dirname)).str.contains('^favorite[0-9]+.csv$')]: # Read through file names
        df = pd.read_csv(os.path.join(dirname, fp)) # Read from directory path
        dataframes.append(df.set_index('id')) # Append dataframe
    favorites = pd.concat(dataframes, axis=1, sort=False) # Concatenate csv
    return favorites


def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 0-1000)

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = combine_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    """

    df_c =  df.copy().replace('\(no ', np.nan, regex=True) # Deep copy & Data Cleaning
    cols = np.array(df_c.columns)[np.array(df_c.columns) != 'name'] # Array of cols without 'name'
    
    ind_prop = df_c[cols].apply(prop_complete, axis=1) # Individual EC, apply prop to rows
    ind_extra = (ind_prop >= 0.75).replace(True, 5).rename('extra credit') # Individual EC Series
    
    class_prop = df_c[cols].apply(prop_complete, axis=0) # Class EC, apply prop to col
    class_extra = np.any(class_prop > 0.90) # Any question 90% completion
    if class_extra: # If 90% completion for any question, class extra credit
        ind_extra = ind_extra + 1
    
    df_extra = pd.concat([df['name'], ind_extra], axis=1)
    return df_extra

# Helper function to check proportion of completion
def prop_complete(lst):
    """
    Given a row/col, check prop of not empty string.
    
    :param lst: col/row to check
    :return: prop of completion
    """
    return np.count_nonzero(~lst.isna()) / len(lst)


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------


def at_least_once(pets, procedure_history):
    """
    How many pets have procedure performed at this clinic at least once.

    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = at_least_once(pets, procedure_history)
    >>> out < len(pets)
    True
    """
    
    proc_hist = procedure_history.groupby('PetID', as_index=False).count()[['PetID', 'Date']].rename(columns={'Date': 'Count'}) # Pet procedure counts
    pet_proc = pd.merge(pets, proc_hist, how = "left", on='PetID') # Pet procedure dataframe
    return np.count_nonzero(~pet_proc['Count'].isna())


def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    
    pets_owners = pd.merge(pets, owners.rename(columns={'Name':'First Name'}), on='OwnerID')
    owned = pets_owners.groupby(['OwnerID', 'First Name']).aggregate({'Name':concat_pets}).reset_index('OwnerID', drop=True)#.loc['Lee']
    return owned['Name']


def total_cost_per_owner(owners, pets, procedure_history, procedure_detail):
    """
    total cost per owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')

    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_owner(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['OwnerID'])
    True
    """
    
    proc_full = pd.merge(procedure_detail, procedure_history, on=['ProcedureType', 'ProcedureSubCode']) # Procedure costs
    pet_proc_full = pd.merge(pets, proc_full, how = "left", on='PetID') # Pets procedures
    pet_owner_all = pd.merge(pet_proc_full, owners.rename(columns={'Name':'First Name'}), on='OwnerID') # Every information
    return pet_owner_all.groupby('OwnerID')['Price'].sum()

# Helper function to concatenate pet names
def concat_pets(strs):
    """
    Concatenate pet names.
    
    :param strs: strings to parse in
    :return: string if one pet name, list if more
    """
    
    if len(strs) ==  1: # If only one string
        return np.sum(strs)
    else: # If more
        return list(strs)


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!


GRADED_FUNCTIONS = {
    'q01': [
        'car_null_hypoth', 'car_alt_hypoth',
        'car_test_stat', 'car_p_value'
    ],
    'q02': ['clean_apps', 'store_info'],
    'q03': ['std_reviews_by_app_cat','su_and_spread'],
    'q04': ['read_survey', 'com_stats'],
    'q05': ['combine_surveys', 'check_credit'],
    'q06': ['at_least_once', 'pet_name_by_owner', 'total_cost_per_owner']
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
