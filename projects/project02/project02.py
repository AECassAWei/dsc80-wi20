import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_san(infp, outfp):
    """
    get_san takes in a filepath containing all flights and a filepath where
    filtered dataset #1 is written (that is, all flights arriving or departing
    from San Diego International Airport in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'santest.tmp')
    >>> get_san(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (53, 31)
    >>> os.remove(outfp)
    """
    header = False # Does not have headers
    with open(outfp, "w") as file:
        iters = pd.read_csv(infp, chunksize=1000, dtype={'ORIGIN_AIRPORT':str, 'DESTINATION_AIRPORT':str})
        for df in iters:
            san_2015 = df[(df['YEAR'] == 2015) & ((df['ORIGIN_AIRPORT'] == 'SAN') | (df['DESTINATION_AIRPORT'] == 'SAN'))] # Year=2015, DEP/ARR=SAN
            if header: # Have headers already
                san_2015.to_csv(file, mode='a', index=False, header=False)
            else: # Does not have headers
                san_2015.to_csv(file, mode='a', index=False)
                header = True
    return None


def get_sw_jb(infp, outfp):
    """
    get_sw_jb takes in a filepath containing all flights and a filepath where
    filtered dataset #2 is written (that is, all flights flown by either
    JetBlue or Southwest Airline in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'jbswtest.tmp')
    >>> get_sw_jb(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (73, 31)
    >>> os.remove(outfp)
    """
    header = False # Does not have headers
    with open(outfp, "w") as file:
        iters = pd.read_csv(infp, chunksize=1000, dtype={'AIRLINE':str, 'ORIGIN_AIRPORT':str, 'DESTINATION_AIRPORT':str})
        for df in iters:
            jbsw_2015 = df[(df['YEAR'] == 2015) & ((df['AIRLINE'] == 'B6') | (df['AIRLINE'] == 'WN'))] # Year=2015, Airline=JetBlue/SouthWest
            if header: # Have headers already
                jbsw_2015.to_csv(file, mode='a', index=False, header=False)
            else: # Does not have headers
                jbsw_2015.to_csv(file, mode='a', index=False)
                header = True
    return None


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def data_kinds():
    """
    data_kinds outputs a (hard-coded) dictionary of data kinds, keyed by column
    name, with values Q, O, N (for 'Quantitative', 'Ordinal', or 'Nominal').

    :Example:
    >>> out = data_kinds()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'O', 'N', 'Q'}
    True
    """

    return ({'YEAR': 'Q',
             'MONTH': 'O', # Month can be ordinal and quantitative
             'DAY': 'O', 
             'DAY_OF_WEEK': 'O', # DayOfWeek can be ordinal and quantitative
             'AIRLINE': 'N',
             'FLIGHT_NUMBER': 'N', 
             'TAIL_NUMBER': 'N',
             'ORIGIN_AIRPORT': 'N',
             'DESTINATION_AIRPORT': 'N',
             'SCHEDULED_DEPARTURE': 'Q',
             'DEPARTURE_TIME': 'Q',
             'DEPARTURE_DELAY': 'Q',
             'TAXI_OUT': 'Q',
             'WHEELS_OFF': 'Q',
             'SCHEDULED_TIME': 'Q',
             'ELAPSED_TIME': 'Q',
             'AIR_TIME': 'Q',
             'DISTANCE': 'Q',
             'WHEELS_ON': 'Q',
             'TAXI_IN': 'Q',
             'SCHEDULED_ARRIVAL': 'Q',
             'ARRIVAL_TIME': 'Q',
             'ARRIVAL_DELAY': 'Q',
             'DIVERTED': 'Q', # boolean would be ordinal? Where is ordinal number???
             'CANCELLED': 'Q',# boolean would be ordinal?
             'CANCELLATION_REASON': 'N',
             'AIR_SYSTEM_DELAY': 'Q',
             'SECURITY_DELAY': 'Q',
             'AIRLINE_DELAY': 'Q',
             'LATE_AIRCRAFT_DELAY': 'Q',
             'WEATHER_DELAY': 'Q'})


def data_types():
    """
    data_types outputs a (hard-coded) dictionary of data types, keyed by column
    name, with values str, int, float.

    :Example:
    >>> out = data_types()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'int', 'str', 'float', 'bool'}
    True
    """

    return ({'YEAR': int,
             'MONTH': int,
             'DAY': int,
             'DAY_OF_WEEK': int,
             'AIRLINE': str,
             'FLIGHT_NUMBER': str,
             'TAIL_NUMBER': str,
             'ORIGIN_AIRPORT': str,
             'DESTINATION_AIRPORT': str,
             'SCHEDULED_DEPARTURE': int,
             'DEPARTURE_TIME': float,
             'DEPARTURE_DELAY': float,
             'TAXI_OUT': float,
             'WHEELS_OFF': float,
             'SCHEDULED_TIME': int,
             'ELAPSED_TIME': float,
             'AIR_TIME': float,
             'DISTANCE': int,
             'WHEELS_ON': float,
             'TAXI_IN': float,
             'SCHEDULED_ARRIVAL': int,
             'ARRIVAL_TIME': float,
             'ARRIVAL_DELAY': float,
             'DIVERTED': bool,
             'CANCELLED': bool,
             'CANCELLATION_REASON': str,
             'AIR_SYSTEM_DELAY': float,
             'SECURITY_DELAY': float,
             'AIRLINE_DELAY': float,
             'LATE_AIRCRAFT_DELAY': float,
             'WEATHER_DELAY': float})


# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------

def basic_stats(flights):
    """
    basic_stats takes flights and outputs a dataframe that contains statistics
    for flights arriving/departing for SAN.
    That is, the output should have have two rows, indexed by ARRIVING and
    DEPARTING, and have the following columns:

    * number of arriving/departing flights to/from SAN (count).
    * mean flight (arrival) delay of arriving/departing flights to/from SAN
      (mean_delay).
    * median flight (arrival) delay of arriving/departing flights to/from SAN
      (median_delay).
    * the airline code of the airline with the longest flight (arrival) delay
      among all flights arriving/departing to/from SAN (airline).
    * a list of the three months with the greatest number of arriving/departing
      flights to/from SAN, sorted from greatest to least (top_months).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = basic_stats(flights)
    >>> out.index.tolist() == ['ARRIVING', 'DEPARTING']
    True
    >>> cols = ['count', 'mean_delay', 'median_delay', 'airline', 'top_months']
    >>> out.columns.tolist() == cols
    True
    """
    flights_AD = flights.copy()
    flights_AD['DEP_ARR'] = flights_AD['DESTINATION_AIRPORT'].apply(lambda des: 'ARRIVING' if des == 'SAN' else 'DEPARTING')
    
    # Helper function to get the airline with longest flight
    def airline(series):
        """
        Get the airline code with the longest flight.

        :param series: arrival delay series
        :return: airline code of longest arrival delay
        """
        return flights_AD.loc[series.idxmax]['AIRLINE']

    # Helper function to get the top three months of greatest num of flights
    def top_months(series):
        """
        Get the top three months of greatest number of flights.

        :param series: arrival delay series
        :return: list of top three months
        """
        pivot_tab = flights_AD.pivot_table(
            values='YEAR', 
            index='MONTH',
            columns=series,
            aggfunc='count'
        ).sort_values(by=series.name, ascending=False) # Sort by number of flights
        return list(pivot_tab.index[:3]) # Return top three index

    agg = (flights_AD
     .groupby('DEP_ARR')
     .aggregate({'YEAR':'count', # Number of arriving and departing flights 
                 'ARRIVAL_DELAY':['mean', 'median', airline], # mean, meadian, longest delay airline
                 'DEP_ARR':top_months} # Top three month of number of flight
    )).rename(columns={'mean':'mean_delay', 'median':'median_delay'}) # Rename column names
    agg.columns = agg.columns.droplevel(0) # Drop top column categories
    return agg


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def depart_arrive_stats(flights):
    """
    depart_arrive_stats takes in a dataframe like flights and calculates the
    following quantities in a series (with the index in parentheses):
    - The proportion of flights from/to SAN that
      leave late, but arrive early or on-time (late1).
    - The proportion of flights from/to SAN that
      leaves early, or on-time, but arrives late (late2).
    - The proportion of flights from/to SAN that
      both left late and arrived late (late3).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats(flights)
    >>> out.index.tolist() == ['late1', 'late2', 'late3']
    True
    >>> isinstance(out, pd.Series)
    True
    >>> out.max() < 0.30
    True
    """
    flights_late = flights[['MONTH', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']].copy() # Deep copy

    # Helper function: leave late, arrive early/on time
    def late1(series):
        """
        Get proportion of plane leave late, arrive early/on time.

        :param series: departure delay
        :return: proportion of leave late, arrive early/on time
        """

        count = flights_late[(series > 0) & (flights_late['ARRIVAL_DELAY'] <= 0)].shape[0]
        return count / flights_late.shape[0]

    # Helper function: leave early/on time, arrive late
    def late2(series):
        """
        Get proportion of plane leave early/on time, arrive late.

        :param series: departure delay
        :return: proportion of leave early/on time, arrive late
        """

        count = flights_late[(series <= 0) & (flights_late['ARRIVAL_DELAY'] > 0)].shape[0]
        return count / flights_late.shape[0]


    # Helper function: leave late, arrive late
    def late3(series):
        """
        Get proportion of plane leave late, arrive late.

        :param series: departure delay
        :return: proportion of leave late, arrive late
        """

        count = flights_late[(series > 0) & (flights_late['ARRIVAL_DELAY'] > 0)].shape[0]
        return count / flights_late.shape[0]
    
    dep_delay = flights_late['DEPARTURE_DELAY']
    return pd.Series({'late1':late1(dep_delay), 'late2':late2(dep_delay), 'late3':late3(dep_delay)})


def depart_arrive_stats_by_month(flights):
    """
    depart_arrive_stats_by_month takes in a dataframe like flights and
    calculates the quantities in depart_arrive_stats, broken down by month

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats_by_month(flights)
    >>> out.columns.tolist() == ['late1', 'late2', 'late3']
    True
    >>> set(out.index) <= set(range(1, 13))
    True
    """
    flights_late = flights[['MONTH', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']].copy() # Deep copy

    # Helper function: leave late, arrive early/on time
    def late1(series):
        """
        Get proportion of plane leave late, arrive early/on time.

        :param series: departure delay
        :return: proportion of leave late, arrive early/on time
        """

        count = flights_late[(series > 0) & (flights_late['ARRIVAL_DELAY'] <= 0)].shape[0]
        return count / flights_late.shape[0]

    # Helper function: leave early/on time, arrive late
    def late2(series):
        """
        Get proportion of plane leave early/on time, arrive late.

        :param series: departure delay
        :return: proportion of leave early/on time, arrive late
        """

        count = flights_late[(series <= 0) & (flights_late['ARRIVAL_DELAY'] > 0)].shape[0]
        return count / flights_late.shape[0]


    # Helper function: leave late, arrive late
    def late3(series):
        """
        Get proportion of plane leave late, arrive late.

        :param series: departure delay
        :return: proportion of leave late, arrive late
        """

        count = flights_late[(series > 0) & (flights_late['ARRIVAL_DELAY'] > 0)].shape[0]
        return count / flights_late.shape[0]
    
    agg_late = (flights_late
     .groupby('MONTH')
     .aggregate({'DEPARTURE_DELAY':[late1, late2, late3]}))
    agg_late.columns = agg_late.columns.droplevel(0) # Drop top column categories
    return agg_late


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def cnts_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, how many flights were there (in 2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = cnts_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    >>> (out >= 0).all().all()
    True
        """
    count_tab = flights.pivot_table(
        values='YEAR', 
        index='DAY_OF_WEEK',
        columns='AIRLINE',
        aggfunc='count'
    )
    return count_tab


def mean_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, what is the average ARRIVAL_DELAY (in
    2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = mean_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    """
    mean_tab = flights.pivot_table(
        values='ARRIVAL_DELAY', 
        index='DAY_OF_WEEK',
        columns='AIRLINE',
        aggfunc='mean'
    )
    return mean_tab


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def predict_null_arrival_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the ARRIVAL_DELAY is null and otherwise False.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `ARRIVAL_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('ARRIVAL_DELAY', axis=1).apply(predict_null_arrival_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """
    return ...


def predict_null_airline_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the AIRLINE_DELAY is null and otherwise False. Since the
    function doesn't depend on AIRLINE_DELAY, it should work a row even if that
    index is dropped.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `AIRLINE_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('AIRLINE_DELAY', axis=1).apply(predict_null_airline_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """

    return ...


# ---------------------------------------------------------------------
# Question #7
# ---------------------------------------------------------------------

def perm4missing(flights, col, N):
    """
    perm4missing takes in flights, a column col, and a number N and returns the
    p-value of the test (using N simulations) that determines if
    DEPARTURE_DELAY is MAR dependent on col.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = perm4missing(flights, 'AIRLINE', 100)
    >>> 0 <= out <= 1
    True
    """

    return ...


def dependent_cols():
    """
    dependent_cols gives a list of columns on which DEPARTURE_DELAY is MAR
    dependent on.

    :Example:
    >>> out = dependent_cols()
    >>> isinstance(out, list)
    True
    >>> cols = 'YEAR DAY_OF_WEEK AIRLINE DIVERTED CANCELLATION_REASON'.split()
    >>> set(out) <= set(cols)
    True
    """

    return ...


def missing_types():
    """
    missing_types returns a Series
    - indexed by the following columns of flights:
    CANCELLED, CANCELLATION_REASON, TAIL_NUMBER, ARRIVAL_TIME.
    - The values contain the most-likely missingness type of each column.
    - The unique values of this Series should be MD, MCAR, MAR, MNAR, NaN.

    :param:
    :returns: A series with index and values as described above.

    :Example:
    >>> out = missing_types()
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) - set(['MD', 'MCAR', 'MAR', 'NMAR', np.NaN]) == set()
    True
    """

    return ...


# ---------------------------------------------------------------------
# Question #8
# ---------------------------------------------------------------------

def prop_delayed_by_airline(jb_sw):
    """
    prop_delayed_by_airline takes in a dataframe like jb_sw and returns a
    DataFrame indexed by airline that contains the proportion of each airline's
    flights that are delayed.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> (out >= 0).all().all() and (out <= 1).all().all()
    True
    >>> len(out.columns) == 1
    True
    """

    return ...


def prop_delayed_by_airline_airport(jb_sw):
    """
    prop_delayed_by_airline_airport that takes in a dataframe like jb_sw and
    returns a DataFrame, with columns given by airports, indexed by airline,
    that contains the proportion of each airline's flights that are delayed at
    each airport.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline_airport(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> ((out >= 0) | (out <= 1) | (out.isnull())).all().all()
    True
    >>> len(out.columns) == 6
    True
    """

    return ...


# ---------------------------------------------------------------------
# Question #9
# ---------------------------------------------------------------------

def verify_simpson(df, group1, group2, occur):
    """
    verify_simpson verifies whether a dataset displays Simpson's Paradox.

    :param df: a dataframe
    :param group1: the first group being aggregated
    :param group2: the second group being aggregated
    :param occur: a column of df with values {0,1}, denoting
    if an event occurred.
    :returns: a boolean. True if simpson's paradox is present,
    otherwise False.

    :Example:
    >>> df = pd.DataFrame([[4,2,1], [1,2,0], [1,4,0], [4,4,1]], columns=[1,2,3])
    >>> verify_simpson(df, 1, 2, 3) in [True, False]
    True
    >>> verify_simpson(df, 1, 2, 3)
    False
    """

    return ...


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def search_simpsons(jb_sw, N):
    """
    search_simpsons takes in the jb_sw dataset and a number N, and returns a
    list of N airports for which the proportion of flight delays between
    JetBlue and Southwest satisfies Simpson's Paradox.

    Only consider airports that have '3 letter codes',
    Only consider airports that have at least one JetBlue and Southwest flight.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=1000)
    >>> pair = search_simpsons(jb_sw, 2)
    >>> len(pair) == 2
    True
    """

    return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_san', 'get_sw_jb'],
    'q02': ['data_kinds', 'data_types'],
    'q03': ['basic_stats'],
    'q04': ['depart_arrive_stats', 'depart_arrive_stats_by_month'],
    'q05': ['cnts_by_airline_dow', 'mean_by_airline_dow'],
    'q06': ['predict_null_arrival_delay', 'predict_null_airline_delay'],
    'q07': ['perm4missing', 'dependent_cols', 'missing_types'],
    'q08': ['prop_delayed_by_airline', 'prop_delayed_by_airline_airport'],
    'q09': ['verify_simpson'],
    'q10': ['search_simpsons']
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
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
