import pandas as pd
import numpy as np
import seaborn as sns
import os


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def simple_pipeline(data):
    '''
    simple_pipeline takes in a dataframe like data and returns a tuple 
    consisting of the pipeline and the predictions your model makes 
    on data (as trained on data).

    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = simple_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], FunctionTransformer)
    True
    >>> preds.shape[0] == data.shape[0]
    True
    '''
    log_func = FunctionTransformer(np.log)
    sim_pl = Pipeline(steps=[
        ('log', log_func), 
        ('linear_reg', LinearRegression())
    ])
    pred = sim_pl.fit(data[['c2']], data['y'])
    predic = sim_pl.predict(data[['c2']])
    return (sim_pl, predic)

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def multi_type_pipeline(data):
    '''
    multi_type_pipeline that takes in a dataframe like data and 
    returns a tuple consisting of the pipeline and the predictions 
    your model makes on data (as trained on data).

    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = multi_type_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], ColumnTransformer)
    True
    >>> data.shape[0] == preds.shape[0]
    True
    '''
    log_func = FunctionTransformer(np.log)
    c2_transformer = Pipeline(steps=[
        ('log', log_func)
    ])

    group_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())     # output from Ordinal becomes input to OneHot
    ])

    # preprocessing pipeline (put them together)
    preproc = ColumnTransformer(transformers=[
        ('log', c2_transformer, ['c2']),
        ('onehot', group_transformer, ['group'])],
        remainder='passthrough')

    mul_pl = Pipeline(steps=[('preprocessor', preproc), ('regressor', LinearRegression())])
    mul_pl.fit(data.drop('y', axis=1), data['y'])
    predic_2 = mul_pl.predict(data.drop('y', axis=1))
    return (mul_pl, predic_2)

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin


class StdScalerByGroup(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 2, 2], 'c2': [3, 1, 2, 0]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> std.grps_ is not None
        True
        """
        # X may not be a pandas dataframe (e.g. a np.array)
        X = pd.DataFrame(data=X)
        
        # A dictionary of means/standard-deviations for each column, for each group.
        X = X.rename(columns={X.columns[0]:'col1'}) # Rename col name to be consistent
        grouped = X.groupby('col1').aggregate(['mean', 'std']) # Grouped to get mean & std
        dic = {}
        for key in grouped.columns:
            dic.update({key:dict(grouped[key])})
        
        self.grps_ = dic

        return self

    def transform(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 3, 4], 'c2': [1, 2, 3, 4]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> out = std.transform(X)
        >>> out.shape == (4, 2)
        True
        >>> np.isclose(out.abs(), 0.707107, atol=0.001).all().all()
        True
        """

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")
        
        # Define a helper function here?
        # Helper function to calculate z score
        def z_score(group, col, score):
            # print(group, col, score)
            mean = self.grps_.get((col, 'mean')).get(group)
            std = self.grps_.get((col, 'std')).get(group)
            z = (score - mean) / std # Standardize
            # print(mean, std, z)
            return z
        
        # X may not be a dataframe (e.g. np.array)
        X = pd.DataFrame(X)
        X = X.rename(columns={X.columns[0]:'col1'})
        for col in X.drop('col1', axis=1).columns:
            X.loc[:,col] = X.drop('col1', axis=1).apply(lambda x: z_score(X.loc[x.name, 'col1'], col, x[col]), axis=1)
        return X.drop('col1', axis=1)

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def eval_toy_model():
    """
    hardcoded answers to question 4

    :Example:
    >>> out = eval_toy_model()
    >>> len(out) == 3
    True
    """

    return [(2.7551086974518104, None), (2.314833616435528, None), (2.3157339477823844, None)] # Only changed this line of code, added None for R^2


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def tree_reg_perf(galton):
    """

    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = tree_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    >>> out['train_err'].iloc[-1] < out['test_err'].iloc[-1]
    True
    """
    X = galton.drop('childHeight', axis=1)
    y = galton.childHeight
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    rows = []
    for k in range(1, 21):
        reg = DecisionTreeRegressor(max_depth=k)
        pl = Pipeline([
           ('DT_reg', reg)
        ])

        pl.fit(X_train, y_train)
        preds_train = pl.predict(X_train)
        preds_test = pl.predict(X_test)
        rmse_train = np.sqrt(np.mean((preds_train - y_train)**2))
        rmse_test = np.sqrt(np.mean((preds_test - y_test)**2))
        rows.append(pd.Series({'train_err':rmse_train, 'test_err':rmse_test}, name=k))

    return pd.DataFrame(rows)


def knn_reg_perf(galton):
    """
    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = knn_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    """
    X = galton.drop('childHeight', axis=1)
    y = galton.childHeight
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    rows = []
    for k in range(1, 21):
        reg = KNeighborsRegressor(n_neighbors=k)
        pl = Pipeline([
           ('KN_reg', reg)
        ])


        pl.fit(X_train, y_train)
        preds_train = pl.predict(X_train)
        preds_test = pl.predict(X_test)
        rmse_train = np.sqrt(np.mean((preds_train - y_train)**2))
        rmse_test = np.sqrt(np.mean((preds_test - y_test)**2))
        rows.append(pd.Series({'train_err':rmse_train, 'test_err':rmse_test}, name=k))
        
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def titanic_model(titanic):
    """
    :Example:
    >>> fp = os.path.join('data', 'titanic.csv')
    >>> data = pd.read_csv(fp)
    >>> pl = titanic_model(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> from sklearn.base import BaseEstimator
    >>> isinstance(pl.steps[-1][-1], BaseEstimator)
    True
    >>> preds = pl.predict(data.drop('Survived', axis=1))
    >>> ((preds == 0)|(preds == 1)).all()
    True
    """
    class PropByColumn(BaseEstimator, TransformerMixin):

        def __init__(self):
            pass

        def fit(self, X, y=None):

            return self

        def transform(self, X, y=None):

            # try:
                # getattr(self, "grps_")
            # except AttributeError:
                # raise RuntimeError("You must fit the transformer before tranforming the data!")

            # print(X.columns)
            X = pd.DataFrame(X)

            X = X.rename(columns={X.columns[0]:'col1'}) # Rename col name to be consistent
            props = []
            for col in X.drop('col1', axis=1).columns:
                rate = X[col].value_counts(normalize=True)
                surv_rate = dict(rate.replace(np.nan, 0))
                prop = X[col].replace(surv_rate)#(X[col] / X.groupby('col1')[col].transform('count'))
                props.append(prop)
            return pd.DataFrame(data=props).T

    def strip_call(df):
        name = list(df['Name'].apply(lambda x: x.split('.')).values)
        df_name = pd.DataFrame(name)
        name = df_name.iloc[:,0]
        name = name.replace(['Lady', 'Sir'], 'Royal')
        name = name.replace(['Mlle', 'Ms'], 'Miss')
        name = name.replace(['Mme'], 'Miss')
        return np.array(name.to_frame())

    def ffare(fare):
        df_fare = pd.DataFrame(fare)
        series = pd.qcut(df_fare.iloc[:,0], 5, labels = [1, 2, 3, 4, 5])
        return np.array(series.to_frame())

    def aage(age):
        df_age = pd.DataFrame(age)
        bins = [0, 6, 12, 18, 25, 40, 60, np.inf]
        lbs = [0, 1, 2, 3, 4, 5, 6]
        series = pd.cut(df_age.iloc[:,0], bins, labels=lbs)
        return np.array(series.to_frame())

    # Data Cleaning
    # cleaned = titanic.copy()
    # title = cleaned['Name'].apply(strip_call)
    # cleaned = cleaned.assign(Title=title) # Assign a title column

    # Train/Test set
    X = titanic.drop(['Survived'], axis=1)
    y = titanic.Survived
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Imputation
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Proportion Transformation
    prop_trans = Pipeline(steps=[
        ('prop', PropByColumn())
    ])

    # Name ordinal encoder
    name_func = FunctionTransformer(strip_call, validate=False)
    name_ordtrans = Pipeline(steps=[
        ('title', name_func),
        ('ordinal', OrdinalEncoder()), 
        ('onehot', OneHotEncoder(handle_unknown ='ignore'))
    ])

    # Fare transformation
    fare_func = FunctionTransformer(ffare, validate=False)
    fare_trans = Pipeline(steps=[
        ('imputation', imp),
        ('fare', fare_func)
    ])

    # Age transformation
    age_func = FunctionTransformer(aage, validate=False)
    age_trans = Pipeline(steps=[
        ('imputation', imp),
        ('age', age_func)
    ])

    # Log transformation
    log_func = FunctionTransformer(np.log)
    log_trans = Pipeline(steps=[
        ('imputation', imp),
        ('log', log_func)
    ])

    # Standardization Transformation
    std_trans = Pipeline(steps=[
        ('imputation', imp),
        ('std', StdScalerByGroup())
    ])

    # Polynomial Transformation
    poly_trans = Pipeline(steps=[
        ('imputation', imp),
        ('poly', PolynomialFeatures(interaction_only=True, include_bias=False))
    ])

    # Proportion Polynomial Transformation
    prop_poly_trans = Pipeline(steps=[
        ('imputation', imp),
        ('prop', PropByColumn()), 
        ('poly', PolynomialFeatures(interaction_only=True, include_bias=False))
    ])

    # Categorical transformation
    group_trans = Pipeline(steps=[
        ('ordinal', OrdinalEncoder()),  
        ('onehot', OneHotEncoder(handle_unknown ='ignore'))
    ])


    # preprocessing pipeline (put them together)
    preproc = (ColumnTransformer(transformers=[
        # ('prop_title', name_ordtrans, ['Name']),
        ('prop_cont_class', prop_trans, ['Pclass', 'Age', 'Fare']),
        ('prop_cat_class', prop_trans, ['Pclass', 'Sex', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']),
        ('prop_cont_sex', prop_trans, ['Sex', 'Age', 'Fare']),
        ('prop_cat_sex', prop_trans, ['Sex', 'Pclass', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']),
        ('std_age_class', std_trans, ['Pclass', 'Age']),
        ('std_fare_class', std_trans, ['Pclass', 'Fare']),
        ('fare', fare_trans, ['Fare']),
        ('age', age_trans, ['Age']),
        ('poly', poly_trans, ['Age', 'Fare']),
        ('log', log_trans, ['Age']),
        ('prop_poly', prop_poly_trans, ['Pclass', 'Age', 'Fare']),
        ('onehot', group_trans, ['Sex'])], remainder='drop'))
    
    pl = Pipeline(steps=[('preprocessor', preproc), ('regressor', GradientBoostingClassifier())])
    # pl = Pipeline(steps=[('regressor', LinearRegression())])
    pl.fit(X_train, y_train)
    pl.fit(X, y)
    # predic_4 = new_pl.predict(data.loc[:, ~data.columns.isin(['y'])])
    # scores_train = cross_val_score(pl, X, y, cv=5)
    # scores_train  # R^2
    # y_pred = pl.predict(X_test)
    # round(accuracy_score(y_pred, y_test) * 100, 2)
    return pl

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def json_reader(file, iterations):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> isinstance(reviews, list)
    True
    >>> isinstance(labels, list)
    True
    >>> len(labels) == len(reviews)
    True
    """

    return ...


def create_classifier_multi(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
    True
    """
    
    return ...


def to_binary(labels):
    """
    :Example
    >>> lst = [1, 2, 3, 4, 5]
    >>> to_binary(lst)
    >>> print(lst)
    [0, 0, 0, 1, 1]
    """
    
    return ...


def create_classifier_binary(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> to_binary(labels)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
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
    'q01': ['simple_pipeline'],
    'q02': ['multi_type_pipeline'],
    'q03': ['StdScalerByGroup'],
    'q04': ['eval_toy_model'],
    'q05': ['tree_reg_perf', 'knn_reg_perf'],
    'q06': ['titanic_model']
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