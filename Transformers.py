#https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/one_hot.py
#pip install category_encoders
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from feature_engine.encoding import RareLabelEncoder

from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

from category_encoders import one_hot



class QuantileTransformerDf(QuantileTransformer):
    """DataFrame Wrapper around QuantileTransformer
    Function based on: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
    """

    def __init__(self, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False,
                 subsample=1e5, random_state=42, copy=True, dataframe_as_output=True,dtype='float32'):
        super(QuantileTransformerDf, self).__init__(n_quantiles=n_quantiles,
                                                    output_distribution=output_distribution,
                                                    ignore_implicit_zeros=ignore_implicit_zeros,
                                                    subsample=subsample,
                                                    random_state=random_state,
                                                    copy=copy)
        self.dataframe_as_output = dataframe_as_output
        self.dtype=dtype

    def transform(self, X, y=None):
        X = X.copy()
        z = super(QuantileTransformerDf, self).transform(X.values)
        if self.dataframe_as_output:
            return pd.DataFrame(z, index=X.index, columns=X.columns,dtype=self.dtype)
        else:
            return z


class RareLabelNanEncoder(BaseEstimator, TransformerMixin):
    """This function based on:
    https://feature-engine.readthedocs.io/en/latest/encoding/RareLabelEncoder.html
    Additionally makes possible rare label encoding even with missing values,
    if impute_missing_label=False missing values in output dataframe is np.nan
    if impute_missing_label=True missing values in output dataframe is 'MISSING
    """

    def __init__(self, categories=None, tol=0.05, minimum_occurrences=None, n_categories=10, max_n_categories=None,
                 replace_with='Rare', impute_missing_label=False, additional_categories_list=None):
        """
        :param categories:
        :param tol: The minimum frequency a label should have to be considered frequent. Categories with frequencies lower than tol will be grouped
        :param minimum_occurrences: defined minimum number of value occurrences for single feature
        :param n_categories: The minimum number of categories a variable should have for the encoder to find frequent labels. If the variable contains less categories, all of them will be considered frequent.
        :param max_n_categories: The maximum number of categories that should be considered frequent. If None, all categories with frequency above the tolerance (tol) will be considered frequent.
        :param replace_with: The category name that will be used to replace infrequent categories.
        :param impute_missing_label: if  False missing values in output dataframe is np.nan if True missing values in output dataframe is 'MISSING
        :param additional_categories_list: add list with feature if you want  feature for default founded categorical features
        """
        super().__init__()
        self.categories = categories
        self.additional_categories_list = additional_categories_list
        self.impute_missing_label = impute_missing_label
        self.new_categories = []
        self.number_of_samples = None
        self.minimum_occurrences = minimum_occurrences

        # original RareLabelEncoder parameters
        self.tol = tol
        self.n_categories = n_categories
        self.max_n_categories = max_n_categories
        self.replace_with = replace_with

    def fit(self, X, y=None):
        X = X.copy()
        self.number_of_samples = X.shape[0]  # number of rows in dataframe

        if self.categories is None:
            self.categories = X.select_dtypes(include=['object']).columns.tolist()
            # option to add some additional feature if you need
            if self.additional_categories_list is not None:
                self.categories = self.categories + self.additional_categories_list

        # option to define minimum value occurrence for single feature- usefull for huge datasets with high cardinality
        if self.minimum_occurrences is not None:
            self.tol = float(self.minimum_occurrences / self.number_of_samples)
            print(f'Value of minimum_occurrences is defined. New value of tol is:{self.tol}')

        return self

    def transform(self, X, y=None):
        pd.options.mode.chained_assignment = None  # default='warn' - turn off warning about  data overwrite
        for category in self.categories:
            x = X[category].copy()  # not use copy to intentionally change value
            idx_nan = x.loc[pd.isnull(x)].index  # find nan values in analyzed feature column

            # replace missing values
            x[idx_nan] = 'MISS'
            encoder = RareLabelEncoder(tol=self.tol, n_categories=self.n_categories,
                                       max_n_categories=self.max_n_categories,
                                       replace_with=self.replace_with)

            x = x.to_frame(name=category)  # convert pd.series to dataframe
            x = encoder.fit_transform(x)
            X[category] = x
            if not self.impute_missing_label:
                X[category].loc[idx_nan] = np.nan
        pd.options.mode.chained_assignment = 'warn'  # default='warn' - turn on warning about  data overwrite
        return X


class OneHotNanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories='auto', drop=True, dtype=np.float64):
        super().__init__()
        self.categories = categories
        self.drop = drop
        self.dtype = dtype
        self.new_categories = []

    def fit(self, X, y=None):

        if self.categories == 'auto':
            self.categories = X.select_dtypes(include=['object']).columns.tolist()

        # get new categories based on train data
        for category in self.categories:
            labels = X[category].unique().tolist()
            labels = [str(x) for x in labels]  # converting nan to 'nan'
            if 'nan' in labels:
                labels.remove('nan')  # remove nan labels

            for label in labels:
                new_label = str(category) + '_' + str(label)
                self.new_categories.append(new_label)

        self.new_categories=list(set(self.new_categories)) #get unique elements
        print(f'new_categories={self.new_categories}')

        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.new_categories]=np.nan
        """
        for category in self.categories:
            labels = X[category].unique().tolist()
            labels = [str(x) for x in labels]  # converting nan to 'nan
            if 'nan' in labels:
                labels.remove('nan')  # remove nan labels

            for label in labels:
                new_label = str(category) + '_' + str(label)
                self.new_categories.append(new_label)
                X[new_label] = np.where(X[category] == label, 1, 0)
                X.loc[X[category].isna(), new_label] = np.nan
        if self.drop:
            X = X.drop(columns=self.categories)  # drop encoded columns
        # X[self.new_categories] = X[self.new_categories].astype(self.dtype)
        """
        return X


class IterativeImputerDf(IterativeImputer):
    """DataFrame Wrapper around QuantileTransformer
    Function based on: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
    """

    def __init__(self, min_value=-np.inf,  # values from 0 to 1 for categorical for numeric
                 max_value=np.inf,
                 random_state=42,
                 max_iter=10,
                 tol=1e-3,
                 initial_strategy='median',
                 verbose=1, dataframe_as_output=True):
        super(IterativeImputerDf, self).__init__(min_value=min_value,
                                                 max_value=max_value,
                                                 random_state=random_state,
                                                 max_iter=max_iter,
                                                 tol=tol,
                                                 initial_strategy=initial_strategy,
                                                 verbose=verbose)
        self.dataframe_as_output = dataframe_as_output
        print('dataframe_as_output=', self.dataframe_as_output)

    def transform(self, X, y=None):
        X = X.copy()
        z = super(IterativeImputerDf, self).transform(X.values)

        if self.dataframe_as_output:
            return pd.DataFrame(z, index=self.index, columns=self.columns)
        else:
            return z

    def fit_transform(self, X, y=None):
        X = X.copy()
        z = super(IterativeImputerDf, self).fit_transform(X.values)
        self.index = X.index
        self.columns = X.columns

        if self.dataframe_as_output:
            return pd.DataFrame(z, index=self.index, columns=self.columns)
        else:
            return z

