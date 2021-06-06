import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import tensorflow as tf
from numpy import load, save
from category_encoders import OneHotEncoder

# Own functions and classes
from Transformers import QuantileTransformerDf, IterativeImputerDf, RareLabelNanEncoder



def make_submission(trained_regressor,
                    test_csv_file='data/test.csv',
                    preprocessing_pipe_file='preprocessing_pipe.pkl',
                    submission_file_name='data/submission.csv'):
    X_test = pd.read_csv(test_csv_file)

    X_test['MSSubClass'] = X_test['MSSubClass'].astype(dtype='object')  # convert column to categorical
    y_idx = X_test['Id'].copy()  # copy indexes to use in submission file

    X_test = X_test.drop(['Id'], axis=1)  # remove Id column
    preprocessing_pipe = joblib.load(preprocessing_pipe_file, mmap_mode=None)

    # input data preprocessing
    X_test_encoded = preprocessing_pipe.transform(X_test)
    # predict y values
    y_predict = trained_regressor.predict(X_test_encoded)

    y_predict = pd.DataFrame(y_predict, columns=['SalePrice'])

    Y_predict = pd.concat([y_idx, y_predict], axis=1)
    Y_predict.to_csv(submission_file_name, index=False)


def get_Xy(XY_train_enc_file, test_size=0.1):
    #XY_train_enc_file = 'data/XY_train_enc_' + str(n_categories) + '_' + str(tol) + '.csv'

    df = pd.read_csv(XY_train_enc_file)
    X = df.drop(['SalePrice'], axis=1).to_numpy().astype(np.float64)
    y = df['SalePrice'].copy().to_numpy().astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        shuffle=True,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test

def debug_array(array):
    array=pd.DataFrame(array)
    print(array.head())
    print(array.info())
    print(array.describe())

def make_preprocessing(config):
    train_enc_file = f'X_train_enc_rare_tol_{config["rare_tol"]}_n_categories_{config["n_categories"]}_max_iter_{config["max_iter"]}_iter_tol_{config["iter_tol"]}.npy'
    train_enc_path = '/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/' + train_enc_file
    y_train_path = '/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/y_train.npy'

    try:
        X_train_encoded = load(file=train_enc_path)
        Y_train = load(file=y_train_path)

    except:
        df_train = pd.read_csv('/home/peterpirog/PycharmProjects/BostonHousesTune/data/train.csv')

        # csv preprcessing
        df_train['MSSubClass'] = df_train['MSSubClass'].astype(dtype='category')  # convert feature to categorical
        X_train = df_train.drop(['Id', 'SalePrice'], axis=1)
        Y_train = df_train['SalePrice'].astype(dtype=np.float32)

        # PREPROCESSING
        # STEP 1 -  categorical features rare labels encoding
        rle = RareLabelNanEncoder(categories=None, tol=config['rare_tol'],
                                  minimum_occurrences=None,
                                  n_categories=config['n_categories'],
                                  max_n_categories=None,
                                  replace_with='Rare',
                                  impute_missing_label=False,
                                  additional_categories_list=None)

        # STEP 2 - categorical features one hot encoding
        # https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/one_hot.py
        ohe = OneHotEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True,
                            handle_missing='return_nan',  # options are 'error', 'return_nan', 'value', and 'indicator'.
                            handle_unknown='return_nan',  # options are 'error', 'return_nan', 'value', and 'indicator'
                            use_cat_names=False)

        # STEP 3 - numerical values quantile transformation with skewness removing
        q_trans = QuantileTransformerDf(n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False,
                                        subsample=1e5, random_state=42, copy=True, dataframe_as_output=True,
                                        dtype=np.float32)

        # STEP 4 - missing values multivariate imputation
        imp = IterativeImputerDf(min_value=0,  # values from 0 to 1 for categorical for numeric
                                 max_value=1,
                                 random_state=42,
                                 initial_strategy='median',
                                 max_iter=config['max_iter'],
                                 tol=config['iter_tol'],
                                 verbose=0, dataframe_as_output=False)
        # Step 5 PCA
        pca = PCA(n_components=config['n_components'], svd_solver='full')

        # STEP 6 MAKE PIPELINE AND TRAIN IT
        pipe = Pipeline([
            ('rare_lab', rle),
            ('one_hot', ohe),
            ('q_trans', q_trans),
            ('imputer', imp),
            ('pca', pca)
        ])

        # Pipeline training
        pipe.fit(X_train)
        X_train_encoded = pipe.transform(X_train).astype(dtype=np.float32)
        # save X_train_encoded array
        save(file=train_enc_path, arr=X_train_encoded)
        save(file=y_train_path, arr=Y_train)

    # STEP 7 SPLITTING DATA FOR KERAS
    X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, Y_train,
                                                        shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test





if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    get_Xy(n_categories=2)
    # Rare encoder options
    n_categories = 2
    tol = 0.05
    xy_train_filename = 'data/XY_train_enc_' + str(n_categories) + '_' + str(tol) + '.csv'

    # load X_train, y_train data
    X_train = pd.read_csv('data/X_train_enc.csv')

    df_train = pd.read_csv(xy_train_filename)
    y_train = df_train['SalePrice'].copy().to_numpy()

    regressor = RandomForestRegressor(n_estimators=200, max_depth=25, n_jobs=-1)

    regressor.fit(X=X_train, y=y_train)

    #keras regressor
    model=tf.keras.models.load_model(filepath='my_model.h5')

    make_submission(trained_regressor=regressor,
                    test_csv_file='data/test.csv',
                    preprocessing_pipe_file='preprocessing_pipe.pkl',
                    submission_file_name='data/submission.csv')
