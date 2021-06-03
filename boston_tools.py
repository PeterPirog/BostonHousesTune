import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf


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
