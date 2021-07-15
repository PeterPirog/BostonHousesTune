﻿import numpy as np
from numpy import load, save
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
import xgbfir

import joblib
from category_encoders import OneHotEncoder
from Transformers import QuantileTransformerDf, IterativeImputerDf, RareLabelNanEncoder

import pandas as pd
pd.set_option('display.max_columns', None)


def rmsle(y_true, y_pred, **kwargs):
    # Implementation of rmsle error
    # Convert nparrays  to tensors

    # Clip values to prevent log from values below 0
    y_true = np.clip(y_true, a_min=0, a_max=np.inf)
    y_pred = np.clip(y_pred, a_min=0, a_max=np.inf)
    return -np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


if __name__ == "__main__":
    # https://machinelearningmastery.com/xgboost-for-regression/

    df = pd.read_csv('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/encoded_train_data.csv')


    print(f'The df shape is:{df.shape}')
    #print(df.head())
    X=df.drop(['SalePrice'], axis=1)
    print(f'The X shape is:{X.shape}')
    y=df['SalePrice']
    print(f'The y shape is:{y.shape}')




    # define model
    model = XGBRegressor(n_estimators=100,
                         max_depth=4,
                         eta=0.1,
                         subsample=0.1,
                         colsample_bytree=0.1)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate_model

    scores = cross_val_score(model, X, y,
                             scoring=make_scorer(rmsle),  # 'neg_mean_absolute_error'
                             cv=cv,
                             n_jobs=-1)


    model.fit(X,y)
    # force scores to be positive
    scores = abs(scores)
    print('Mean RMSLE: %.3f (%.3f)' % (scores.mean(), scores.std()))

    # saving to file with proper feature names
    xgbfir.saveXgbFI(model, feature_names=X.columns, OutputXlsxFile='feature_analysis.xlsx')

    """
    model.save('encoder.h5')

    history_dict = history.history
    # print(f'keys:{history_dict.keys()}')

    error = np.array(history.history['mae'])
    loss = np.array(history.history['loss'])
    val_error = np.array(history.history['val_mae'])
    val_loss = np.array(history.history['val_loss'])

    start_iter = 20
    plt.plot(loss[start_iter:], 'b', label="Błąd trenowania")
    plt.plot(val_loss[start_iter:], 'bo', label="Błąd walidacji")
    plt.xlabel("Epoki")
    plt.ylabel('Strata')
    plt.legend()
    plt.show()
    """
