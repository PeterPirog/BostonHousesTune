import numpy as np
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


def make_xgb_preprocessing(config):
    x_encoded_file = f'x_train_enc_rare_tol_{config["rare_tol"]}_n_categories_{config["n_categories"]}_max_iter_' \
                     f'{config["max_iter"]}_iter_tol_{config["iter_tol"]}.csv'
    y_output_file=f'y_train.csv'

    x_train_enc_path = '/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/' + x_encoded_file
    y_train_path = '/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/' + y_output_file

    try:
        df_x_enc=pd.read_csv(x_train_enc_path)
        df_y = pd.read_csv(y_train_path)


    except:
        df_train = pd.read_csv('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/preprocessed_train_data.csv')
        X_train=df_train.drop(['SalePrice'], axis=1).copy()
        Y_train=df_train['SalePrice'].copy()
        #print(X_train.head())
        # print(Y_train.head())



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
                            handle_missing='return_nan',
                            # options are 'error', 'return_nan', 'value', and 'indicator'.
                            handle_unknown='return_nan',
                            # options are 'error', 'return_nan', 'value', and 'indicator'
                            use_cat_names=False)

        # STEP 3 - numerical values quantile transformation with skewness removing
        q_trans = QuantileTransformerDf(n_quantiles=1000, output_distribution='uniform',
                                        ignore_implicit_zeros=False,
                                        subsample=1e5, random_state=42, copy=True, dataframe_as_output=True,
                                        dtype=np.float32)

        # STEP 4 - missing values multivariate imputation
        imp = IterativeImputerDf(min_value=0,  # values from 0 to 1 for categorical for numeric
                                 max_value=1,
                                 random_state=42,
                                 initial_strategy='median',
                                 max_iter=config['max_iter'],
                                 tol=config['iter_tol'],
                                 verbose=3, dataframe_as_output=True)  # if config["output"] == 'df' else False

        # STEP 5 MAKE PIPELINE AND TRAIN IT
        pipeline = Pipeline([
            ('rare_lab', rle),
            ('one_hot', ohe),
            ('q_trans', q_trans),
            ('imputer', imp)
        ])

        # Pipeline training

        X_train_encoded=pipeline.fit_transform(X_train)
        print('X_train_encoded',type(X_train_encoded))
        # save X_train_encoded array
        #save(file=train_enc_path, arr=X_train_encoded)
        #save(file=y_train_path, arr=Y_train)

        # save trained pipeline
        #joblib.dump(pipeline, pipeline_path)
        df_train_encoded = pd.concat([X_train_encoded,Y_train], axis=1)
        print(df_train_encoded.head())
        df_train_encoded.to_csv(path_or_buf='/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/encoded_train_data.csv',
                      sep=',',
                      header=True,
                      index=False)
        df_train_encoded.to_excel('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/encoded_train_data.xlsx',
                        sheet_name='output_data',
                        index=False)

    # STEP 7 SPLITTING DATA FOR KERAS
    X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, Y_train,
                                                        shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)

    return X_train_encoded,Y_train
    #return X_train, X_test, y_train, y_test


def rmsle(y_true, y_pred, **kwargs):
    # Implementation of rmsle error
    # Convert nparrays  to tensors

    # Clip values to prevent log from values below 0
    y_true = np.clip(y_true, a_min=0, a_max=np.inf)
    y_pred = np.clip(y_pred, a_min=0, a_max=np.inf)
    return -np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


if __name__ == "__main__":
    # https://machinelearningmastery.com/xgboost-for-regression/
    config = {
        # PREPROCESSING
        # Rare label encoder
        "rare_tol": 0.05,
        "n_categories": 1,
        # Iterative imputer
        "max_iter": 30,
        "iter_tol": 0.01,
        "output": 'df'
    }

    #X_train, X_test, y_train, y_test = make_xgb_preprocessing(config=config)
    X,y = make_xgb_preprocessing(config=config)

    #X = pd.concat([X_train, X_test], axis=0)
    #y = pd.concat([y_train, y_test], axis=0)

    print(f'The input shape is:{X.shape}')
    print(X.head())
    """
    # define model
    model = XGBRegressor(n_estimators=200,
                         max_depth=7,
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
    """

    #model.fit(X,y)
    # force scores to be positive
    #scores = abs(scores)
    #print('Mean RMSLE: %.3f (%.3f)' % (scores.mean(), scores.std()))

    # saving to file with proper feature names
    #xgbfir.saveXgbFI(model, feature_names=X.columns, OutputXlsxFile='X_fi.xlsx')


    """
    model.save('encoder.h5')

    history_dict = history.history
    # print(f'keys:{history_dict.keys()}')

    error = np.array(history.history['mae'])
    loss = np.array(history.history['loss'])
    val_error = np.array(history.history['val_mae'])
    val_loss = np.array(history.history['val_loss'])

    start_iter = 20
    plt.plot(loss[start_iter:], 'b', label="B????d trenowania")
    plt.plot(val_loss[start_iter:], 'bo', label="B????d walidacji")
    plt.xlabel("Epoki")
    plt.ylabel('Strata')
    plt.legend()
    plt.show()
    """
