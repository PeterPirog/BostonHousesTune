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



def make_xgb_preprocessing(config):
    train_enc_file = f'X_train_enc_rare_tol_{config["rare_tol"]}_n_categories_{config["n_categories"]}_max_iter_{config["max_iter"]}_iter_tol_{config["iter_tol"]}_no_pca.npy'
    train_enc_path = '/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/' + train_enc_file
    y_train_path = '/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/y_train.npy'

    pipeline_file = f'pipeline_{config["rare_tol"]}_n_categories_{config["n_categories"]}_max_iter_{config["max_iter"]}_iter_tol_{config["iter_tol"]}_no_pca.pkl'
    pipeline_path = '/home/peterpirog/PycharmProjects/BostonHousesTune/data/' + pipeline_file

    try:
        joblib.load(pipeline_path, mmap_mode=None)
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
        pipeline.fit(X_train)
        X_train_encoded = pipeline.transform(X_train).astype(dtype=np.float32)
        # save X_train_encoded array
        save(file=train_enc_path, arr=X_train_encoded)
        save(file=y_train_path, arr=Y_train)

        # save trained pipeline
        joblib.dump(pipeline, pipeline_path)

    # STEP 7 SPLITTING DATA FOR KERAS
    X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, Y_train,
                                                        shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


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
        "n_categories": 2,
        # Iterative imputer
        "max_iter": 30,
        "iter_tol": 0.001,
        "output": 'df'
    }

    X_train, X_test, y_train, y_test = make_xgb_preprocessing(config=config)
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    print(f'The input shape is:{X.shape}')
    print(X.head())

    # define model
    model = XGBRegressor(n_estimators=200,
                         max_depth=7,
                         eta=0.1,
                         subsample=0.1,
                         colsample_bytree=0.1)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate_model
    """
    scores = cross_val_score(model, X, y,
                             scoring=make_scorer(rmsle),  # 'neg_mean_absolute_error'
                             cv=cv,
                             n_jobs=-1)
    """

    model.fit(X,y)
    # force scores to be positive
    #scores = abs(scores)
    #print('Mean RMSLE: %.3f (%.3f)' % (scores.mean(), scores.std()))

    # saving to file with proper feature names
    xgbfir.saveXgbFI(model, feature_names=X.columns, OutputXlsxFile='X_fi.xlsx')


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
