import pandas as pd
import numpy as np
import joblib

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from Transformers import QuantileTransformerDf, IterativeImputerDf, RareLabelNanEncoder
from category_encoders import OneHotEncoder

#DESCRIPTION
#UPGRADEED BY DIFFERENT PREPROCESSING PIPELINES

if __name__ == '__main__':
    verbose=False

    #Rare encoder options
    n_categories=6
    tol=0.05
    x_train_filename='data/X_train_enc_'+str(n_categories)+'_'+str(tol)
    xy_train_filename='data/XY_train_enc_'+str(n_categories)+'_'+str(tol)
    print(f"preprocessing rare values with n_categories={n_categories}, tol={tol}")


    # make all dataframe columns visible
    pd.set_option('display.max_columns', None)

    df_train = pd.read_csv('data/train.csv')

    # csv preprcessing
    df_train['MSSubClass'] = df_train['MSSubClass'].astype(dtype='category') #convert fature to categorical
    X_train = df_train.drop(['Id', 'SalePrice'], axis=1)
    Y_train = df_train['SalePrice'].astype(dtype=np.float32)

    if verbose:
        print(X_train.head())
        print(X_train.info())
    # STEP 1 -  categorical features rare labels encoding
    rle = RareLabelNanEncoder(categories=None, tol=tol, minimum_occurrences=None, n_categories=n_categories,
                              max_n_categories=None,
                              replace_with='Rare', impute_missing_label=False, additional_categories_list=None)

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
                             max_iter=25,
                             tol=1e-5,
                             verbose=2, dataframe_as_output=True)

    pipe = Pipeline([
        ('rare_lab', rle),
        ('one_hot', ohe),
        ('q_trans', q_trans),
        ('imputer', imp)
    ])

    #Pipeline training
    pipe.fit(X_train)
    X_train_encoded = pipe.transform(X_train)
    X_train_encoded=X_train_encoded.astype(dtype='float32')

    if verbose:
        print(f'\ndf_original=\n{X_train}')
        print(f'df_imputed=\n{X_train_encoded.head()}')
    # save encoded data to csv and xls files
    X_train_encoded.to_csv(x_train_filename+'.csv', index=False)
    X_train_encoded.to_excel(x_train_filename+'.xlsx',sheet_name='X_encoded',index=False)

    #concatenate columns of encoded X_train with Y_train
    XY_encoded=pd.concat([X_train_encoded,Y_train],axis=1).astype('float32')
    XY_encoded.to_csv(xy_train_filename+'.csv', index=False)
    XY_encoded.to_excel(xy_train_filename+'.xlsx',sheet_name='XY_encoded',index=False)

    # SAVE PIPELINE
    joblib.dump(pipe, 'preprocessing_pipe_'+str(n_categories)+'_'+str(tol)+'.pkl')

    # LOAD PIPELINE
    #loaded_pipe = joblib.load('preprocessing_pipe.pkl', mmap_mode=None)

