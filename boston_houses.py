import pandas as pd
import numpy as np
import joblib

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from Transformers import QuantileTransformerDf, IterativeImputerDf, RareLabelNanEncoder
from category_encoders import OneHotEncoder

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    df_train=pd.read_csv('train.csv')

    #csv preporcessing
    #del df_train['Id']
    df_train['MSSubClass']=df_train['MSSubClass'].astype(dtype='object')
    X_train=df_train.drop(['Id','SalePrice'],axis=1)
    Y_train=df_train['SalePrice'].astype(dtype=np.float32)



    print(Y_train.head())
    print(X_train.info())
    # STEP 1 -  categorical features rare labels encoding
    rle = RareLabelNanEncoder(categories=None, tol=0.05, minimum_occurrences=None, n_categories=10,
                              max_n_categories=None,
                              replace_with='Rare', impute_missing_label=False, additional_categories_list=None)

    # STEP 2 - categorical features one hot encoding
    #https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/one_hot.py
    ohe=OneHotEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_missing='return_nan', #options are 'error', 'return_nan', 'value', and 'indicator'.
                       handle_unknown='return_nan',#options are 'error', 'return_nan', 'value', and 'indicator'
                       use_cat_names=False)

    # STEP 3 - numerical values quantile transformation with skewness removing
    q_trans = QuantileTransformerDf(n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False,
                                    subsample=1e5, random_state=42, copy=True, dataframe_as_output=True,dtype=np.float32)

    # STEP 4 - missing values multivariate imputation
    imp = IterativeImputerDf(min_value=0,  # values from 0 to 1 for categorical for numeric
                             max_value=1,
                             random_state=42,
                             max_iter=100,
                             tol=1e-4,
                             verbose=2, dataframe_as_output=True)

    pipe = Pipeline([
        ('rare_lab', rle),
        ('one_hot', ohe),
        ('q_trans', q_trans),
        ('imputer', imp)
    ])

    pipe.fit(X_train)
    X_train_encoded = pipe.transform(X_train)
    print(f'\ndf_original=\n{X_train}')
    print(f'df_imputed=\n{X_train_encoded.head()}')
    #save encoded data to csv
    X_train_encoded.to_csv('x_train_enc.csv',index=False)

    #SAVE PIPELINE
    joblib.dump(pipe, 'pipe.pkl')

    #LOAD PIPELINE
    loaded_pipe=joblib.load('pipe.pkl', mmap_mode=None)
"""
    #Prepare test data
    X_test=[['A', 1.1, np.nan, 0.3, 1.5],[np.nan, 1.0, np.nan, 0.5, 2.5]]
    df_test = pd.DataFrame(data=X_test, columns=columns)
    df_test_imputed=loaded_pipe.transform(df_test)

    print(f'df_test_original=\n{df_test}')
    print(f'df_test_imputed=\n{df_test_imputed.head()}')
"""