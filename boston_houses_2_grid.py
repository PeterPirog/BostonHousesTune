import pandas as pd
import numpy as np
import joblib
import os
#explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.tree import DecisionTreeRegressor,export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error,make_scorer,r2_score,mean_squared_error,mean_absolute_error
from sklearn import ensemble



if __name__ == '__main__':
    verbose=True

    pd.set_option('display.max_columns', None)
    df_train=pd.read_csv('data/train.csv')

    #csv preporcessing
    #del df_train['Id']
    #df_train['MSSubClass']=df_train['MSSubClass'].astype(dtype='object')
    X_train=pd.read_csv('x_train_enc.csv')
    Y_train=df_train['SalePrice'].astype(dtype=np.float32)



    #print(X_train.head())
    #print(X_train.info())

    X_test=pd.read_csv('data/test.csv')

    #csv preporcessing
    #del df_train['Id']
    X_test['MSSubClass']=X_test['MSSubClass'].astype(dtype='object')
    Y_column=X_test['Id'].copy()
    X_test=X_test.drop(['Id'],axis=1)
    #Y_train=df_train['SalePrice'].astype(dtype=np.float32)
    print(X_test.info())

    #LOAD PIPELINE
    loaded_pipe=joblib.load('pipe.pkl', mmap_mode=None)
    X_test_enc=loaded_pipe.transform(X_test)

    X_test_enc.to_csv('x_test_enc.csv', index=False)

    #print(f'X_test_enc={X_test_enc.head()}')
    tree_reg=DecisionTreeRegressor(max_depth=10,criterion='mae',min_samples_split=2,min_samples_leaf=3)

    #hyperparameters
    rf_param_grid=dict(
        max_depth=[5,6,7,8,9,10,12,15,17,20,22,23,25],
        min_samples_split=[2,4,8,10,12,14,16],
        min_samples_leaf=[1,2,3,4,6,8,10]
    )
    scorer=make_scorer(mean_absolute_error,greater_is_better=False)
    clf=GridSearchCV(estimator=tree_reg,
                     param_grid=rf_param_grid,
                     scoring=scorer,
                     cv=5,
                     n_jobs=-1)
    search=clf.fit(X=X_train,y=Y_train)

    print(f'search.best_params_: {search.best_params_}')
    print(f'search.best_score_: {search.best_score_}')
    """
    tree_reg.fit(X=X_train,y=Y_train)

    export_graphviz(tree_reg,out_file='regression_tree.dot',
                    feature_names=X_train.columns,
                    rounded=True,
                    filled=True)
    os.system('dot -Tpng regression_tree.dot -o graph1.png')


    Y_predict=tree_reg.predict(X_test_enc)
    Y_predict=pd.DataFrame(Y_predict,columns=['SalePrice'])

    Y_predict=pd.concat([Y_column,Y_predict],axis=1)
    Y_predict.to_csv('Y_test_submission.csv', index=False)

    #error=100*np.abs((Y_predict-Y_train)/Y_train)
    #print(f'mean={np.mean(error)}, std={np.std(error)}')
    """