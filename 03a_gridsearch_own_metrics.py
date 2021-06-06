# https://www.youtube.com/watch?v=5nYqK-HaoKY
import pandas as pd
import joblib

from sklearn import ensemble, model_selection
from sklearn.metrics import make_scorer
from trash.own_metrics import rmsle

if __name__ == "__main__":
    #Rare encoder options
    n_categories=6
    tol=0.05
    xy_train_filename='data/XY_train_enc_'+str(n_categories)+'_'+str(tol)+'.csv'

    df = pd.read_csv(xy_train_filename)
    #X = df.drop(['SalePrice', 'cluster'], axis=1) for data with cluster info
    X = df.drop(['SalePrice'], axis=1)
    y = df['SalePrice']

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    regresor = ensemble.RandomForestRegressor(n_jobs=-1)
    param_grid = {
        "n_estimators": [150, 200, 250, 300, 350, 400,450],
        "max_depth": [4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20,25,30]
        #"criterion": ["mae"]  # mae
    }

    model = model_selection.GridSearchCV(
        estimator=regresor,
        param_grid=param_grid,
        scoring=make_scorer(rmsle, greater_is_better=False),#"neg_mean_absolute_error"
        verbose=10,
        n_jobs=-1,
        cv=kf
    )
    model.fit(X, y)

    print('best_params:', model.best_params_)
    print('best_score:', model.best_score_)
    #print('cv_results:', model.cv_results_)

    joblib.dump(model, 'grid_search_model.pkl')
