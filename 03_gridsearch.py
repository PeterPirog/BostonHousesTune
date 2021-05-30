# https://www.youtube.com/watch?v=5nYqK-HaoKY
import pandas as pd
import numpy as np
import joblib

from sklearn import ensemble, metrics, model_selection

if __name__ == "__main__":
    df = pd.read_csv("data/XY_enc_cluster.csv")
    X = df.drop(['SalePrice', 'cluster'], axis=1)
    y = df['SalePrice']
    test_fold = df['cluster'].to_numpy()

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    regresor = ensemble.RandomForestRegressor(n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 150, 200, 250, 300, 350, 400],
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20],
        "criterion": ["mae"]  # mae
    }

    model = model_selection.GridSearchCV(
        estimator=regresor,
        param_grid=param_grid,
        # scoring="accuracy",
        verbose=10,
        n_jobs=-1,
        cv=kf
    )
    model.fit(X, y)

    print('best_params:', model.best_params_)
    print('best_score:', model.best_score_)
    print('cv_results:', model.cv_results_)

    joblib.dump(model, 'grid_search_model.pkl')
