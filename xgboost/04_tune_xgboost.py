# UBUNTU
#from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred, **kwargs):
    # Implementation of rmsle error
    # Convert nparrays  to tensors

    # Clip values to prevent log from values below 0
    y_true = np.clip(y_true, a_min=0, a_max=np.inf)
    y_pred = np.clip(y_pred, a_min=0, a_max=np.inf)
    return -np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def train_boston(config):
    df = pd.read_csv('/home/peterpirog/PycharmProjects/BostonHousesTune/preprocessing/encoded_train_data.csv')

    X = df.drop(['SalePrice'], axis=1)
    y = df['SalePrice']

    # define model
    # https: // towardsdatascience.com / selecting - optimal - parameters -for -xgboost - model - training - c7cd9ed5e45e
    model = XGBRegressor(n_estimators=config["n_estimators"],
                         max_depth=config["max_depth"],
                         eta=config["eta"],
                         subsample=config["subsample"],
                         colsample_bytree=1.0)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate_model


    scores = cross_val_score(model, X, y,
                             scoring=make_scorer(rmsle),  # 'neg_mean_absolute_error'
                             cv=cv,
                             n_jobs=-1)

    # force scores to be positive
    scores = abs(scores)

    print('Mean RMSLE: %.4f (%.4f)' % (scores.mean(), scores.std()))

    # Creating own metric
    ray.tune.report(_metric=scores.mean()+2*scores.std())


if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch

    try:
        ray.init()
    except:
        ray.shutdown()
        ray.init()

    sched_asha = ASHAScheduler(time_attr="training_iteration",
                               max_t=500,
                               grace_period=16,
                               # mode='max', #find maximum, do not define here if you define in tune.run
                               reduction_factor=3,
                               # brackets=1
                               )

    analysis = tune.run(
        train_boston,
        search_alg=HyperOptSearch(),
        name="xgboost",
        #scheduler=sched_asha, - no need scheduler if there is no iterations
        # Checkpoint settings
        keep_checkpoints_num=3,
        checkpoint_freq=3,
        checkpoint_at_end=True,
        verbose=3,
        # Optimalization
        # metric="val_rmsle",  # mean_accuracy
        mode="min",  # max
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 100
        },
        num_samples=200,  # number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='/home/peterpirog/PycharmProjects/BostonHousesTune/xgboost/ray_results',# default value is ~/ray_results /root/ray_results/  or ~/ray_results
        resources_per_trial={
            "cpu": 16  # ,
            # "gpu": 0
        },
        config={
            "n_estimators": tune.randint(10, 250),
            "max_depth": tune.randint(1, 10),
            "eta": tune.quniform(0.1, 1.0, 0.1),
            "subsample": tune.quniform(0.1, 1.0, 0.1)
        }

    )
    print("Best hyperparameters found were: ", analysis.best_config)
    # tensorboard --logdir /home/peterpirog/PycharmProjects/BostonHousesTune/xgboost/ray_results/xgboost --bind_all
    #https://towardsdatascience.com/beyond-grid-search-hypercharge-hyperparameter-tuning-for-xgboost-7c78f7a2929d