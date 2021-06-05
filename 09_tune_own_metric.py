# UBUNTU
from ray.tune.integration.keras import TuneReportCallback
import numpy as np
from numpy import load, save
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder

# Own functions and classes
from Transformers import QuantileTransformerDf, IterativeImputerDf, RareLabelNanEncoder
from own_metrics import rmsle
#from own_metrics import rmsle_tf, mre_tf

# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
"""
def rmsle(y_pred, y_test):

    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=np.inf)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    y_test = tf.clip_by_value(y_test, clip_value_min=0, clip_value_max=np.inf)

    return tf.math.sqrt(tf.reduce_mean((tf.math.log1p(y_pred) - tf.math.log1p(y_test)) ** 2))
"""

def train_boston(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf # tensorflow >= 2.5
    # print('Is cuda available for trainer:', tf.config.list_physical_devices('GPU'))
    epochs = 10000 #this values is not important because training will be stopped by EarlyStopping callback

    #define train path to reuse
    train_enc_file=f'X_train_enc_rare_tol_{config["rare_tol"]}_n_categories_{config["n_categories"]}_max_iter_{config["max_iter"]}_iter_tol_{config["iter_tol"]}.npy'
    train_enc_path='/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/'+train_enc_file
    y_train_path='/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/y_train.npy'

    try:
        X_train_encoded=load(file=train_enc_path)
        Y_train=load(file=y_train_path)
        #load only Y_train
        #df_train = pd.read_csv('/home/peterpirog/PycharmProjects/BostonHousesTune/data/train.csv')
        #Y_train = df_train['SalePrice'].astype(dtype=np.float32)
    except:
        df_train = pd.read_csv('/home/peterpirog/PycharmProjects/BostonHousesTune/data/train.csv')

        # csv preprcessing
        df_train['MSSubClass'] = df_train['MSSubClass'].astype(dtype='category')  # convert fature to categorical
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
                                 initial_strategy='median',
                                 max_iter=config['max_iter'],
                                 tol=config['iter_tol'],
                                 verbose=0, dataframe_as_output=False)
        # Step 5 PCA
        pca = PCA(n_components=config['n_components'], svd_solver='full')

        # STEP 6 MAKE PIPELINE AND TRAIN IT
        pipe = Pipeline([
            ('rare_lab', rle),
            ('one_hot', ohe),
            ('q_trans', q_trans),
            ('imputer', imp),
            ('pca', pca)
        ])

        # Pipeline training
        pipe.fit(X_train)
        X_train_encoded = pipe.transform(X_train).astype(dtype=np.float32)
        #save X_train_encoded array
        save(file=train_enc_path,arr=X_train_encoded)
        save(file=y_train_path, arr=Y_train)


    # STEP 7 SPLITTING DATA FOR KERAS
    X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, Y_train,
                                                        shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)

    # STEP 8 DEFINE NEURAL NET MODEL
    epochs = 1000

    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    # layer 1
    x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                              activation=config["activation1"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    # layer 2
    x = tf.keras.layers.Dense(units=config["hidden2"], kernel_initializer='glorot_normal',
                              activation=config["activation2"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)
    outputs = tf.keras.layers.Dense(units=1, activation=config["activation_output"])(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss=rmsle,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[rmsle])

    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=10),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.1,
                                                           patience=5)]


    history=model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        shuffle=True,
        epochs=epochs,
        verbose=0,
        validation_data=(X_test, y_test),  # tf reduce mean ignore tabnanny
        callbacks=callbacks_list)

    #Creating own metric
    history_dict=history.history
    loss=np.array(history.history['loss'])
    val_loss=np.array(history.history['val_loss'])
    result = np.mean(val_loss[-5] + np.abs(val_loss[-5] - loss[-5]))
    ray.tune.report(_metric=result)

if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import tensorflow as tf

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
        name="exp",
        scheduler=sched_asha,
        # Checkpoint settings
        keep_checkpoints_num=3,
        checkpoint_freq=3,
        checkpoint_at_end=True,
        verbose=3,
        # Optimalization
        #metric="val_rmsle",  # mean_accuracy
        mode="min",  # max
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 500
        },
        num_samples=5000,  # number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='/home/peterpirog/PycharmProjects/BostonHousesTune/ray_results',  # default value is ~/ray_results /root/ray_results/  or ~/ray_results
        resources_per_trial={
            "cpu": 1  # ,
            # "gpu": 0
        },
        config={
            # preprocessing parameters
            # RARE LABEL ENCODER  https://feature-engine.readthedocs.io/en/latest/encoding/RareLabelEncoder.html
            "rare_tol": tune.choice([0.05, 0.02, 0.01]),
            # The minimum frequency a label should have to be considered frequent. Categories with frequencies lower than tol will be grouped
            "n_categories": tune.choice([1, 2, 10]),
            # he minimum number of categories a variable should have for the encoder to find
            # frequent labels. If the variable contains less categories, all of them will be considered frequent.

            # ITERATIVE IMPUTER https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
            "max_iter": tune.choice([10,15,20]),
            "iter_tol": tune.choice([1e-3]),

            # PCA DECOMPOSITION https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
            "n_components": tune.choice([0.999]),
            # number of components such that the amount of variance that needs to be explained is
            # greater than the percentage specified by n_components.

            # NEURAL NET PARAMETERS
            "batch": tune.choice([4, 8]),
            "lr": tune.choice([0.01]),
            # Layer 1 params
            "hidden1": tune.randint(16, 200),
            "activation1": tune.choice(["elu"]),
            "dropout1": tune.quniform(0.05, 0.5, 0.01),
            # Layer 2 params
            "hidden2": tune.randint(16, 200),
            "dropout2": tune.quniform(0.05, 0.5, 0.01),
            "activation2": tune.choice(["elu"]),
            "activation_output": tune.choice(["relu",None])
        }

    )
    print("Best hyperparameters found were: ", analysis.best_config)
