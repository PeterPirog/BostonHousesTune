# UBUNTU
from ray.tune.integration.keras import TuneReportCallback
from ray import tune
import numpy as np
from numpy import load, save
import pandas as pd
#from tensorflow.keras.activations import relu,linear,exponential

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder

# Own functions and classes
#from Transformers import QuantileTransformerDf, IterativeImputerDf, RareLabelNanEncoder
from nn_tools import build_mixed_model,rmsle,make_preprocessing

def train_boston(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf # tensorflow >= 2.5


    #if config['hidden1']<config['hidden2'] or config['hidden2']<config['hidden3']:
    #    ray.tune.report(_metric=1)
    #    exit()

    epochs = 10000 #this values is not important because training will be stopped by EarlyStopping callback

    X_train, X_test, y_train, y_test=make_preprocessing(config=config)

    epochs = 1000
    model = build_mixed_model(config=config, X_train=X_train)
    # Define callbacks
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=15),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.1,
                                                           patience=10)]
                      #tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                      #                                   monitor='val_loss',
                      #                                   save_best_only=True),
                      #TuneReportCallback({'val_loss':'val_loss'})]

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
        name="exp_mix_metric",
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
        num_samples=2000,  # number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='/home/peterpirog/PycharmProjects/BostonHousesTune/ray_results',
        resources_per_trial={
            "cpu": 1  # ,
            # "gpu": 0
        },
        config={
            # preprocessing parameters
            # RARE LABEL ENCODER  https://feature-engine.readthedocs.io/en/latest/encoding/RareLabelEncoder.html
            # The minimum frequency a label should have to be considered frequent. Categories with frequencies lower than tol will be grouped
            "rare_tol": tune.choice([0.01]),
            # The minimum number of categories a variable should have for the encoder to find
            # frequent labels. If the variable contains less categories, all of them will be considered frequent.
            "n_categories": tune.choice([2]),


            # ITERATIVE IMPUTER https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
            "max_iter": tune.choice([20]),
            "iter_tol": tune.choice([0.001]),

            # PCA DECOMPOSITION https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
            "n_components": tune.choice([0.999]),
            # number of components such that the amount of variance that needs to be explained is
            # greater than the percentage specified by n_components.

            # NEURAL NET PARAMETERS
            # training parameters
            "batch": tune.choice([8]),
            "lr": tune.choice([0.01]),
                #CONVOLUTION PART
            # Layer conv 1 params
            "filter1": tune.choice([16,32,64]),
            "kernel1": tune.choice([3,5,7,9,11]),
            "activation_c1": tune.choice(["elu"]),
            "dropout_c1": tune.quniform(0.04, 0.5, 0.02),
            # Layer 2 params
            "filter2": tune.choice([16,32,64]),
            "kernel2":tune.choice([3,5,7,9,11]),
            "activation_c2": tune.choice(["elu"]),
            # Layer 3 params
            "filter3": tune.choice([16,32,64]),
            "kernel3": tune.choice([3,5,7,9,11]),
            "activation_c3": tune.choice(["elu"]),
            "dropout_c3": tune.quniform(0.04, 0.5, 0.02),
            # output layers
            "hidden_conv": tune.randint(16, 101),
            "activation_hidden_conv": tune.choice(["elu"]),
                # DENSE PART
            # Layer 1 params
            "hidden1": tune.randint(16, 151),
            "activation1": tune.choice(["elu"]),
            "dropout1": tune.quniform(0.04, 0.5, 0.02),
            # Layer 2 params
            "hidden2": tune.randint(16, 151),
            "dropout2": tune.quniform(0.04, 0.5, 0.02),
            "activation2": tune.choice(["elu"]),
            # Layer 3 params
            "hidden3": tune.randint(16, 151),
            "dropout3": tune.quniform(0.04, 0.5, 0.02),
            "activation3": tune.choice(["elu"]),
                #OUTPUT
            "activation_output": tune.choice(["linear"])}
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    #tensorboard --logdir /home/peterpirog/PycharmProjects/BostonHousesTune/ray_results/exp_mix_metric --bind_all
