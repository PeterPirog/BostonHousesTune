#https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
from tensorflow.keras.datasets import mnist
import tensorflow as tf
#from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import scipy.stats as st
import tensorflow as tf  # tensorflow >= 2.5
from sklearn.model_selection import KFold
from boston_tools import get_Xy
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.utils import plot_model
#from own_metrics import rmsle,mre,rmslemax,mremax
from nn_tools import build_dense2L_model


if __name__ == "__main__":


    # https://github.com/tensorflow/tensorflow/issues/32159

    # print('Is cuda available for trainer:', tf.config.list_physical_devices('GPU'))

    config = {
        # preprocessing parameters
        "n_categories": 6,
        # training parameters
        "batch": 30,
        "lr": 0.01,
        # Layer 1 params
        "hidden1": 180,
        "activation1": "elu",
        "dropout1": 0.2,
        # Layer 2 params
        "hidden2": 150,
        "dropout2": 0.2,
        "activation2": "elu",
        "activation_output": "elu"}

    epochs = 1000



    XY_train_enc_file = f'data/XY_train_enc_{str(config["n_categories"])}_0.05.csv'
    X_train, X_test, y_train, y_test = get_Xy(XY_train_enc_file=XY_train_enc_file)

    #PCA decomposition
    pca = PCA(n_components=0.9999,svd_solver='full')
    pca.fit_transform(X_train)

    print(f'number of features to processing:{len(pca.explained_variance_)}')

    X_train=pca.transform(X_train)
    X_test = pca.transform(X_test)

    #Call function to make dense 2 layer net
    model=build_dense2L_model(config=config,X_train=X_train)

    #Define callbacks
    callbacks_list=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=15),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.1,
                                                         patience=10),
    
                    tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                       monitor='val_rmsle',
                                                       save_best_only=True)]


    history=model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        shuffle=True,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list)

    history_dict=history.history
    #print(f'keys:{history_dict.keys()}')


    error=np.array(history.history['accuracy'])
    loss=np.array(history.history['loss'])
    val_error=np.array(history.history['val_accuracy'])
    val_loss=np.array(history.history['val_loss'])

    start_iter=20
    plt.plot(loss[start_iter:],'b', label="Błąd trenowania")
    plt.plot(val_loss[start_iter:],'bo', label="Błąd walidacji")
    plt.xlabel("Epoki")
    plt.ylabel('Strata')
    plt.legend()
    plt.show()

    result=np.mean(val_loss[-3]+np.abs(val_loss[-3]-loss[-3]))
    print(f'Combined loss is:{result}')
