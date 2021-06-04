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



# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y_test):

    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=np.inf)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    y_test = tf.clip_by_value(y_test, clip_value_min=0, clip_value_max=np.inf)

    return tf.math.sqrt(tf.reduce_mean((tf.math.log1p(y_pred) - tf.math.log1p(y_test)) ** 2))

if __name__ == "__main__":

    # https://github.com/tensorflow/tensorflow/issues/32159

    # print('Is cuda available for trainer:', tf.config.list_physical_devices('GPU'))

    config = {
        # preprocessing parameters
        "n_categories": 6,
        # training parameters
        "batch": 8,
        "lr": 0.01,
        # Layer 1 params
        "hidden1": 120,
        "activation1": "elu",
        "dropout1": 0.4, #0.08
        # Layer 2 params
        "hidden2": 97,
        "dropout2": 0.4,#0.075
        "activation2": "elu",
        "activation_output": "elu"}

    epochs = 1000


    # choose preprocessed features file
    #For UBUNTU
    #XY_train_enc_file = f'/home/peterpirog/PycharmProjects/BostonHousesTune/data/XY_train_enc_' \
    #                    f'{str(config["n_categories"])}_0.05.csv'

    XY_train_enc_file = f'data/XY_train_enc_{str(config["n_categories"])}_0.05.csv'
    X_train, X_test, y_train, y_test = get_Xy(XY_train_enc_file=XY_train_enc_file)

    #PCA decomposition
    pca = PCA(n_components='mle',svd_solver='full')
    pca.fit_transform(X_train)

    print(f'number of features to processing:{len(pca.explained_variance_)}')

    X_train=pca.transform(X_train)
    X_test = pca.transform(X_test)
    print(f'X_train.shape={X_train.shape}')


    # define model
    #inputs = tf.keras.layers.Input(shape=((X_train.shape[1],1)))
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))
    x=tf.keras.layers.Reshape(target_shape=(None,X_train.shape[1],1))(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Conv1D(filters=16,kernel_size=5,activation='selu',kernel_initializer='lecun_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=5,activation='selu',kernel_initializer='lecun_normal')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    #x=tf.keras.layers.MaxPool1D(pool_size=2)(x)

    """
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

    """
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(units=1, activation='selu',kernel_initializer='lecun_normal')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.summary()

    model.compile(
        loss=rmsle,  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[rmsle])  # accuracy mean_squared_logarithmic_error tf.keras.metrics.MeanSquaredLogarithmicError()

    #Define callbacks
    callbacks_list=[tf.keras.callbacks.EarlyStopping(monitor='val_rmsle',
                                                     patience=15),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rmsle',
                                                         factor=0.1,
                                                         patience=10)]#,
    """
                    tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                       monitor='val_rmsle',
                                                       save_best_only=True)]
    """

    history=model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list)

    history_dict=history.history
    #print(f'keys:{history_dict.keys()}')


    error=np.array(history.history['rmsle'])
    loss=np.array(history.history['loss'])
    val_error=np.array(history.history['val_rmsle'])
    val_loss=np.array(history.history['val_loss'])

    start_iter=20
    plt.plot(loss[start_iter:],'b', label="Błąd trenowania")
    plt.plot(val_loss[start_iter:],'bo', label="Błąd walidacji")
    plt.xlabel("Epoki")
    plt.ylabel('Strata')
    plt.legend()
    plt.show()
