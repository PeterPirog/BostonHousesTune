# https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/

import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
from boston_tools import get_Xy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nn_tools import build_conv3L_model,rmsle,make_preprocessing


if __name__ == "__main__":
    config = {
        #PREPROCESSING
            #Rare label encoder
        "rare_tol":0.01,
        "n_categories": 10,
            #Iterative imputer
        "max_iter":20,
        "iter_tol":0.001,
            #PCA decomposition
        "n_components":0.999,
        #NEURAL NET
        # training parameters
        "batch": 8,
        "lr": 0.01,
        # Layer 1 params
        "filter1": 64,
        "kernel1": 7,
        "activation_c1": "elu",
        "dropout_c1": 0.5,
        # Layer 2 params
        "filter2": 32,
        "kernel2": 3,
        "activation_c2": "elu",
        # Layer 3 params
        "filter3": 16,
        "kernel3": 2,
        "activation_c3": "elu",
        # output layers
        "hidden_conv": 32,
        "activation_hidden_conv": "elu",
        "activation_output": "linear"}

    """

    

    XY_train_enc_file = f'data/XY_train_enc_{str(config["n_categories"])}_0.05.csv'
    X_train, X_test, y_train, y_test = get_Xy(XY_train_enc_file=XY_train_enc_file)

    # PCA decomposition
    pca = PCA(n_components=0.999, svd_solver='full')
    pca.fit_transform(X_train)


    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    """

    X_train, X_test, y_train, y_test=make_preprocessing(config=config)

    epochs = 1000
    model = build_conv3L_model(config=config, X_train=X_train)
    model.summary()
    # Define callbacks
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=15),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.1,
                                                           patience=10),
                      tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                         monitor='val_loss',
                                                         save_best_only=True)]


    history = model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list)

    history_dict = history.history
    # print(f'keys:{history_dict.keys()}')

    error = np.array(history.history['rmsle'])
    loss = np.array(history.history['loss'])
    val_error = np.array(history.history['val_rmsle'])
    val_loss = np.array(history.history['val_loss'])

    start_iter = 20
    plt.plot(loss[start_iter:], 'b', label="Błąd trenowania")
    plt.plot(val_loss[start_iter:], 'bo', label="Błąd walidacji")
    plt.xlabel("Epoki")
    plt.ylabel('Strata')
    plt.legend()
    plt.show()
