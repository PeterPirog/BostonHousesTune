# https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/

import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
from boston_tools import get_Xy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nn_tools import build_mixed_model,rmsle,make_preprocessing


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
            #CONVOLUTION PART
        # Layer 1 params
        "filter1": 64,
        "kernel1": 9,
        "activation_c1": "elu",
        "dropout_c1": 0.24,
        # Layer 2 params
        "filter2": 16,
        "kernel2": 7,
        "activation_c2": "elu",
        # Layer 3 params
        "filter3": 16,
        "kernel3": 2,
        "activation_c3": "elu",
        "dropout_c3": 0.04,
        # output layers
        "hidden_conv": 87,
        "activation_hidden_conv": "elu",
            #DENSE PART
        # Layer 1 params
        "hidden1": 120,#142
        "activation1": "elu",
        "dropout1": 0.44,
        # Layer 2 params
        "hidden2": 26,
        "dropout2": 0.3,
        "activation2": "elu",
        # Layer 3 params
        "hidden3": 94,
        "dropout3": 0.28,
        "activation3": "elu",
        #OUTPUT
        "activation_output": "linear"}


    X_train, X_test, y_train, y_test=make_preprocessing(config=config)

    epochs = 1000
    model = build_mixed_model(config=config, X_train=X_train)
    model.summary()
    # Define callbacks
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=15),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.8,
                                                           patience=10),
                      tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                         monitor='val_loss',
                                                         save_best_only=True)]
    tf.keras.utils.plot_model(model,to_file="model.png")

    history = model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        shuffle=True,
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
