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

            #DENSE PART
        # Layer 1 params
        "hidden1": 100,#142
        "activation1": "elu",
        #"dropout1": 0.3,
        # Layer 2 params
        "hidden_enc": 50,
        #"dropout2": 0.3,
        "activation_enc": "sigmoid",
        # Layer 3 params
        #"hidden3": 10,
        #"dropout3": 0.28,
        #"activation3": "elu",
        #OUTPUT
        "activation_output": "linear"}


    X_train, X_test, y_train, y_test=make_preprocessing(config=config)
    print(f'The input shape is:{X_train.shape[1]}')

    epochs = 1000

    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))
    #x = tf.keras.layers.LayerNormalization()(inputs)
    # layer 1
    x = tf.keras.layers.Dense(units=config['hidden1'], kernel_initializer='glorot_normal',
                              activation=config['activation_enc'])(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    # layer 2
    enc_layer = tf.keras.layers.Dense(units=config['hidden_enc'], kernel_initializer='glorot_normal',
                              activation=config['activation_enc'])(x)
    #x = tf.keras.layers.LayerNormalization()(enc_layer)
    x = tf.keras.layers.Dense(units=config['hidden1'], kernel_initializer='glorot_normal',
                              activation=config['activation1'])(x)
    # Output layer
    outputs = tf.keras.layers.Dense(units=X_train.shape[1], activation='linear')(x)


    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")
    encoder = tf.keras.Model(inputs=inputs, outputs=enc_layer, name="encoder")

    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics='mae')


    model.summary()
    # Define callbacks
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='loss',  #'val_loss
                                                       patience=15),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', #'val_loss
                                                           factor=0.8,
                                                           patience=10),
                      tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                         monitor='val_loss',
                                                         save_best_only=True)]
    tf.keras.utils.plot_model(model,to_file="model.png")

    history = model.fit(
        X_train,
        X_train,
        batch_size=config["batch"],
        shuffle=True,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, X_test),
        callbacks=callbacks_list)

    model.save('encoder.h5')

    history_dict = history.history
    # print(f'keys:{history_dict.keys()}')

    error = np.array(history.history['mae'])
    loss = np.array(history.history['loss'])
    val_error = np.array(history.history['val_mae'])
    val_loss = np.array(history.history['val_loss'])

    start_iter = 20
    plt.plot(loss[start_iter:], 'b', label="Błąd trenowania")
    plt.plot(val_loss[start_iter:], 'bo', label="Błąd walidacji")
    plt.xlabel("Epoki")
    plt.ylabel('Strata')
    plt.legend()
    plt.show()
