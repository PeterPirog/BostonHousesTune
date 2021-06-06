#https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
#from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
from boston_tools import make_preprocessing
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from trash.own_metrics import rmsle,mre

if __name__ == "__main__":


    # https://github.com/tensorflow/tensorflow/issues/32159

    # print('Is cuda available for trainer:', tf.config.list_physical_devices('GPU'))

    config = {
        # preprocessing parameters
            #rare labels encoder
        "n_categories": 2,
        "rare_tol":0.01,
            #iterative imputer
        "max_iter":20,
        "iter_tol":0.001,
            #pca decomposition
        "n_components":0.999,
        # training parameters
        "batch": 4,
        "lr": 0.01,
        # Layer 1 params
        "hidden1": 137,#133
        "activation1": "elu",
        "dropout1": 0.08, #0.42
        # Layer 2 params
        "hidden2": 116,#37
        "dropout2": 0.33,#0.2
        "activation2": "elu",
        # Layer 2 params
        "hidden3": 24,  # 36
        "dropout3": 0.15,  # 0.18
        "activation3": "elu",
        "activation_output": "linear"}



    X_train, X_test, y_train, y_test=make_preprocessing(config=config)

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
    # layer 3
    x = tf.keras.layers.Dense(units=config["hidden3"], kernel_initializer='glorot_normal',
                              activation=config["activation3"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout3"])(x)

    #Output layer
    outputs = tf.keras.layers.Dense(units=1, activation=config["activation_output"])(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss=rmsle,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[rmsle])

    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)

    model.compile(
        loss=rmsle,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[mre])

    #Define callbacks
    callbacks_list=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=15),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
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
        shuffle=True,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list)

    history_dict=history.history
    #print(f'keys:{history_dict.keys()}')


    error=np.array(history.history['mre'])
    loss=np.array(history.history['loss'])
    val_error=np.array(history.history['val_mre'])
    val_loss=np.array(history.history['val_loss'])

    start_iter=40
    plt.plot(loss[start_iter:],'b', label="Błąd trenowania")
    plt.plot(val_loss[start_iter:],'bo', label="Błąd walidacji")
    plt.xlabel("Epoki")
    plt.ylabel('Strata')
    plt.legend()
    plt.show()

    result=np.mean(val_loss[-3]+np.abs(val_loss[-3]-loss[-3]))
    print(f'Combined loss is:{result}')
