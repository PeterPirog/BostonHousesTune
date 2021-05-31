
from tensorflow.keras.datasets import mnist
import tensorflow as tf
#from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
from boston_tools import get_Xy

# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y_test):
    return tf.math.sqrt(tf.reduce_mean((tf.math.log1p(y_pred) - tf.math.log1p(y_test)) ** 2))

class Rmsle(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self._data = []

        def on_epoch_end(self, batch, logs={}):
            X_val, y_val = self.validation_data[0], self.validation_data[1]
            y_predict = np.asarray(model.predict(X_val))

            y_val = np.argmax(y_val, axis=1)
            y_predict = np.argmax(y_predict, axis=1)

            self._data.append({
                'rmsle': rmsle(y_val, y_predict),
            })
            return

        def get_data(self):
            return self._data

if __name__ == "__main__":

    # https://github.com/tensorflow/tensorflow/issues/32159

    # print('Is cuda available for trainer:', tf.config.list_physical_devices('GPU'))

    config = {
        # preprocessing parameters
        "n_categories": 3,
        # training parameters
        "batch": 4,
        "lr": 0.01,
        # Layer 1 params
        "hidden1": 65,
        "activation1": "elu",
        "dropout1": 0.1,
        # Layer 2 params
        "hidden2": 30,
        "dropout2": 0.04,  # tune.choice([0.01, 0.02, 0.05, 0.1, 0.2])
        "activation2": "elu",
        "activation_output": "elu"}

    num_classes = 1
    epochs = 1000

    # choose preprocessed features file
    XY_train_enc_file = f'/home/peterpirog/PycharmProjects/BostonHousesTune/data/XY_train_enc_' \
                        f'{str(config["n_categories"])}_0.05.csv'
    X_train, X_test, y_train, y_test = get_Xy(XY_train_enc_file=XY_train_enc_file)

    # define model
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

    outputs = tf.keras.layers.Dense(units=num_classes, activation="elu")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")



    model.compile(
        loss=rmsle,  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(lr=config["lr"]),
        metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])  # accuracy mean_squared_logarithmic_error

    callbacks_list=[tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_logarithmic_error',
                                                     patience=15),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_logarithmic_error',
                                                         factor=0.1,
                                                         patience=10),
                    tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                       monitor='val_mean_squared_logarithmic_error',
                                                       save_best_only=True)]
    model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list)
