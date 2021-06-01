# https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
from tensorflow.keras.datasets import mnist
import tensorflow as tf
# from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import scipy.stats as st
import tensorflow as tf  # tensorflow >= 2.5
from sklearn.model_selection import KFold
from boston_tools import get_Xy


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y_test):
    return tf.math.sqrt(tf.reduce_mean((tf.math.log1p(y_pred) - tf.math.log1p(y_test)) ** 2))


# def rmsle(y_pred, y_test):
#    m = tf.reduce_mean(tf.boolean_mask((tf.math.log1p(y_pred) - tf.math.log1p(y_test)) ** 2, tf.math.is_nan))
#    return tf.math.sqrt(m)

"""
class KerasModel(tf.keras.Model):
    def __init__(self, config):
        super(KerasModel, self).__init__()

        # parameters layer 1
        self.hidden1 = config['hidden1']
        self.activation1 = config['activation1']
        self.dropout1 = config['dropout1']
        # parameters Layer 2
        self.hidden2 = config['hidden2']
        self.activation2 = config['activation2']
        self.dropout2 = config['dropout2']
        # output parameter
        self.activation_output = config["activation_output"]

        # layers definition
        self.normalization_inp = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(units=self.hidden1, kernel_initializer='glorot_normal',
                                            activation=self.activation1)
        self.normalization1 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(self.dropout1)
        self.dense2 = tf.keras.layers.Dense(units=self.hidden2, kernel_initializer='glorot_normal',
                                            activation=self.activation2)
        self.normalization2 = tf.keras.layers.LayerNormalization()
        self.dropout2 = tf.keras.layers.Dropout(self.dropout2)
        self.dense_out = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_normal',
                                               activation=self.activation_output)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.normalization1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.normalization2(x)
        if training:
            x = self.dropout2(x, training=training)
        return self.dense_out(x)
"""

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
        "hidden1": 60,
        "activation1": "elu",
        "dropout1": 0.2,
        # Layer 2 params
        "hidden2": 22,
        "dropout2": 0.1,
        "activation2": "elu",
        "activation_output": "elu"}

    num_classes = 1
    epochs = 1000
    num_folds = 5
    error_per_fold = []
    loss_per_fold = []

    # choose preprocessed features file
    # For UBUNTU
    # XY_train_enc_file = f'/home/peterpirog/PycharmProjects/BostonHousesTune/data/XY_train_enc_' \
    #                    f'{str(config["n_categories"])}_0.05.csv'

    XY_train_enc_file = f'data/XY_train_enc_{str(config["n_categories"])}_0.05.csv'
    X_train, X_test, y_train, y_test = get_Xy(XY_train_enc_file=XY_train_enc_file)
    inputs = np.concatenate((X_train, X_test), axis=0)
    targets = np.concatenate((y_train, y_test), axis=0)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    print(f'kfold split {kfold.split(inputs, targets)}')
    for train, test in kfold.split(inputs, targets):

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
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
            metrics=[rmsle])  # accuracy mean_squared_logarithmic_error tf.keras.metrics.MeanSquaredLogarithmicError()

        # Define callbacks
        callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_rmsle',
                                                           patience=15),
                          tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rmsle',
                                                               factor=0.1,
                                                               patience=10),
                          tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                             monitor='val_rmsle',
                                                             save_best_only=True)]
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        history = model.fit(
            X_train,
            y_train,
            batch_size=config["batch"],
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=callbacks_list)

        error = np.array(history.history['rmsle'])[-1]
        loss = np.array(history.history['loss'])[-1]
        val_error = np.array(history.history['val_rmsle'])[-1]
        val_loss = np.array(history.history['val_loss'])[-1]

        # Increase fold number
        fold_no = fold_no + 1

        print(f'val_error={val_error}')
        error_per_fold.append(val_error)
        loss_per_fold.append(val_loss)

    error_per_fold = np.array(error_per_fold)
    error_per_fold = error_per_fold[~np.isnan(error_per_fold)]  # remove nan values

    upper_bound = \
    st.t.interval(0.95, len(error_per_fold) - 1, loc=np.mean(error_per_fold), scale=st.sem(error_per_fold))[1]
    print(f'Error={error_per_fold},loss={loss_per_fold}')
    print(f'Upper bound of error is:{upper_bound}')
