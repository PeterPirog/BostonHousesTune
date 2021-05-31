from tensorflow.keras.datasets import mnist
from ray.tune.integration.keras import TuneReportCallback
import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
from boston_tools import get_Xy


# https://stackoverflow.com/questions/37657260/how-to-implement-custom-metric-in-keras

# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y_test):
    return tf.math.sqrt(tf.reduce_mean((tf.math.log1p(y_pred) - tf.math.log1p(y_test)) ** 2))


def train_boston(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf
    # print('Is cuda available for trainer:', tf.config.list_physical_devices('GPU'))
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
    x = tf.keras.layers.Dense(units=config["hidden1"],kernel_initializer='glorot_normal',
                              activation=config["activation1"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    # layer 2
    x = tf.keras.layers.Dense(units=config["hidden2"],kernel_initializer='glorot_normal',
                              activation=config["activation2"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)

    outputs = tf.keras.layers.Dense(units=num_classes, activation="elu")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

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

    metric = Rmsle()
    model.compile(
        loss=rmsle,  # mean_squared_logarithmic_error "mse"
        optimizer=tf.keras.optimizers.Adam(lr=config["lr"]),
        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])  # accuracy mean_squared_logarithmic_error

    model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=epochs,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=[TuneReportCallback({
            "mean_absolute_percentage_error": "val_mean_absolute_percentage_error"
            ##optional values ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        })])  # "mean_accuracy": "val_accuracy"


if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    import tensorflow as tf

    print('Is cuda available for container:', tf.config.list_physical_devices('GPU'))

    # mnist.load_data()  # we do this on the driver because it's not threadsafe
    """
    ray.init(num_cpus=8,
             num_gpus=1,
             include_dashboard=True,  # if you use docker use docker run -p 8265:8265 -p 6379:6379
             dashboard_host='0.0.0.0')
    """
    # ray.init(address='auto', _redis_password='5241590000000000')
    try:
        ray.init()
    except:
        ray.shutdown()
        ray.init()

    sched_asha = ASHAScheduler(time_attr="training_iteration",
                               max_t=500,
                               grace_period=10,
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
        metric="mean_absolute_percentage_error",  # mean_accuracy
        mode="min",  # max
        stop={
            # "mean_accuracy": 0.99,
            "training_iteration": 500
        },
        num_samples=500, #number of samples from hyperparameter space
        reuse_actors=True,
        # Data and resources
        local_dir='~/ray_results',  # default value is ~/ray_results /root/ray_results/
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        config={
            # preprocessing parameters
            "n_categories": tune.choice([1, 2, 3, 6]),
            # training parameters
            "batch": tune.choice([4, 8]),
            "lr": tune.loguniform(1e-5, 1e-2),
            # Layer 1 params
            "hidden1": tune.randint(16, 129),
            "activation1": tune.choice(["elu"]),
            "dropout1": tune.uniform(0.01, 0.15),
            # Layer 2 params
            "hidden2": tune.randint(16, 129),
            "dropout2": tune.uniform(0.01, 0.15),  # tune.choice([0.01, 0.02, 0.05, 0.1, 0.2])
            "activation2": tune.choice(["elu"]),
            "activation_output": tune.choice(["elu"]) #,tf.keras.activations.exponential
            # "layers": tune.choice([2]),tf.keras.activations.exponential

        }

    )
    print("Best hyperparameters found were: ", analysis.best_config)