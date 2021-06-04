from sklearn.metrics import make_scorer
import tensorflow as tf
import numpy as np
import scipy.stats as st
import math


def mre(y_true, y_pred):
    # implementation of mean relative error in percents
    MRE = np.abs((y_pred - y_true) / y_true)
    MRE = np.mean(MRE)
    return 100 * MRE


def mre_tf(y_true, y_pred):
    # implementation of mean relative error in percents
    # Convert nparrays  to tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    MRE = tf.math.abs((y_pred - y_true) / y_true)
    MRE = tf.reduce_mean(MRE)
    return 100 * MRE


def mre_bound(y_true, y_pred):
    N_samples = np.shape(y_true)[0]
    print(N_samples)
    err = np.abs((y_pred - y_true) / y_true)
    upper_bound = st.t.interval(0.95, len(err) - 1, loc=np.mean(err), scale=st.sem(err))[1]
    return upper_bound


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def rmsle_tf(y_true, y_pred):
    # Convert nparrays  to tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Clip values to prevent log from values below 0
    y_true = tf.clip_by_value(y_true, clip_value_min=0, clip_value_max=np.inf)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=np.inf)
    return tf.math.sqrt(tf.reduce_mean((tf.math.log1p(y_pred) - tf.math.log1p(y_true)) ** 2))


if __name__ == '__main__':
    y_true = np.array([1, 2, 3])
    y_pred = np.array([0.9, 1.9, 1.5])

    err = rmsle(y_true, y_pred)
    print(f'err={err}')

    mre_score = make_scorer(score_func=mre,
                            greater_is_better=False,
                            needs_proba=False)

    mre_bound_score = make_scorer(score_func=mre_bound,
                                  greater_is_better=False,
                                  needs_proba=False)
