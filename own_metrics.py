from sklearn.metrics import make_scorer
import numpy as np
import scipy.stats as st
import math


def mre(y_true, y_pred):
    # implementation of mean relative error 1.0 is 100% of error
    MRE = np.abs((y_pred - y_true) / y_true)
    MRE = np.mean(MRE)
    return MRE


def mre_bound(y_true, y_pred):
    N_samples = np.shape(y_true)[0]
    print(N_samples)
    err = np.abs((y_pred - y_true) / y_true)
    upper_bound = st.t.interval(0.95, len(err) - 1, loc=np.mean(err), scale=st.sem(err))[1]
    return upper_bound


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y_test) :
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))


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
