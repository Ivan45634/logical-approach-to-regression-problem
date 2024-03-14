import time
import math
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed


def _without_replacement(rng, n_max, size):
    bag = set()
    while size > 0:
        vals = np.unique(rng.integers(0, n_max, size))
        cur = size if vals.shape[0] > size else vals.shape[0]
        size -= cur
        for val in vals[:cur]:
            bag.add(val)
    return np.array(list(bag))


# def _generate_indices(rng, n, size, bootstrap):
#     if bootstrap:
#         return rng.integers(0, n, size)
#     return _without_replacement(rng, n, size)


# def _generate_bagging_indices(rng, n, fraction, bootstrap):
#     size = math.ceil(fraction * n)
#     return _generate_indices(rng, n, size, bootstrap)

def generate_bagging_indices(rng, n_samples, sample_fraction, bootstrap):
    """Generates bootstrap indices for bagging."""
    if bootstrap:
        indices = rng.choice(n_samples, size=int(n_samples*sample_fraction), replace=True)
    else:
        indices = rng.choice(n_samples, size=int(n_samples*sample_fraction), replace=False)
    return indices


def _fit_k_estimators(k_estimators, seeds, ensemble, X, y, G, coefs, trace):
    estimators = []
    estimators_features = []
    estimators_fit_time = []

    for i in range(k_estimators):
        rng = np.random.default_rng(seeds[i])
        samples = _generate_bagging_indices(
            rng, X.shape[0], ensemble.sub_sample_size, ensemble.bootstrap)
        features = _generate_bagging_indices(
            rng, X.shape[1], ensemble.sub_feature_size, ensemble.bootstrap)

        estimator = ensemble.make_estimator(random_state=seeds[i])

        if trace:
            start = time.time()
            estimator.fit(X[samples][:, features], G[samples], coefs)

            estimators_fit_time.append(time.time() - start)
        else:
            estimator.fit(X[samples][:, features], G[samples], coefs)

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features, estimators_fit_time


def _predict_k_estimators(estimators, estimators_features, X):
    return sum(
        estimator.predict(X[:, features])
        for estimator, features in zip(estimators, estimators_features)
    )
