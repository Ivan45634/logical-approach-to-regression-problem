import os
import pandas as pd
import numpy as np
import seaborn as sns
from copy import copy
from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, RegressorMixin
from runc import *
from ensembles import GradientBoostingMSE
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from scipy.optimize import minimize
import numpy.ma as mask
from sklearn.dummy import DummyRegressor

import os
import pandas as pd
import numpy as np
import seaborn as sns
from copy import copy
from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, RegressorMixin
from runc import *
from ensembles import GradientBoostingMSE
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from scipy.optimize import minimize
import numpy.ma as mask
from sklearn.dummy import DummyRegressor


class BoostingElementaryPredicates(BaseEstimator, RegressorMixin):
    def __init__(self, num_iter, m):
        self.num_iter = num_iter
        self.m = m
        self.h = []  # weak learners
        self.gamma = []  # coefficients for the learners
        self.covers = []  # covers for the learners
        # self.runc = RuncDualizer()
        self.base_value = None
        self.key_objects = []  # "bad" objects for the learners

    def fit(self, X, y):
        self.base_value = y_hat = y.mean()
        # print(y)
        # print('------')

        for _ in range(self.num_iter):
            # print(y_hat)
            residuals = y - y_hat
            # print(residuals)
            # print('------')
            max_residual_idx = np.argmax(np.abs(residuals))

            self.runc = RuncDualizer()  # create new Dualizer

            sorted_residual_indices = np.argsort(np.abs(residuals))
            min_m_residual_indices = sorted_residual_indices[:self.m]
            key_object = X[max_residual_idx]

            for row_idx in min_m_residual_indices:
                row = X[row_idx]
                non_zero_indices = np.where(row != X[max_residual_idx])[0]
                if len(non_zero_indices) > 0:
                    self.runc.add_input_row(list(row))
            comp_rows = []
            for idx in min_m_residual_indices:
                comp_row = []
                for j in range(X.shape[1]):
                    if X[idx, j] == key_object[j]:
                        comp_row.extend([1, 0, 0])  # Равно
                    elif X[idx, j] < key_object[j]:
                        comp_row.extend([0, 1, 0])  # Меньше
                    else:
                        comp_row.extend([0, 0, 1])  # Больше
                comp_rows.append(comp_row)
                self.runc.add_input_row(comp_row)
            print(len(comp_rows), len(comp_rows[0]))
            print(comp_rows)

            h_m, min_residual_sum, gamma_m = 0, float('inf'), 0
            # X_other = X[sorted_residual_indices[self.m:]]

            for i in range(20):
                covers = self.runc.enumerate_covers()
                if len(covers) == 0:
                    break

                for cover in covers:
                    h_mask = np.isin(np.arange(X.shape[1]), cover)
                    base_estimator = residuals[max_residual_idx] * np.where(
                        (X[:, h_mask] == X[max_residual_idx][h_mask]).all(axis=1), 1, 0)
                    residual_sum_maybe = ((y - y_hat - base_estimator) ** 2).mean()
                    if residual_sum_maybe < min_residual_sum:
                        h_m, min_residual_sum = base_estimator, residual_sum_maybe
                        best_cover = cover
                        # gamma_m = self.optimize(y, y_hat, h_m)[0]
            # print(min_residual_sum)
            self.h.append(h_m)
            gamma_m = self.optimize(y, y_hat, h_m)[0]
            self.gamma.append(gamma_m)
            self.covers.append(best_cover)
            self.key_objects.append(X[max_residual_idx])
            y_hat += gamma_m * h_m

            del (self.runc)

        return self

    def optimize(self, y, y_hat, h):
        loss = lambda gamma: ((y - y_hat - gamma * h) ** 2).mean()
        result = minimize(loss, x0=0.0)
        return result.x

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_value)
        for i in range(len(self.h)):
            h_mask = np.isin(np.arange(X.shape[1]), self.covers[i])
            res = np.unique(self.h[i][self.h[i] != 0.])
            base_estimator = res * np.where((X[:, h_mask] == self.key_objects[i][h_mask]).all(axis=1), 1, 0)
            y_pred += base_estimator.reshape(-1)

        return y_pred


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

def main():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    y = dataset['MPG']
    X = dataset.drop(columns='MPG')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    m_values = list(range(70, 80, 1)) # Равномерная сетка от 5 до 59
    r2_scores = []

    for m in m_values:
        model = BoostingElementaryPredicates(num_iter=11, m=m)
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred = model.predict(X_test.to_numpy())
        r2 = r2_score(y_test.to_numpy(), y_pred)
        r2_scores.append(r2)

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, r2_scores, marker='o', linestyle='-', color='b')
    plt.title('Зависимость r2_score от значения параметра m')
    plt.xlabel('Значение параметра p')
    plt.ylabel('r2_score')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()