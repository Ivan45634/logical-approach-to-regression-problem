import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from runc import RuncDualizer


class BoostingElementaryPredicates(BaseEstimator, RegressorMixin):
    def __init__(self, num_iter=None, m=None, max_cov=None):
        self.num_iter = num_iter
        self.m = m
        self.max_cov = max_cov
        self.h = []  # weak learners
        self.gamma = []  # coefficients for the learners
        self.covers = [] # covers for the learners
        # self.runc = RuncDualizer()
        self.base_value = None
        self.coeffs = []
        self.key_objects = [] # "bad" objects for the learners
        self.train_losses = []  # train loss for each iteration
        self.test_losses = []  # test loss for each iteration
        self.est_res = []
        
    def fit(self, X, y):
        self.base_value = y_hat = y.mean()
        n = X.shape[1]

        for _ in range(self.num_iter):
            residuals = y - y_hat
            max_residual_idx = np.argmax(np.abs(residuals))

            self.runc = RuncDualizer() # create new Dualizer
           
            sorted_residual_indices = np.argsort(np.abs(residuals))
            min_m_residual_indices = sorted_residual_indices[:self.m]
            key_object = X[max_residual_idx]
            
            for idx in min_m_residual_indices:
                comp_row = []
                for j in range(n):
                    if X[idx, j] < key_object[j]:
                        comp_row.append(j)  # Меньше
                    elif X[idx, j] > key_object[j]:
                        comp_row.append(j+n)  # Больше
                # print(comp_row)
                if len(comp_row) > 0:
                    self.runc.add_input_row(comp_row)

            h_m, min_residual_sum, gamma_m = 0, float('inf'), 0

            while True:
                covers = self.runc.enumerate_covers()
                # print(covers)

                if self.max_cov is not None and len(covers) > self.max_cov: #TODO: добавить стохастику в построение и выбор покрытий
                    covers = covers[:self.max_cov]
                
                if len(covers) == 0:
                    break

                for cover in covers:
                    h_mask_l = np.isin(np.arange(n), cover)
                    h_mask_g = np.isin(np.arange(n, 2*n), cover)
                    H_l = np.where((X[:, h_mask_l] >= X[max_residual_idx][h_mask_l]).all(axis=1), 1, 0)
                    H_g = np.where((X[:, h_mask_g] <= X[max_residual_idx][h_mask_g]).all(axis=1), 1, 0)
                    base_estimator = residuals[max_residual_idx] * H_l * H_g
                    residual_sum_maybe = ((y - y_hat - base_estimator) ** 2).mean()
                    if residual_sum_maybe < min_residual_sum:
                        h_m, min_residual_sum = base_estimator, residual_sum_maybe
                        best_cover = cover
            self.h.append(h_m)
            # gamma_m = self.optimize(y, y_hat, h_m)[0]
            opt_coeffs = self.optimize(y, y_hat, h)  # function returns the optimal (a, b)
            self.coeffs.append(opt_coeffs)
            self.coeffs.append(opt_coeffs)
            self.gamma.append(gamma_m)
            self.covers.append(best_cover)
            self.key_objects.append(X[max_residual_idx])
            self.est_res.append(residuals[max_residual_idx])
            # y_hat += gamma_m * h_m
            y_hat = self.update_predictions(y_hat, opt_coeffs, h_m)
            
        # del self.runc
        
        return self

    def optimize(self, y, y_hat, h):
        loss = lambda a, b: ((y - y_hat - a * h + b) ** 2).mean()
        result = minimize(loss, x0=0.0)
        return result.x, result.y

    def update_predictions(self, y_hat, coeffs, h_m):
        # Assuming h(x) returns either 0 or 1, update y_hat based on the added learner and its coefficients (a, b)
        a, b = coeffs
        return y_hat + a * h_m + b
    
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_value)
        for i in range(len(self.h)):
            n = X.shape[1]
            h_mask_l = np.isin(np.arange(n), self.covers[i])
            h_mask_g = np.isin(np.arange(n, 2*n), self.covers[i])
            H_l = np.where((X[:, h_mask_l] >= self.key_objects[i][h_mask_l]).all(axis=1), 1, 0)
            H_g = np.where((X[:, h_mask_g] <= self.key_objects[i][h_mask_g]).all(axis=1), 1, 0)
            base_estimator = self.est_res[i] * H_l * H_g
            y_pred += self.gamma[i] * base_estimator.reshape(-1)

        return y_pred
    
#--------------------------#

class BoostingElementaryPredicatesv2(BaseEstimator, RegressorMixin):
    """
    Модель бустинга на элементарных предикатах. Алгоритм использует итеративный подход
    для улучшения предсказаний, минимизируя остатки на тренировочных данных.
    
    Параметры:
    - num_iter: количество итераций алгоритма
    - m: количество объектов с наименьшими остатками для участия в матрице сравнения
    - max_cov: максимальное количество покрытий
    """
    
    def __init__(self, num_iter=None, m=None, max_cov=None):
        self.num_iter = num_iter
        self.m = m
        self.max_cov = max_cov
        self.h = []
        self.gamma = []  # coefficients for the learners
        self.covers = [] # covers for the learners
        # self.runc = RuncDualizer()
        self.base_value = None
        self.key_objects = [] # "bad" objects for the learners
        self.train_losses = []  # train loss for each iteration
        self.test_losses = []  # test loss for each iteration
        self.est_res = []

    def fit(self, X, y):
        n = X.shape[1]
        self.base_value = y_hat = y.mean()

        for _ in range(self.num_iter):
            residuals = y - y_hat
            min_residual_idx = np.argmin(residuals)
            key_object = X[min_residual_idx]

            self.runc = RuncDualizer()

            # Отбор объектов
            if self.m > 0 and self.m < len(residuals):
                candidates = np.argsort(-residuals)[:self.m]
            else:
                candidates = np.arange(len(residuals))

            for idx in candidates:
                # Создание матрицы сравнения опорного объекта с остальными
                comp_row = []
                for j in range(n):
                    if X[idx, j] < key_object[j]:
                        comp_row.append(j)  # Меньше
                    elif X[idx, j] > key_object[j]:
                        comp_row.append(j + n)  # Больше
                if len(comp_row) > 0:
                    self.runc.add_input_row(comp_row)
            
            min_residual_sum = float('inf')
            num_processed_covers = 0

            while True:
                covers = self.runc.enumerate_covers()

                if len(covers) == 0:
                    break  # Если покрытий больше нет, выход из цикла

                if self.max_cov is not None and num_processed_covers >= self.max_cov:
                    break
                
                for cover in covers:
                    # Прежде чем обработать покрытие, проверяем, не превышен ли лимит
                    if num_processed_covers >= self.max_cov:
                        break
                    a_opt, b_opt, residual_sum, base_estimator = self.evaluate_coefs_and_residual_sum(cover, X, residuals, min_residual_idx)
                    if residual_sum < min_residual_sum:
                        best_cover = cover
                        best_gamma = a_opt, b_opt
                        min_residual_sum = residual_sum
                        h_m = base_estimator
                    
                    num_processed_covers += 1
                else:
                    continue
                # Если внутренний цикл прерван из-за достижения лимита, прерываем и внешний цикл
                if num_processed_covers >= self.max_cov:
                    break
            
            self.h.append(best_cover)
            self.gamma.append(best_gamma)
            self.covers.append(best_cover)
            self.key_objects.append(X[min_residual_idx])
            self.est_res.append(residuals[min_residual_idx])
            # y_hat += gamma_m * h_m
            a_m, b_m = best_gamma
            y_hat += a_m * h_m + b_m
        
        return self


    def evaluate_coefs_and_residual_sum(self, cover, X, residuals, min_residual_idx):

        def loss(params):
            a, b = params
            return ((a * base_estimator + b - residuals)**2).mean()
        
        
        n = X.shape[1]

        h_mask_l = np.isin(np.arange(n), cover)
        h_mask_g = np.isin(np.arange(n, 2*n), cover)
        H_l = np.where((X[:, h_mask_l] >= X[min_residual_idx][h_mask_l]).all(axis=1), 1, 0)
        H_g = np.where((X[:, h_mask_g] <= X[min_residual_idx][h_mask_g]).all(axis=1), 1, 0)
        base_estimator = H_l * H_g
        # result = minimize(loss, x0=np.array([1.0, 0.1]), method='BFGS') - численный поиск опт. коэфф.
        # a_opt, b_opt = result.x
        # min_residual_sum = loss(result.x)

        B = base_estimator 
        r = residuals
        sum_r_B = np.sum(r * B)  # Сумма r_i * B(S_i)
        sum_B = np.sum(B)        # Сумма B(S_i)

        sum_r_B1 = np.sum(r * (B - 1))  # Сумма r_i * (B(S_i) - 1)
        sum_B1 = np.sum(B - 1)          # Сумма (B(S_i) - 1)

        if sum_B == 0 or sum_B1 == 0:
            raise ValueError("Division by zero in formula calculation for a_opt or b_opt.")

        a_opt = (sum_r_B / sum_B) - (sum_r_B1 / sum_B1)
        b_opt = sum_r_B1 / sum_B1

        params = [a_opt, b_opt]

        min_residual_sum = loss(params)

        return a_opt, b_opt, min_residual_sum, base_estimator

    
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_value)
        for i in range(len(self.h)):
            n = X.shape[1]
            h_mask_l = np.isin(np.arange(n), self.covers[i])
            h_mask_g = np.isin(np.arange(n, 2*n), self.covers[i])
            H_l = np.where((X[:, h_mask_l] >= self.key_objects[i][h_mask_l]).all(axis=1), 1, 0)
            H_g = np.where((X[:, h_mask_g] <= self.key_objects[i][h_mask_g]).all(axis=1), 1, 0)
            base_estimator = H_l * H_g
            # print(self.gamma[i])
            a_i, b_i = self.gamma[i]
            y_pred += a_i * base_estimator + b_i

        return y_pred
    
#--------------------------#

class BoostingElementaryPredicates1(BaseEstimator, RegressorMixin):
    def __init__(self, num_iter, m):
        self.num_iter = num_iter
        self.m = m
        self.h = []  # weak learners
        self.gamma = []  # coefficients for the learners
        self.covers = [] # covers for the learners
        # self.runc = RuncDualizer()
        self.base_value = None
        self.key_objects = [] # "bad" objects for the learners
        self.train_losses = []  # train loss for each iteration
        self.test_losses = []  # test loss for each iteration
        self.est_res = []
        
    def fit_predict(self, X_train, y_train, X_test, y_test):
        self.base_value = y_hat = y_train.mean()
        n = X_train.shape[1]

        print(self.num_iter)
        for _ in range(self.num_iter):
            residuals = y_train - y_hat
            max_residual_idx = np.argmax(np.abs(residuals))

            self.runc = RuncDualizer()  # create new Dualizer
           
            sorted_residual_indices = np.argsort(np.abs(residuals))
            min_m_residual_indices = sorted_residual_indices[:self.m]
            key_object = X_train[max_residual_idx]
            
            # print(residuals[min_m_residual_indices])
            # print(residuals[max_residual_idx])
            
            # for row_idx in min_m_residual_indices:
            #     row = X[row_idx]
            #     non_zero_indices = np.where(row != X[max_residual_idx])[0]
            #     if len(non_zero_indices) > 0:
            #         self.runc.add_input_row(list(row))
            # comp_rows=[]
            for idx in min_m_residual_indices:
                comp_row = []
                for j in range(n):
                    # if X[idx, j] == key_object[j]:
                    #     comp_row.append(j)  # Равно
                    if X_train[idx, j] < key_object[j]:
                        comp_row.append(j)  # Меньше
                    elif X_train[idx, j] > key_object[j]:
                        comp_row.append(j+n)  # Больше
                # comp_rows.append(comp_row)
                # print(comp_row)
                if len(comp_row) > 0:
                    self.runc.add_input_row(comp_row)

            h_m, min_residual_sum, gamma_m = 0, float('inf'), 0
            while True:
                covers = self.runc.enumerate_covers()
                if len(covers) == 0:
                    break

                for cover in covers:
                    # n = X_train.shape[1]
                    h_mask_l = np.isin(np.arange(n), cover)
                    h_mask_g = np.isin(np.arange(n, 2*n), cover)
                    H_l = np.where((X_train[:, h_mask_l] >= X_train[max_residual_idx][h_mask_l]).all(axis=1), 1, 0)
                    H_g = np.where((X_train[:, h_mask_g] <= X_train[max_residual_idx][h_mask_g]).all(axis=1), 1, 0)
                    base_estimator = residuals[max_residual_idx] * H_l * H_g
                    residual_sum_maybe = ((y_train - y_hat - base_estimator) ** 2).mean()
                    if residual_sum_maybe < min_residual_sum:
                        h_m, min_residual_sum = base_estimator, residual_sum_maybe
                        best_cover = cover

            self.h.append(h_m)
            gamma_m = self.optimize(y_train, y_hat, h_m)[0]
            self.gamma.append(gamma_m)
            self.covers.append(best_cover)
            self.key_objects.append(X_train[max_residual_idx])
            self.est_res.append(residuals[max_residual_idx])
            y_hat += gamma_m * h_m

            # Calculate train and test loss for current iteration
            train_loss = ((y_train - self.predict(X_train)) ** 2).mean()
            test_loss = ((y_test - self.predict(X_test)) ** 2).mean()
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

        # Plot the train and test loss
        x = np.arange(self.num_iter) + 1
        plt.plot(x, self.train_losses, label='Train Loss')
        plt.plot(x, self.test_losses, label='Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
        return self

    def optimize(self, y, y_hat, h):
        loss = lambda gamma: ((y - y_hat - gamma * h) ** 2).mean()
        result = minimize(loss, x0=0.0)
        return result.x
    
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_value)
        for i in range(len(self.h)):
            # h_mask = np.isin(np.arange(X.shape[1]), self.covers[i])
            # res = np.unique(self.h[i][self.h[i] != 0.])
            # base_estimator = res * np.where((X[:, h_mask] == self.key_objects[i][h_mask]).all(axis=1), 1, 0)
            # y_pred += base_estimator.reshape(-1)
            n = X.shape[1]
            h_mask_l = np.isin(np.arange(n), self.covers[i])
            h_mask_g = np.isin(np.arange(n, 2*n), self.covers[i])
            H_l = np.where((X[:, h_mask_l] >= self.key_objects[i][h_mask_l]).all(axis=1), 1, 0)
            H_g = np.where((X[:, h_mask_g] <= self.key_objects[i][h_mask_g]).all(axis=1), 1, 0)
            # res = np.unique(self.h[i][self.h[i] != 0.])
            base_estimator = self.est_res[i] * H_l * H_g
            y_pred += self.gamma[i] * base_estimator.reshape(-1)

        return y_pred
