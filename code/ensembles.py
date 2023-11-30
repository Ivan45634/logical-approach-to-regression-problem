import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar

class RandomForestMSE:
    def __init__(self, n_estimators, feature_subsample_size=1.0,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """

        self.n_estimators = n_estimators
        self.feature_subsample_size = feature_subsample_size
        self.tree_list = []
        for _ in range(n_estimators):
            self.tree_list.append(DecisionTreeRegressor(**trees_parameters))
        
    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """

        feat_indeces = np.arange(0, X.shape[1])
        self.matrix_of_feature_index = np.zeros((self.n_estimators, int(X.shape[1] * self.feature_subsample_size)), dtype=int)
        for i in range(self.n_estimators):
            obj_indeces = np.random.randint(0, X.shape[0], X.shape[0])
            np.random.shuffle(feat_indeces)
            self.matrix_of_feature_index[i] = feat_indeces[:int(X.shape[1] * self.feature_subsample_size)]
            self.tree_list[i].fit(X[obj_indeces][:, self.matrix_of_feature_index[i]], y[obj_indeces])
        
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        ans_matrix = np.zeros((X.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
            ans_matrix[:, i] = self.tree_list[i].predict(X[:, self.matrix_of_feature_index[i]])
        return np.mean(ans_matrix, axis=1)


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, feature_subsample_size=1.0,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_subsample_size = feature_subsample_size
        self.tree_list = []
        self.weigths = np.zeros(n_estimators)
        for _ in range(self.n_estimators):
            self.tree_list.append(DecisionTreeRegressor(**trees_parameters))
        
    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """

        feat_indeces = np.arange(0, X.shape[1])
        self.matrix_of_feature_index = np.zeros((self.n_estimators, int(X.shape[1] * self.feature_subsample_size)), dtype=int)
        F_sum = np.zeros(len(y))

        for i in range(self.n_estimators):
            np.random.shuffle(feat_indeces)
            self.matrix_of_feature_index[i] = feat_indeces[:int(X.shape[1] * self.feature_subsample_size)]

            y_new = y - F_sum
            self.tree_list[i].fit(X[:, self.matrix_of_feature_index[i]], y_new)
            y_pred = self.tree_list[i].predict(X[:, self.matrix_of_feature_index[i]])

            def func(x):
                return np.sum((y - F_sum - x * y_pred) ** 2) / 2.0
            
            self.weigths[i] = minimize_scalar(func).x * self.learning_rate
            F_sum = F_sum + self.weigths[i] * y_pred

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        ans = np.zeros(X.shape[0])
        for i, alg in enumerate(self.tree_list):
            ans = ans + self.weigths[i] * alg.predict(X[:, self.matrix_of_feature_index[i]])
        return ans
