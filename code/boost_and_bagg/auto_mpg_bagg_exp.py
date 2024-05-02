import numpy as np
from ucimlrepo import fetch_ucirepo
from runc import RuncDualizer
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from models import *
from utilites import *
from own_forest import *
from bagging_trees import BaggingElementaryPredicates



def hp_tuning(X_train, X_test, y_train, y_test, model, params, dataset_name, model_name, n_iter=10, cv=5):
    scoring_rmse = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
    scoring_r2 = make_scorer(r2_score)

    randomized_search = RandomizedSearchCV(estimator=model,
                                           param_distributions=params,
                                           n_iter=n_iter, 
                                           cv=cv, 
                                           random_state=42)
    randomized_search.fit(X_train, y_train) 

    best_model = randomized_search.best_estimator_
    
    cv_rmse_scores = cross_val_score(best_model, X_test, y_test, cv=cv, scoring=scoring_rmse)
    cv_r2_scores = cross_val_score(best_model, X_test, y_test, cv=cv, scoring=scoring_r2)

    mean_rmse = np.mean(-cv_rmse_scores)
    std_rmse = np.std(cv_rmse_scores)

    mean_r2 = np.mean(cv_r2_scores)
    std_r2 = np.std(cv_r2_scores)
    
    print(f"{model_name} on {dataset_name}: RMSE: {mean_rmse:.3f} (+/- {std_rmse:.3f}), R2: {mean_r2:.3f} (+/- {std_r2:.3f})")

    return best_model, randomized_search.best_params_, mean_rmse, std_rmse, mean_r2, std_r2

auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

data = pd.concat([X, y], axis=1)

data = data.dropna()

X_train, X_val, X_test, y_train, y_val, y_test = preprocess(data, 'mpg')

dataset_name = 'Auto_MPG_1'

be_params = {
    'base_estimator': [BoostingElementaryPredicatesv2(num_iter=i, m=j, max_cov=500, learning_rate=0.5) for i in np.linspace(300, 600, 5).astype(int) for j in range(10, 21)],
    # 'base_estimator': [BoostingElementaryPredicatesv2(num_iter=533, m=20, max_cov=500, learning_rate=0.5)],
    'n_estimators': sp_randint(10, 200),
    'sub_sample_size': uniform(0.5, 0.5),  # от 0.5 до 1.0
    'sub_feature_size': uniform(0.5, 0.5),  # от 0.5 до 1.0
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
    'n_jobs': [1]  # 1 processor
}

bg_params = {
    'n_estimators': [10, 50, 100],
    'base_estimator': [DecisionTreeRegressor(max_depth=d) for d in range(1, 11)],
}

rf_params = {
    'n_estimators': [50, 100, 150], 
    'max_depth': [3, 5, 7]
}

models = [ 
    ("BaggingElementaryPredicates", BaggingElementaryPredicates(base_estimator=BoostingElementaryPredicatesv2(num_iter=533, m=20, max_cov=500, learning_rate=0.5)), be_params),
    ("BaggingRegressor", BaggingRegressor(n_jobs=1), bg_params),
    ("RandomForestRegressor", RandomForestRegressor(n_jobs=1), rf_params)
]

results = []

filename = f"results_{dataset_name}.txt"

with open(filename, 'w') as file:
    for model_name, model, params in models:
        # Используйте новые значения, возвращаемые функцией hp_tuning
        _, best_params, mean_rmse, std_rmse, mean_r2, std_r2 = hp_tuning(X_train, X_test, y_train, y_test, model, params, dataset_name, model_name)
        
        results.append((dataset_name, model_name, f"{mean_rmse:.3f} +- {std_rmse:.3f}", f"{mean_r2:.3f} +- {std_r2:.3f}"))
        
        # Форматируйте строку для записи в файл
        file.write(f"Best hyperparameters for {model_name} on {dataset_name}: {best_params}\n")
        file.write(f"{model_name} Results:\n")
        file.write(f"Train/Test RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}\n")
        file.write(f"Train/Test R2: {mean_r2:.3f} ± {std_r2:.3f}\n")
        file.write("-----------------------------------\n")

    results_df = pd.DataFrame(results, columns=['Dataset', 'Model', 'RMSE (mean ± std)', 'R2 (mean ± std)'])
    file.write(f"\nSummary Results:\n")
    file.write(results_df.to_string(index=False))

print(f"Results have been saved to {filename}")
