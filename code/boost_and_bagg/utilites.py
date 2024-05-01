from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, r2_score, mean_squared_error


def preprocess(data, target_name, drop=True, index_col=None):
    if isinstance(data, (str, )):
        if index_col is not None:
            df = pd.read_csv(data, index_col=index_col).reset_index()
        else:
            df = pd.read_csv(data)

    else:
        df = data.copy()
    
    if drop and 'index' in df.columns:
        df = df.drop(columns='index')

    # Splitting the dataset into features and target
    X_raw = df.drop(columns=[target_name])
    
    y = df[target_name].to_numpy()
    # y = df[target_name]

    # Normalizing numerical features
    numeric_cols = X_raw.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    X_raw[numeric_cols] = scaler.fit_transform(X_raw[numeric_cols])

    # Encoding categorical variables
    X_raw = pd.get_dummies(X_raw, drop_first=True)

    # Splitting data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, train_size=0.8, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=42)

    return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train, y_val, y_test


# def preprocess(data, target_name='TARGET', drop=False, index_col=0):
#     if isinstance(data, (pd.DataFrame, )):
#         df = data.copy()
#     elif isinstance(data, (str, )):
#         df = pd.read_csv(data, index_col=index_col)
#         if drop:
#             df = df.drop(columns='index')
#     else:
#         raise ValueError("Unsupported data type.")

#     X = df.drop(columns=target_name)
#     y = df[target_name]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

#     y_train = y_train.to_numpy()
#     y_test = y_test.to_numpy()

#     scaler = StandardScaler().fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test

# def preprocess(data, target_name, drop=True, index_col=0):
#     if isinstance(data, (str, )):
#         if index_col is not None:
#             df = pd.read_csv(data, index_col=0).reset_index()
#         else:
#             df = pd.read_csv(data)
#         #df = pd.read_csv(data, index_col=0).reset_index()
#         if drop:
#             df = df.drop(columns='index')
#         X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=target_name), df[target_name], train_size=0.8)

#         X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5)

#         y_train = y_train.to_numpy()
#         y_test = y_test.to_numpy()
#         y_val = y_val.to_numpy()

#         scaler = StandardScaler().fit(X_train)

#         X_train = scaler.transform(X_train)
#         X_test = scaler.transform(X_test)
#         X_val = scaler.transform(X_val)
#     else:
#         X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=target_name), data[target_name], train_size=0.8)

#         X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5)

#         y_train = y_train.to_numpy()
#         y_test = y_test.to_numpy()
#         y_val = y_val.to_numpy()

#         scaler = StandardScaler().fit(X_train)

#         X_train = scaler.transform(X_train)
#         X_test = scaler.transform(X_test)
#         X_val = scaler.transform(X_val)

#     return X_train, y_train, X_val, y_val, X_test, y_test

def hp_tuning(X_train, X_test, y_train, y_test, model, params, dataset_name, model_name, n_iter=10):
    scoring_fnc = make_scorer(rmse, greater_is_better=False)
    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=params, 
                                           n_iter=n_iter, scoring=scoring_fnc, cv=5, random_state=42)
    randomized_search.fit(X_train, y_train)

    print(f"Best hyperparameters for {model_name} on {dataset_name}: {randomized_search.best_params_}")
    best_model = randomized_search.best_estimator_

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\n--Train RMSE for {model_name} on {dataset_name}--\n{train_rmse}")
    print(f"--Test RMSE for {model_name} on {dataset_name}--\n{test_rmse}\n")
    print(f"--Train R2 for {model_name} on {dataset_name}--\n{train_r2}")
    print(f"--Test R2 for {model_name} on {dataset_name}--\n{test_r2}\n")

    return best_model, randomized_search.best_params_, train_rmse, test_rmse, train_r2, test_r2 
    

def train_and_validate(model, X_train, y_train, X_val, y_val, fraction, num_bags, num_iter, m):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse
