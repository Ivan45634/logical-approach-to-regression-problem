from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def preprocess(data, target_name='TARGET', drop=False, index_col=0):
    if isinstance(data, (pd.DataFrame, )):
        df = data.copy()
    elif isinstance(data, (str, )):
        df = pd.read_csv(data, index_col=index_col)
        if drop:
            df = df.drop(columns='index')
    else:
        raise ValueError("Unsupported data type.")

    X = df.drop(columns=target_name)
    y = df[target_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_and_validate(model, X_train, y_train, X_val, y_val, fraction, num_bags, num_iter, m):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse
