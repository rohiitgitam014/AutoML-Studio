import pandas as pd

def preprocess_data(df, target_column):
    df = df.dropna(axis=1, how='all')  # Drop columns with all NaN
    df = df.dropna()  # Drop rows with any NaN

    target = df[target_column]
    task_type = "classification" if target.dtype == "object" or target.nunique() < 15 else "regression"

    X = df.drop(columns=[target_column])
    X = pd.get_dummies(X)  # One-hot encode categorical features
    X = X.astype(float)

    return task_type, df, X, target
