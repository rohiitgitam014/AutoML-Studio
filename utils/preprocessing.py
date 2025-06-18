import pandas as pd
import numpy as np

def preprocess_data(df, target_column, threshold=0.3):
    df = df.copy()

    # 1. Drop completely empty columns
    df = df.dropna(axis=1, how='all')  

    # 2. Fill missing values
    for col in df.columns:
        if col == target_column:
            continue

        missing_ratio = df[col].isna().mean()

        if df[col].dtype in [np.float64, np.int64]:
            if missing_ratio <= threshold:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 3. Drop rows where target is still missing (if any)
    df = df.dropna(subset=[target_column])

    # 4. Determine task type
    target = df[target_column]
    task_type = "classification" if target.dtype == "object" or target.nunique() < 15 else "regression"

    # 5. Prepare features
    X = df.drop(columns=[target_column])
    X = pd.get_dummies(X)  # One-hot encode categorical features
    X = X.astype(float)

    return task_type, df, X, target
