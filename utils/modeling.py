import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score, mean_absolute_error

def run_models(X, y, task_type):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "K-Nearest Neighbors Regressor (KNN)": KNeighborsRegressor(),
        }

    for name, model in models.items():
        st.subheader(f"🔹 {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "classification":
            st.text("Classification Report:".upper())
            report = classification_report(y_test, y_pred, output_dict= True)
            report = pd.DataFrame(report).transform()
            st.table(report)
        else:
            st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.2f}")
            st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            st.write(f"**Mean squared error**:{mean_squared_error(y_test, y_pred): 2f}")
            st.write(f"**Mean absolute error**:{mean_absolute_error(y_test, y_pred): 2f}")

