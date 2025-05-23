import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_data
from utils.eda import run_eda
from utils.modeling import run_models

st.set_page_config(page_title="ML Server", layout="wide")
st.title("📊 ML Project Server App")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Original Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select the target column", df.columns)

    if target_column:
        # Preprocess dataset: returns task type, cleaned df, features (X), target (y)
        task_type, df_processed, X, y = preprocess_data(df, target_column)

        st.subheader("Cleaned Dataset Preview (After Preprocessing)")
        st.dataframe(df_processed.head())

        st.markdown("### 🔍 Exploratory Data Analysis")
        run_eda(df_processed, target_column)

        st.markdown("### 🤖 Model Training and Evaluation")
        run_models(X, y, task_type)
