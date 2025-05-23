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
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select the target column", df.columns)
    
    if target_column:
        task_type, df_processed, X, y = preprocess_data(df, target_column)
        
        st.markdown("### 🔍 Exploratory Data Analysis")
        run_eda(df, target_column)

        st.markdown("### 🤖 Model Training and Evaluation")
        run_models(X, y, task_type)
