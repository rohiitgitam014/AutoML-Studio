import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def run_eda(df, target_column):
    st.write("## 🧪 Exploratory Data Analysis")

    # Target distribution
    st.write("### 🎯 Target Variable Distribution")
    fig, ax = plt.subplots()
    df[target_column].value_counts().plot(kind="bar", ax=ax, color='skyblue')
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Correlation heatmap
    st.write("### 🔥 Correlation Heatmap (Numerical features only)")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include='number')
    sns.heatmap(numeric_df.corr(), ax=ax, annot=True, cmap="coolwarm")
    st.pyplot(fig)

    # Boxplots for numeric features
    st.write("### 📦 Boxplots for Numerical Features")
    numeric_cols = numeric_df.columns.tolist()
    for col in numeric_cols:
        if col != target_column:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax, color='orange')
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)

    # Histograms for numeric features
    st.write("### 🧮 Histograms for Numeric Features")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        df[col].hist(bins=30, ax=ax, color='teal')
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Countplots for categorical features
    st.write("### 📊 Countplots for Categorical Features")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=col, ax=ax, palette="viridis")
        ax.set_title(f"Countplot of {col}")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    # Pairplot (limited to 4 features for performance)
    st.write("### 🔗 Pairplot (up to 4 numeric features)")
    if len(numeric_cols) >= 2:
        selected_cols = numeric_cols[:4]
        fig = sns.pairplot(df[selected_cols + [target_column]], hue=target_column if df[target_column].nunique() <= 10 else None)
        st.pyplot(fig)

