import streamlit as st
import plotly.express as px
import pandas as pd

def run_eda(df, target_column):
    st.write("## 🧪 Exploratory Data Analysis")

    # Target distribution (Bar chart)
    st.write("### 🎯 Target Variable Distribution")
    target_counts = df[target_column].value_counts().reset_index()
    target_counts.columns = [target_column, 'count']
    fig = px.bar(target_counts, x=target_column, y='count', color=target_column,
                 labels={target_column: "Target", "count": "Count"},
                 title="Target Variable Distribution")
    st.plotly_chart(fig)

    # Correlation heatmap (numerical only)
    st.write("### 🔥 Correlation Heatmap (Numerical features only)")
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap")
    st.plotly_chart(fig)

    # Boxplots for numeric features
    st.write("### 📦 Boxplots for Numerical Features")
    numeric_cols = numeric_df.columns.tolist()
    for col in numeric_cols:
        if col != target_column:
            fig = px.box(df, y=col, title=f"Boxplot of {col}", color_discrete_sequence=['orange'])
            st.plotly_chart(fig)

    # Histograms for numeric features
    st.write("### 🧮 Histograms for Numeric Features")
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}", color_discrete_sequence=['teal'])
        st.plotly_chart(fig)

    # Countplots for categorical features
    st.write("### 📊 Countplots for Categorical Features")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        fig = px.histogram(df, x=col, title=f"Countplot of {col}", color=col,
                           color_discrete_sequence=px.colors.qualitative.Viridis)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)


