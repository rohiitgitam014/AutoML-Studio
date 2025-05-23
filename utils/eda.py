import streamlit as st
import plotly.express as px
import pandas as pd

def run_eda(df, target_column):
    st.write("## 🧪 Exploratory Data Analysis")

    numeric_df = df.select_dtypes(include='number')
    numeric_cols = numeric_df.columns.tolist()

    # 🔥 Correlation heatmap
    if len(numeric_cols) > 1:
        st.write("### 🔥 Correlation Heatmap")
        corr = numeric_df.corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                        title="Correlation Heatmap")
        st.plotly_chart(fig)

    # 📦 Boxplots for numeric features
    st.write("### 📦 Boxplots for Numerical Features")
    for col in numeric_cols:
        if col != target_column:
            fig = px.box(df, y=col, title=f"Boxplot of {col}", color_discrete_sequence=['orange'])
            st.plotly_chart(fig)

    # 🧮 Histograms for numeric features
    st.write("### 🧮 Histograms for Numeric Features")
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}", color_discrete_sequence=['teal'])
        st.plotly_chart(fig)

    # 🎻 Violin plots
    st.write("### 🎻 Violin Plots for Numerical Features")
    for col in numeric_cols:
        if col != target_column:
            fig = px.violin(df, y=col, box=True, points="all", title=f"Violin Plot of {col}")
            st.plotly_chart(fig)

    # 📊 Bar plots for all non-target categorical features
    st.write("### 📊 Bar Plots for Categorical Features (Excluding Target)")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target_column in cat_cols:
        cat_cols.remove(target_column)

    if not cat_cols:
        st.info("No non-target categorical columns to display bar plots for.")
    else:
        for col in cat_cols:
            count_data = df[col].value_counts().reset_index()
            count_data.columns = [col, 'count']
            fig = px.bar(count_data, x=col, y='count', color=col,
                         labels={col: col, 'count': 'Count'},
                         title=f"Bar Plot of {col}",
                         color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)

    # 🥧 Pie chart for target (if categorical and small # of classes)
    if df[target_column].nunique() <= 10 and df[target_column].dtype == 'object':
        st.write("### 🥧 Pie Chart of Target Variable")
        fig = px.pie(df, names=target_column, title=f"Pie Chart of {target_column}")
        st.plotly_chart(fig)

    # 📈 Scatterplots (Numeric vs Target)
    st.write("### 📈 Scatterplots (Numeric vs Target)")
    for col in numeric_cols:
        if col != target_column:
            fig = px.scatter(df, x=col, y=target_column,
                             title=f"Scatterplot: {col} vs {target_column}",
                             color=target_column if df[target_column].nunique() <= 10 else None,
                             color_continuous_scale='Viridis')
            st.plotly_chart(fig)

    # 🔁 All 2D Scatter Plots Between Numeric Features (Including Target)
    st.write("### 📌 All 2D Scatter Plots Between Numeric Features")
    max_pairs = 20  # Adjust to control rendering load
    pair_count = 0
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            if pair_count >= max_pairs:
                st.info(f"Only showing first {max_pairs} scatter pairs to prevent overload.")
                break
            x_col = numeric_cols[i]
            y_col = numeric_cols[j]
            fig = px.scatter(df, x=x_col, y=y_col,
                             color=target_column if df[target_column].nunique() <= 10 and target_column in df.columns else None,
                             title=f"Scatter Plot: {x_col} vs {y_col}",
                             color_continuous_scale='Viridis')
            st.plotly_chart(fig)
            pair_count += 1

    # 🔗 Scatter Matrix (if enough features)
    if len(numeric_cols) >= 2:
        st.write("### 🔗 Scatter Matrix")
        pair_cols = numeric_cols[:5]
        fig = px.scatter_matrix(df, dimensions=pair_cols,
                                color=target_column if df[target_column].nunique() <= 10 else None,
                                title="Scatter Matrix")
        st.plotly_chart(fig)

    # 🌊 KDE/Density plots
    st.write("### 🌊 Density Plots (KDE Style)")
    for col in numeric_cols:
        fig = px.density_contour(df, x=col, title=f"Density Contour of {col}")
        st.plotly_chart(fig)
