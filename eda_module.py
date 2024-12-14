# 2. eda_module.py
# This module handles the Exploratory Data Analysis (EDA) functionality.

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from eda_functions import *

# EDA functions
def perform_eda(dataframe):
    """Perform EDA on the selected table."""
    if dataframe.empty:
        st.warning("The table is empty.")
        return

    st.subheader("Exploratory Data Analysis")

    # Numeric column statistics
    st.write("### Numeric Columns Summary")
    numeric_cols = dataframe.select_dtypes(include=['number'])
    if not numeric_cols.empty:
        st.dataframe(numeric_cols.describe())
    else:
        st.write("No numeric columns found.")

        # Categorical column analysis v2
    st.write("### Categorical Columns Analysis")
    categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns

    # Streamlit Sidebar for User to Select Columns for Visualization
    selected_columns = st.sidebar.multiselect(
        'Select Categorical Columns for Visualization',
        categorical_columns.tolist(),
        default=categorical_columns.tolist()  # Default to show all categorical columns
    )

    # Check if user has selected columns
    if selected_columns:
        for column in selected_columns:
            # Display the chart title
            st.subheader(f'Distribution of {column}')
            
            # Create the bar plot for each categorical column
            # Reset index and rename columns properly
            value_counts = dataframe[column].value_counts().reset_index()
            value_counts.columns = [column, 'count']  # Rename columns to match the plot's x and y

            # Create the Plotly bar chart
            fig = px.bar(value_counts,
                        x=column, y='count', 
                        labels={column: column, 'count': 'Frequency'},
                        title=f'Distribution of {column}')
            
            # Display the plotly chart in the app
            st.plotly_chart(fig)
    else:
        st.write("Please select categorical columns from the sidebar.")

def extensive_eda(dataframe):
    # Page Navigation for EDA
    st.title("Extensive EDA for Machine Learning")

    # Sidebar for selecting EDA analysis
    analysis_type = st.selectbox(
        "Select an EDA Analysis Step",
        options=["Correlation Heatmap","Outlier Detection", "Missing Value Detection", "Descriptive Statistics", "Distributions", "Check Skewness", "Class Imbalance", "Detect Outliers", "Isolation Forest Outliers", "Feature Importance"]
    )

    # Correlation Matrix
    if analysis_type == "Correlation Heatmap":
        plot_heatmap(dataframe)

    # Outlier Detection (using IQR method)
    elif analysis_type == "Outlier Detection":
        st.subheader("Outlier Detection (Using IQR Method)")
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for column in numeric_columns:
            Q1 = dataframe[column].quantile(0.25)
            Q3 = dataframe[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
            outlier_info[column] = len(outliers)
        
        # Display the number of outliers per column
        st.write(outlier_info)
        
        # Visualize outliers for each numeric column
        for column in numeric_columns:
            st.subheader(f"Outliers in {column}")
            fig, ax = plt.subplots()
            sns.boxplot(x=dataframe[column], ax=ax)
            st.pyplot(fig)
        
    # Missing Value Detection
    elif analysis_type == "Missing Value Detection":
        st.subheader("Missing Value Heatmap")
        # Visualize missing values
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(dataframe.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)
        
        # Show the number of missing values per column
        missing_values = dataframe.isnull().sum()
        st.write("Missing values per column:")
        st.write(missing_values)

    # Descriptive Statistics
    elif analysis_type == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        st.write(dataframe.describe())

    # Distribution of Numerical Variables
    elif analysis_type == "Distributions":
        st.subheader("Distributions of Numerical Variables")
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            st.subheader(f"Distribution of {column}")
            fig, ax = plt.subplots()
            sns.histplot(dataframe[column], kde=True, ax=ax)
            st.pyplot(fig)

    elif analysis_type == "Check Skewness":
        check_skewness_kurtosis(dataframe)
    
    elif analysis_type == "Class Imbalance":
        # Allow user to select the target column from a dropdown
        target_column = st.selectbox("Select Target Column", dataframe.columns)
        plot_class_imbalance(dataframe, target_column)

    elif analysis_type == "Detect Outliers":
        detect_outliers_zscore(dataframe)

    elif analysis_type == "Isolation Forest Outliers":
        isolation_forest_outliers(dataframe)
    
    elif analysis_type == "Feature Importance":
        target_column = st.selectbox("Select Target Column", dataframe.columns)
        feature_importance(dataframe, target_column)


# Function to fetch data from Snowflake
def fetch_table_data(conn, database_name, schema_name, table_name):
    """Fetch the content of a specific table from Snowflake."""
    try:
        query = f"SELECT * FROM {database_name}.{schema_name}.{table_name};"
        cursor = conn.cursor()
        cursor.execute(query)
        # Convert data to Pandas DataFrame
        df = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])
        return df
    except Exception as e:
        st.error(f"Error fetching table data: {e}")
        return pd.DataFrame()