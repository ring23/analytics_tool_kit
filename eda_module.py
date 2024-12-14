# 2. eda_module.py
# This module handles the Exploratory Data Analysis (EDA) functionality.

import pandas as pd
import streamlit as st

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

    # Categorical column analysis
    st.write("### Categorical Columns Analysis")
    categorical_cols = dataframe.select_dtypes(include=['object', 'category'])
    if not categorical_cols.empty:
        for col in categorical_cols.columns:
            st.write(f"#### {col}")
            st.bar_chart(categorical_cols[col].value_counts())
    else:
        st.write("No categorical columns found.")

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