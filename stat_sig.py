import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
import streamlit as st
import snowflake.connector
import pandas as pd
from scipy.stats import ttest_ind, chisquare
from statsmodels.stats.proportion import proportions_ztest
from snowflake_connector import *

# Function to fetch tables from Snowflake
def get_tables(conn):
    query = "SHOW TABLES"
    df = pd.read_sql(query, conn)
    return df['name'].tolist()

# Function to fetch columns from a Snowflake table
def get_columns(conn, table_name, database, schema):
    query = f"SHOW COLUMNS IN TABLE {database}.{schema}.{table_name}"
    df = pd.read_sql(query, conn)
    return df['column_name'].tolist()

# Helper functions for statistical tests
def run_t_test(data_source, group_a, group_b):
    t_stat, p_value = ttest_ind(group_a, group_b)
    st.write(f"T-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        st.success("Result: Statistically Significant")
    else:
        st.warning("Result: Not Statistically Significant")


def run_chi_squared_test(observed, expected):
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    st.write(f"Chi-squared Statistic: {chi2_stat}, P-value: {p_value}")
    if p_value < 0.05:
        st.success("Result: Statistically Significant")
    else:
        st.warning("Result: Not Statistically Significant")


def run_proportion_test(successes, totals):
    z_stat, p_value = proportions_ztest(count=successes, nobs=totals)
    st.write(f"Z-statistic: {z_stat}, P-value: {p_value}")
    if p_value < 0.05:
        st.success("Result: Statistically Significant")
    else:
        st.warning("Result: Not Statistically Significant")


# Function to handle manual input
def handle_manual_input(test_type):
    if test_type == "T-Test":
        st.subheader("T-Test Input")
        mean_a = st.number_input("Group A Mean")
        std_a = st.number_input("Group A Standard Deviation")
        size_a = st.number_input("Group A Sample Size", min_value=1, step=1)
        mean_b = st.number_input("Group B Mean")
        std_b = st.number_input("Group B Standard Deviation")
        size_b = st.number_input("Group B Sample Size", min_value=1, step=1)

        if st.button("Run T-Test"):
            group_a = [mean_a] * size_a
            group_b = [mean_b] * size_b
            run_t_test("Manual", group_a, group_b)

    elif test_type == "Chi-Squared Test":
        st.subheader("Chi-Squared Test Input")
        observed_input = st.text_area("Enter Observed Frequencies (comma-separated):")
        expected_input = st.text_area("Enter Expected Frequencies (comma-separated):")

        if st.button("Run Chi-Squared Test"):
            observed = [float(x) for x in observed_input.split(",")]
            expected = [float(x) for x in expected_input.split(",")]
            run_chi_squared_test(observed, expected)

    elif test_type == "Proportion Test":
        st.subheader("Proportion Test Input")
        successes_a = st.number_input("Group A Successes", min_value=0, step=1)
        size_a = st.number_input("Group A Sample Size", min_value=1, step=1)
        successes_b = st.number_input("Group B Successes", min_value=0, step=1)
        size_b = st.number_input("Group B Sample Size", min_value=1, step=1)

        if st.button("Run Proportion Test"):
            count = [successes_a, successes_b]
            nobs = [size_a, size_b]
            run_proportion_test(count, nobs)


# Function to handle Snowflake input
def handle_snowflake_input(test_type, conn, database, schema, table, columns):
    if test_type == "T-Test":
        group_a_col = st.selectbox("Select Group A Column", columns)
        group_b_col = st.selectbox("Select Group B Column", columns)

        if st.button("Run T-Test"):
            query = f"SELECT {group_a_col}, {group_b_col} FROM {database}.{schema}.{table}"
            df = pd.read_sql(query, conn)
            run_t_test("Snowflake", df[group_a_col], df[group_b_col])

    elif test_type == "Chi-Squared Test":
        selected_columns = st.multiselect("Select Columns for Observed Frequencies", columns)

        if selected_columns:
            query = f"SELECT {', '.join(selected_columns)} FROM {database}.{schema}.{table}"
            df = pd.read_sql(query, conn)
            observed = df.sum(axis=0).apply(pd.to_numeric, errors='coerce').dropna()
            st.write("Observed Frequencies:", observed)

            expected_method = st.radio("Expected Frequencies:", ["Equal", "Custom"])
            if expected_method == "Equal":
                expected = [observed.sum() / len(observed)] * len(observed)
            elif expected_method == "Custom":
                expected_input = st.text_input("Enter Expected Frequencies (comma-separated):")
                if expected_input:
                    expected = [float(x) for x in expected_input.split(",")]

            if st.button("Run Chi-Squared Test"):
                run_chi_squared_test(observed, expected)

    elif test_type == "Proportion Test":
        # Select columns for both groups
        success_col_a = st.selectbox("Select Group A Success Column:", columns, key="group_a_success")
        total_col_a = st.selectbox("Select Group A Total Column:", columns, key="group_a_total")
        success_col_b = st.selectbox("Select Group B Success Column:", columns, key="group_b_success")
        total_col_b = st.selectbox("Select Group B Total Column:", columns, key="group_b_total")

        if st.button("Run Proportion Test"):
            query = f"""
                SELECT 
                    SUM({success_col_a}) AS success_a, 
                    SUM({total_col_a}) AS total_a, 
                    SUM({success_col_b}) AS success_b, 
                    SUM({total_col_b}) AS total_b
                FROM {database}.{schema}.{table}
            """
            df = pd.read_sql(query, conn)

            # Extract aggregated values
            successes = [df.iloc[0]['SUCCESS_A'], df.iloc[0]['SUCCESS_B']]  # If Snowflake returns uppercase columns
            totals = [df.iloc[0]['TOTAL_A'], df.iloc[0]['TOTAL_B']]

            # Perform proportion test
            z_stat, p_value = proportions_ztest(count=successes, nobs=totals)
            st.write(f"Z-statistic: {z_stat}, P-value: {p_value}")
            if p_value < 0.05:
                st.success("Result: Statistically Significant")
            else:
                st.warning("Result: Not Statistically Significant")


# Main Statistical Significance Calculator
def statistical_significance_calculator():
    st.title("Statistical Significance Calculator")

    data_source = st.radio("Select Data Input Method:", ["Manual Input", "Use Snowflake Table"])

    if data_source == "Manual Input":
        test_type = st.radio("Choose Test Type:", ["T-Test", "Chi-Squared Test", "Proportion Test"])
        handle_manual_input(test_type)

    elif data_source == "Use Snowflake Table":
        conn = get_snowflake_connection()
        if conn:
            st.success("Connected to Snowflake!")

            databases = list_databases(conn)
            selected_database = st.selectbox("Select a database:", databases)

            schemas = list_schemas(conn, selected_database)
            selected_schema = st.selectbox("Select a schema:", schemas)

            tables = list_tables(conn, selected_database, selected_schema)
            selected_table = st.selectbox("Select a table:", tables)

            columns = get_columns(conn, selected_table, selected_database, selected_schema)
            test_type = st.radio("Choose Test Type:", ["T-Test", "Chi-Squared Test", "Proportion Test"])
            handle_snowflake_input(test_type, conn, selected_database, selected_schema, selected_table, columns)

            conn.close()

