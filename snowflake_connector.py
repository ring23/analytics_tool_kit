# 1. snowflake_connector.py
# This module handles the connection to Snowflake.

import streamlit as st
import snowflake.connector
import pandas as pd

def get_snowflake_connection():
    """Establish a connection to Snowflake."""
    return snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"]
    )

def list_warehouses(conn):
    """Retrieve the list of available warehouses."""
    query = "SHOW WAREHOUSES"
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        warehouses = [row[0] for row in cursor.fetchall()]  # Assuming the warehouse name is in the second column
        return warehouses
    except Exception as e:
        st.error(f"Error fetching warehouses: {e}")
        return []
    finally:
        cursor.close()

def list_databases(conn):
    """Fetch the list of databases in Snowflake."""
    try:
        query = "SHOW DATABASES;"
        cursor = conn.cursor()
        cursor.execute(query)
        databases = cursor.fetchall()
        return [row[1] for row in databases]
    except Exception as e:
        st.error(f"Error fetching database list: {e}")
        return []

def list_schemas(conn, database_name):
    """Fetch the list of schemas for a given database."""
    try:
        query = f"SHOW SCHEMAS IN DATABASE {database_name};"
        cursor = conn.cursor()
        cursor.execute(query)
        schemas = cursor.fetchall()
        return [row[1] for row in schemas]
    except Exception as e:
        st.error(f"Error fetching schema list: {e}")
        return []

def list_tables(conn, database_name, schema_name):
    """Fetch the list of tables for a given database and schema."""
    try:
        query = f"SHOW TABLES IN SCHEMA {database_name}.{schema_name};"
        cursor = conn.cursor()
        cursor.execute(query)
        tables = cursor.fetchall()
        return [row[1] for row in tables]
    except Exception as e:
        st.error(f"Error fetching table list: {e}")
        return []

# Fetch stages from Snowflake database and schema
def list_stages(database, schema):
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    # Set the current database and schema
    cursor.execute(f"USE DATABASE {database}")
    cursor.execute(f"USE SCHEMA {schema}")
    # Query to get the stages
    query = f"SELECT STAGE_NAME FROM INFORMATION_SCHEMA.STAGES"
    cursor.execute(query)
    stages = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return [stage[0] for stage in stages]

# Clean the DataFrame to handle missing values based on data type
def clean_dataframe_for_snowflake(df):
    """
    Replace NaN with None (NULL) for compatibility with Snowflake.
    Ensure numeric columns have valid values.
    Clean column names to ensure compatibility with Snowflake (e.g., uppercase).
    """
    # Clean column names (convert to uppercase)
    df.columns = [col.upper() for col in df.columns]

    # Replace NaN with None (SQL NULL) for compatibility
    df = df.where(pd.notnull(df), None)

    # Ensure numeric columns contain only numeric or NULL values
    for column in df.select_dtypes(include=['number']).columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert invalid entries to NaN
        df[column] = df[column].where(pd.notnull(df[column]), None)  # Replace NaN with None

    return df
