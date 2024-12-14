# 1. snowflake_connector.py
# This module handles the connection to Snowflake.

import streamlit as st
import snowflake.connector

def get_snowflake_connection():
    """Establish a connection to Snowflake."""
    return snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"]
    )

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