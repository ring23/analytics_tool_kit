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

def list_tables(conn):
    """Fetch the list of tables in the user's Snowflake database."""
    try:
        query = "SHOW TABLES;"
        cursor = conn.cursor()
        cursor.execute(query)
        tables = cursor.fetchall()
        return [row[1] for row in tables]
    except Exception as e:
        st.error(f"Error fetching table list: {e}")
        return []
