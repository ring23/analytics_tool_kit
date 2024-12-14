import streamlit as st
import snowflake.connector

def connect_to_snowflake():
    conn = snowflake.connector.connect(
        user="cmeringolo",
        password="U*i9o0u8",
        account="oycfeqg-uo43734"
    )
    return conn

st.title("Snowflake Test")
st.write("Connecting to Snowflake...")
try:
    conn = connect_to_snowflake()
    st.success("Connected successfully!")
    cursor = conn.cursor()
    cursor.execute("SELECT CURRENT_VERSION()")
    version = cursor.fetchone()
    st.write(f"Snowflake version: {version[0]}")
except Exception as e:
    st.error(f"Connection failed: {e}")
