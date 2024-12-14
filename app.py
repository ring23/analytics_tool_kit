import streamlit as st
import snowflake.connector

def connect_to_snowflake():
    conn = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"]
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
