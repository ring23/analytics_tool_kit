# 3. app.py
# Main Streamlit application to tie it all together.

import streamlit as st
from snowflake_connector import get_snowflake_connection, list_tables
from eda_module import fetch_table_data, perform_eda

st.title("Analytics Tool Kit")
st.subheader("Exploratory Data Analysis")

# Connect to Snowflake
st.write("Connecting to Snowflake...")
conn = get_snowflake_connection()
if conn:
    st.success("Connected to Snowflake!")

    # Fetch available tables
    st.write("Fetching table list...")
    tables = list_tables(conn)

    if tables:
        # Table selection
        selected_table = st.selectbox("Select a table for EDA:", tables)

        # Fetch data and perform EDA
        if selected_table:
            st.write(f"You selected: {selected_table}")
            table_data = fetch_table_data(conn, selected_table)

            if not table_data.empty:
                perform_eda(table_data)
            else:
                st.warning("The selected table has no data.")
    else:
        st.warning("No tables found in the database.")
else:
    st.error("Failed to connect to Snowflake. Check your credentials.")
