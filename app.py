# 3. app.py
# Main Streamlit application to tie it all together.

import streamlit as st
from snowflake_connector import get_snowflake_connection, list_databases, list_schemas, list_tables
from eda_module import fetch_table_data, perform_eda, extensive_eda
from ml_module import ml_page

st.title("Analytics Tool Kit")
st.subheader("Exploratory Data Analysis")

# Page navigation with st.selectbox
page = st.sidebar.selectbox("Select a Page", options=["Home", "EDA", "Other Pages", "ML Development"])

if page == "Home":
    # Connect to Snowflake
    st.write("Connecting to Snowflake...")
    conn = get_snowflake_connection()
    
    if conn:
        st.success("Connected to Snowflake!")

        # Fetch available databases
        databases = list_databases(conn)
        if databases:
            selected_database = st.selectbox("Select a database:", databases)

            # Fetch schemas for the selected database
            if selected_database:
                schemas = list_schemas(conn, selected_database)
                if schemas:
                    selected_schema = st.selectbox("Select a schema:", schemas)

                    # Fetch tables for the selected database and schema
                    if selected_schema:
                        tables = list_tables(conn, selected_database, selected_schema)
                        if tables:
                            selected_table = st.selectbox("Select a table for EDA:", tables)

                            # Fetch data and perform EDA
                            if selected_table:
                                st.write(f"You selected: {selected_database}.{selected_schema}.{selected_table}")
                                try:
                                    # Fetch table data
                                    table_data = fetch_table_data(conn, selected_database, selected_schema, selected_table)
                                    
                                    # Store the fetched table data in session_state
                                    if not table_data.empty:
                                        st.session_state.table_data = table_data  # Store in session_state
                                        st.success("Table data fetched successfully.")
                                    else:
                                        st.warning("The selected table has no data.")
                                except Exception as e:
                                    st.error(f"Error during data fetch or EDA: {e}")
                        else:
                            st.warning("No tables found in the selected schema.")
                else:
                    st.warning("No schemas found in the selected database.")
        else:
            st.warning("No databases found.")
    else:
        st.error("Failed to connect to Snowflake. Check your credentials.")

elif page == "EDA":
    # Check if table_data exists in session_state before using it
    if 'table_data' in st.session_state:
        table_data = st.session_state.table_data
        extensive_eda(table_data)  # Call the EDA function
    else:
        st.warning("No data available for EDA. Please fetch the data from the 'Home' page first.")

elif page == "ML Development":
    # Check if table_data exists in session_state before using it
    if 'table_data' in st.session_state:
        table_data = st.session_state.table_data
        ml_page(table_data)  # Call the EDA function
    else:
        st.warning("No data available for EDA. Please fetch the data from the 'Home' page first.")


# Other pages for different functionalities (like the categorical visuals page)
elif page == "Other Pages":
    st.title("Other Functionality")
    st.write("Add more functionality here.")

