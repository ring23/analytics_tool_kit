import streamlit as st
from snowflake_connector import *
from inference_functions import *
from schedule_inference_functions import *

def schedule_inference_page(conn):
    conn = get_snowflake_connection()
    # Warehouse, Database, Schema, Model, and Input Table Selection
    warehouse = st.selectbox("Select Warehouse", list_warehouses(conn))
    database = st.selectbox("Select Database", list_databases(conn))
    schema = st.selectbox("Select Schema", list_schemas(conn, database))
    stage = st.selectbox("Select Stage", list_stages(database, schema))
    model_file = st.selectbox("Select Model File", list_files_in_stage(database, schema, stage))


    # Dynamic Table Selection based on Database & Schema
    input_table = st.selectbox("Select Input Table", list_tables(conn,database, schema))

    # Inference Type Selection
    inference_type = st.selectbox("Select Inference Type", ["Real-Time", "Batch", "One-Time"])
    schedule = None

    # Batch Inference Schedule
    if inference_type == "Batch":
        schedule = st.selectbox("Select Schedule", ["DAILY", "WEEKLY", "MONTHLY"])

    # Task/Stream and Stored Procedure Names
    task_name = st.text_input("Enter Task Name")
    stream_name = st.text_input("Enter Stream Name")
    proc_name = st.text_input("Enter Stored Procedure Name")
    # Allow user to select or create an output table
    output_table_options = list_tables(conn, database, schema)  # Fetch the list of existing tables
    new_table_name = st.text_input("Or create a new table name (leave blank to select existing)", "")
    output_table = new_table_name if new_table_name else st.selectbox("Select Output Table", output_table_options)

    # Save Configuration Button
    if st.button("Save Configuration"):
        try:
            # Backend logic to handle task/stream creation and stored procedure
            save_inference_configuration(conn, warehouse, database, schema, stage, model_file, 
                                        input_table, inference_type, schedule, task_name, stream_name, proc_name, output_table)
        except Exception as e:
            st.error(f"Error saving configuration: {e}")