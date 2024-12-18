import streamlit as st
from snowflake_connector import *
import pickle
import json

def save_inference_configuration(conn, warehouse, database, schema, stage_location, model_file,
                                  input_table, inference_type, schedule, task_name, stream_name, proc_name, output_table):
    # Set up Snowflake connection
    cursor = conn.cursor()

    try:
        # Ensure schedule is defined for batch inference only
        if inference_type == "Batch" and schedule is None:
            st.error("Please select a schedule for batch inference.")
            return
        if inference_type != "Batch":
            schedule = None  # No schedule needed for Real-Time or One-Time inference

        # Step 1: Create the Stream or Task
        if inference_type == "Real-Time":
            create_stream_query = f"""
            CREATE OR REPLACE STREAM {database}.{schema}.{stream_name}
            ON TABLE {database}.{schema}.{input_table}
            SHOW_INITIAL_ROWS = TRUE;
            """
            cursor.execute(create_stream_query)
            conn.commit()
        elif inference_type == "Batch":
            create_task_query = f"""
            CREATE OR REPLACE TASK {database}.{schema}.{task_name}
            SCHEDULE = '{schedule}'  -- e.g., 'DAILY', 'WEEKLY', 'MONTHLY'
            WAREHOUSE = {warehouse}
            COMMENT = 'Scheduled batch inference task'
            AS
            CALL {database}.{schema}.{proc_name}();
            """
            cursor.execute(create_task_query)
            conn.commit()

        # Step 2: Create the Stored Procedure for Inference
        procedure_sql = f"""
        CREATE OR REPLACE PROCEDURE {database}.{schema}.{proc_name}()
        RETURNS STRING
        LANGUAGE JAVASCRIPT
        EXECUTE AS CALLER
        AS
        $$
        try {{
            // Query the input table for all rows
            var input_query = "SELECT * FROM {database}.{schema}.{input_table}";
            var input_statement = snowflake.createStatement({{ sqlText: input_query }});
            var input_result_set = input_statement.execute();

            // Check if the output table exists; if not, create it
            var check_table_query = `SELECT COUNT(*) 
                                     FROM information_schema.tables 
                                     WHERE table_schema = '{schema}' 
                                     AND table_name = '{output_table}'`;
            var check_statement = snowflake.createStatement({{ sqlText: check_table_query }});
            var check_result = check_statement.execute();
            check_result.next();

            if (check_result.getColumnValue(1) == 0) {{
                var create_table_query = `
                    CREATE TABLE {database}.{schema}.{output_table} (
                        job_id STRING,
                        input_data STRING,  -- Adjust column type as needed
                        inference_result STRING,  -- Adjust column type as needed
                        created_at TIMESTAMP
                    )`;
                var create_statement = snowflake.createStatement({{ sqlText: create_table_query }});
                create_statement.execute();
            }}

            // Process rows from the input table
            while (input_result_set.next()) {{
                var row_data = input_result_set.getColumnValue(1);  // Assuming the first column holds input data
                var model_file = '{model_file}';

                // Call external function for inference
                var inference_statement = snowflake.createStatement({{
                    sqlText: "SELECT run_inference(:1, :2)",
                    binds: [row_data, model_file]  // Pass bind variables as an array
                }});
                var inference_result_set = inference_statement.execute();
                inference_result_set.next(); // Ensure there's a result
                var inference_result = inference_result_set.getColumnValue(1);

                // Insert inference results into the output table
                var insert_query = `
                    INSERT INTO {database}.{schema}.{output_table} (job_id, input_data, inference_result, created_at)
                    VALUES ('{proc_name}', ?, ?, CURRENT_TIMESTAMP())`;
                var insert_statement = snowflake.createStatement({{
                    sqlText: insert_query,
                    binds: [row_data, inference_result]  // Pass bind variables as an array
                }});
                insert_statement.execute();
            }}

            return 'Inference executed successfully!';
        }} catch (err) {{
            return 'Error during inference: ' + err.message;
        }}
        $$;
        """
        cursor.execute(procedure_sql)
        conn.commit()

        # Success message
        st.success(f"Configuration saved! Task/Stream and Stored Procedure '{proc_name}' created successfully.")
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
    finally:
        cursor.close()
        conn.close()


def generate_inference_notebook(database, schema, input_table, stream_name, model_file, output_table):
    # Create the notebook content in JSON format with the necessary structure
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell_1",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import snowflake.connector\n",
                    "import pandas as pd\n\n",
                    "# Connect to Snowflake\n",
                    "conn = snowflake.connector.connect(\n",
                    "    user='your_user',\n",
                    "    password='your_password',\n",
                    "    account='your_account',\n",
                    "    warehouse='your_warehouse',\n",
                    "    database='{database}',\n",
                    "    schema='{schema}'\n",
                    ")\n\n",
                    "# Query to get data from the stream\n",
                    "stream_query = \"SELECT * FROM {database}.{schema}.{input_table} WHERE METADATA$ACTION = 'INSERT'\"\n",
                    "df = pd.read_sql(stream_query, conn)\n\n",
                    "# Inference logic: Replace with your actual model inference logic\n",
                    "def run_inference(data, model_file):\n",
                    "    # Example logic for inference - replace with actual logic\n",
                    "    return \"inference_result_mock\"\n\n",
                    "# Apply inference to the dataset\n",
                    "df['inference_result'] = df['column_name'].apply(lambda x: run_inference(x, '{model_file}'))\n\n",
                    "# Display the results (For testing, you can replace with actual actions later)\n",
                    "print(df.head())\n\n",
                    "# Close connection\n",
                    "conn.close()\n"
                ]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Define the notebook filename
    notebook_filename = "real_time_inference_notebook.ipynb"
    
    # Write the notebook content to the .ipynb file
    with open(notebook_filename, "w") as f:
        json.dump(notebook_content, f, indent=4)

    # Return the filename
    return notebook_filename










