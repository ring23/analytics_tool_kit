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


def generate_inference_notebook(database, schema, input_table, stream_name, model_file, output_table, warehouse, inference_type, schedule, output_file="inference_notebook.ipynb"):
    # Placeholder instructions in the first Markdown cell
    cell_1 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Instructions\n",
            "This notebook is designed to run scoring on a saved model in Snowflake.\n",
            "Follow the steps in each cell to perform the inference."
        ]
    }

    # Imports in the second Python cell
    cell_2 = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Import necessary libraries\n",
            "from snowflake.snowpark import Session\n",
            "import pandas as pd\n"
        ]
    }

    # Initialize cell_3 and cell_4 as None to ensure they are always present
    cell_3 = None
    cell_4 = None

    # Logic for Real-Time vs Batch inference
    if inference_type == 'Real-Time':
        # Create a Snowflake STREAM for real-time inference
        stream_sql = f"CREATE OR REPLACE STREAM {database}.{schema}.{stream_name} ON TABLE {database}.{schema}.{input_table} SHOW_INITIAL_ROWS = TRUE APPEND_ONLY = TRUE;"
        cell_3 = {
            "cell_type": "code",
            "metadata": {
                "language": "sql"
            },
            "execution_count": None,
            "outputs": [],
            "source": [
                "-- Create Stream for Real-Time Inference\n",
                f"{stream_sql}\n"
            ]
        }
        # Add a new cell to run the SELECT * query from the stream
        select_sql = f"SELECT * FROM {database}.{schema}.{stream_name} WHERE METADATA$ACTION = 'INSERT';"
        cell_4 = {
            "cell_type": "code",
            "metadata": {
                "language": "sql"
            },
            "execution_count": None,
            "outputs": [],
            "source": [
                "-- Select data from the Stream\n",
                f"{select_sql}\n"
            ]
        }

    elif inference_type == 'Batch':
        # Create a Snowflake TASK for batch inference
        task_sql = f"""
        CREATE OR REPLACE TASK {schema}.{stream_name}_task
        WAREHOUSE = {warehouse}
        SCHEDULE = '{schedule}' 
        AS
        CALL run_inference_procedure('{model_file}', '{output_table}');
        """
        cell_4 = {
            "cell_type": "code",
            "metadata": {
                "language": "sql"
            },
            "execution_count": None,
            "outputs": [],
            "source": [
                "-- Create Task for Batch Inference\n",
                f"{task_sql}\n"
            ]
        }

    # Load results into a DataFrame in the fifth Python cell
    cell_5 = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Execute the query and load the results into a Snowpark DataFrame\n",
            "# Ensure the session is already initialized in the notebook environment\n",
            "df_snowflake = cell4.to_pandas()\n",
            "df_snowflake.head()\n"
        ]
    }

    # Notebook structure
    notebook_content = {
        "cells": [cell_1, cell_2] + ([cell_3] if cell_3 else []) + ([cell_4] if cell_4 else []) + [cell_5],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Save the notebook to a file
    with open(output_file, "w") as f:
        json.dump(notebook_content, f, indent=4)

    # Return the filename
    return output_file









