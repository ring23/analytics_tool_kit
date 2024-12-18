import streamlit as st
from datetime import datetime
import pandas as pd
from snowflake_connector import *
from inference_functions import *
import re

def real_time_inference_page():
    st.title("Configure Real-Time or Scheduled Inference")

    # Step 1: Select Model and Input Table
    st.header("1. Select Model and Input Table")
    conn = get_snowflake_connection()

    # Select warehouse from Snowflake
    warehouses = list_warehouses(conn)
    selected_warehouse = st.selectbox("Select Warehouse for Task", warehouses)

    # Select model from Snowflake Stage
    databases = list_databases(conn)
    selected_database = st.selectbox("Select Database for Model", databases)
    
    if selected_database:
        schemas = list_schemas(conn, selected_database)
        selected_schema = st.selectbox("Select Schema for Model", schemas)
        
        if selected_schema:
            stages = list_stages(selected_database, selected_schema)
            selected_stage = st.selectbox("Select Stage for Model", stages)

            if selected_stage:
                model_files = list_files_in_stage(selected_database, selected_schema, selected_stage)
                selected_model_file = st.selectbox("Select Model File", model_files)

                if selected_model_file:
                    st.success(f"Selected Model File: {selected_model_file}")
                    actual_model_file_name = os.path.basename(selected_model_file)
                    if st.button("Load Selected Model"):
                        pipeline = load_model_from_stage(selected_database, selected_schema, selected_stage, actual_model_file_name)

                        if pipeline and isinstance(pipeline, Pipeline):
                            st.session_state['selected_pipeline'] = pipeline
                            st.success("Model/Pipeline loaded successfully!")
                        else:
                            st.error("The selected file is not a valid pipeline.")

    # Select input table from Snowflake
    st.header("2. Select Input Table for Inference")
    datasets_database = st.selectbox("Select Database for Dataset", databases, key="dataset_db")
    if datasets_database:
        datasets_schema = st.selectbox("Select Schema for Dataset", list_schemas(conn, datasets_database), key="dataset_schema")

        if datasets_schema:
            tables = list_tables(conn, datasets_database, datasets_schema)
            selected_table = st.selectbox("Select Table for Dataset", tables, key="dataset_table")

            if selected_table:
                st.success(f"Selected Table: {selected_table}")

    # Step 2: Configure Inference Type
    st.header("3. Configure Inference Settings")
    inference_type = st.selectbox(
        "Select Inference Type",
        ["Real-time Inference", "Batch Inference"]
    )

    if inference_type == "Batch Inference":
        schedule = st.selectbox(
            "Select Schedule for Inference",
            ["Daily", "Weekly", "Monthly"]
        )
    else:
        schedule = None

    # Step 3: Save Scheduled Inference Details to Snowflake
    st.header("4. Save and Schedule Inference")
    if st.button("Save Configuration"):
        # Prepare job_data dictionary
        job_data = {
            'job_id': f"{selected_model_file}_{selected_table}_{inference_type}_{schedule}",  # Unique job ID (you can refine this logic)
            'model_file': selected_model_file,
            'input_table': selected_table,
            'inference_type': inference_type,
            'schedule': schedule,
            'status': 'active'  # Assuming the job starts active, you could modify this as needed
        }

        # Save inference settings to Snowflake (pass the job data dictionary)
        save_inference_job_to_snowflake(selected_database, selected_schema, selected_warehouse, job_data)

    st.success(f"Inference job saved successfully!")

def create_valid_job_id(model_file, table, inference_type, schedule):
    """Sanitize job ID to remove invalid characters for Snowflake."""
    job_id = f"{model_file}_{table}_{inference_type}_{schedule}"
    
    # Replace invalid characters with underscores
    job_id = re.sub(r'[^a-zA-Z0-9_]', '_', job_id)  # Replaces non-alphanumeric characters with underscores
    
    return job_id

def clean_job_data(job_data):
    """Ensure job data is properly formatted for Snowflake."""
    if job_data['schedule'] is None:
        job_data['schedule'] = 'None'  # Or some other default value like ''
    return job_data

def create_stream_and_task(job_data, database, schema):
    """Create a stream and task for real-time or scheduled inference jobs."""
    try:
        # Connection setup
        conn = get_snowflake_connection()
        cursor = conn.cursor()

        # Create Stream for real-time inference
        if job_data['inference_type'] == 'Real-time Inference':
            stream_name = f"stream_{job_data['job_id']}"
            stream_sql = f"""
            CREATE OR REPLACE STREAM {database}.{schema}.{stream_name}
            ON TABLE {database}.{schema}.{job_data['input_table']}
            SHOW_INITIAL_ROWS = TRUE;
            """
            cursor.execute(stream_sql)
            st.success(f"Stream {stream_name} created successfully for real-time inference!")

            # Create Task to trigger stored procedure when data changes
            task_name = f"task_{job_data['job_id']}"
            task_sql = f"""
            CREATE OR REPLACE TASK {database}.{schema}.{task_name}
            WAREHOUSE = <your_warehouse>  -- Replace with the desired warehouse
            SCHEDULE = 'USING CRON 0 * * * * UTC'  -- Hourly schedule, adjust as needed
            COMMENT = 'Task for real-time inference'
            WHEN SYSTEM$STREAM_HAS_DATA('{database}.{schema}.{stream_name}')
            AS
            CALL {database}.{schema}.sp_inference_{job_data['job_id']}();
            """
            cursor.execute(task_sql)
            st.success(f"Task {task_name} created successfully for real-time inference!")

        # Create Task for scheduled inference (Hourly, Daily, Weekly, Monthly)
        elif job_data['inference_type'] == 'Batch Inference' and job_data['schedule']:
            task_name = f"task_{job_data['job_id']}"
            cron_expression = get_cron_expression(job_data['schedule'])
            
            task_sql = f"""
            CREATE OR REPLACE TASK {database}.{schema}.{task_name}
            WAREHOUSE = <your_warehouse>  -- Replace with the desired warehouse
            SCHEDULE = '{cron_expression}'
            COMMENT = 'Task for scheduled inference'
            AS
            CALL {database}.{schema}.sp_inference_{job_data['job_id']}();
            """
            cursor.execute(task_sql)
            st.success(f"Task {task_name} created successfully for {job_data['schedule']} inference!")

        conn.commit()

    except Exception as e:
        st.error(f"Error creating stream and task: {e}")
    
    finally:
        cursor.close()
        conn.close()

def create_stored_procedure(job_data, database, schema, warehouse):
    """Create a stored procedure for the inference job in Snowflake."""
    try:
        # Connection setup
        conn = get_snowflake_connection()
        cursor = conn.cursor()

        # Define names for objects
        proc_name = f"sp_inference_{job_data['job_id']}"
        stream_name = f"stream_{job_data['input_table']}"
        task_name = f"task_{job_data['job_id']}"

        # Step 1: Create the stored procedure
        procedure_sql = f"""
        CREATE OR REPLACE PROCEDURE {database}.{schema}.{proc_name}()
        RETURNS STRING
        LANGUAGE JAVASCRIPT
        EXECUTE AS CALLER
        AS
        $$ 
        try {{
            // Log initial job data for debugging
            console.log('Job Data:', '{job_data['job_id']}');

            // Stream table definition
            var stream_query = "SELECT * FROM {database}.{schema}.{stream_name} WHERE METADATA$ACTION = 'INSERT'"; // Consider only insert actions
            console.log('Stream Query:', stream_query);  // Log the stream query

            // Read data from the stream
            var statement = snowflake.createStatement({{ sqlText: stream_query }});
            var result_set = statement.execute();

            // Log the number of rows found in the stream
            console.log('Number of rows in stream:', result_set.getRowCount());

            // Process each row in the stream
            while (result_set.next()) {{
                // Retrieve columns (adjust indices based on your schema)
                var row_data = result_set.getColumnValue(1);  // Assuming the first column contains the input data
                var model_file = '{job_data['model_file']}';

                // Log row data and model file
                console.log('Processing row data:', row_data);
                console.log('Model File:', model_file);

                // Run inference - assuming run_inference is a predefined function
                var inference_result = run_inference(model_file, row_data);

                // Log the inference result
                console.log('Inference Result:', inference_result);

                // Check if INFERENCE_RESULTS table exists
                var table_check_query = `SELECT * FROM INFORMATION_SCHEMA.TABLES 
                                         WHERE TABLE_SCHEMA = '{schema}' 
                                         AND TABLE_NAME = 'INFERENCE_RESULTS';`;
                var table_check_statement = snowflake.createStatement({{ sqlText: table_check_query }});
                var table_check_result = table_check_statement.execute();

                // Log the table check result
                console.log('Table check result:', table_check_result.next() ? 'Exists' : 'Does not exist');

                // If table does not exist, create it
                if (!table_check_result.next()) {{
                    var create_table_query = `CREATE TABLE {database}.{schema}.INFERENCE_RESULTS (
                                              job_id STRING,
                                              input_data VARIANT,
                                              inference_result VARIANT,
                                              created_at TIMESTAMP_LTZ);`;
                    var create_table_statement = snowflake.createStatement({{ sqlText: create_table_query }});
                    create_table_statement.execute();
                    console.log('Created INFERENCE_RESULTS table');
                }}

                // Insert inference result into the table
                var insert_query = `INSERT INTO {database}.{schema}.INFERENCE_RESULTS (job_id, input_data, inference_result, created_at)
                                    VALUES ('{job_data['job_id']}', :row_data, :inference_result, CURRENT_TIMESTAMP())`;
                var insert_statement = snowflake.createStatement({{ sqlText: insert_query, binds: [row_data, inference_result] }});
                insert_statement.execute();
                console.log('Inserted inference result into INFERENCE_RESULTS');
            }}

            // Update the job status once processing is done
            var update_query = `UPDATE {database}.{schema}.INFERENCE_JOBS 
                                SET status = 'completed', updated_at = CURRENT_TIMESTAMP()
                                WHERE job_id = '{job_data['job_id']}';`;
            var update_statement = snowflake.createStatement({{ sqlText: update_query }});
            update_statement.execute();
            console.log('Updated job status to completed');

            return 'Inference executed successfully!';
        }} catch (err) {{
            console.log('Error during inference:', err.message);
            return 'Error during inference: ' + err.message;
        }}
        $$;
        """

        # Log the procedure SQL before execution
        st.write("Stored Procedure SQL Before Execution:")
        st.code(procedure_sql, language="sql")

        # Execute procedure creation
        cursor.execute(procedure_sql)
        conn.commit()

        # Step 2: Create a stream on the input table
        create_stream_query = f"""
        CREATE OR REPLACE STREAM {database}.{schema}.{stream_name}
        ON TABLE {database}.{schema}.{job_data['input_table']}
        SHOW_INITIAL_ROWS = TRUE;  -- Make sure initial rows are captured
        """
        st.write("Stream Creation SQL:")
        st.code(create_stream_query, language="sql")

        # Execute stream creation
        cursor.execute(create_stream_query)
        conn.commit()

        # Step 3: Create a task to run the stored procedure (e.g., every minute for real-time inference)
        create_task_query = f"""
        CREATE OR REPLACE TASK {database}.{schema}.{task_name}
        SCHEDULE = '1 MINUTE'  -- Adjust as per requirement (e.g., hourly, daily)
        WAREHOUSE = {warehouse}
        COMMENT = 'Task for real-time inference processing'
        AS
        CALL {database}.{schema}.{proc_name}();
        """
        st.write("Task Creation SQL:")
        st.code(create_task_query, language="sql")

        # Execute task creation
        cursor.execute(create_task_query)
        conn.commit()

        # Step 4: Activate the task
        resume_task_query = f"ALTER TASK {database}.{schema}.{task_name} RESUME;";
        cursor.execute(resume_task_query)
        conn.commit()

        # Success message
        st.success(f"Stored procedure '{proc_name}' created successfully!")
        st.success(f"Stream '{stream_name}' created successfully!")
        st.success(f"Task '{task_name}' scheduled to run every minute!")

    except Exception as e:
        # Log the error message for debugging
        st.error(f"Error creating stored procedure: {e}")
        st.error(f"Detailed error: {e}")
    finally:
        cursor.close()
        conn.close()


def save_inference_job_to_snowflake(database, schema, warehouse, job_data):
    """Save inference job configuration to Snowflake, create procedure, stream, and task."""
    try:
        # Clean job data and sanitize job ID
        job_data = clean_job_data(job_data)
        job_data['job_id'] = create_valid_job_id(job_data['model_file'], job_data['input_table'], job_data['inference_type'], job_data['schedule'])

        # Connect to Snowflake
        conn = get_snowflake_connection()
        cursor = conn.cursor()

        # Create the INFERENCE_JOBS table if not exists
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {database}.{schema}.INFERENCE_JOBS (
            job_id STRING PRIMARY KEY,
            model_file STRING,
            input_table STRING,
            inference_type STRING,
            schedule STRING,
            status STRING DEFAULT 'inactive',
            created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        );
        """
        cursor.execute(create_table_query)

        # Create Stored Procedure for the job
        create_stored_procedure(job_data, database, schema, warehouse)

        # Insert job data into INFERENCE_JOBS table
        insert_query = f"""
        INSERT INTO {database}.{schema}.INFERENCE_JOBS (job_id, model_file, input_table, inference_type, schedule, status)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        
        cursor.execute(insert_query, (
            job_data['job_id'],
            job_data['model_file'],
            job_data['input_table'],
            job_data['inference_type'],
            job_data['schedule'],
            job_data['status']
        ))

        # Create the stream and task for the inference job
        create_stream_and_task(job_data, database, schema)

        conn.commit()
        st.success(f"Inference job saved with stored procedure, stream, and task! Job ID: {job_data['job_id']}")

    except Exception as e:
        st.error(f"Error saving inference job to Snowflake: {e}")
    
    finally:
        cursor.close()
        conn.close()

def get_task_schedule(schedule):
    if schedule == "Daily":
        return "USING CRON 0 0 * * * UTC"  # Runs every day at midnight
    elif schedule == "Weekly":
        return "USING CRON 0 0 * * 1 UTC"  # Runs every Monday at midnight
    elif schedule == "Monthly":
        return "USING CRON 0 0 1 * * UTC"  # Runs on the first day of every month
    else:
        return "USING CRON 0 0 * * * UTC"  # Default to daily
    
def get_cron_expression(schedule):
    """Get the cron expression for the selected schedule."""
    cron_map = {
        'Hourly': '0 * * * *',
        'Daily': '0 0 * * *',
        'Weekly': '0 0 * * 0',
        'Monthly': '0 0 1 * *',
    }
    return cron_map.get(schedule, '0 * * * *')  # Default to hourly if not found

