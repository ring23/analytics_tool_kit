import streamlit as st
from snowflake_connector import *
from inference_functions import *
#Streamlit UI for Inference Automation Setup
# Streamlit UI for Inference Automation Setup
def setup_inference_automation():
    st.title("Inference Automation Setup")
    
    st.write("Configure when and how to run inference on your dataset using your trained model.")

    # --- Step 1: Select Model ---
    st.header("1. Select Model File")

    model_database = st.selectbox("Select Database for Model", list_databases(get_snowflake_connection()))
    model_schema = st.selectbox("Select Schema for Model", list_schemas(get_snowflake_connection(), model_database))
    model_stage = st.selectbox("Select Stage for Model", list_stages(model_database, model_schema))

    # List model files
    model_files = list_files_in_stage(model_database, model_schema, model_stage)
    selected_model_file = st.selectbox("Select Model File", model_files)

    if selected_model_file:
        st.success(f"Selected Model: {selected_model_file}")
        if st.button("Load Model"):
            model = load_model_from_stage(model_database, model_schema, model_stage, os.path.basename(selected_model_file))
            if model is not None:
                st.session_state['trained_model'] = model
                st.success("Model loaded successfully!")
            else:
                st.error("Error loading model.")

    # --- Step 2: Configure Inference Frequency ---
    st.header("2. Configure Inference Frequency")

    inference_frequency = st.radio("Select Inference Trigger", 
                                  ("Scheduled Inference", "Real-Time Inference"))

    if inference_frequency == "Scheduled Inference":
        # User selects a schedule
        st.subheader("Scheduled Inference")
        schedule = st.selectbox("Select Frequency", ["Daily", "Weekly", "Monthly"])
        time_of_day = st.time_input("Select Time of Day", value=pd.to_datetime("12:00:00").time())

        if st.button("Create Scheduled Inference Task"):
            create_scheduled_task(schedule, time_of_day)

    elif inference_frequency == "Real-Time Inference":
        # User selects a real-time trigger
        st.subheader("Real-Time Inference")
        trigger_on_changes = st.checkbox("Trigger Inference on Data Changes")

        if trigger_on_changes and st.button("Create Real-Time Trigger"):
            create_real_time_inference_trigger()

# Function to preprocess the dataset before inference
def preprocess_input_data(model, df):
    """
    Preprocess the dataset using the saved model's preprocessing pipeline.
    """
    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        # Extract the preprocessor from the pipeline
        preprocessor = model.named_steps['preprocessor']
        st.info("Applying preprocessing pipeline to the dataset...")
        try:
            df_processed = preprocessor.transform(df)
            return df_processed
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            return None
    else:
        st.warning("No preprocessing pipeline found in the model. Passing raw data.")
        return df

# Function to perform inference with the model
def perform_inference(model, input_df):
    """
    Perform inference using the loaded model and preprocessed input data.
    """
    try:
        # Preprocess the input data before inference
        input_data = preprocess_input_data(model, input_df)
        
        if input_data is not None:
            st.info("Running inference...")
            predictions = model.predict(input_data)
            return predictions
        else:
            st.error("Preprocessing failed. Cannot run inference.")
            return None
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None

# Function to create a scheduled Snowflake Task for inference
def create_scheduled_task(schedule, time_of_day):
    task_name = f"run_inference_{schedule.lower()}"
    cron_schedule = ""
    
    if schedule == "Daily":
        cron_schedule = f"USING CRON '{time_of_day.strftime('%M %H * * *')}' UTC"
    elif schedule == "Weekly":
        cron_schedule = f"USING CRON '{time_of_day.strftime('%M %H * * 1')}' UTC"  # 1 = Monday
    elif schedule == "Monthly":
        cron_schedule = f"USING CRON '{time_of_day.strftime('%M %H 1 * *')}' UTC"  # 1 = Day of the month

    # Create Snowflake Task to run inference on schedule
    query = f"""
    CREATE OR REPLACE TASK {task_name}
    WAREHOUSE = your_warehouse
    SCHEDULE = '{cron_schedule}'
    AS
    CALL run_inference_procedure();
    """
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        st.success(f"Scheduled Task '{task_name}' created successfully!")
    except Exception as e:
        st.error(f"Error creating scheduled task: {e}")
    finally:
        cursor.close()
        conn.close()

# Function to create a Snowflake Stream and Task for real-time inference
def create_real_time_inference_trigger():
    stream_name = "data_changes_stream"
    task_name = "real_time_inference_task"

    # Create Snowflake Stream on the target table (replace with actual table name)
    create_stream_query = f"""
    CREATE OR REPLACE STREAM {stream_name}
    ON TABLE target_table
    SHOW_INITIAL_ROWS = TRUE;
    """
    
    # Create Snowflake Task to run inference when the stream captures changes
    create_task_query = f"""
    CREATE OR REPLACE TASK {task_name}
    WAREHOUSE = your_warehouse
    AS
    BEGIN
        IF EXISTS (SELECT 1 FROM {stream_name} WHERE METADATA$ACTION IN ('INSERT', 'UPDATE')) THEN
            CALL run_inference_procedure();
        END IF;
    END;
    """
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute(create_stream_query)
        cursor.execute(create_task_query)
        st.success(f"Real-Time Inference Trigger '{task_name}' created successfully!")
    except Exception as e:
        st.error(f"Error creating real-time trigger: {e}")
    finally:
        cursor.close()
        conn.close()