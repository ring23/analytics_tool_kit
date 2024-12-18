import streamlit as st
from snowflake_connector import *
import pandas as pd
from sklearn.pipeline import Pipeline
from inference_functions import *

# Prediction Page
def prediction_page():
    st.title("Apply Trained Model to Selected Table")

    # Connect to Snowflake
    conn = get_snowflake_connection()

    # --- Step 1: Select Model/Pipeline ---
    st.header("1. Select Model/Pipeline")
    databases = list_databases(conn)
    selected_database = st.selectbox("Select Database for Model", databases, key="model_db")

    if selected_database:
        schemas = list_schemas(conn, selected_database)
        selected_schema = st.selectbox("Select Schema for Model", schemas, key="model_schema")

        if selected_schema:
            stages = list_stages(selected_database, selected_schema)
            selected_stage = st.selectbox("Select Stage for Model", stages)

            if selected_stage:
                # List files in the stage
                model_files = list_files_in_stage(selected_database, selected_schema, selected_stage)
                selected_model_file = st.selectbox("Select Model/Pipeline File", model_files)

                if selected_model_file:
                    st.success(f"Selected Model File: {selected_model_file}")

                    # Extract the actual file name from the full path (remove subdirectory prefix)
                    actual_model_file_name = os.path.basename(selected_model_file)

                    if st.button("Load Selected Model"):
                        pipeline = load_model_from_stage(
                            selected_database,
                            selected_schema,
                            selected_stage,
                            actual_model_file_name  # Pass only the file name
                        )

                        if pipeline and isinstance(pipeline, Pipeline):
                            st.session_state['selected_pipeline'] = pipeline
                            st.success("Model/Pipeline loaded successfully!")
                        else:
                            st.error("The selected file is not a valid pipeline.")

    # --- Step 2: Select Dataset for Inference ---
    st.header("2. Select Dataset for Inference")
    if 'selected_pipeline' in st.session_state:
        pipeline = st.session_state['selected_pipeline']

        datasets_database = st.selectbox("Select Database for Dataset", databases, key="dataset_db")
        if datasets_database:
            datasets_schema = st.selectbox("Select Schema for Dataset", list_schemas(conn, datasets_database), key="dataset_schema")

            if datasets_schema:
                tables = list_tables(conn, datasets_database, datasets_schema)
                selected_table = st.selectbox("Select Table for Dataset", tables, key="dataset_table")

                if selected_table:
                    st.success(f"Selected Table: {selected_table}")
                    if st.button("Load Dataset"):
                        # Read table in to dataframe
                        query = f"SELECT * FROM {datasets_database}.{datasets_schema}.{selected_table}"
                        df = pd.read_sql(query, conn)
                        st.session_state['dataset_df'] = df
                        st.dataframe(df.head())

    # --- Step 3: Run Inference ---
    st.header("3. Run Inference")
    if 'dataset_df' in st.session_state and 'selected_pipeline' in st.session_state:
        df = st.session_state['dataset_df']
        pipeline = st.session_state['selected_pipeline']

        # Ensure the pipeline has a preprocessor and required columns
        preprocessor = pipeline.named_steps.get('preprocessor', None)
        if preprocessor is None:
            st.error("The pipeline does not contain a preprocessor step.")
            return

        required_columns = preprocessor.feature_names_in_
        if not set(required_columns).issubset(df.columns):
            st.error(f"The table must contain the following columns: {required_columns}")
            return

        # Filter to required columns
        input_df = df[required_columns]

        # Apply the pipeline to make predictions
        if st.button("Run Inference"):
            try:
                predictions = pipeline.predict(input_df)
                result_df = input_df.copy()
                result_df['Prediction'] = predictions
                st.session_state['scored_df'] = result_df
                st.success("Inference completed successfully!")
                st.dataframe(result_df.head())
            except Exception as e:
                st.error(f"Error during inference: {e}")

    # --- Save Results Section ---
    st.subheader("Save Scored Results")

    # Initialize scored_df to prevent errors
    if "scored_df" not in st.session_state:
        st.session_state['scored_df'] = None

    save_database = st.selectbox("Select Database to Save Results", list_databases(get_snowflake_connection()), key="save_db")
    save_schema = st.selectbox("Select Schema to Save Results", list_schemas(get_snowflake_connection(), save_database), key="save_schema")
    new_table_name = st.text_input("Enter New Table Name")

    # Save the results only if 'scored_df' exists
    if st.button("Save Results"):
        if st.session_state['scored_df'] is not None:
            if new_table_name:
                try:
                    # Clean the DataFrame
                    cleaned_df = clean_dataframe_for_snowflake(st.session_state['scored_df'])

                    # Save to Snowflake
                    save_to_snowflake(save_database, save_schema, new_table_name, cleaned_df)
                    st.success(f"Scored results saved to {save_database}.{save_schema}.{new_table_name}!")
                except Exception as e:
                    st.error(f"Error saving data: {e}")
            else:
                st.error("Please provide a valid table name.")
        else:
            st.error("No scored results found. Please run inference first.")