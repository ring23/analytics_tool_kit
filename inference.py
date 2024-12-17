from snowflake_connector import *
import streamlit as st
import pickle
import pandas as pd
import os
import gzip
import tempfile
# Fetch files from a Snowflake stage
def list_files_in_stage(database, schema, stage_name):
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Formulate the LIST command
        query = f"LIST @{database}.{schema}.{stage_name};"
        print(f"Executing query: {query}")  # Debugging output

        # Execute the query
        cursor.execute(query)

        # Fetch and display the results
        files = [row[0] for row in cursor.fetchall()]  # Extract file names
        print(f"Files in stage: {files}")

        return files

    except snowflake.connector.errors.ProgrammingError as e:
        print(f"SQL error occurred: {str(e)}")
        return []

    finally:
        cursor.close()
        conn.close()

# Function to query dataset from Snowflake
def get_dataset(database, schema, table):
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    query = f"SELECT * FROM {database}.{schema}.{table}"
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(rows, columns=columns)
    return df

# Function to load a saved model
def load_model(model_name):
    # Assuming models are serialized and stored in a binary format in Snowflake stage
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    query = f"SELECT MODEL_BINARY FROM {model_name}"  # Modify to your Snowflake schema
    cursor.execute(query)
    model_binary = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    # Deserialize the model (assuming it is a pickled model)
    import pickle
    model = pickle.loads(model_binary)
    return model


# Function to run inference on the dataset
def run_inference(model, dataset):
    if model is None:
        st.error("Model is not loaded correctly, cannot run inference.")
        return None
    
    # Check if model has 'predict' method
    if hasattr(model, 'predict'):
        try:
            return model.predict(dataset)
        except Exception as e:
            st.error(f"Error during inference: {e}")
            return None
    else:
        st.error("The loaded model does not have a 'predict' method.")
        return None

# Streamlit UI for dataset selection and inference
def select_dataset_and_inference():
    database = st.selectbox("Select Database", ["db1", "db2"])  # Dynamically populate
    schema = st.selectbox("Select Schema", ["schema1", "schema2"])  # Dynamically populate
    table = st.selectbox("Select Dataset", ["dataset1", "dataset2"])  # Dynamically populate

    df = get_dataset(database, schema, table)
    st.write(f"Dataset preview:", df.head())

def load_model_from_stage(database, schema, stage, file_name, local_directory=None):
    """Download the model file from the Snowflake stage and load it using pickle."""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    downloaded_file_path = None

    try:
        # Determine the local directory
        local_dir = local_directory if local_directory else tempfile.gettempdir()
        os.makedirs(local_dir, exist_ok=True)

        # Fetch model file from stage
        query = f"GET @{database}.{schema}.{stage}/{file_name} file://{local_dir}/"
        cursor.execute(query)

        # Resolve file path
        downloaded_file_path = os.path.join(local_dir, os.path.basename(file_name))

        # Load the model (handling gzipped pickle files)
        with gzip.open(downloaded_file_path, 'rb') as model_file:
            model = pickle.load(model_file)
            return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    finally:
        cursor.close()
        conn.close()
        # Clean up
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            os.remove(downloaded_file_path)

def save_scored_data_to_snowflake(database, schema, table_name, df):
    try:
        # Create a connection to Snowflake
        conn = get_snowflake_connection()

        # Check if the table exists in the schema
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM {database}.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}'
        """)
        table_exists = cursor.fetchone()[0] > 0

        # If the table does not exist, create it
        if not table_exists:
            create_table_sql = f"CREATE TABLE {database}.{schema}.{table_name} ("
            columns = ', '.join([f"{col} {get_snowflake_column_type(df[col])}" for col in df.columns])
            create_table_sql += columns + ")"
            cursor.execute(create_table_sql)
            st.success(f"Table '{table_name}' created successfully.")

        # Prepare the data to insert (convert DataFrame to records)
        records = df.to_dict(orient='records')

        # Ensure the columns in the DataFrame match the table schema
        columns = ', '.join(df.columns)  # Column names in the table
        values = ', '.join(['%s'] * len(df.columns))  # Placeholder for values

        # Create SQL INSERT statement
        insert_sql = f"INSERT INTO {database}.{schema}.{table_name} ({columns}) VALUES ({values})"

        # Execute the insert statement for each row
        for record in records:
            cursor.execute(insert_sql, tuple(record.values()))  # Execute the insert with the row values
        conn.commit()

        cursor.close()
        conn.close()

        st.success(f"Scored results saved to {table_name}!")

    except Exception as e:
        st.error(f"Error saving data: {e}")

def get_snowflake_column_type(series):
    """
    Determines the appropriate Snowflake column type based on the pandas series type.
    """
    if pd.api.types.is_integer_dtype(series):
        return "NUMBER"
    elif pd.api.types.is_float_dtype(series):
        return "FLOAT"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP_LTZ"
    else:
        return "STRING"

# Streamlit UI to select a file in a stage
# Main function for selecting model and running inference
def select_model_and_infer():
    st.title("Model Inference Tool")
    st.write("This page allows you to load a model from a Snowflake stage, run inference on a dataset, and save the results.")

    # Connect to Snowflake
    conn = get_snowflake_connection()

    # --- Step 1: Select Model File ---
    st.header("1. Select Model File")
    
    # Dropdowns for Database and Schema
    databases = list_databases(conn)
    selected_database = st.selectbox("Select Database", databases)

    if selected_database:
        schemas = list_schemas(conn, selected_database)
        selected_schema = st.selectbox("Select Schema", schemas)

        if selected_schema:
            # List Stages
            stages = list_stages(selected_database, selected_schema)
            selected_stage = st.selectbox("Select Stage", stages)

            if selected_stage:
                # List Files in Stage
                files = list_files_in_stage(selected_database, selected_schema, selected_stage)
                selected_file = st.selectbox("Select Model File", files)

                if selected_file:
                    st.success(f"Selected Model File: {selected_file}")
                    file_name = os.path.basename(selected_file)
                    # Load the Model
                    if st.button("Load Model"):
                        model = load_model_from_stage(selected_database, selected_schema, selected_stage, file_name)
                        st.session_state['trained_pipeline'] = load_model_from_stage(selected_database, selected_schema, selected_stage, file_name)
                        # model = load_model_from_stage("ANALYTICS_TOOL_KIT", "PUBLIC", "MY_STAGE2", "best_model3.pkl.gz")
                        # st.session_state['trained_pipeline'] = load_model_from_stage("ANALYTICS_TOOL_KIT", "PUBLIC", "MY_STAGE2", "best_model3.pkl.gz")
                        if model is not None:
                            st.success("Model loaded successfully!")
                            st.session_state.model = model
                        else:
                            st.error("Model is empty or not loaded correctly.")
                            return None

    # --- Step 2: Select Dataset for Inference ---
    st.header("2. Select Dataset")

    if 'model' in st.session_state:
        datasets_database = st.selectbox("Select Database for Dataset", databases, key="dataset_db")

        if datasets_database:
            datasets_schema = st.selectbox("Select Schema for Dataset", list_schemas(conn, datasets_database), key="dataset_schema")

            if datasets_schema:
                tables = list_tables(conn, datasets_database, datasets_schema)
                selected_table = st.selectbox("Select Table", tables, key="dataset_table")

                if selected_table:
                    st.success(f"Selected Table: {selected_table}")
                    if st.button("Load Dataset"):
                        df = get_dataset(datasets_database, datasets_schema, selected_table)
                        st.session_state.df = df
                        st.dataframe(df.head())

    # --- Step 3: Run Inference and Save Results ---
    st.header("3. Run Inference and Save Results")

    if 'df' in st.session_state and 'trained_pipeline' in st.session_state:
        st.write("Dataset and Model are ready for inference.")

        # Apply the saved pipeline to the newly selected dataset
        if st.button("Run Inference"):
            try:
                # --- Apply the saved pipeline to the new dataset ---
                df_selected = st.session_state.df.copy()

                # Use the saved pipeline to preprocess the data and make predictions
                predictions = st.session_state.trained_pipeline.predict(df_selected)  # Directly call predict on the pipeline

                scored_df = df_selected.copy()
                scored_df['Prediction'] = predictions
                st.session_state.scored_df = scored_df

                st.dataframe(scored_df.head())
                st.success("Inference complete!")

            except Exception as e:
                st.error(f"Error during inference: {e}")

    # Saving the scored file
    st.subheader("Save Scored Results")
    save_database = st.selectbox("Select Database to Save Results", list_databases(get_snowflake_connection()), key="save_db")
    save_schema = st.selectbox("Select Schema to Save Results", list_schemas(get_snowflake_connection(), save_database), key="save_schema")
    new_table_name = st.text_input("Enter New Table Name")

    if st.button("Save Results"):
        if new_table_name:
            save_scored_data_to_snowflake(save_database, save_schema, new_table_name, st.session_state.scored_df)
            st.success(f"Scored results saved to {new_table_name}!")
        else:
            st.error("Please provide a valid table name.")