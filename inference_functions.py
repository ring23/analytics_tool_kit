from snowflake_connector import *
import streamlit as st
import pickle
import pandas as pd
import os
import gzip
import tempfile
from inference_automation import *
from sklearn.pipeline import Pipeline
from snowflake.connector.pandas_tools import write_pandas
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
        # Resolve the local directory
        local_dir = local_directory if local_directory else tempfile.gettempdir()
        os.makedirs(local_dir, exist_ok=True)

        # Construct the GET query
        stage_path = f"@{database}.{schema}.{stage}/{file_name}"
        query = f"GET {stage_path} file://{local_dir}/"
        st.write(f"Executing query: {query}")  # Debugging statement
        cursor.execute(query)

        # Check if file downloaded
        downloaded_file_path = os.path.join(local_dir, os.path.basename(file_name))
        if not os.path.exists(downloaded_file_path):
            raise FileNotFoundError(f"File not downloaded: {downloaded_file_path}")

        # Load the model (assuming gzipped pickle format)
        st.write(f"Loading model from: {downloaded_file_path}")  # Debugging statement
        with gzip.open(downloaded_file_path, 'rb') as model_file:
            model = pickle.load(model_file)

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    finally:
        cursor.close()
        conn.close()

        # Only delete the downloaded file if it exists
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
    
def save_to_snowflake(database, schema, table_name, dataframe):
    """
    Save a cleaned Pandas DataFrame to Snowflake using the write_pandas utility.
    """
    conn = get_snowflake_connection()

    try:
        # Set the target schema
        conn.cursor().execute(f"USE SCHEMA {database}.{schema}")

        # Clean the DataFrame to ensure compatibility with Snowflake
        dataframe = clean_dataframe_for_snowflake(dataframe)

        # Write the DataFrame to Snowflake
        success, nchunks, nrows, output = write_pandas(
            conn,
            dataframe,
            table_name.upper(),  # Snowflake expects uppercase table names
            overwrite=True  # Replace the table if it exists
        )

        if not success:
            raise RuntimeError(f"Failed to write to Snowflake: {output}")

    except Exception as e:
        raise RuntimeError(f"Error saving data to Snowflake: {e}")

    finally:
        conn.close()