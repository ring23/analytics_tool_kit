import streamlit as st
import joblib
from snowflake_connector import *
import pickle

# Function to get the list of databases, schemas, and stages from Snowflake
# Corrected code to ensure proper fetching of all schemas
def get_snowflake_databases_schemas_stages():
    """Get the list of databases, schemas, and stages available in Snowflake."""
    conn = get_snowflake_connection()
    databases = []
    schemas = {}
    stages = {}

    try:
        with conn.cursor() as cur:
            # Get the list of databases
            cur.execute("SHOW DATABASES")
            for row in cur:
                databases.append(row[1])  # Database name is in the second column

            # For each database, get the schemas and stages
            for db in databases:
                schemas[db] = []
                stages[db] = {}
                cur.execute(f"SHOW SCHEMAS IN DATABASE {db}")
                for row in cur:
                    schema_name = row[1]  # Schema name is in the second column
                    schemas[db].append(schema_name)

                    # Get the list of stages in this schema
                    stages[db][schema_name] = []
                    cur.execute(f"SHOW STAGES IN SCHEMA {db}.{schema_name}")
                    for stage_row in cur:
                        stages[db][schema_name].append(stage_row[1])  # Stage name is in the second column

    except Exception as e:
        st.error(f"Error fetching databases, schemas, and stages: {e}")
    
    return databases, schemas, stages


# Function to create a new stage in Snowflake
def create_snowflake_stage(db_name, schema_name, stage_name):
    """Creates a new stage in Snowflake."""
    conn = get_snowflake_connection()

    try:
        with conn.cursor() as cur:
            # Set the database and schema to use
            cur.execute(f"USE DATABASE {db_name}")
            cur.execute(f"USE SCHEMA {schema_name}")
            
            # Create the new stage
            cur.execute(f"CREATE STAGE {stage_name}")
            st.success(f"Successfully created new stage: {stage_name} in {db_name}.{schema_name}")
    except Exception as e:
        st.error(f"Failed to create new stage: {e}")

# Function to upload the model to Snowflake stage
def upload_to_snowflake(model_file_name, db_name, schema_name, stage_name):
    """Uploads the model to the Snowflake stage."""
    conn = get_snowflake_connection()

    try:
        with conn.cursor() as cur:
            # Set the database and schema to use
            cur.execute(f"USE DATABASE {db_name}")
            cur.execute(f"USE SCHEMA {schema_name}")
            
            # First, upload the model file to Snowflake stage
            cur.execute(f"PUT file://{model_file_name} @{stage_name} AUTO_COMPRESS=TRUE OVERWRITE = TRUE")
            st.success(f"Successfully uploaded {model_file_name} to Snowflake stage: {stage_name}")
    except Exception as e:
        st.error(f"Failed to upload model to Snowflake stage: {e}")

def deploy_model_page(best_model, session_state):
    """
    Model deployment page for saving and deploying the trained model.

    Parameters:
    - best_model: Trained machine learning model
    - session_state: Streamlit session state object
    """
    st.title("Model Deployment")
    st.write("Save and deploy your trained model for production.")

    # User inputs for database, schema, stage, and model name
    db_name = st.text_input("Enter the name of the database:", value="my_database")
    schema_name = st.text_input("Enter the name of the schema:", value="public")
    stage_name = st.text_input("Enter the name of the stage:", value="my_stage")
    model_name = st.text_input("Enter a name for your model:", value="my_model")

    # Validate that all required inputs are provided
    if not db_name or not schema_name or not stage_name or not model_name:
        st.error("Please enter a valid database name, schema name, stage name, and model name.")
        return

    # Convert the stage name to uppercase to match Snowflake's naming conventions
    stage_name_upper = stage_name.upper()

    # Add button for saving model
    if st.button("Save Model to Snowflake"):
        # Connect to Snowflake
        conn = get_snowflake_connection()

        try:
            with conn.cursor() as cur:
                # Set the database and schema context
                cur.execute(f"USE DATABASE {db_name}")
                cur.execute(f"USE SCHEMA {schema_name}")

                # Fetch the list of stages in the schema
                cur.execute(f"SHOW STAGES IN SCHEMA {db_name}.{schema_name}")
                stages_in_schema = [row[1].upper() for row in cur]  # Get stage names and convert to uppercase

                # Check if the stage exists
                if stage_name_upper not in stages_in_schema:
                    # If the stage doesn't exist, create it
                    cur.execute(f"CREATE STAGE {stage_name_upper}")
                    st.success(f"Stage {stage_name_upper} created in {db_name}.{schema_name}.")
                else:
                    st.success(f"Stage {stage_name_upper} already exists in {db_name}.{schema_name}.")

                # Save model to Snowflake stage
                model_file_name = f"{model_name}.pkl"  # Use model name from user input
                with open(model_file_name, "wb") as file:
                    pickle.dump(best_model, file)

                # Upload model to Snowflake stage
                cur.execute(f"PUT file://{model_file_name} @{stage_name_upper}")
                st.success(f"Model saved to Snowflake stage: {db_name}.{schema_name}.{stage_name_upper}/{model_file_name}")

        except Exception as e:
            st.error(f"Failed to save model to Snowflake: {e}")