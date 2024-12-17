import streamlit as st
from snowflake_connector import *
import pandas as pd
from sklearn.pipeline import Pipeline

# Prediction Page
def prediction_page():
    st.title("Apply Trained Model to Selected Table")

    # Check if trained pipeline exists
    if 'trained_pipeline' not in st.session_state:
        st.error("No trained pipeline found. Please train a model first on the Machine Learning page.")
        return
    
    pipeline = st.session_state['trained_pipeline']  # Load trained pipeline

    st.write(f"Type of stored pipeline: {type(st.session_state['trained_pipeline'])}")

    # Ensure that pipeline is an instance of Pipeline
    if not isinstance(pipeline, Pipeline):
        st.error("Stored object is not a valid pipeline.")
        return

    # Connect to Snowflake
    conn = get_snowflake_connection()

    # List databases and allow the user to select one
    databases = list_databases(conn)
    selected_database = st.selectbox("Select Database", databases)

    if selected_database:
        # List schemas for the selected database
        schemas = list_schemas(conn, selected_database)
        selected_schema = st.selectbox("Select Schema", schemas)

        if selected_schema:
            # List tables for the selected schema
            tables = list_tables(conn, selected_database, selected_schema)
            selected_table = st.selectbox("Select Table", tables)

            if selected_table:
                # Load the table into a DataFrame
                query = f"SELECT * FROM {selected_database}.{selected_schema}.{selected_table}"
                df = pd.read_sql(query, conn)
                st.write("Selected Table Data:", df)

                # Ensure that the pipeline has the required preprocessor step
                preprocessor = pipeline.named_steps.get('preprocessor', None)
                if preprocessor is None:
                    st.error("The pipeline does not contain a preprocessor step.")
                    return
                
                # Get the list of required columns for the model
                required_columns = preprocessor.feature_names_in_
                
                # Check if required columns are present in the dataframe
                if not set(required_columns).issubset(df.columns):
                    st.error(f"The table must contain the following columns: {required_columns}")
                    return

                # Filter to required columns
                input_df = df[required_columns]

                # Apply the pipeline to make predictions
                predictions = pipeline.predict(input_df)

                # Display predictions
                st.subheader("Predictions")
                st.write(predictions)

                # Optionally save predictions
                save_button = st.button("Save Predictions to CSV")
                if save_button:
                    result_df = input_df.copy()
                    result_df['Prediction'] = predictions
                    result_df.to_csv("predictions.csv", index=False)
                    st.success("Predictions saved as 'predictions.csv'.")
