import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from eda_functions import plot_class_imbalance, plot_heatmap  # Import EDA functions if needed
from evaluation import *
from explainability import *
from model_config import *


# Function for feature preprocessing
def create_preprocessing_pipeline(numeric_cols, categorical_cols):
    # Define the transformations for numeric and categorical columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def ml_page(df):
    st.title("Machine Learning Model Development")

    # Check if the modified dataframe exists in session_state (from feature engineering).
    if 'modified_df' in st.session_state:
        df_selected = st.session_state.modified_df  # Use the modified dataset
    elif 'temp_df' in st.session_state:
        df_selected = st.session_state.temp_df
    else:
        st.warning("No dataset found. Please go to the Feature Engineering page and commit changes.")
        return
    
    # Select columns to drop
    st.sidebar.subheader("Column Selection")
    columns = df_selected.columns.tolist()
    dropped_columns = st.sidebar.multiselect("Select columns to drop", columns, default=[])

    # Filter DataFrame based on selected columns
    df_selected = df_selected.drop(columns=dropped_columns)
    st.write("Filtered Dataset", df_selected)

    # Target variable selection
    target_column = st.sidebar.selectbox("Select target variable", df_selected.columns)

    # Handle missing values for the target variable explicitly
    if target_column == 'PROFESSION' and df_selected[target_column].isnull().sum() > 0:
        st.sidebar.subheader("Handle Missing Values in Target Variable")
        target_missing_action = st.sidebar.radio("How to handle missing values in the target?", 
                                                 ["Drop rows", "Fill with mode", "Fill with default"])
        
        if target_missing_action == "Drop rows":
            df_selected = df_selected.dropna(subset=[target_column])
        elif target_missing_action == "Fill with mode":
            mode_value = df_selected[target_column].mode()[0]
            df_selected[target_column] = df_selected[target_column].fillna(mode_value)
        elif target_missing_action == "Fill with default":
            default_value = "Unknown"  # Or any other default value
            df_selected[target_column] = df_selected[target_column].fillna(default_value)

    # Handle missing values for other columns (features)
    st.sidebar.subheader("Handle Missing Values in Features")
    missing_action = st.sidebar.radio("What to do with missing values in features?", 
                                     ["Drop rows", "Fill with mean/median", "Fill with mode"])
    
    if missing_action == "Drop rows":
        df_selected = df_selected.dropna()
    elif missing_action == "Fill with mean/median":
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
        df_selected[numeric_cols] = df_selected[numeric_cols].fillna(df_selected[numeric_cols].mean())
    elif missing_action == "Fill with mode":
        categorical_cols = df_selected.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_selected[col] = df_selected[col].fillna(df_selected[col].mode()[0])

    # Train-test split
    X = df_selected.drop(columns=[target_column])
    y = df_selected[target_column]

    # Ensure that there are no missing values in target variable y
    if y.isnull().sum() > 0:
        st.error(f"Target variable '{target_column}' contains missing values. Please handle them before proceeding.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Confirm the types and columns of X_train and X_test
    st.write(f"X_train columns: {X_train.columns.tolist()}")
    st.write(f"X_test columns: {X_test.columns.tolist()}")

    # Ensure X_train and X_test are DataFrames with column names
    X_train = pd.DataFrame(X_train, columns=X.columns)  # Restore column names for X_train
    X_test = pd.DataFrame(X_test, columns=X.columns)  # Restore column names for X_test

    # Define numeric and categorical columns based on the dataframe
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Select the model type
    model_type = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression"])

    # Set the model based on the selected model type
    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(random_state=42)

    # Create the full pipeline: preprocessing + model
    pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessing_pipeline(numeric_cols, categorical_cols)),
        ('model', model)
    ])

    # Debugging step: Check the pipeline before fitting
    st.write("Pipeline before fitting:")
    st.write(pipeline)

    # Train the model when the button is pressed
    if st.sidebar.button("Train Model"):
        # Fit the pipeline with training data
        st.write("Fitting the pipeline with X_train and y_train...")
        pipeline.fit(X_train, y_train)

        # Save the trained pipeline (both preprocessing and model) to session_state
        st.session_state['trained_pipeline'] = pipeline
        # Save the model object to session_state
        st.session_state['best_model'] = pipeline.named_steps['model']

        # Feedback to the user
        st.success("Model and pipeline have been trained and saved successfully!")

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.subheader(f"Model Evaluation - Accuracy: {accuracy:.4f}")
        st.text(report)