import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


def feature_engineering(df_selected):
    # Initialize session state
    if 'temp_df' not in st.session_state:
        st.session_state.temp_df = df_selected.copy()
    if 'history' not in st.session_state:
        st.session_state.history = [df_selected.copy()]
    if 'history_index' not in st.session_state:
        st.session_state.history_index = 0

    # Temporary DataFrame for transformations
    temp_df = st.session_state.temp_df

    # Layout: Two columns for transformations and dataset preview
    col_left, col_right = st.columns([2, 3])
    
    with col_left:
        st.subheader("Feature Engineering Options")
        
        # Undo/Redo Buttons
        undo_redo_col = st.columns([1, 1])
        with undo_redo_col[0]:
            if st.button("Undo"):
                if st.session_state.history_index > 0:
                    st.session_state.history_index -= 1
                    temp_df = st.session_state.history[st.session_state.history_index]
                    st.session_state.temp_df = temp_df
                    st.success("Undone last change.")
        with undo_redo_col[1]:
            if st.button("Redo"):
                if st.session_state.history_index < len(st.session_state.history) - 1:
                    st.session_state.history_index += 1
                    temp_df = st.session_state.history[st.session_state.history_index]
                    st.session_state.temp_df = temp_df
                    st.success("Redone last undone change.")
        
        # Feature Engineering Method Selection
        feature_engineering_method = st.radio(
            "Select a feature engineering method:", 
            ["Predefined Transformations", "Interactive Feature Creation", "One-Hot Encoding"], 
            horizontal=True
        )

        # Predefined Transformations
        if feature_engineering_method == "Predefined Transformations":
            st.markdown("### Predefined Transformations")
            transformation_columns = st.multiselect("Select columns for transformation", temp_df.columns)
            transformation = st.radio("Choose a transformation:", 
                                       ["Standardize", "Log Transform", "Create Polynomial Features"], 
                                       horizontal=True)

            if transformation == "Standardize":
                if st.button("Apply Standardization"):
                    scaler = StandardScaler()
                    temp_df[transformation_columns] = scaler.fit_transform(temp_df[transformation_columns])
                    st.success(f"Standardization applied to {', '.join(transformation_columns)}.")

            elif transformation == "Log Transform":
                if st.button("Apply Log Transformation"):
                    temp_df[transformation_columns] = temp_df[transformation_columns].apply(np.log1p)
                    st.success(f"Log transformation applied to {', '.join(transformation_columns)}.")

            elif transformation == "Create Polynomial Features":
                degree = st.slider("Select polynomial degree", 2, 5, 2)
                new_col_prefix = st.text_input("Prefix for new polynomial features", "POLY")
                if st.button("Apply Polynomial Transformation"):
                    for col in transformation_columns:
                        for i in range(2, degree + 1):
                            column_name = f"{new_col_prefix}_{col}_{i}"
                            temp_df[column_name] = temp_df[col] ** i
                    st.success(f"Polynomial transformation applied to {', '.join(transformation_columns)}.")

        # Interactive Feature Creation
        elif feature_engineering_method == "Interactive Feature Creation":
            st.markdown("### Interactive Feature Creation")
            col1 = st.selectbox("Select first column:", temp_df.columns, key="col1")
            operation = st.selectbox("Select operation:", ["+", "-", "*", "/"], key="operation")
            col2 = st.selectbox("Select second column:", temp_df.columns, key="col2")
            new_feature_name = st.text_input("Name for the new feature", f"{col1}_{operation}_{col2}")

            if st.button("Apply Interactive Feature Creation"):
                if operation == "+":
                    temp_df[new_feature_name] = temp_df[col1] + temp_df[col2]
                elif operation == "-":
                    temp_df[new_feature_name] = temp_df[col1] - temp_df[col2]
                elif operation == "*":
                    temp_df[new_feature_name] = temp_df[col1] * temp_df[col2]
                elif operation == "/":
                    temp_df[new_feature_name] = temp_df[col1] / temp_df[col2]
                st.success(f"New feature '{new_feature_name}' created.")

        # One-Hot Encoding
        elif feature_engineering_method == "One-Hot Encoding":
            st.markdown("### One-Hot Encoding")
            categorical_cols = st.multiselect("Select columns for One-Hot Encoding", temp_df.select_dtypes(include=['object']).columns)
            
            if st.button("Apply One-Hot Encoding"):
                temp_df = pd.get_dummies(temp_df, columns=categorical_cols, drop_first=True)
                st.success(f"One-Hot Encoding applied to {', '.join(categorical_cols)}.")
        
        # Commit Changes
        if st.button("Commit Changes"):
            st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
            st.session_state.history.append(temp_df.copy())
            st.session_state.history_index += 1
            st.session_state.temp_df = temp_df
            st.session_state.modified_df = temp_df  # Save modified dataset here
            st.success("Changes have been committed to the dataset.")

    with col_right:
        st.subheader("Current Dataset")
        st.write(temp_df)
        st.write(f"Shape: {temp_df.shape}")
        st.write(f"Columns: {', '.join(temp_df.columns)}")
