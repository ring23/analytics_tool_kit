import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from eda_functions import plot_class_imbalance, plot_heatmap  # Import EDA functions if needed


def ml_page(df):
    st.title("Machine Learning Model Development")

    # Select columns to drop
    st.sidebar.subheader("Column Selection")
    columns = df.columns.tolist()
    dropped_columns = st.sidebar.multiselect("Select columns to drop", columns, default=[])

    # Filter DataFrame based on selected columns
    df_selected = df.drop(columns=dropped_columns)
    st.write("Filtered Dataset", df_selected)

    # Target variable selection
    target_column = st.sidebar.selectbox("Select target variable", df_selected.columns)

    # Handle missing values
    st.sidebar.subheader("Handle Missing Values")
    missing_action = st.sidebar.radio("What to do with missing values?", ["Drop rows", "Fill with mean/median", "Fill with mode"])
    
    if missing_action == "Drop rows":
        df_selected = df_selected.dropna()
    elif missing_action == "Fill with mean/median":
        # Handle numeric columns
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
        df_selected[numeric_cols] = df_selected[numeric_cols].fillna(df_selected[numeric_cols].mean())
    elif missing_action == "Fill with mode":
        # Handle categorical columns
        categorical_cols = df_selected.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_selected[col] = df_selected[col].fillna(df_selected[col].mode()[0])

    # Show cleaned data
    st.write("Cleaned Dataset", df_selected)
    
    # Preprocessing - Encode categorical variables
    categorical_cols = df_selected.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_selected[col] = le.fit_transform(df_selected[col].astype(str))
        label_encoders[col] = le

    st.write("Dataset after Encoding", df_selected)
    
    # Train-test split
    X = df_selected.drop(columns=[target_column])
    y = df_selected[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose ML model
    st.sidebar.subheader("Model Selection")
    model_type = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression"])

    if model_type == "Random Forest":
        model = RandomForestClassifier()
    elif model_type == "Logistic Regression":
        model = LogisticRegression()

    # Train the model
    if st.sidebar.button("Train Model"):
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.subheader(f"Model Evaluation - Accuracy: {accuracy:.4f}")
        st.text(report)

        # Show feature importance (for tree-based models like Random Forest)
        if model_type == "Random Forest":
            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
            st.subheader("Feature Importance")
            st.bar_chart(feature_importances.sort_values(ascending=False))

