import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from eda_functions import plot_class_imbalance, plot_heatmap  # Import EDA functions if needed
from evaluation import *
from explainability import *
from model_config import *


def ml_page(df):
    st.title("Machine Learning Model Development")

    # Check if the modified dataframe exists in session_state (from feature engineering).
    # If modified dataset exists, use that for modeling, otherwise use base dataset.
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

    # Handle missing values (same process as before)
    st.sidebar.subheader("Handle Missing Values")
    missing_action = st.sidebar.radio("What to do with missing values?", ["Drop rows", "Fill with mean/median", "Fill with mode"])
    
    if missing_action == "Drop rows":
        df_selected = df_selected.dropna()
    elif missing_action == "Fill with mean/median":
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
        df_selected[numeric_cols] = df_selected[numeric_cols].fillna(df_selected[numeric_cols].mean())
    elif missing_action == "Fill with mode":
        categorical_cols = df_selected.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_selected[col] = df_selected[col].fillna(df_selected[col].mode()[0])

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

    # Ensure the model_type matches the expected values
    if model_type not in ["Random Forest", "Logistic Regression"]:
        st.error(f"Invalid model type selected: {model_type}. Please choose from 'Random Forest' or 'Logistic Regression'.")
        return  # Exit early to prevent further processing
    
    # Hyperparameter tuning options
    if model_type == "Random Forest":
        param_grid = {"n_estimators": [100, 200], "max_depth": [10, 20, 30]}
    elif model_type == "Logistic Regression":
        param_grid = {"C": [0.1, 1, 10]}

    model = create_model(model_type)

    # Hyperparameter tuning (optional)
    use_random_search = st.sidebar.checkbox("Use Randomized Search for Hyperparameter Tuning")
    if use_random_search:
        best_model, best_params = tune_model(model, X_train, y_train, param_grid, use_random_search=True)
        st.write(f"Best Model Parameters: {best_params}")
    else:
        best_model, best_params = tune_model(model, X_train, y_train, param_grid, use_random_search=False)
        st.write(f"Best Model Parameters: {best_params}")

    # Train the model
    if st.sidebar.button("Train Model"):
        model.fit(X_train, y_train)

        # Save the trained model to session_state for deployment
        st.session_state['best_model'] = model

        # Feedback to the user
        st.success("Model has been trained and saved successfully!")

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Evaluate model
        accuracy, report, roc_auc = evaluate_classification_model(best_model, X_test, y_test)

        st.subheader(f"Model Evaluation - Accuracy: {accuracy:.4f}")
        st.text(report)
        if roc_auc is not None:
            st.write(f"ROC AUC: {roc_auc:.4f}")

        # Show residual plot for regression (if regression model is selected)
        if model_type == "Logistic Regression":
            plot_residuals(y_test, y_pred)
            show_confusion_matrix(best_model, X_test, y_test)

        # Show feature importance (for tree-based models like Random Forest)
        if model_type == "Random Forest":
            feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
            st.subheader("Feature Importance")
            st.bar_chart(feature_importances.sort_values(ascending=False))
        
        st.subheader(f"ROC CURVE")
        plot_roc_curve(best_model, X_test, y_test)
        st.subheader(f"Cross Validation")
        cross_validation(best_model, X, y)
        st.subheader(f"Feature Importance")
        plot_feature_importance_final(best_model, X_train, X.columns)
        st.subheader(f"Permutation Plot")
        permutation_importance_plot(best_model,X_train, y_train, 'accuracy')
        st.subheader(f"Calibration Curve")
        calibration_curve_plot(best_model,X_test, y_test)


