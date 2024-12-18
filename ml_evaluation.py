import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import functions from explainability and evaluation files
from explainability import *
from evaluation import *

def display_evaluation_and_explainability():
    # Check if trained model exists in session_state
    if 'trained_pipeline' in st.session_state:
        trained_pipeline = st.session_state['trained_pipeline']
        
        if isinstance(trained_pipeline, Pipeline):
            # Assuming the last step in pipeline is the trained model
            model = trained_pipeline[-1]
            X_test = st.session_state['X_test']  # Assuming X_test is stored in session_state
            y_test = st.session_state['y_test']  # Assuming y_test is stored in session_state

            # Step 1: Model Evaluation - Classification Metrics
            st.subheader("Model Evaluation")

            # Evaluate the classification model
            accuracy, report, roc_auc = evaluate_classification_model(model, X_test, y_test)
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write("Classification Report:")
            st.text(report)
            
            # ROC Curve
            plot_roc_curve(model, X_test, y_test)

            # Confusion Matrix
            show_confusion_matrix(model, X_test, y_test)

            # Cross-Validation
            st.subheader("Cross-Validation")
            cross_validation(model, X_test, y_test)

            # Step 2: Model Explainability - Feature Importance
            st.subheader("Model Explainability")

            # Extract feature names from the preprocessor (after transformation)
            if hasattr(trained_pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                feature_names = trained_pipeline.named_steps['preprocessor'].get_feature_names_out()
            else:
                feature_names = X_test.columns  # If not a sparse matrix, fallback to original

            # Feature Importance
            plot_feature_importance(model, X_test, feature_names)
            
            # Calibration Curve
            calibration_curve_plot(model, X_test, y_test)

            # Residuals (for regression models)
            if hasattr(model, 'predict') and len(np.unique(y_test)) > 2:
                y_pred = model.predict(X_test)
                plot_residuals(y_test, y_pred)
        
        else:
            st.error("The trained pipeline is not a valid model pipeline.")
    else:
        st.error("No trained model found in session state.")


