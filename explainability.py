import shap
import random
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.calibration import calibration_curve

# Function to plot feature importance (for tree-based models)
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        sorted_idx = feature_importance.argsort()
        plt.barh(feature_names[sorted_idx], feature_importance[sorted_idx])
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.show()


def plot_feature_importance_final(model, X, feature_names):
    """
    Function to plot feature importance for both Logistic Regression and Random Forest models.
    
    Parameters:
    - model: Trained model (LogisticRegression or RandomForestClassifier)
    - X: Feature matrix (input data)
    - feature_names: List of feature names
    """
    # For Logistic Regression: Coefficients
    if isinstance(model, LogisticRegression):
        importance = np.abs(model.coef_[0])
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance (Logistic Regression)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        st.pyplot(fig)

    # For Random Forest: Feature Importance
    elif isinstance(model, RandomForestClassifier):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance (Random Forest)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        st.pyplot(fig)


def plot_feature_importance(model, X, feature_names):
    """
    Function to plot feature importance for both Logistic Regression and Random Forest models.
    
    Parameters:
    - model: Trained model (LogisticRegression or RandomForestClassifier)
    - X: Feature matrix (input data)
    - feature_names: List of feature names
    """
    # For Logistic Regression: Coefficients
    if isinstance(model, LogisticRegression):
        importance = np.abs(model.coef_[0])
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance (Logistic Regression)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        st.pyplot(fig)

    # For Random Forest: Feature Importance
    elif isinstance(model, RandomForestClassifier):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance (Random Forest)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        st.pyplot(fig)


def calibration_curve_plot(best_model, X_test, y_test):
    """
    Plot calibration curves for binary or multi-class classification.
    
    Parameters:
    - best_model: Trained model
    - X_test: Test feature matrix
    - y_test: Test labels
    """
    # Check the number of classes in the model
    is_binary = len(best_model.classes_) == 2

    plt.figure(figsize=(10, 8))

    if is_binary:
        # Binary classification case
        class_label = 1  # By default, evaluate the positive class
        y_prob = best_model.predict_proba(X_test)[:, class_label]  # Probability of the positive class
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_prob, n_bins=10
        )

        plt.plot(mean_predicted_value, fraction_of_positives, marker="o", label="Positive Class")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.title("Calibration Curve (Binary Classification)")
    else:
        # Multi-class classification case
        for class_label in range(len(best_model.classes_)):
            y_prob = best_model.predict_proba(X_test)[:, class_label]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test == class_label, y_prob, n_bins=10  # Convert to binary for the specific class
            )

            plt.plot(mean_predicted_value, fraction_of_positives, marker="o", label=f"Class {class_label}")

        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.title("Calibration Curve (Multi-class Classification)")

    # Common plot formatting
    plt.xlabel("Mean Predicted Value")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.grid()
    
    # Display the plot using Streamlit
    st.pyplot(plt)