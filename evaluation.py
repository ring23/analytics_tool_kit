import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay, roc_curve, auc, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize
import numpy as np
import streamlit as st
import shap


# Function to plot residuals (for regression models)
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    st.pyplot(plt)

# Function to evaluate classification models
def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    # CALCULATE ROC AUC
    if hasattr(model, 'predict_proba'):
        if len(set(y_test)) == 2:  # Binary classification
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:  # Multi-class classification
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')  # One-vs-Rest
    else:
        roc_auc = None
    return accuracy, report, roc_auc

def show_confusion_matrix(model, X_test, y_test):
    """
    This function computes and displays the confusion matrix for a given model.

    Parameters:
    - model: Trained classification model
    - X_test: Test features
    - y_test: True labels of the test set
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique class labels from y_test (fallback if model.classes_ is unavailable)
    classes = np.unique(y_test)
    
    # Create a ConfusionMatrixDisplay object for plotting
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    st.pyplot(fig)

def plot_roc_curve(model, X_test, y_test):
    # Check if the problem is binary or multi-class
    if len(set(y_test)) == 2:  # Binary classification
        # Predict probabilities for the positive class
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        st.pyplot(plt)
        
        return roc_auc
    
    else:  # Multi-class classification
        # Binarize the labels for multi-class ROC curve
        y_test_bin = label_binarize(y_test, classes=model.classes_)
        n_classes = y_test_bin.shape[1]
        
        # Predict probabilities for all classes
        y_prob = model.predict_proba(X_test)
        
        # Initialize variables for ROC curve
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {model.classes_[i]} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (Multi-Class)')
        plt.legend(loc='lower right')
        st.pyplot(plt)
        
        return roc_auc
    
def cross_validation(model, X, y, cv=5, scoring='accuracy'):
    """
    Function to perform cross-validation on a model and display results.
    
    Parameters:
    - model: The model to evaluate (e.g., LogisticRegression, RandomForestClassifier)
    - X: The feature matrix (input data)
    - y: The target vector (labels)
    - cv: Number of cross-validation folds (default is 5)
    - scoring: Scoring metric (default is 'accuracy')
    """
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv), scoring=scoring)
    
    # Display results as text
    st.write(f"Cross-Validation Results ({scoring}):")
    st.write(f"Mean {scoring.capitalize()}: {np.mean(cv_scores):.4f}")
    st.write(f"Standard Deviation: {np.std(cv_scores):.4f}")
    
    # Display boxplot for cross-validation results
    st.subheader(f"Boxplot of {scoring.capitalize()} scores")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(cv_scores)
    ax.set_title(f'{scoring.capitalize()} Scores Distribution')
    ax.set_ylabel(scoring.capitalize())
    st.pyplot(fig)