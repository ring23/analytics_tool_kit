from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, LeaveOneOut
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import streamlit as st
import pandas as pd

# Function to create model
def create_model(model_type="Random Forest"):
    if model_type == "Random Forest":
        return RandomForestClassifier()
    elif model_type == "Logistic Regression":
        return LogisticRegression(max_iter=10000)
    elif model_type == "SVC":
        return SVC()
    else:
        raise ValueError("Invalid model type")

# Function for hyperparameter tuning
def tune_model(model, X_train, y_train, param_grid, use_random_search=False, n_splits=5):
    # Check class distribution
    class_counts = pd.Series(y_train).value_counts()
    min_class_size = class_counts.min()

    # Check if the class sizes are too small for the number of splits
    if min_class_size < n_splits:
        # If classes are too small, use LeaveOneOut cross-validation
        print(f"Classes too small for {n_splits}-fold cross-validation. Using LeaveOneOut instead.")
        cv = LeaveOneOut()
    elif class_counts.max() / class_counts.sum() > 0.9:  # If dataset is highly imbalanced
        print(f"Classes are imbalanced. Performing oversampling using SMOTE.")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        cv = StratifiedKFold(n_splits=min(min_class_size, n_splits), shuffle=True, random_state=42)
    else:
        # Normal case: Use StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Print class distribution after resampling (if applied)
    print(f"Class distribution after resampling (if applied): \n{pd.Series(y_train).value_counts()}")

    # Model tuning with GridSearchCV or RandomizedSearchCV
    if use_random_search:
        from sklearn.model_selection import RandomizedSearchCV
        model_tuner = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=10, random_state=42)
    else:
        from sklearn.model_selection import GridSearchCV
        model_tuner = GridSearchCV(model, param_grid, cv=cv)

    # Fit the model
    model_tuner.fit(X_train, y_train)

    # Return best model and parameters
    return model_tuner.best_estimator_, model_tuner.best_params_
