import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import ttest_ind, f_oneway
from sklearn.ensemble import RandomForestClassifier

def plot_pairplot(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis object
    # Create the pairplot with the specific axis
    sns.pairplot(df[numeric_columns])
    # Pass the figure to Streamlit for rendering
    st.pyplot(fig)

def correlation_with_target(df, target_column):
    correlation = df.corr()[target_column].sort_values(ascending=False)
    st.write(f"Correlation with Target: {target_column}")
    st.bar_chart(correlation)

def check_skewness_kurtosis(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        skew_value = skew(df[column].dropna())
        kurt_value = kurtosis(df[column].dropna())
        st.write(f"Skewness of {column}: {skew_value:.2f}")
        st.write(f"Kurtosis of {column}: {kurt_value:.2f}")

def plot_class_imbalance(df, target_column):
    st.subheader(f"Class Imbalance in {target_column}")
    class_counts = df[target_column].value_counts()
    st.bar_chart(class_counts)


def detect_outliers_zscore(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        z_scores = stats.zscore(df[column].dropna())
        outliers = df[(z_scores > 3) | (z_scores < -3)]
        st.write(f"Outliers detected in {column} (Z-score > 3 or < -3): {outliers.shape[0]} rows")


def plot_heatmap(df):
    # Filter for numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    # Allow the user to select which columns to include in the heatmap
    selected_columns = st.multiselect(
        "Select Columns for Heatmap", 
        options=numeric_df.columns, 
        default=numeric_df.columns.tolist()  # Default: include all numeric columns
    )
    # If no columns are selected, show a message and return
    if not selected_columns:
        st.write("Please select at least one column to include in the heatmap.")
        return
    # Calculate the correlation matrix for the selected columns
    corr = numeric_df[selected_columns].corr()
    st.subheader("Correlation Heatmap")
    # Create the heatmap
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
    # Display the heatmap in the Streamlit app
    st.pyplot(fig)

def isolation_forest_outliers(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        iso_forest = IsolationForest(contamination=0.05)  # 5% contamination
        df['outlier'] = iso_forest.fit_predict(df[[column]])
        outliers = df[df['outlier'] == -1]
        st.write(f"Outliers detected in {column} using Isolation Forest: {outliers.shape[0]}")
        st.write(outliers)

def feature_importance(df, target_column):
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle non-numeric columns by encoding them using LabelEncoder
    label_encoder = LabelEncoder()
    
    # Apply LabelEncoder to each non-numeric column
    for column in X.columns:
        if X[column].dtype == 'object':  # Check if the column is non-numeric
            X[column] = label_encoder.fit_transform(X[column].astype(str))  # Convert to numeric labels
    
    # Initialize the RandomForest model
    model = RandomForestClassifier()
    # Fit the model
    model.fit(X, y)

    # Extract feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    # Plot feature importances
    st.bar_chart(feature_importances.sort_values(ascending=False))
