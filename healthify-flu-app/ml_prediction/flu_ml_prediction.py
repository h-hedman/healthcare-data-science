# Author: Healthify
# Date: 2025-03-09
# Goal: Predict high flu incidence by age and race using logistic regression
# --------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Handling class imbalance
import seaborn as sns
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------------------------------
# Define file paths
age_file = 'cleaned_flu_ml_age.csv' # r"G:\My Drive\Projects\OMSCS\Courses\CS 6440 - Introduction to Health Informatics\Assignments\IHI - Practicum Project\ml_prediction\cleaned_flu_ml_age.csv"
race_file = 'cleaned_flu_ml_race.csv' # r"G:\My Drive\Projects\OMSCS\Courses\CS 6440 - Introduction to Health Informatics\Assignments\IHI - Practicum Project\ml_prediction\cleaned_flu_ml_race.csv"
# --------------------------------------------------------------------------------------------------
# Define output CSV file
output_dir = 'out' # r"G:\My Drive\Projects\OMSCS\Courses\CS 6440 - Introduction to Health Informatics\Assignments\IHI - Practicum Project\ml_prediction"
output_csv = os.path.join(output_dir, "flu_ml_results.csv")
# --------------------------------------------------------------------------------------------------
# Load datasets
df_age = pd.read_csv(age_file)
df_race = pd.read_csv(race_file)
# --------------------------------------------------------------------------------------------------
# Function to prepare data for logistic regression
def prepare_data(df, feature_col, additional_features=None):
    """
    pre-process the data for logistic regression
    """
    df = df.dropna(subset=[feature_col, "high_incidence"])  # Drop missing values

    # Encode categorical feature
    le = LabelEncoder()
    df.loc[:, feature_col] = le.fit_transform(df[feature_col])  # Fix SettingWithCopyWarning

    # Select features
    feature_cols = [feature_col]  
    if additional_features:
        feature_cols.extend(additional_features)  # Add more predictors
    
    X = df[feature_cols]  # Predictor variables
    y = df["high_incidence"]  # Binary target

    return X, y, le  # Return label encoder for reference

# Prepare data for age-based prediction (including YEAR and WEEK)
X_age, y_age, age_encoder = prepare_data(df_age, "clean_age_category" , [ "WEEK"])#"YEAR.1",

# Prepare data for race-based prediction (including YEAR and WEEK)
X_race, y_race, race_encoder = prepare_data(df_race, "RACE.CATEGORY" , [ "WEEK"])#"YEAR.1",
# --------------------------------------------------------------------------------------------------
# Function to train, evaluate, and save results to CSV
def train_and_evaluate(X, y, dataset_name, output_csv):
    """
    BAM!! Let's start training and make a CSV with our output!
    """
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train logistic regression model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_resampled, y_train_resampled)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute accuracy & classification report
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)  # Convert to dict

    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    report_df["dataset"] = dataset_name  # Add dataset label
    report_df["accuracy"] = accuracy  # Add accuracy as a separate column

    # Save results to a CSV file (append mode)
    report_df.to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=True)

    # Print confirmation
    print(f"Results saved to: {output_csv}")

    # Generate confusion matrix
    plot_confusion_matrix(y_test, y_pred, dataset_name)
# --------------------------------------------------------------------------------------------------
# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, dataset_name):
    """
    plot confusion matrix for graphical analysis
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.show()
# --------------------------------------------------------------------------------------------------
# Train and evaluate models for age-based and race-based datasets
train_and_evaluate(X_age, y_age, "Age-Based Model", output_csv)
train_and_evaluate(X_race, y_race, "Race-Based Model", output_csv)
# --------------------------------------------------------------------------------------------------
