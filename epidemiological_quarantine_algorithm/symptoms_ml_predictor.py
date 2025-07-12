# -------------------------------------------------------------------------------------------------------
# Author: Hayden Hedman
# Date: 2025-07-09 (first version used in fall of 2021)
# Description: Conducts machine learning-based triage on a public health epidemiological quarantine algorithm dataset
# Note: All data is randomly generated and does NOT represent real individuals or cases.
# -------------------------------------------------------------------------------------------------------
# load libraries
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import numpy as np
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# File paths and seed
# -----------------------------------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_PATH = INPUT_DIR / "historical_bus_registry.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "ml_predictions.csv"
SENSITIVITY_PATH = OUTPUT_DIR / "sensitivity_summary.csv"
SEED = 88
TODAY = datetime.today().date()
# -----------------------------------------------------------------------------------------------------
# Load raw dataset
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# File paths and seed
# -----------------------------------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = INPUT_DIR / "historical_bus_registry.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "ml_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "ml_summary_table.csv"
SEED = 88
TODAY = datetime.today().date()

# -----------------------------------------------------------------------------------------------------
# Load raw dataset
# -----------------------------------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["arrival_date"])
print("flu_status:\n", df["flu_status"].value_counts())
print("covid_status:\n", df["covid_status"].value_counts())
print("varicella_status:\n", df["varicella_status"].value_counts())
print("lice_status:\n", df["lice_status"].value_counts())
print("temp (summary):\n", df["temp"].describe())

# -----------------------------------------------------------------------------------------------------
# Derive triage_reason (inline rule-based labeling)
# -----------------------------------------------------------------------------------------------------
df["days_since_arrival"] = df["arrival_date"].apply(lambda x: (TODAY - x.date()).days)

def triage_reason(row):
    fever = row["temp"] >= 100
    symptoms = row["symptoms"].strip().lower() != "none"
    days = (TODAY - row["arrival_date"].date()).days

    if row["lice_status"] == "positive":
        return "Lice"
    if row["varicella_status"] == "positive":
        return "Varicella"
    if row["covid_status"] == "positive":
        if days <= 11 or fever:
            return "COVID"
    if row["flu_status"] == "positive":
        if days <= 8 or fever:
            return "Flu"
    if fever and symptoms and days <= 5:
        return "Symptom-Based"
    return "None"

df["triage_reason"] = df.apply(triage_reason, axis=1)

# -----------------------------------------------------------------------------------------------------
# Filter to labeled data (drop 'None')
# -----------------------------------------------------------------------------------------------------
df = df[df["triage_reason"] != "None"]

# -----------------------------------------------------------------------------------------------------
# Define features and target
# -----------------------------------------------------------------------------------------------------
features = ["temp", "symptoms", "days_since_arrival"]
target = "triage_reason"
X = df[features]
y = df[target]

# -----------------------------------------------------------------------------------------------------
# Preprocessing and pipeline
# -----------------------------------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("symptoms_vec", CountVectorizer(), "symptoms"),
        ("scale_num", StandardScaler(), ["temp", "days_since_arrival"])
    ],
    remainder="drop"
)

clf = RandomForestClassifier(class_weight="balanced", random_state=SEED)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", clf)
])

# -----------------------------------------------------------------------------------------------------
# Train/test split and model fitting
# -----------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------------------------------------------
# Predict and export line-level predictions
# -----------------------------------------------------------------------------------------------------
df["ml_predicted_reason"] = pipeline.predict(X)

df_out = df[[
    "child_id", "arrival_date", "triage_reason", "ml_predicted_reason",
    "symptoms", "temp", "days_since_arrival"
]]

df_out.to_csv(PREDICTIONS_PATH, index=False)
print(f"\nLine-level ML predictions saved to: {PREDICTIONS_PATH}")

# -----------------------------------------------------------------------------------------------------
# Evaluation and summary table export
# -----------------------------------------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

# Print full classification report to console
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Save full classification report to CSV
from sklearn.metrics import classification_report

report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose().reset_index()
report_df = report_df.rename(columns={"index": "Triage Reason"})
report_df.to_csv(SUMMARY_PATH, index=False)

print(f"\nModel summary table saved to: {SUMMARY_PATH}")
# -----------------------------------------------------------------------------------------------------
# Compute Odds Ratios
# -----------------------------------------------------------------------------------------------------
def compute_pathogen_or_table(df, target_label, output_dir="data"):
    """
    Compute and save the odds ratio table for a binary logistic regression
    predicting target_label vs all other triage outcomes.
    Removes intercept row for clean output.
    """
    from pathlib import Path
    import statsmodels.api as sm
    import numpy as np

    df = df.copy()
    df["target"] = (df["triage_reason"] == target_label).astype(int)

    X = df[["temp", "days_since_arrival"]].copy()
    X["symptom_flag"] = (df["symptoms"].str.lower() != "none").astype(int)
    X = sm.add_constant(X)

    model = sm.Logit(df["target"], X)
    result = model.fit(disp=0)

    or_table = pd.DataFrame({
        "OR": np.exp(result.params),
        "2.5%": np.exp(result.conf_int()[0]),
        "97.5%": np.exp(result.conf_int()[1])
    })

    # Drop the intercept ("const") row
    or_table = or_table.drop(index="const", errors="ignore").round(3)

    # Save to file
    output_path = OUTPUT_DIR / f"or_table_{target_label.lower()}.csv"
    or_table.to_csv(output_path, index=True)
    print(f"Complete OR table saved to: {output_path}")

    return or_table

# -----------------------------------------------------------------------------------------------------
# Sensitivity Analysis for Fever Thresholds
# -----------------------------------------------------------------------------------------------------
def run_fever_sensitivity_analysis(
    df,
    thresholds=[99.5, 100, 100.5],
    date_col="arrival_date",
    output_filename="sensitivity_summary.csv"
):
    """
    Run fever threshold sensitivity analysis on triage classification.
    Returns and saves a summary DataFrame with distribution of triage reasons at each threshold,
    excluding 'None' rows.
    """
    from pathlib import Path
    from datetime import datetime

    TODAY = datetime.today().date()

    df = df.copy()
    df["days_since_arrival"] = df[date_col].apply(lambda x: (TODAY - x.date()).days)

    def triage_reason_thresh(row, threshold):
        fever = row["temp"] >= threshold
        symptoms = row["symptoms"].strip().lower() != "none"
        days = row["days_since_arrival"]

        if row["lice_status"] == "positive":
            return "Lice"
        if row["varicella_status"] == "positive":
            return "Varicella"
        if row["covid_status"] == "positive":
            if days <= 11 or fever:
                return "COVID"
        if row["flu_status"] == "positive":
            if days <= 8 or fever:
                return "Flu"
        if fever and symptoms and days <= 5:
            return "Symptom-Based"
        return "None"

    results = []
    for threshold in thresholds:
        df[f"triage_{threshold}"] = df.apply(lambda row: triage_reason_thresh(row, threshold), axis=1)
        counts = df[f"triage_{threshold}"].value_counts(normalize=True).reset_index()
        counts.columns = ["Triage Reason", "Proportion"]
        counts["Fever Threshold"] = threshold
        results.append(counts)

    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df.pivot(index="Triage Reason", columns="Fever Threshold", values="Proportion").fillna(0)

    # Remove 'None' row if it exists
    result_df = result_df.drop(index="None", errors="ignore")

    result_df = result_df.round(3)

    # Save output to /data folder
    output_path = OUTPUT_DIR / output_filename
    result_df.to_csv(output_path)
    print(f"Completed - Sensitivity analysis saved to: {output_path}")

    return result_df
# -----------------------------------------------------------------------------------------------------
# Generate Epi Curve Summary 
# -----------------------------------------------------------------------------------------------------
def generate_epi_curve_summary(
    df,
    date_col="arrival_date",
    output_filename="epi_curve_summary.csv"
):
    """
    Generate and save a monthly epi curve summary of triage cases by disease.
    Output format: rows = month, columns = triage_reason, values = count
    """
    from pathlib import Path

    df = df.copy()
    df["month"] = df[date_col].dt.to_period("M").astype(str)

    # Group and count
    grouped = (
        df.groupby(["month", "triage_reason"])
        .size()
        .reset_index(name="count")
    )

    # Pivot to wide format: one column per triage reason
    summary = grouped.pivot(index="month", columns="triage_reason", values="count").fillna(0).astype(int)

    # Save to CSV
    output_path = OUTPUT_DIR / output_filename
    summary.to_csv(output_path)
    print(f"Epi curve summary saved to: {output_path}")

    return summary
# -----------------------------------------------------------------------------------------------------
# Run and Save Additional Analyses
# -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    covid_or = compute_pathogen_or_table(df, "COVID")
    flu_or = compute_pathogen_or_table(df, "Flu")
    sensitivity_df = run_fever_sensitivity_analysis(df)
    epi_summary = generate_epi_curve_summary(df)
# -----------------------------------------------------------------------------------------------------
