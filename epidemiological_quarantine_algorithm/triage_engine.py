# -------------------------------------------------------------------------------------------------------
# Author: Hayden Hedman
# Date: 2025-07-09 (first version used in 2021)
# Description: Triage engine for a public health epidemiological quarantine algorithm dataset used 
#              to determine quarantine and isolation decisions based on bus registry and facility census data.
# Note: All data is randomly generated and does NOT represent real individuals or cases.
# -------------------------------------------------------------------------------------------------------

# Load libraries
import pandas as pd
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------------------------------------------
# Define base directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input" 
OUTPUT_DIR = BASE_DIR / "output"  # still saving to 'output' since it's pre-model

# -------------------------------------------------------------------------------------------------------
# Load CSVs from input/
bus_df = pd.read_csv(INPUT_DIR / "bus_registry.csv", parse_dates=["arrival_date"])
facility_df = pd.read_csv(INPUT_DIR / "facility_census.csv")

# -------------------------------------------------------------------------------------------------------
# Set today's date
TODAY = datetime.today().date()

# -------------------------------------------------------------------------------------------------------
# Print heads for sanity (SAN) check
print("\nBus Registry Sample:")
print(bus_df.head())

print("\nFacility Census Sample:")
print(facility_df.head())

# -------------------------------------------------------------------------------------------------------
# Merge on child_id to get full context
df = pd.merge(bus_df, facility_df, on="child_id", how="left")

# -------------------------------------------------------------------------------------------------------
# Helper: check recent arrival (e.g., last 5 days)
def is_recent(arrival_date, window=5):
    return (TODAY - arrival_date.date()).days <= window

# -------------------------------------------------------------------------------------------------------
# Triage logic
def triage_row(row):
    days_since_arrival = (TODAY - row['arrival_date'].date()).days
    has_fever = row['temp'] >= 100
    has_symptoms = row['symptoms'].strip().lower() != 'none'

    if row['lice_status'] == 'positive':
        return ('Isolate (Lice)', 'Lice')
    if row['varicella_status'] == 'positive':
        return ('Isolate (Varicella)', 'Varicella')
    if row['covid_status'] == 'positive':
        if days_since_arrival <= 11:
            return ('Quarantine (COVID - Active)', 'COVID')
        elif has_fever:
            return ('Quarantine (COVID - Fever Post Isolation)', 'COVID')
        else:
            return ('Clear (COVID - Past Window)', 'COVID')
    if row['flu_status'] == 'positive':
        if days_since_arrival <= 8:
            return ('Quarantine (Flu - Active)', 'Flu')
        elif has_fever:
            return ('Quarantine (Flu - Fever Post Isolation)', 'Flu')
        else:
            return ('Clear (Flu - Past Window)', 'Flu')
    if has_fever and has_symptoms and days_since_arrival <= 5:
        return ('Quarantine (Symptomatic)', 'Symptom-Based')
    return ('Clear', 'None')

# -------------------------------------------------------------------------------------------------------
# Apply triage logic
df[['triage_decision', 'triage_reason']] = df.apply(triage_row, axis=1, result_type='expand')

# -------------------------------------------------------------------------------------------------------
# Prepare report
report_cols = [
    'child_id',
    'arrival_date',
    'facility_id',
    'room_id',
    'temp',
    'symptoms',
    'triage_reason',
    'triage_decision'
]

# -------------------------------------------------------------------------------------------------------
# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save output to output (requires additional processing in Dash app)
report_path = OUTPUT_DIR / "triage_report.csv"
df[report_cols].to_csv(report_path, index=False)
# -------------------------------------------------------------------------------------------------------
# Console confirmation
print(f"\nCompleted! Triage summary report saved to: {report_path}")
# -------------------------------------------------------------------------------------------------------
