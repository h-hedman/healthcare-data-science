# Quarantine Engine

A synthetic epidemiological triage engine for simulating disease-based quarantine and isolation decisions. This system is based on an original algorithm developed by Hayden Hedman during an emergency CDC response. It combines rule-based logic and machine learning to emulate real-time public health operations in high-throughput intake environments.

---

## Project Overview

This project simulates a CDC-style quarantine decision pipeline using synthetic data. The pipeline replicates end-to-end processes including:

- Synthetic data generation of bus arrivals, symptoms, temperatures, and facility census
- Rule-based quarantine and isolation logic derived from pathogen timelines and symptom profiles
- Machine learning triage modeling to assess generalization and accuracy of automated classification
- Epidemiological reporting outputs suitable for public health analysts or deployment in operational dashboards

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `data_generator.py` | Generates synthetic bus registry and facility census data |
| `triage_engine.py` | Applies original rule-based quarantine and isolation logic |
| `symptoms_ml_predictor.py` | Trains and evaluates a Random Forest model to replicate triage classification |

---

## Outputs

- `bus_registry.csv` / `facility_census.csv` – simulated population-level intake data
- `triage_report.csv` – line-level quarantine decisions with associated reasoning
- `ml_predictions.csv` – predicted vs. rule-labeled triage reasons
- `triage_quarantine_count_table.csv` – summary of quarantine/isolation by pathogen
- `epi_curve_summary.csv` – monthly trend of triage cases by classification

---

## Sample Triage Logic

- **COVID**: quarantine if ≤11 days since arrival or presents with fever
- **Flu**: quarantine if ≤8 days since arrival or presents with fever
- **Lice / Varicella**: immediate isolation upon detection
- **Symptomatic (unspecified)**: fever + symptoms within 5 days of arrival
- **Clear**: all other cases

---

## Running the Pipeline

```bash
# Step 1: Generate synthetic data
python data_generator.py

# Step 2: Apply triage logic
python triage_engine.py

# Step 3: Train and evaluate ML model
python symptoms_ml_predictor.py
