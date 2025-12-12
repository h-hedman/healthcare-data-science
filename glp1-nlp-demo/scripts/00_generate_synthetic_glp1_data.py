# -------------------------------------------------------------------
"""
GLP-1 Synthetic Data Generation Script

Author: Hayden Hedman
Date: 2024-11-10

Purpose:
Generate a small, reproducible set of synthetic healthcare data to support
demonstration of a lightweight NLP pipeline for GLP-1–related labeling.
All outputs are intentionally synthetic and written to the project-level
/data directory.

This script is intended to be run once prior to executing downstream
analysis or NLP pipelines.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
SEED = 64
np.random.seed(SEED)
random.seed(SEED)

# -------------------------------------------------------------------
# Global parameters
# -------------------------------------------------------------------
N_PATIENTS = 320
PATIENT_IDS = list(range(1, N_PATIENTS + 1))

START_DATE = pd.Timestamp("2022-10-01")
N_DAYS = 60
END_DATE = START_DATE + timedelta(days=N_DAYS - 1)


# -------------------------------------------------------------------
# Resolve project root and data directory
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_clinical_notes():
    """
    Generate short, repetitive clinical note fragments typical of real
    EHR text. Notes are intentionally simple and template-driven.
    """
    note_templates = [
        "History of type 2 diabetes mellitus.",
        "No prior diagnosis of diabetes noted.",
        "GLP-1 therapy discussed during visit.",
        "Semaglutide initiated at prior visit.",
        "Started GLP-1 therapy for weight management.",
        "Patient reports increased appetite and recent weight gain.",
        "Reports mild nausea following injections.",
        "Denies nausea, vomiting, or abdominal pain.",
        "No gastrointestinal adverse effects reported.",
        "Follow-up visit after GLP-1 initiation.",
        "Patient tolerating medication without issues.",
        "Reports intermittent gastrointestinal symptoms.",
        "Discontinued GLP-1 due to GI intolerance.",
        "Lifestyle modification counseling provided.",
        "HbA1c previously elevated per chart review."
    ]

    rows = []

    for pid in PATIENT_IDS:
        n_notes = random.choice([1, 2, 3])
        note_dates = np.random.choice(
            pd.date_range(START_DATE, END_DATE),
            size=n_notes,
            replace=False
        )

        for nd in note_dates:
            fragments = random.sample(
                note_templates,
                k=random.choice([1, 2])
            )
            rows.append({
                "patient_id": pid,
                "note_date": nd,
                "note_text": " ".join(fragments)
            })

    df = pd.DataFrame(rows).sort_values(["patient_id", "note_date"])
    out_path = DATA_DIR / "clinical_notes.txt"
    df.to_csv(out_path, sep="|", index=False)

    print(f"[INFO] Wrote {len(df)} clinical notes")
    return df


def generate_prescriptions():
    """
    Generate synthetic GLP-1 prescription exposure records.
    """
    glp1_drugs = ["semaglutide", "liraglutide"]
    rows = []

    for pid in PATIENT_IDS:
        if random.random() < 0.65:  # peak GLP-1 uptake period
            start = START_DATE + timedelta(
                days=random.randint(0, N_DAYS - 10)
            )
            rows.append({
                "patient_id": pid,
                "drug_name": random.choice(glp1_drugs),
                "start_date": start,
                "end_date": None
            })

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "prescriptions.csv"
    df.to_csv(out_path, index=False)

    print(f"[INFO] Wrote {len(df)} prescription records")
    return df


def generate_diagnoses():
    """
    Generate synthetic ICD-10 diagnosis codes with dates prior to
    the GLP-1 exposure window.
    """
    icd_codes = {
        "E11.9": "Type 2 diabetes mellitus",
        "E66.9": "Obesity",
        "R73.03": "Prediabetes"
    }

    rows = []

    for pid in PATIENT_IDS:
        n_dx = random.choice([1, 2, 3])
        dx_dates = np.random.choice(
            pd.date_range(START_DATE - timedelta(days=180), START_DATE),
            size=n_dx,
            replace=False
        )

        for d in dx_dates:
            rows.append({
                "patient_id": pid,
                "icd_code": random.choice(list(icd_codes.keys())),
                "diagnosis_date": d
            })

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "diagnoses_icd.csv"
    df.to_csv(out_path, index=False)

    print(f"[INFO] Wrote {len(df)} diagnosis records")
    return df


def generate_labels(dx_df, rx_df):
    """
    Generate derived patient-level labels.
    """
    diabetes_codes = {"E11.9", "R73.03"}

    patient_ids = sorted(
        set(dx_df["patient_id"]).union(rx_df["patient_id"])
    )

    rows = []

    for pid in patient_ids:
        has_diabetes = (
            (dx_df["patient_id"] == pid) &
            (dx_df["icd_code"].isin(diabetes_codes))
        ).any()

        glp1_exposed = (rx_df["patient_id"] == pid).any()

        rows.append({
            "patient_id": pid,
            "has_diabetes": has_diabetes,
            "glp1_exposed": glp1_exposed
        })

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "labels.csv"
    df.to_csv(out_path, index=False)

    print(f"[INFO] Wrote {len(df)} patient-level labels")
    return df


def main():
    print("[INFO] Starting synthetic GLP-1 data generation")
    print(f"[INFO] Study window: {START_DATE.date()} to {END_DATE.date()}")
    print(f"[INFO] Output directory: {DATA_DIR}")

    notes_df = generate_clinical_notes()
    rx_df = generate_prescriptions()
    dx_df = generate_diagnoses()
    generate_labels(dx_df, rx_df)

    print("[INFO] Synthetic data generation complete")
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
# -------------------------------------------------------------------
