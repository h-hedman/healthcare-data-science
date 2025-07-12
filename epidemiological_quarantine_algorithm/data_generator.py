# Author: Hayden Hedman
# Date: 2025-07-09 (first version used in 2021)
# Description: Generates synthetic data for a public health epidemiological quarantine algorithm response
# Note: All data is randomly generated and does NOT represent real individuals or cases.
# -------------------------------------------------------------------------------------------------------
# Load libararies
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
# -------------------------------------------------------------------------------------------------------
# Automatically detect project root
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_DIR, "input")
BUS_OUTPUT_PATH = os.path.join(INPUT_DIR, "bus_registry.csv")
# -------------------------------------------------------------------------------------------------------
# Configurable Parameters
SYMPTOM_POOL = ["fever", "cough", "rash", "fatigue", "headache", "none"]
STATUS_CHOICES = ["positive", "negative", "unknown"]
FACILITY_ID_RANGE = (1, 18)
BUS_COUNT_RANGE = (3, 5)
KIDS_PER_BUS_RANGE = (15, 42)
# -------------------------------------------------------------------------------------------------------
# Utility Functions
def generate_symptoms():
    symptoms = random.sample(SYMPTOM_POOL, k=random.randint(1, 3))
    if "none" in symptoms and len(symptoms) > 1:
        symptoms.remove("none")
    return ", ".join(symptoms)

def assign_status(infection_weight=0.35):
    return np.random.choice(STATUS_CHOICES, p=[
        infection_weight,
        1 - infection_weight - 0.05,
        0.05
    ])
# -------------------------------------------------------------------------------------------------------
# Main Generator Function
def generate_bus_registry(output_path=BUS_OUTPUT_PATH, seed=88):
    random.seed(seed)
    np.random.seed(seed)

    today = datetime.today()
    start_date = today - timedelta(days=random.randint(5, 8))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    child_counter = 1
    num_buses = random.randint(*BUS_COUNT_RANGE)

    for bus_num in range(1, num_buses + 1):
        arrival_date = start_date + timedelta(days=random.randint(10 * (bus_num - 1), 14 * bus_num))
        kids_on_bus = random.randint(*KIDS_PER_BUS_RANGE)

        for _ in range(kids_on_bus):
            child_id = f"C{child_counter:05d}"
            facility_source_id = random.randint(*FACILITY_ID_RANGE)

            record = {
                "child_id": child_id,
                "bus_id": f"Bus_{bus_num}",
                "arrival_date": arrival_date.strftime("%Y-%m-%d"),
                "facility_source_id": facility_source_id,
                "symptoms": generate_symptoms(),
                "notes": random.choice(["", "prior exposure", "travel fatigue", ""]),

                "flu_status": assign_status(0.12),
                "covid_status": assign_status(0.08),
                "varicella_status": assign_status(0.07),
                "lice_status": assign_status(0.10),

                "temp": round(np.random.normal(98.6, 1.3), 1)
            }

            records.append(record)
            child_counter += 1

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Confirmed: bus_registry.csv created with {len(df)} records at {output_path}")
#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Facility Census Generator
def generate_facility_census(output_path=None, seed=123, n_total=1500, bus_df=None):
    if output_path is None:
        PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(INPUT_DIR, "facility_census.csv")


    random.seed(seed)
    np.random.seed(seed)

    FACILITY_IDS = list(range(1, 19))
    ROOM_PREFIXES = ['A', 'B', 'C', 'D', 'E']
    EXIT_NOTES = ['', 'reunified', 'transferred', 'aged out']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []

    # First: add all bus kids to facility census
    if bus_df is not None:
        for _, row in bus_df.iterrows():
            intake_date = datetime.strptime(row["arrival_date"], "%Y-%m-%d")
            stay_length = random.randint(10, 40)
            est_exit_date = intake_date + timedelta(days=stay_length)
            has_exited = random.random() < 0.3

            records.append({
                "child_id": row["child_id"],
                "facility_id": row["facility_source_id"],
                "room_id": f"{random.choice(ROOM_PREFIXES)}-{random.randint(1, 150):03d}",
                "intake_date": intake_date.strftime("%Y-%m-%d"),
                "est_exit_date": est_exit_date.strftime("%Y-%m-%d"),
                "current_status": "exited" if has_exited else "in_facility",
                "notes": random.choice(EXIT_NOTES) if has_exited else ""
            })

    # Then: add random non-bus kids until  hit n_total
    starting_id = 50000
    while len(records) < n_total:
        child_id = f"C{starting_id:05d}"
        if bus_df is not None and child_id in bus_df["child_id"].values:
            starting_id += 1
            continue  # skip if ID already exists

        intake_date = datetime.today() - timedelta(days=random.randint(1, 30))
        stay_length = random.randint(10, 40)
        est_exit_date = intake_date + timedelta(days=stay_length)
        has_exited = random.random() < 0.3

        records.append({
            "child_id": child_id,
            "facility_id": random.choice(FACILITY_IDS),
            "room_id": f"{random.choice(ROOM_PREFIXES)}-{random.randint(1, 150):03d}",
            "intake_date": intake_date.strftime("%Y-%m-%d"),
            "est_exit_date": est_exit_date.strftime("%Y-%m-%d"),
            "current_status": "exited" if has_exited else "in_facility",
            "notes": random.choice(EXIT_NOTES) if has_exited else ""
        })

        starting_id += 1

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Confirmed: facility_census.csv created with {len(df)} records at {output_path}")
#------------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_historical_case_data(output_path=None, seed=42):
    if output_path is None:
        output_path = os.path.join(INPUT_DIR, "historical_bus_registry.csv")


    random.seed(seed)
    np.random.seed(seed)

    start_date = datetime.today() - timedelta(days=4*365)
    end_date = datetime.today()
    current_date = start_date

    records = []
    child_counter = 100000  # avoid overlap with recent bus data

    bus_load_id = 1
    while current_date <= end_date:
        kids_on_bus = random.randint(15, 45)
        infection_rate = random.uniform(0.05, 0.75)
        infected_kids = int(kids_on_bus * infection_rate)

        # Random disease mix distribution (normalized)
        disease_probs = np.random.dirichlet(np.ones(4), size=1)[0]
        flu_prob, covid_prob, varicella_prob, lice_prob = disease_probs

        for i in range(kids_on_bus):
            child_id = f"H{child_counter:06d}"
            facility_id = random.randint(*FACILITY_ID_RANGE)
            symptoms = generate_symptoms()

            # Assign infection based on infected count + random mix
            is_infected = i < infected_kids

            record = {
                "child_id": child_id,
                "bus_id": f"HistBus_{bus_load_id}",
                "arrival_date": current_date.strftime("%Y-%m-%d"),
                "facility_source_id": facility_id,
                "symptoms": symptoms,
                "notes": random.choice(["", "prior exposure", "travel fatigue", ""]),
                "flu_status": assign_status(flu_prob if is_infected else 0.02),
                "covid_status": assign_status(covid_prob if is_infected else 0.02),
                "varicella_status": assign_status(varicella_prob if is_infected else 0.01),
                "lice_status": assign_status(lice_prob if is_infected else 0.01),
                "temp": round(np.random.normal(98.6, 1.3), 1)
            }

            records.append(record)
            child_counter += 1

        current_date += timedelta(days=random.randint(10, 14))
        bus_load_id += 1

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Confirmed: historical_bus_registry.csv created with {len(df)} records at {output_path}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main Entry Point
# For directions on running this script see README.md
if __name__ == "__main__":
    generate_bus_registry()
    bus_df = pd.read_csv(os.path.join(INPUT_DIR, "bus_registry.csv"))
    generate_facility_census(bus_df=bus_df)
    generate_historical_case_data()
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
