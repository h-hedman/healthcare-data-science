

# =========================================================================================================
#Author: Hayden Hedman
#Synthetic Dataset Generator for Occupational Cleaning Exposure and Asthma-Like Symptoms
#Anchored to Vizcaya et al. (2011), workforce-based asthma symptoms among cleaners.
#Anchor source paper: Vizcaya et al. 2011, https://doi.org/10.1136/oem.2010.063271
#Script Overview Goals::
#(1) Simulates demographic/workplace/exposure distributions similar to the anchor study.
#(2) Encodes exposures using low vs. high frequency indicators (0/1).
#(3) Generates an outcome variable representing "asthma-like symptoms":
#   (current asthma OR wheeze without cold) ~ 10–12% overall prevalence.
#(4) Adds known elevated risk for:
#   - Hospital cleaners
#   - Frequent hydrochloric acid use
#   - Frequent ammonia/degreaser use
# =========================================================================================================
# Load libraries
import numpy as np
import pandas as pd
import os
from pathlib import Path
# =========================================================================================================
def generate_synthetic_cleaner_dataset(N=20000, seed=64):
    np.random.seed(seed)

    # -------------------------------------------
    # 1. Demographic Variables (anchored to Vizcaya Table 1)
    # -------------------------------------------

    # Sex: ~80-85% female
    sex = np.random.choice(["female", "male"], size=N, p=[0.82, 0.18])

    # Age, mean ~45, sd ~10 (used whole numbers, clipped to 18-70)
    age = np.clip(np.random.normal(45, 10, size=N), 18, 70).round().astype(int)

    # Smoking status: ~30% current, ~10% former, ~60% never
    smoking = np.random.choice(
        ["never", "former", "current"], size=N, p=[0.60, 0.10, 0.30]
    )

    # Nationality proxy (foreign-born / non-native language worker) ~30%
    # Literature eximates varied from 20 to 60%+ depending on region; 30% is a moderate assumption.
        #Eggerth et al., 2019 — Latino building cleaners; immigrant-heavy workforce
        #Am J Ind Med 62(7):600–608
        #Jørgensen et al., 2011 — Immigrant vs Danish cleaners; major disparities
        #Int Arch Occup Environ Health 84(6):665–674
        #Speiser et al., 2022 — Environmental health needs of Latinas in cleaning jobs
        #Environ Health Insights 16:11786302221100045

    foreign_born = np.random.binomial(1, 0.30, size=N)

    # -------------------------------------------
    # 2. Workplace Type (Based on Table 3 trends)
    # -------------------------------------------
    workplace = np.random.choice(
        ["home", "common_areas", "hospital", "industry", "school", "other_healthcare"],
        size=N,
        p=[0.28, 0.25, 0.24, 0.15, 0.05, 0.03]
    )

    # -------------------------------------------
    # 3. Cleaning Product Exposure (Frequency Encoded: low=0, high=1)
    # -------------------------------------------
    hcl_high = np.random.binomial(1, 0.10, size=N)  # Hydrochloric acid use is relatively uncommon, strong effect
    ammonia_high = np.random.binomial(1, 0.15, size=N)
    degreaser_high = np.random.binomial(1, 0.18, size=N)
    multipurpose_high = np.random.binomial(1, 0.40, size=N)  # Common background exposure
    wax_high = np.random.binomial(1, 0.10, size=N)

    # -------------------------------------------
    # 4. Linear Risk Model (Logistic Probability of Symptoms)
    # Calibrated to produce ~11% population prevalence.
    # -------------------------------------------
    logit = (
        -3.60                                # Baseline log-odds (calibrated)
        + 0.55 * (workplace == "hospital")    # Hospital exposure → higher risk
        + 0.85 * hcl_high                     # Hydrochloric acid → strongest effect
        + 0.45 * ammonia_high
        + 0.65 * degreaser_high
        + 0.30 * multipurpose_high
        + 0.10 * wax_high
        + 0.15 * (smoking == "current")       # Smoking modest contributor
        + 0.10 * foreign_born                 # Mild susceptibility gradient
    )

    probability = 1 / (1 + np.exp(-logit))
    asthma_like_symptoms = np.random.binomial(1, probability)

    # -------------------------------------------
    # 5. Assemble DataFrame
    # -------------------------------------------
    df = pd.DataFrame({
        "sex": sex,
        "age": age,
        "smoking": smoking,
        "foreign_born": foreign_born,
        "workplace": workplace,
        "hcl_high": hcl_high,
        "ammonia_high": ammonia_high,
        "degreaser_high": degreaser_high,
        "multipurpose_high": multipurpose_high,
        "wax_high": wax_high,
        "asthma_like_symptoms": asthma_like_symptoms
    })

    return df
# =========================================================================================================
if __name__ == "__main__":
    # Determine the script's directory
    script_dir = Path(__file__).parent
    # Define relative paths
    data_dir = script_dir.parent / 'data'  # Moves up one level to root and then into 'data'
    output_path = data_dir / "synthetic_cleaner_cohort.csv"

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate the dataset
    df = generate_synthetic_cleaner_dataset(N=20000)

    # Save dataset to the determined output path
    df.to_csv(output_path, index=False)

    print(f"Synthetic dataset saved to: {output_path}")
    print(df.head())
    
    # ===========================
    # Summary statistics for Table 2
    # ===========================
    def summarize_simulated_data(df):
        print("\n=== Demographics ===")
        print("Sex distribution (%):")
        print(df["sex"].value_counts(normalize=True).mul(100).round(2))
        
        print("\nAge:")
        print(f"Mean = {df['age'].mean():.2f}, SD = {df['age'].std():.2f}")
        
        print("\nSmoking status (%):")
        print(df["smoking"].value_counts(normalize=True).mul(100).round(2))
        
        print("\nForeign-born (%):")
        print(df["foreign_born"].mean() * 100)

        print("\n=== Workplace Distribution (%): ===")
        print(df["workplace"].value_counts(normalize=True).mul(100).round(2))

        print("\n=== Chemical Exposures (% high exposure): ===")
        chem_cols = ["hcl_high", "ammonia_high", "degreaser_high", 
                    "multipurpose_high", "wax_high"]
        print(df[chem_cols].mean().mul(100).round(2))

        print("\n=== Outcome ===")
        prev = df["asthma_like_symptoms"].mean() * 100
        print(f"Asthma-like symptoms prevalence: {prev:.2f}%")

        return {
            "sex": df["sex"].value_counts(normalize=True).mul(100).round(2).to_dict(),
            "age_mean": df["age"].mean(),
            "age_sd": df["age"].std(),
            "smoking": df["smoking"].value_counts(normalize=True).mul(100).round(2).to_dict(),
            "foreign_born_pct": df["foreign_born"].mean() * 100,
            "workplace": df["workplace"].value_counts(normalize=True).mul(100).round(2).to_dict(),
            "chemical_exposures": df[chem_cols].mean().mul(100).round(2).to_dict(),
            "prevalence_pct": prev
        }

    # Run summary
    summary_stats = summarize_simulated_data(df)
# =========================================================================================================
