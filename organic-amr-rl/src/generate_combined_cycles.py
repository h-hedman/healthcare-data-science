"""
Generate combined cycle-level dataset for GLMM analysis in R.
Author: Hayden Hedman
Date: 2025-12-10
"""

import pandas as pd
from pathlib import Path

# -------------------------
# Resolve project directories
# -------------------------
CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

RAW_CSV = PROJECT_ROOT / "outputs" / "raw_csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "summary_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load all three agent files
# -------------------------
files = {
    "Random": RAW_CSV / "random_policy_results.csv",
    "Rule-Based": RAW_CSV / "rule_based_results.csv",
    "Q-Learner": RAW_CSV / "q_learner_results.csv",
}

df_list = []

for agent, path in files.items():
    df = pd.read_csv(path)
    df["agent"] = agent  # add column
    df_list.append(df)

# -------------------------
# Combine into one dataset
# -------------------------
combined = pd.concat(df_list, ignore_index=True)

# -------------------------
# Save output
# -------------------------
out_path = OUT_DIR / "combined_agent_cycles.csv"
combined.to_csv(out_path, index=False)

print(f"Saved combined dataset → {out_path}")
