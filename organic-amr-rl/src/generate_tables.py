"""
Generate Summary Tables and Bootstrap Comparisons
Author: Hayden Hedman
Revised: 2025-12-10

Workflow:
1. Load raw simulation outputs from outputs/raw_csv/.
2. Compute per-episode metrics:
       - AUC_MIC_chloro
       - AUC_MIC_polyB
       - copper_burden
       - mean_growth_inhibition
3. Summarize statistics by agent.
4. Run bootstrap contrasts (95% CI).
5. Save all tables into outputs/summary_tables/.
"""

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Resolve dynamic project paths
# -------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parent               # /src/scripts/
PROJECT_ROOT = SCRIPTS_DIR.parent               # /src/

RAW_CSV_DIR = PROJECT_ROOT / "outputs" / "raw_csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "summary_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_FILES = {
    "Random": RAW_CSV_DIR / "random_policy_results.csv",
    "Rule-Based": RAW_CSV_DIR / "rule_based_results.csv",
    "Q-Learner": RAW_CSV_DIR / "q_learner_results.csv",
}

# -------------------------------------------------------------------
# Helper: AUC using trapezoid rule
# -------------------------------------------------------------------
def compute_auc(cycles, values):
    """Compute area under the curve (AUC) using trapezoid rule."""
    # Ensure cycles are sorted to avoid subtle bugs
    order = np.argsort(cycles)
    cycles = cycles[order]
    values = values[order]
    # Use np.trapezoid to avoid deprecation warning
    return np.trapezoid(values, cycles)

# -------------------------------------------------------------------
# Compute per-episode summary metrics
# -------------------------------------------------------------------
def compute_episode_metrics(df):
    """
    Returns a DataFrame of:
       agent, episode, AUC_MIC_chloro, AUC_MIC_polyB,
       copper_burden, mean_growth_inhibition
    """

    records = []

    for ep, group in df.groupby("episode"):

        cycles = group["cycle"].to_numpy()

        auc_chloro = compute_auc(cycles, group["MIC_chloro"].to_numpy())
        auc_polyB  = compute_auc(cycles, group["MIC_polyB"].to_numpy())

        copper_burden = group["copper"].sum()
        mean_growth   = group["growth_inhibition"].mean()

        records.append({
            "agent": group["agent"].iloc[0],
            "episode": int(ep),
            "AUC_MIC_chloro": auc_chloro,
            "AUC_MIC_polyB": auc_polyB,
            "copper_burden": copper_burden,
            "mean_growth_inhibition": mean_growth,
        })

    return pd.DataFrame(records)

# -------------------------------------------------------------------
# Bootstrap: median difference + CI
# -------------------------------------------------------------------
def bootstrap_difference(values_a, values_b, n_boot=8000):
    """
    Compute bootstrap median difference and 95% CI.
    Difference = A - B

    Handles unequal sample sizes by resampling both groups
    to the size of the smaller group on each bootstrap draw.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    nA = len(values_a)
    nB = len(values_b)
    n = min(nA, nB)

    diffs = np.empty(n_boot)

    for i in range(n_boot):
        sampleA = np.random.choice(values_a, size=n, replace=True)
        sampleB = np.random.choice(values_b, size=n, replace=True)
        diffs[i] = np.median(sampleA - sampleB)

    return (
        float(np.median(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )
# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():

    print("\n===========================================")
    print(" Loading agent data + computing metrics…")
    print("===========================================\n")

    all_ep_tables = []

    # Loop through all agents
    for agent_name, csv_path in AGENT_FILES.items():

        if not csv_path.exists():
            print(f"[ERROR] Missing CSV for {agent_name}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df["agent"] = agent_name

        ep_table = compute_episode_metrics(df)
        all_ep_tables.append(ep_table)

        print(f"Processed: {agent_name}")

    # Combine all per-episode tables
    combined = pd.concat(all_ep_tables, ignore_index=True)

    # Save per-episode metrics
    episode_out = OUT_DIR / "episode_metrics.csv"
    combined.to_csv(episode_out, index=False)
    print(f"\nSaved → {episode_out.name}")

    # --------------------------------------------------------------
    # AGENT SUMMARY TABLE
    # --------------------------------------------------------------
    print("\nGenerating agent-level summary table…")

    summary = (
        combined.groupby("agent")[[
            "AUC_MIC_chloro",
            "AUC_MIC_polyB",
            "copper_burden",
            "mean_growth_inhibition"
        ]]
        .agg(["mean", "median", "std"])
        .round(3)
    )

    summary_out = OUT_DIR / "agent_summary_metrics.csv"
    summary.to_csv(summary_out)
    print(f"Saved → {summary_out.name}")

    # --------------------------------------------------------------
    # BOOTSTRAP CONTRASTS
    # --------------------------------------------------------------
    print("\nRunning bootstrap contrasts…\n")

    comparisons = [
        ("Q-Learner", "Random"),
        ("Q-Learner", "Rule-Based"),
        ("Rule-Based", "Random"),
    ]

    results = []

    for A, B in comparisons:

        dfA = combined[combined["agent"] == A]
        dfB = combined[combined["agent"] == B]

        for metric in ["AUC_MIC_chloro", "AUC_MIC_polyB"]:

            median, low, high = bootstrap_difference(
                dfA[metric].values,
                dfB[metric].values
            )

            results.append({
                "contrast": f"{A} - {B}",
                "metric": metric,
                "median_difference": round(median, 3),
                "CI_lower": round(low, 3),
                "CI_upper": round(high, 3),
            })

            print(f"{A} - {B} | {metric}: {median:.3f} ({low:.3f} to {high:.3f})")

    boot_df = pd.DataFrame(results)

    boot_out = OUT_DIR / "bootstrap_auc_summary.csv"
    boot_df.to_csv(boot_out, index=False)

    print(f"\nSaved → {boot_out.name}")
    print("\nAll tables generated successfully.\n")

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
# -------------------------------------------------------------------
