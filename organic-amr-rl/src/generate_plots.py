"""
Generate Manuscript Figures for Copper-Driven Antibiotic Resistance Study
Author: Hayden Hedman
Revised: 2026-03-03

Produces clean, high-resolution figures (300 dpi) for manuscript use:
- MIC (Chloramphenicol)
- MIC (Polymyxin B)
- Copper concentration
- Growth inhibition

Each plot overlays:
  - Random Policy
  - Rule-Based Policy
  - Q-Learner

Now includes:
  - Mean trajectory
  - ± Standard Deviation shaded bands

Figures saved to:
    /src/outputs/figures/
"""

# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import sys

# --------------------------------------------------------------------
# Dynamically resolve paths
# --------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

RAW_CSV_DIR = PROJECT_ROOT / "outputs" / "raw_csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Color Theme (Viridis)
# --------------------------------------------------------------------
viridis = cm.get_cmap("viridis")

AGENT_COLORS = {
    "Random": viridis(0.15),
    "Rule-Based": viridis(0.55),
    "Q-Learner": viridis(0.85),
}

AGENT_FILES = {
    "Random": RAW_CSV_DIR / "random_policy_results.csv",
    "Rule-Based": RAW_CSV_DIR / "rule_based_results.csv",
    "Q-Learner": RAW_CSV_DIR / "q_learner_results.csv",
}

# --------------------------------------------------------------------
# Compute mean ± SD by cycle
# --------------------------------------------------------------------
def compute_mean_sd_curve(df, value_col):
    grouped = df.groupby("cycle")[value_col]
    means = grouped.mean()
    sds = grouped.std()

    cycles = means.index.values
    mean_vals = means.values
    sd_vals = sds.values

    return cycles, mean_vals, sd_vals


# --------------------------------------------------------------------
# Core plotting function
# --------------------------------------------------------------------
def plot_mean_curves(outcome_col, ylabel, output_name):
    """
    Generate a single outcome figure overlaying all three agents.
    Includes mean ± SD shaded bands.
    """

    plt.figure(figsize=(8, 6))

    for agent_name, csv_path in AGENT_FILES.items():

        if not csv_path.exists():
            print(f"[Warning] Missing CSV for {agent_name}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        cycles, means, sds = compute_mean_sd_curve(df, outcome_col)

        color = AGENT_COLORS[agent_name]

        # Mean line
        plt.plot(
            cycles,
            means,
            label=agent_name,
            color=color,
            linewidth=3.0,
            alpha=0.95,
        )

        # Shaded SD band
        plt.fill_between(
            cycles,
            means - sds,
            means + sds,
            color=color,
            alpha=0.20
        )

    plt.xlabel("Cycle", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax * 1.05)

    plt.legend(fontsize=16, loc="upper left")
    plt.grid(False)

    plt.tight_layout()

    out_path = FIGURE_DIR / output_name
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved figure → {out_path}")


# --------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------
def main():

    print("Generating manuscript-quality figures with variability bands...\n")

    plot_mean_curves(
        outcome_col="MIC_chloro",
        ylabel="MIC (Chloramphenicol)",
        output_name="fig_MIC_chloramphenicol.png"
    )

    plot_mean_curves(
        outcome_col="MIC_polyB",
        ylabel="MIC (Polymyxin B)",
        output_name="fig_MIC_polymyxinB.png"
    )

    plot_mean_curves(
        outcome_col="copper",
        ylabel="Copper Concentration (mg/L)",
        output_name="fig_copper_concentration.png"
    )

    plot_mean_curves(
        outcome_col="growth_inhibition",
        ylabel="Relative Susceptibility",
        output_name="fig_relative_susceptibility.png"
    )

    print("\nAll figures generated successfully.")


# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
# --------------------------------------------------------------------