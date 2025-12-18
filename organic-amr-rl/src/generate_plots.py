"""
Generate Manuscript Figures for Copper-Driven Antibiotic Resistance Study
Author: Hayden Hedman
Revised: 2025-12-10

Produces clean, high-resolution figures (300 dpi) for manuscript use:
- MIC (Chloramphenicol)
- MIC (Polymyxin B)
- Copper concentration
- Growth inhibition

Each plot overlays:
  - Random Policy
  - Rule-Based Policy
  - Q-Learner

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
SCRIPTS_DIR = CURRENT_FILE.parent          # /src/scripts/
PROJECT_ROOT = SCRIPTS_DIR.parent          # /src/

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
# Helper: smoothed mean curves (optional)
# --------------------------------------------------------------------
def smooth_curve(values, window=3):
    """Simple moving average smoother."""
    if window <= 1:
        return values
    return np.convolve(values, np.ones(window)/window, mode="same")

# --------------------------------------------------------------------
# Helper — compute mean curve by cycle
# --------------------------------------------------------------------
def compute_mean_curve(df, value_col, smoothing=False):
    grouped = df.groupby("cycle")[value_col].mean()
    cycles = grouped.index.values
    means = grouped.values

    if smoothing:
        means = smooth_curve(means, window=3)

    return cycles, means

# --------------------------------------------------------------------
# Core plotting function
# --------------------------------------------------------------------
def plot_mean_curves(outcome_col, ylabel, output_name, smoothing=False):
    """
    Generate a single outcome figure overlaying all three agents.
    Highly legible manuscript-quality output.
    """

    plt.figure(figsize=(8, 6))

    for agent_name, csv_path in AGENT_FILES.items():

        if not csv_path.exists():
            print(f"[Warning] Missing CSV for {agent_name}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        cycles, means = compute_mean_curve(df, outcome_col, smoothing=smoothing)

        plt.plot(
            cycles,
            means,
            label=agent_name,
            color=AGENT_COLORS[agent_name],
            linewidth=3.0,
            alpha=0.90,
        )

    plt.xlabel("Cycle", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Ensure Y-axis has visual breathing room
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

    print("Generating manuscript-quality figures...\n")

    # Option to toggle mild smoothing on/off
    SMOOTH = False

    plot_mean_curves(
        outcome_col="MIC_chloro",
        ylabel="MIC (Chloramphenicol)",
        output_name="fig_MIC_chloramphenicol.png",
        smoothing=SMOOTH
    )

    plot_mean_curves(
        outcome_col="MIC_polyB",
        ylabel="MIC (Polymyxin B)",
        output_name="fig_MIC_polymyxinB.png",
        smoothing=SMOOTH
    )

    plot_mean_curves(
        outcome_col="copper",
        ylabel="Copper Concentration (mg/L)",
        output_name="fig_copper_concentration.png",
        smoothing=SMOOTH
    )

    plot_mean_curves(
        outcome_col="growth_inhibition",
        ylabel="Relative Susceptibility",
        output_name="fig_relative_susceptibility.png",
        smoothing=SMOOTH
    )

    print("\nAll figures generated successfully.")

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
# --------------------------------------------------------------------
