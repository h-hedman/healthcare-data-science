# =============================================================================
# 04_figures.py
# FluSurv-NET Influenza Hospitalization Rates — Publication Figures
#
# Purpose:
#   Generate all publication-ready figures for the IDR manuscript.
#   Figures are saved as 300 DPI PNG to outputs/figures/.
#   No embedded titles — captions go in manuscript text per MDPI convention.
#   Axis labels and annotations are self-contained within each figure.
#
# Inputs (from data/cleaned/):
#   flusurvnet_cleaned.csv
#   cleaned_ts_overall.csv
#
# Inputs (from outputs/tables/):
#   table2_rr_race_phase.csv
#   stl_components.csv
#   stl_amplitude_summary.csv
#
# Outputs (to outputs/figures/):
#   fig1_weekly_rate_timeseries.png   — Overall time series with phase shading
#   fig2_age_peak_rates_by_phase.png  — Age group peak rates by phase
#   fig3_rr_forest_plot.png           — Race/ethnicity RR forest plot
#   fig4_stl_decomposition.png        — STL decomposition pre vs post
#   figS1_season_trajectories.png     — Season-level spaghetti plot
#   figS2_virus_rates_by_phase.png    — Virus type rates by phase
#
# Style conventions:
#   - Viridis colormap for sequential/categorical groupings
#   - Phase shading: pre=steel blue, disruption=salmon, recovery=sage green
#   - 300 DPI, bbox_inches='tight'
#   - Font sizes: 11pt labels, 10pt ticks, 9pt annotations
#   - No embedded figure titles
#
# Author: Hayden
# Seed: 88
# =============================================================================

import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# RANDOM SEED
# -----------------------------------------------------------------------------
import numpy as np
np.random.seed(88)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PATHS = {
    "cleaned" : PROJECT_ROOT / "data" / "cleaned",
    "tables"  : PROJECT_ROOT / "outputs" / "tables",
    "figures" : PROJECT_ROOT / "outputs" / "figures",
    "logs"    : PROJECT_ROOT / "outputs" / "logs",
}
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
LOG_FILE = PATHS["logs"] / "04_figures_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

def section(title):
    bar = "=" * 70
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)

# =============================================================================
# STYLE CONSTANTS
# =============================================================================

DPI         = 300
FACECOLOR   = "white"
LABEL_SIZE  = 13
TICK_SIZE   = 11
ANNOT_SIZE  = 10
LEGEND_SIZE = 10

# Phase shading — lighter alpha, used sparingly
PHASE_COLORS = {
    "pre_pandemic" : "#AEC6CF",
    "disruption"   : "#FFB3A7",
    "recovery"     : "#B5D5B5",
}

PHASE_LABELS = {
    "pre_pandemic" : "Pre-pandemic",
    "disruption"   : "Pandemic disruption",
    "recovery"     : "Post-pandemic recovery",
}

PHASE_SPANS = [
    ("pre_pandemic", "2009-10-04", "2019-05-18"),
    ("disruption",   "2019-09-29", "2022-05-14"),
    ("recovery",     "2022-09-25", "2026-03-07"),
]

# Viridis discrete palette
import matplotlib.cm as cm
VIRIDIS = cm.get_cmap("viridis")

def viridis_colors(n, vmax=0.85):
    return [VIRIDIS(i / max(n - 1, 1) * vmax) for i in range(n)]

RACE_ORDER = [
    "Black",
    "American Indian/Alaska Native",
    "Hispanic/Latino",
    "White",
    "Asian/Pacific Islander",
]
RACE_SHORT = {
    "Black"                          : "Black",
    "American Indian/Alaska Native"  : "AIAN",
    "Hispanic/Latino"                : "Hispanic/Latino",
    "White"                          : "White (ref)",
    "Asian/Pacific Islander"         : "Asian/PI",
}

AGE_ORDER_STANDARD = ["0-4 yr", "5-17 yr", "18-49 yr", "50-64 yr", ">= 65 yr"]
AGE_LABELS_CLEAN   = ["0–4",    "5–17",    "18–49",    "50–64",    "65+"]

VIRUS_ORDER = ["Influenza A", "Influenza B", "A(H1N1)pdm09", "A(H3N2)"]

SEASON_ORDER = [
    "2009-10","2010-11","2011-12","2012-13","2013-14",
    "2014-15","2015-16","2016-17","2017-18","2018-19",
    "2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"
]

# Phase line colors — used across Fig 1, Fig 3, Fig S1
PHASE_LINE = {
    "pre_pandemic" : "#2166ac",
    "disruption"   : "#d6604d",
    "recovery"     : "#4dac26",
}

# =============================================================================
# HELPERS
# =============================================================================

def save_fig(fig, name):
    path = PATHS["figures"] / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
    plt.close(fig)
    log.info(f"  Saved: {path}")


def add_phase_shading(ax, alpha=0.12):
    """Add subtle phase background shading to a date-axis plot."""
    for phase, start, end in PHASE_SPANS:
        ax.axvspan(
            pd.Timestamp(start), pd.Timestamp(end),
            color=PHASE_COLORS[phase], alpha=alpha, zorder=0, linewidth=0
        )


def clean_axes(ax):
    """Remove grid lines and top/right spines."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def phase_legend_patches():
    return [
        mpatches.Patch(color=PHASE_COLORS[p], alpha=0.55, label=PHASE_LABELS[p])
        for p in ["pre_pandemic", "disruption", "recovery"]
    ]


def load_csv(filename, label):
    path = PATHS["cleaned"] / filename
    if not path.exists():
        path = PATHS["tables"] / filename
    if not path.exists():
        log.error(f"  File not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    if "epiweek_date" in df.columns:
        df["epiweek_date"] = pd.to_datetime(df["epiweek_date"], errors="coerce")
    log.info(f"  Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# =============================================================================
# FIG 1 — OVERALL WEEKLY RATE TIME SERIES
# =============================================================================

def fig1_timeseries(df):
    section("FIG 1 — Overall Weekly Rate Time Series")

    overall = df[
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall")
    ].sort_values("epiweek_date").copy()

    # Viridis colors: pre=dark purple, disruption=teal, recovery=teal-green (avoid yellow)
    vc = viridis_colors(5)
    phase_vc = {
        "pre_pandemic" : vc[0],   # dark purple
        "disruption"   : vc[2],   # mid teal
        "recovery"     : vc[3],   # teal-green (avoids bright yellow)
    }

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=FACECOLOR)

    for phase in ["pre_pandemic", "disruption", "recovery"]:
        sub = overall[overall["phase"] == phase]
        ax.plot(
            sub["epiweek_date"], sub["weekly_rate"],
            color=phase_vc[phase], linewidth=2.0, alpha=0.92, zorder=3
        )

    # 2020-21 suppressed — no marker; handled in caption/methods

    # Axis
    ax.set_xlabel("Influenza season (year)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Weekly hospitalization rate\n(per 100,000 population)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlim(pd.Timestamp("2009-07-01"), pd.Timestamp("2026-06-01"))
    ax.set_ylim(bottom=0)
    clean_axes(ax)

    # Legend: colored lines only
    line_handles = [
        mlines.Line2D([], [], color=phase_vc["pre_pandemic"],  linewidth=2.5, label="Pre-pandemic"),
        mlines.Line2D([], [], color=phase_vc["disruption"],    linewidth=2.5, label="Pandemic disruption"),
        mlines.Line2D([], [], color=phase_vc["recovery"],      linewidth=2.5, label="Post-pandemic recovery"),
    ]
    ax.legend(
        handles=line_handles,
        fontsize=LEGEND_SIZE, framealpha=0.92,
        loc="upper left", ncol=1
    )

    fig.tight_layout()
    save_fig(fig, "fig1_weekly_rate_timeseries.png")

# =============================================================================
# FIG 2 — AGE GROUP PEAK RATES BY PHASE
# =============================================================================

def fig2_age_peak_rates(df):
    section("FIG 2 — Age Group Peak Rates by Phase")

    age_mask = (
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall") &
        (df["age_cat"].isin(AGE_ORDER_STANDARD)) &
        (~df["season"].isin({"2020-21"}))
    )
    sub = df[age_mask].copy()

    season_peak = (
        sub.groupby(["season", "phase", "age_cat"])["weekly_rate"]
        .max()
        .reset_index(name="peak_rate")
    )
    phase_peak = (
        season_peak.groupby(["phase", "age_cat"])["peak_rate"]
        .mean()
        .reset_index(name="mean_peak")
    )

    phases   = ["pre_pandemic", "disruption", "recovery"]
    n_phases = len(phases)
    n_ages   = len(AGE_ORDER_STANDARD)
    colors   = viridis_colors(n_ages)

    x      = np.arange(n_phases)
    width  = 0.15
    offset = np.linspace(-(n_ages - 1) * width / 2, (n_ages - 1) * width / 2, n_ages)

    # x-axis labels with clean year ranges
    phase_xlabels = {
        "pre_pandemic" : "Pre-pandemic\n(2009–2019)",
        "disruption"   : "Pandemic disruption\n(2019–2022)",
        "recovery"     : "Post-pandemic recovery\n(2022–2026)",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=FACECOLOR)

    for i, age in enumerate(AGE_ORDER_STANDARD):
        vals = []
        for phase in phases:
            row = phase_peak[
                (phase_peak["phase"] == phase) &
                (phase_peak["age_cat"] == age)
            ]
            vals.append(row["mean_peak"].values[0] if len(row) > 0 else 0)

        ax.bar(
            x + offset[i], vals, width=width * 0.9,
            color=colors[i],
            label=AGE_LABELS_CLEAN[i],
            zorder=3, edgecolor="white", linewidth=0.4
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [phase_xlabels[p] for p in phases],
        fontsize=TICK_SIZE
    )
    ax.set_ylabel("Mean peak weekly rate\n(per 100,000 population)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylim(bottom=0)
    clean_axes(ax)

    ax.legend(
        title="Age group (years)", title_fontsize=ANNOT_SIZE,
        fontsize=LEGEND_SIZE, framealpha=0.92,
        loc="upper left", ncol=2
    )

    fig.tight_layout()
    save_fig(fig, "fig2_age_peak_rates_by_phase.png")


# =============================================================================
# FIG 3 — RACE/ETHNICITY RR FOREST PLOT
# =============================================================================

def fig3_rr_forest(table2):
    section("FIG 3 — Race/Ethnicity Rate Ratio Forest Plot")

    t2 = table2[~table2["race_group"].str.contains("reference", case=False)].copy()

    def parse_rr(val):
        try:
            return float(val)
        except Exception:
            return np.nan

    def parse_ci(val):
        try:
            parts = str(val).split("–")
            return float(parts[0]), float(parts[1])
        except Exception:
            return np.nan, np.nan

    t2["rr_val"] = t2["rr"].apply(parse_rr)
    t2["ci_lo"]  = t2["ci_95"].apply(lambda x: parse_ci(x)[0])
    t2["ci_hi"]  = t2["ci_95"].apply(lambda x: parse_ci(x)[1])

    phase_list = [
        "Pre-pandemic (2009\u201310 to 2018\u201319)",
        "Pandemic disruption (2019\u201320 to 2021\u201322)",
        "Post-pandemic recovery (2022\u201323 to 2025\u201326)",
    ]
    phase_short = {
        phase_list[0]: "Pre-pandemic",
        phase_list[1]: "Pandemic disruption",
        phase_list[2]: "Post-pandemic recovery",
    }
    vc = viridis_colors(3)
    phase_vc = {
        phase_list[0]: vc[0],
        phase_list[1]: vc[1],
        phase_list[2]: vc[2],
    }

    races    = [r for r in RACE_ORDER if r != "White"]
    n_races  = len(races)

    fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor=FACECOLOR)

    y          = 0
    group_gap  = 0.55
    phase_gap  = 1.4
    yticks     = []
    yticklabels = []

    for ri, race in enumerate(races):
        race_rows  = t2[t2["race_group"] == race]
        phase_ys   = []
        for pi, phase in enumerate(phase_list):
            row  = race_rows[race_rows["phase"] == phase]
            ypos = y
            phase_ys.append(ypos)
            if len(row) > 0:
                rr   = row["rr_val"].values[0]
                cilo = row["ci_lo"].values[0]
                cihi = row["ci_hi"].values[0]
                color = phase_vc[phase]
                if pd.notna(rr):
                    ax.plot([cilo, cihi], [ypos, ypos],
                            color=color, linewidth=2.2, zorder=3, solid_capstyle="round")
                    ax.scatter(rr, ypos, color=color,
                               s=70, zorder=4, marker="D")
            y -= group_gap

        # y-tick label centred on this race's group
        yticks.append(np.mean(phase_ys))
        yticklabels.append(RACE_SHORT.get(race, race))
        y -= phase_gap

    # Reference line — dashed dark gray, visible but not competing with data
    ax.axvline(x=1.0, color="#444444", linewidth=1.8, linestyle="--", alpha=0.9, zorder=2)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=TICK_SIZE + 1)
    ax.set_xlabel("Rate ratio vs. White (age-adjusted cumulative rate)",
                  fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.set_xlim(0.3, 4.2)
    clean_axes(ax)

    # Legend — include White reference as a distinct entry
    legend_handles = [
        mlines.Line2D([], [], color=phase_vc[p],
                      linewidth=2.5, marker="D", markersize=6,
                      label=phase_short[p])
        for p in phase_list
    ]
    ref_handle = mlines.Line2D(
        [], [], color="#444444", linewidth=1.8, linestyle="--",
        label="White (reference, RR = 1.0)"
    )
    ax.legend(
        handles=legend_handles + [ref_handle],
        fontsize=LEGEND_SIZE, framealpha=0.92,
        loc="upper right"
    )

    fig.tight_layout()
    save_fig(fig, "fig3_rr_forest_plot.png")


# =============================================================================
# FIG 4 — STL DECOMPOSITION PANEL
# =============================================================================

def fig4_stl(stl_components):
    section("FIG 4 — STL Decomposition Pre vs Post-Pandemic")

    pre  = stl_components[stl_components["series"] == "pre_pandemic"].copy()
    post = stl_components[stl_components["series"] == "post_pandemic"].copy()

    for df_s in [pre, post]:
        df_s["epiweek_date"] = pd.to_datetime(df_s["epiweek_date"])
        df_s.sort_values("epiweek_date", inplace=True)

    components = ["observed", "trend", "seasonal", "residual"]
    comp_labels = {
        "observed" : "Observed rate",
        "trend"    : "Trend",
        "seasonal" : "Seasonal",
        "residual" : "Residual",
    }

    vc = viridis_colors(3)
    comp_colors = {
        "pre_pandemic"  : vc[0],
        "post_pandemic" : vc[2],
    }

    fig, axes = plt.subplots(
        len(components), 2,
        figsize=(13, 11),
        facecolor=FACECOLOR,
        sharex="col"
    )

    series_data = [
        ("pre_pandemic",  pre,  "Pre-pandemic (2009–10 to 2018–19)"),
        ("post_pandemic", post, "Post-pandemic recovery (2022–23 to 2024–25)"),
    ]

    for col, (series_key, data, col_title) in enumerate(series_data):
        color = comp_colors[series_key]
        for row, comp in enumerate(components):
            ax = axes[row, col]
            ax.plot(data["epiweek_date"], data[comp],
                    color=color, linewidth=1.4, alpha=0.92)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_ylabel(comp_labels[comp], fontsize=LABEL_SIZE - 1)
            ax.tick_params(labelsize=TICK_SIZE - 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)

            if row == 0:
                ax.set_title(col_title, fontsize=LABEL_SIZE,
                             fontweight="bold", pad=8)

        axes[-1, col].set_xlabel("Epiweek", fontsize=LABEL_SIZE - 1)
        # Rotate post-pandemic x-ticks to prevent label overlap/cutoff
        if col == 1:
            axes[-1, col].tick_params(axis="x", rotation=30)
            for label in axes[-1, col].get_xticklabels():
                label.set_ha("right")

    fig.subplots_adjust(hspace=0.38, wspace=0.3)
    save_fig(fig, "fig4_stl_decomposition.png")


# =============================================================================
# FIG S1 — EXCLUDED (season trajectory spaghetti — not interpretable at print)
# =============================================================================

def figS1_trajectories(df):
    section("FIG S1 — Skipped (excluded from manuscript)")
    log.info("  figS1 excluded per review: season spaghetti not interpretable at print scale.")


# =============================================================================
# FIG S2 — VIRUS TYPE WEEKLY RATES BY PHASE
# =============================================================================

def figS2_virus(df):
    section("FIG S2 — Virus Type Weekly Rates by Phase")

    virus_mask = (
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"].isin(VIRUS_ORDER)) &
        (~df["season"].isin({"2020-21"}))
    )
    sub = df[virus_mask].copy()

    phases  = ["pre_pandemic", "disruption", "recovery"]
    colors  = viridis_colors(len(VIRUS_ORDER))

    summary = (
        sub.groupby(["phase", "virus_cat"])["weekly_rate"]
        .mean()
        .reset_index(name="mean_weekly")
    )

    n_virus = len(VIRUS_ORDER)
    x       = np.arange(len(phases))
    width   = 0.18
    offset  = np.linspace(-(n_virus - 1) * width / 2,
                           (n_virus - 1) * width / 2, n_virus)

    phase_xlabels = {
        "pre_pandemic" : "Pre-pandemic\n(2009–2019)",
        "disruption"   : "Pandemic disruption\n(2019–2022)",
        "recovery"     : "Post-pandemic recovery\n(2022–2026)",
    }

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=FACECOLOR)

    for i, virus in enumerate(VIRUS_ORDER):
        vals = []
        for phase in phases:
            row = summary[
                (summary["phase"] == phase) &
                (summary["virus_cat"] == virus)
            ]
            vals.append(row["mean_weekly"].values[0] if len(row) > 0 else 0)

        ax.bar(
            x + offset[i], vals, width=width * 0.9,
            color=colors[i], label=virus,
            zorder=3, edgecolor="white", linewidth=0.4
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [phase_xlabels[p] for p in phases], fontsize=TICK_SIZE
    )
    ax.set_ylabel("Mean weekly hospitalization rate\n(per 100,000 population)",
                  fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_ylim(bottom=0)
    clean_axes(ax)

    ax.legend(
        title="Virus type", title_fontsize=ANNOT_SIZE,
        fontsize=LEGEND_SIZE, framealpha=0.92, loc="upper left"
    )

    fig.tight_layout()
    save_fig(fig, "figS2_virus_rates_by_phase.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    run_start = datetime.now()
    section("04_figures.py — Publication Figures")
    log.info(f"Run started  : {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Project root : {PROJECT_ROOT}")
    log.info(f"Output path  : {PATHS['figures']}")

    # --- Load data ---
    section("LOAD DATA")
    df      = load_csv("flusurvnet_cleaned.csv", "flusurvnet_cleaned")
    ts_df   = load_csv("cleaned_ts_overall.csv", "cleaned_ts_overall")
    table2  = load_csv("table2_rr_race_phase.csv", "table2_rr_race_phase")
    stl_comp = load_csv("stl_components.csv", "stl_components")

    # Restore season_week_num as numeric
    for col in ["season_week_num", "week", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Generate figures ---
    fig1_timeseries(df)
    fig2_age_peak_rates(df)
    fig3_rr_forest(table2)
    fig4_stl(stl_comp)
    figS1_trajectories(df)
    figS2_virus(df)

    # --- Summary ---
    section("FINAL SUMMARY")
    figs = list(PATHS["figures"].glob("*.png"))
    log.info(f"  Figures written: {len(figs)}")
    for f in sorted(figs):
        log.info(f"    {f.name}")
    log.info(f"  Log file     : {LOG_FILE}")
    log.info(f"  Run completed: {(datetime.now() - run_start).seconds}s")
    section("DONE — proceed to 05_aim3_ml.py")


if __name__ == "__main__":
    main()
