# =============================================================================
# 03_aim1_stats.py
# FluSurv-NET Influenza Hospitalization Rates — Aim 1 Descriptive & Inferential
#
# Purpose:
#   Produce publication-ready descriptive tables and inferential statistical
#   analyses for Aim 1 (phase characterization) and Aim 2 (disparity) of the
#   IDR manuscript. All outputs are CSV tables formatted for direct inclusion
#   in the manuscript or supplement.
#
# Inputs (from data/cleaned/):
#   flusurvnet_cleaned.csv      — full cleaned dataset (all strata)
#   cleaned_ts_overall.csv      — deduplicated Overall time series (ITS-ready)
#
# Outputs (to outputs/tables/):
#   table1_phase_summary.csv            — Table 1: phase × stratum summary
#   table1_publication.csv              — Table 1: publication-ready wide format
#   table1_supp_season_detail.csv       — Supp Table S1: season-level detail
#   table_s2_season_detail_publication.csv — Supp Table S1: publication-ready
#   table2_rr_race_phase.csv            — Table 2: race/ethnicity RRs vs White
#   table2_publication.csv              — Table 2: publication-ready
#   dispersion_check.csv                — variance/mean ratios by stratum
#   phase_comparison_ols.csv            — OLS phase comparison results
#   phase_cohens_d.csv                  — Cohen's d effect sizes (FIXED)
#   lmm_phase_results.csv               — LMM fixed effects: phase trend
#   table_s3a_ols_publication.csv       — Supp Table S3a: OLS, publication-ready
#   table_s3b_lmm_publication.csv       — Supp Table S3b: LMM, publication-ready
#   table_s3c_cohens_d_publication.csv  — Supp Table S3c: Cohen's d, pub-ready (NEW)
#   stl_components.csv                  — STL decomposition components
#   stl_amplitude_summary.csv           — STL seasonal amplitude by phase
#   stl_amplitude_publication.csv       — STL amplitude: publication-ready (NEW)
#
# Outputs (to outputs/logs/):
#   03_aim1_stats_log.txt
#
# Statistical methods:
#   [S01] Dispersion check         — variance/mean ratio on weekly_rate by stratum
#   [S02] Table 1                  — phase-level summary: median(IQR), mean(SD),
#                                    peak rate, peak week, cumulative rate,
#                                    season duration; all five strata
#   [S03] Supplementary Table S1   — season-level detail (17 rows)
#   [S04] Table 2                  — age-adjusted RR vs White by phase,
#                                    bootstrapped 95% CIs (n=1000, seed=88)
#   [S05] Season-level OLS         — mean_weekly ~ phase dummies; primary +
#                                    sensitivity models; Cohen's d pairwise
#   [S05b] Linear Mixed Model      — weekly_rate ~ phase + season_week_num
#                                    + (1|season); statsmodels MixedLM
#   [S06] STL decomposition        — statsmodels STL on overall weekly series;
#                                    pre- and post-pandemic separately;
#                                    seasonal amplitude and trend quantified
#
# Author: Hayden
# Seed: 88
# =============================================================================

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# statsmodels — required for LMM and STL
try:
    import statsmodels.formula.api as smf
    from statsmodels.tsa.seasonal import STL
    import statsmodels.api as sm
except ImportError:
    print("statsmodels not found. Install with: pip install statsmodels")
    sys.exit(1)

# -----------------------------------------------------------------------------
# RANDOM SEED
# -----------------------------------------------------------------------------
SEED = 88
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# DYNAMIC PATH RESOLUTION
# -----------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PATHS = {
    "cleaned" : PROJECT_ROOT / "data" / "cleaned",
    "tables"  : PROJECT_ROOT / "outputs" / "tables",
    "logs"    : PROJECT_ROOT / "outputs" / "logs",
}

for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
LOG_FILE = PATHS["logs"] / "03_aim1_stats_log.txt"

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

def section(title: str) -> None:
    bar = "=" * 70
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)


# =============================================================================
# CONSTANTS
# =============================================================================

# Phase ordering for table display — consistent throughout
PHASE_ORDER = ["pre_pandemic", "disruption", "recovery"]

# Phase display labels for manuscript tables
PHASE_LABELS = {
    "pre_pandemic" : "Pre-pandemic (2009–10 to 2018–19)",
    "disruption"   : "Pandemic disruption (2019–20 to 2021–22)",
    "recovery"     : "Post-pandemic recovery (2022–23 to 2025–26)",
}

# Reference group for race/ethnicity rate ratios (Table 2)
RACE_REFERENCE = "White"

# Bootstrap iterations for Table 2 CIs
N_BOOTSTRAP = 1000

# Season duration threshold: minimum weekly_rate to count a week as "active"
# 0.1 per 100k is a standard low-level detection threshold in FluSurv-NET work
SEASON_ACTIVE_THRESHOLD = 0.1

# Suppressed seasons — excluded from rate-based summaries where weekly_rate
# is structurally NaN (2020-21); included in cumulative rate summaries
SUPPRESSED_WEEKLY_SEASONS = {"2020-21"}

# Partial seasons — included in descriptive summaries but flagged
PARTIAL_SEASONS = {"2025-26"}

# Age group display labels for manuscript tables
AGE_DISPLAY_LABELS = {
    "0-4 yr"   : "0–4",
    "5-17 yr"  : "5–17",
    "18-49 yr" : "18–49",
    "50-64 yr" : "50–64",
    ">= 65 yr" : "65+",
}

# STL decomposition parameters
# period=52 assumes a full year cycle; for traditional 30-week seasons this
# will still capture the dominant annual periodicity. robust=True downweights
# outliers (pandemic seasons) in LOESS fitting.
STL_PERIOD  = 52
STL_ROBUST  = True

# Pre/post split for STL — disruption phase excluded from both series to
# produce clean pre-pandemic baseline and post-pandemic recovery estimates
STL_PRE_SEASONS  = [s for s in [
    "2009-10","2010-11","2011-12","2012-13","2013-14",
    "2014-15","2015-16","2016-17","2017-18","2018-19"
]]
STL_POST_SEASONS = [
    "2022-23","2023-24","2024-25"   # exclude 2025-26 (partial)
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_cleaned(filename: str, label: str) -> pd.DataFrame:
    """Load a cleaned CSV with type restoration and shape logging."""
    path = PATHS["cleaned"] / filename
    if not path.exists():
        log.error(f"  File not found: {path}")
        log.error("  Ensure 02_cleaning.py has been run successfully first.")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    if "epiweek_date" in df.columns:
        df["epiweek_date"] = pd.to_datetime(df["epiweek_date"], errors="coerce")
    for col in ["year", "week", "season_year", "season_week_num",
                "pandemic_h1n1_flag", "partial_season_flag",
                "weekly_suppressed_flag", "year_round_season_flag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    log.info(f"  Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def fmt_median_iqr(series: pd.Series, decimals: int = 1) -> str:
    """Format as 'median (Q1–Q3)' string, dropping NaN."""
    s = series.dropna()
    if len(s) == 0:
        return "—"
    med = s.median()
    q1  = s.quantile(0.25)
    q3  = s.quantile(0.75)
    return f"{med:.{decimals}f} ({q1:.{decimals}f}–{q3:.{decimals}f})"


def fmt_mean_sd(series: pd.Series, decimals: int = 1) -> str:
    """Format as 'mean (SD)' string, dropping NaN."""
    s = series.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.{decimals}f} ({s.std():.{decimals}f})"


def fmt_n(series: pd.Series) -> str:
    """Non-null count as string."""
    return str(series.notna().sum())


def season_duration(df_season: pd.DataFrame,
                    threshold: float = SEASON_ACTIVE_THRESHOLD) -> int:
    """
    Count weeks in a season where weekly_rate >= threshold.
    Represents 'active season duration' rather than total reported weeks.
    """
    return int((df_season["weekly_rate"] >= threshold).sum())


# =============================================================================
# S01 — DISPERSION CHECK
# =============================================================================

def run_dispersion_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    S01 — Compute variance/mean ratio (VMR) for weekly_rate by stratum.

    VMR interpretation:
      VMR ≈ 1   : Poisson-consistent (equidispersion)
      VMR >> 1  : overdispersed — negative binomial more appropriate
      VMR < 1   : underdispersed — rare in count/rate data

    For rate data (not raw counts), VMR > 2 is a practical threshold
    suggesting the mean alone is a poor summary and spread should be
    reported alongside it — motivating the median/IQR choice in Table 1.

    Also reports skewness to document the right-skewed rate distribution.
    """
    section("S01. DISPERSION CHECK")

    records = []
    strata_configs = [
        ("Overall",  {"age_cat": "Overall", "sex_cat": "Overall",
                      "race_cat": "Overall", "virus_cat": "Overall"}),
        ("Age",      {"sex_cat": "Overall", "race_cat": "Overall",
                      "virus_cat": "Overall"}),
        ("Sex",      {"age_cat": "Overall", "race_cat": "Overall",
                      "virus_cat": "Overall"}),
        ("Race",     {"age_cat": "Overall", "sex_cat": "Overall",
                      "virus_cat": "Overall"}),
        ("Virus",    {"age_cat": "Overall", "sex_cat": "Overall",
                      "race_cat": "Overall"}),
    ]

    for stratum_label, filters in strata_configs:
        mask = pd.Series(True, index=df.index)
        for col, val in filters.items():
            if col in df.columns:
                mask &= (df[col] == val)

        # Exclude structurally suppressed weeks for dispersion calculation
        mask &= ~df["season"].isin(SUPPRESSED_WEEKLY_SEASONS)

        sub = df[mask]["weekly_rate"].dropna()
        if len(sub) < 10:
            continue

        vmr      = sub.var() / sub.mean() if sub.mean() > 0 else np.nan
        skewness = stats.skew(sub)
        interpretation = (
            "overdispersed — NB appropriate" if vmr > 2
            else "near-equidispersion — Poisson acceptable"
        )

        records.append({
            "stratum"        : stratum_label,
            "n_obs"          : len(sub),
            "mean"           : round(sub.mean(), 3),
            "variance"       : round(sub.var(), 3),
            "vmr"            : round(vmr, 2),
            "skewness"       : round(skewness, 2),
            "interpretation" : interpretation,
        })
        log.info(f"  {stratum_label}: VMR={vmr:.2f}, skewness={skewness:.2f} — {interpretation}")

    result = pd.DataFrame(records)
    out_path = PATHS["tables"] / "dispersion_check.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  dispersion_check.csv → {out_path}")
    return result


# =============================================================================
# S02 — TABLE 1: PHASE-LEVEL SUMMARY
# =============================================================================

def build_table1_stratum(df: pd.DataFrame,
                          stratum_col: str,
                          stratum_vals: list,
                          label: str) -> pd.DataFrame:
    """
    Build Table 1 rows for a single stratification variable.
    Returns a tidy DataFrame with one row per phase × stratum_val.

    Metrics reported:
      - n_seasons          : number of seasons contributing
      - n_weeks_obs        : total non-NaN weekly_rate observations
      - weekly_median_iqr  : median (Q1–Q3) weekly rate — primary estimate
      - weekly_mean_sd     : mean (SD) weekly rate — secondary estimate
      - peak_rate_mean_sd  : mean (SD) of per-season peak weekly rates
      - peak_week_median_iqr : median (IQR) of per-season peak MMWR weeks
      - cum_rate_mean_sd   : mean (SD) of per-season cumulative rate
      - season_dur_median_iqr : median (IQR) active season duration (weeks)
    """
    records = []

    for phase in PHASE_ORDER:
        phase_mask = df["phase"] == phase

        for val in stratum_vals:
            val_mask = df[stratum_col] == val if stratum_col else pd.Series(
                True, index=df.index)

            sub = df[phase_mask & val_mask].copy()
            if len(sub) == 0:
                continue

            # Exclude structurally suppressed weekly rates from weekly summaries
            sub_weekly = sub[~sub["season"].isin(SUPPRESSED_WEEKLY_SEASONS)]

            # Per-season aggregates for peak and cumulative metrics
            season_stats = (
                sub_weekly
                .groupby("season")
                .agg(
                    peak_weekly   = ("weekly_rate", "max"),
                    peak_week_val = ("week",
                                     lambda x: x.iloc[
                                         sub_weekly.loc[x.index, "weekly_rate"]
                                         .fillna(0).values.argmax()
                                     ] if len(x) > 0 else np.nan),
                    active_dur    = ("weekly_rate",
                                     lambda x: (x >= SEASON_ACTIVE_THRESHOLD).sum()),
                )
                .reset_index()
            )

            # Cumulative rate: use max cum_rate per season (end-of-season value)
            cum_by_season = (
                sub.groupby("season")["cum_rate"]
                .max()
                .reset_index(name="cum_rate_max")
            )

            records.append({
                "phase"                  : PHASE_LABELS[phase],
                "stratum_variable"       : label,
                "stratum_value"          : AGE_DISPLAY_LABELS.get(val, val),
                "n_seasons"              : sub["season"].nunique(),
                "n_weeks_obs"            : int(sub_weekly["weekly_rate"].notna().sum()),
                "weekly_median_iqr"      : fmt_median_iqr(sub_weekly["weekly_rate"]),
                "weekly_mean_sd"         : fmt_mean_sd(sub_weekly["weekly_rate"]),
                "peak_rate_mean_sd"      : fmt_mean_sd(season_stats["peak_weekly"]),
                "peak_week_median_iqr"   : fmt_median_iqr(season_stats["peak_week_val"], decimals=0),
                "cum_rate_mean_sd"       : fmt_mean_sd(cum_by_season["cum_rate_max"]),
                "season_dur_median_iqr"  : fmt_median_iqr(season_stats["active_dur"], decimals=0),
            })

    return pd.DataFrame(records)


def run_table1(df: pd.DataFrame) -> pd.DataFrame:
    """S02 — Assemble Table 1 across all five strata."""
    section("S02. TABLE 1 — PHASE-LEVEL SUMMARY")

    frames = []

    # --- Overall ---
    overall_mask = (
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall")
    )
    t1_overall = build_table1_stratum(
        df[overall_mask], stratum_col=None,
        stratum_vals=["Overall"], label="Overall"
    )
    frames.append(t1_overall)
    log.info(f"  Overall stratum: {len(t1_overall)} rows")

    # --- Age group ---
    standard_age = ["0-4 yr", "5-17 yr", "18-49 yr", "50-64 yr", ">= 65 yr"]
    age_mask = (
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall") &
        (df["age_cat"].isin(standard_age))
    )
    t1_age = build_table1_stratum(
        df[age_mask], stratum_col="age_cat",
        stratum_vals=standard_age, label="Age group"
    )
    frames.append(t1_age)
    log.info(f"  Age group stratum: {len(t1_age)} rows")

    # --- Sex ---
    sex_mask = (
        (df["age_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall") &
        (df["sex_cat"].isin(["Male", "Female"]))
    )
    log.info(f"  Sex debug: {df[df['sex_cat'].isin(['Male','Female'])]['sex_cat'].value_counts().to_dict()}")
    t1_sex = build_table1_stratum(
        df[sex_mask], stratum_col="sex_cat",
        stratum_vals=["Male", "Female"], label="Sex"
    )
    frames.append(t1_sex)
    log.info(f"  Sex stratum: {len(t1_sex)} rows")

    # --- Race/ethnicity ---
    race_vals = [
        "White", "Black", "Hispanic/Latino",
        "Asian/Pacific Islander", "American Indian/Alaska Native"
    ]
    race_mask = (
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["virus_cat"] == "Overall") &
        (df["race_cat"].isin(race_vals))
    )
    t1_race = build_table1_stratum(
        df[race_mask], stratum_col="race_cat",
        stratum_vals=race_vals, label="Race/ethnicity"
    )
    frames.append(t1_race)
    log.info(f"  Race/ethnicity stratum: {len(t1_race)} rows")

    # --- Virus type ---
    virus_vals = ["Influenza A", "Influenza B", "A(H1N1)pdm09", "A(H3N2)"]
    virus_mask = (
        (df["age_cat"]  == "Overall") &
        (df["sex_cat"]  == "Overall") &
        (df["race_cat"] == "Overall") &
        (df["virus_cat"].isin(virus_vals))
    )
    t1_virus = build_table1_stratum(
        df[virus_mask], stratum_col="virus_cat",
        stratum_vals=virus_vals, label="Virus type"
    )
    frames.append(t1_virus)
    log.info(f"  Virus type stratum: {len(t1_virus)} rows")

    # Combine and write
    table1 = pd.concat(frames, ignore_index=True)

    col_order = [
        "stratum_variable", "stratum_value", "phase",
        "n_seasons", "n_weeks_obs",
        "weekly_median_iqr", "weekly_mean_sd",
        "peak_rate_mean_sd", "peak_week_median_iqr",
        "cum_rate_mean_sd", "season_dur_median_iqr",
    ]
    table1 = table1[[c for c in col_order if c in table1.columns]]

    out_path = PATHS["tables"] / "table1_phase_summary.csv"
    table1.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  Table 1: {len(table1)} rows → {out_path}")

    overall_rows = table1[table1["stratum_variable"] == "Overall"]
    log.info(f"  Table 1 Overall preview:\n{overall_rows.to_string(index=False)}")

    qc_mask = (
        (df["age_cat"]   == ">= 65 yr") &
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall")
    )
    qc = df[qc_mask].groupby("season")["cum_rate"].max().reset_index()
    log.info(f"  QC — 65+ cum_rate by season:\n{qc.to_string(index=False)}")

    export_publication_table1(df, table1)

    return table1


# =============================================================================
# Publication quality Table 1
# =============================================================================

def build_publication_table1(table1: pd.DataFrame) -> pd.DataFrame:
    """
    Build publication-ready wide-format Table 1 with panel structure.
    Output: table1_publication.csv
    """

    PHASE_ABBREV = {
        "Pre-pandemic (2009\u201310 to 2018\u201319)"          : "PRE",
        "Pandemic disruption (2019\u201320 to 2021\u201322)"   : "DISR",
        "Post-pandemic recovery (2022\u201323 to 2025\u201326)": "REC",
    }

    METRICS = [
        ("weekly_median_iqr",    "Median Weekly Rate (IQR)"),
        ("peak_rate_mean_sd",    "Mean Peak Rate (SD)"),
        ("peak_week_median_iqr", "Median Peak Week (IQR)"),
        ("cum_rate_mean_sd",     "Mean Cumulative Rate (SD)"),
        ("season_dur_median_iqr","Median Season Duration (IQR)"),
    ]

    PANELS = [
        ("Panel A. Overall",         "Overall",        ["Overall"]),
        ("Panel B. Age group",       "Age group",      ["0–4", "5–17", "18–49", "50–64", "65+"]),
        ("Panel C. Sex",             "Sex",            ["Male", "Female"]),
        ("Panel D. Virus type",      "Virus type",     ["Influenza A", "Influenza B",
                                                        "A(H1N1)pdm09", "A(H3N2)"]),
        ("Panel E. Race/ethnicity",  "Race/ethnicity", ["White", "Black", "Hispanic/Latino",
                                                        "Asian/Pacific Islander",
                                                        "American Indian/Alaska Native"]),
    ]

    t1 = table1.copy()
    t1["phase_abbrev"] = t1["phase"].map(PHASE_ABBREV)

    phase_order = ["PRE", "DISR", "REC"]
    col_order   = []
    for raw_col, display_name in METRICS:
        unit = "(IQR)" if "IQR" in display_name else "(SD)"
        base = display_name.replace(f" {unit}", "").strip()
        for phase in phase_order:
            col_order.append(f"{base} {phase} {unit}")

    rows = []

    for panel_label, stratum_var, group_vals in PANELS:
        rows.append({"Group": panel_label, **{c: "" for c in col_order}})

        panel_data = t1[t1["stratum_variable"] == stratum_var]

        for group in group_vals:
            group_rows = panel_data[panel_data["stratum_value"] == group]
            if len(group_rows) == 0:
                continue

            row = {"Group": group}
            for raw_col, display_name in METRICS:
                unit = "(IQR)" if "IQR" in display_name else "(SD)"
                base = display_name.replace(f" {unit}", "").strip()
                for phase in phase_order:
                    col_name = f"{base} {phase} {unit}"
                    match = group_rows[group_rows["phase_abbrev"] == phase]
                    row[col_name] = match[raw_col].values[0] if len(match) > 0 else "—"

            rows.append(row)

        rows.append({"Group": "", **{c: "" for c in col_order}})

    pub_table = pd.DataFrame(rows, columns=["Group"] + col_order)
    return pub_table


def export_publication_table1(df: pd.DataFrame, table1: pd.DataFrame) -> None:
    """Call after run_table1 to export the publication-formatted version."""
    pub = build_publication_table1(table1)
    out_path = PATHS["tables"] / "table1_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table1_publication.csv → {out_path} ({len(pub)} rows)")


# =============================================================================
# Publication quality Table 2
# =============================================================================

def build_publication_table2(table2: pd.DataFrame) -> pd.DataFrame:
    """
    Build publication-ready Table 2 with abbreviated phase labels.
    Output: table2_publication.csv
    """

    PHASE_ABBREV = {
        "Pre-pandemic (2009\u201310 to 2018\u201319)"          : "PRE",
        "Pandemic disruption (2019\u201320 to 2021\u201322)"   : "DISR",
        "Post-pandemic recovery (2022\u201323 to 2025\u201326)": "REC",
    }

    RACE_ORDER = [
        "White (reference)",
        "Black",
        "Hispanic/Latino",
        "Asian/Pacific Islander",
        "American Indian/Alaska Native",
    ]

    PHASE_ORDER_PUB = ["PRE", "DISR", "REC"]

    t2 = table2.copy()
    t2["phase_abbrev"] = t2["phase"].map(PHASE_ABBREV)
    t2["race_group"] = t2["race_group"].str.replace(
        r"\(reference\)", "(reference)", regex=True
    )

    rows = []

    for phase in PHASE_ORDER_PUB:
        rows.append({
            "Phase"                                : phase,
            "Race/Ethnicity Group"                 : "",
            "Seasons (n)"                          : "",
            "Mean Age-Adjusted Rate (per 100,000)" : "",
            "Rate Ratio"                           : "",
            "95% CI"                               : "",
            "Denominator Note"                     : "",
        })

        phase_data = t2[t2["phase_abbrev"] == phase]

        for race in RACE_ORDER:
            match = phase_data[
                phase_data["race_group"].str.contains(
                    race.replace(" (reference)", ""), regex=False
                )
            ]
            if len(match) == 0:
                continue

            row_data = match.iloc[0]
            rows.append({
                "Phase"                                : "",
                "Race/Ethnicity Group"                 : race,
                "Seasons (n)"                          : row_data["n_seasons"],
                "Mean Age-Adjusted Rate (per 100,000)" : row_data["mean_adj_rate"],
                "Rate Ratio"                           : row_data["rr"],
                "95% CI"                               : row_data["ci_95"],
                "Denominator Note"                     : row_data["denom_note"],
            })

        rows.append({k: "" for k in [
            "Phase", "Race/Ethnicity Group", "Seasons (n)",
            "Mean Age-Adjusted Rate (per 100,000)",
            "Rate Ratio", "95% CI", "Denominator Note"
        ]})

    pub_table = pd.DataFrame(rows)
    return pub_table


def export_publication_table2(table2: pd.DataFrame) -> None:
    """Call after run_table2 to export the publication-formatted version."""
    pub = build_publication_table2(table2)
    out_path = PATHS["tables"] / "table2_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table2_publication.csv → {out_path} ({len(pub)} rows)")


# =============================================================================
# S03 — SUPPLEMENTARY TABLE S1: SEASON-LEVEL DETAIL
# =============================================================================

def run_supp_table_s1(df: pd.DataFrame) -> pd.DataFrame:
    """
    S03 — Season-level detail table (17 rows × all strata).
    One row per season for the Overall stratum.
    """
    section("S03. SUPPLEMENTARY TABLE S1 — SEASON-LEVEL DETAIL")

    overall_mask = (
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall")
    )
    sub = df[overall_mask].copy()

    records = []
    for season, grp in sub.groupby("season"):
        wr = grp["weekly_rate"].dropna()
        cr = grp["cum_rate"].dropna()
        aa = grp["age_adj_cum_rate"].dropna()

        peak_idx = grp["weekly_rate"].fillna(0).idxmax()
        peak_week_val = grp.loc[peak_idx, "week"] if not grp.empty else np.nan
        peak_rate_val = grp.loc[peak_idx, "weekly_rate"] if not grp.empty else np.nan

        active_dur = int((grp["weekly_rate"] >= SEASON_ACTIVE_THRESHOLD).sum())

        records.append({
            "season"              : season,
            "phase"               : grp["phase"].iloc[0],
            "denom_version"       : grp["denom_version"].iloc[0],
            "partial_season"      : int(grp["partial_season_flag"].iloc[0]),
            "pandemic_h1n1"       : int(grp["pandemic_h1n1_flag"].iloc[0]),
            "n_weeks_reported"    : int(grp["week"].nunique()),
            "n_weeks_active"      : active_dur,
            "weekly_median"       : round(wr.median(), 2) if len(wr) > 0 else np.nan,
            "weekly_q1"           : round(wr.quantile(0.25), 2) if len(wr) > 0 else np.nan,
            "weekly_q3"           : round(wr.quantile(0.75), 2) if len(wr) > 0 else np.nan,
            "weekly_mean"         : round(wr.mean(), 2) if len(wr) > 0 else np.nan,
            "weekly_sd"           : round(wr.std(), 2) if len(wr) > 0 else np.nan,
            "peak_weekly_rate"    : round(float(peak_rate_val), 2) if pd.notna(peak_rate_val) else np.nan,
            "peak_mmwr_week"      : int(peak_week_val) if pd.notna(peak_week_val) else np.nan,
            "cum_rate_eos"        : round(cr.max(), 2) if len(cr) > 0 else np.nan,
            "age_adj_cum_rate_eos": round(aa.max(), 2) if len(aa) > 0 else np.nan,
        })

    result = pd.DataFrame(records).sort_values("season")
    out_path = PATHS["tables"] / "table1_supp_season_detail.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  Supp Table S1: {len(result)} rows → {out_path}")
    log.info(f"\n{result[['season','phase','weekly_median','peak_weekly_rate','cum_rate_eos']].to_string(index=False)}")

    export_publication_supp_s1(result)

    return result


def build_publication_supp_s1(supp_s1: pd.DataFrame) -> pd.DataFrame:
    """Build publication-ready supplementary Table S2."""

    PHASE_ABBREV = {
        "pre_pandemic" : "PRE",
        "disruption"   : "DISR",
        "recovery"     : "REC",
    }

    COL_RENAME = {
        "season"               : "Season",
        "phase"                : "Phase",
        "denom_version"        : "Denom",
        "partial_season"       : "Partial",
        "pandemic_h1n1"        : "H1N1 Season",
        "n_weeks_reported"     : "Weeks (n)",
        "n_weeks_active"       : "Active Weeks (n)",
        "weekly_median"        : "Median Rate",
        "weekly_q1"            : "Q1",
        "weekly_q3"            : "Q3",
        "weekly_mean"          : "Mean Rate",
        "weekly_sd"            : "SD",
        "peak_weekly_rate"     : "Peak Rate",
        "peak_mmwr_week"       : "Peak Week",
        "cum_rate_eos"         : "EOS Cum Rate",
        "age_adj_cum_rate_eos" : "EOS Age-Adj Rate",
    }

    pub = supp_s1.copy()
    pub["phase"] = pub["phase"].map(PHASE_ABBREV).fillna(pub["phase"])
    pub = pub.rename(columns=COL_RENAME)

    return pub


def export_publication_supp_s1(supp_s1: pd.DataFrame) -> None:
    """Export publication-formatted supplementary Table S1."""
    pub = build_publication_supp_s1(supp_s1)
    out_path = PATHS["tables"] / "table_s2_season_detail_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table_s2_season_detail_publication.csv → {out_path} ({len(pub)} rows)")


# =============================================================================
# S04 — TABLE 2: RATE RATIOS VS WHITE BY PHASE
# =============================================================================

def bootstrap_rr(group_rates: np.ndarray,
                 ref_rates: np.ndarray,
                 n_boot: int = N_BOOTSTRAP,
                 seed: int = SEED) -> tuple[float, float, float]:
    """Bootstrap rate ratio (group/reference) with 95% percentile CI."""
    rng = np.random.default_rng(seed)
    n = len(group_rates)
    if n == 0 or np.nansum(ref_rates) == 0:
        return np.nan, np.nan, np.nan

    point = np.nanmean(group_rates) / np.nanmean(ref_rates)

    boot_rrs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        g   = group_rates[idx]
        r   = ref_rates[idx]
        if np.nanmean(r) > 0:
            boot_rrs.append(np.nanmean(g) / np.nanmean(r))

    if len(boot_rrs) < 10:
        return point, np.nan, np.nan

    ci_lo = float(np.percentile(boot_rrs, 2.5))
    ci_hi = float(np.percentile(boot_rrs, 97.5))
    return round(point, 2), round(ci_lo, 2), round(ci_hi, 2)


def run_table2(df: pd.DataFrame) -> pd.DataFrame:
    """S04 — Table 2: age-adjusted cumulative rate ratios vs White, by phase."""
    section("S04. TABLE 2 — RACE/ETHNICITY RATE RATIOS VS WHITE BY PHASE")

    race_mask = (
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["virus_cat"] == "Overall") &
        (df["race_cat"]  != "Overall")
    )
    race_df = df[race_mask].copy()

    seasonal = (
        race_df
        .groupby(["season", "phase", "denom_version", "race_cat"])["age_adj_cum_rate"]
        .max()
        .reset_index(name="age_adj_cum_rate_eos")
    )

    seasonal = seasonal[~seasonal["season"].isin(SUPPRESSED_WEEKLY_SEASONS)]

    wide = seasonal.pivot_table(
        index=["season", "phase", "denom_version"],
        columns="race_cat",
        values="age_adj_cum_rate_eos"
    ).reset_index()
    wide.columns.name = None

    log.info(f"  Seasonal race pivot shape: {wide.shape}")
    log.info(f"  Seasons with White reference available: "
             f"{wide['White'].notna().sum()} of {len(wide)}")

    race_groups = [
        "Black", "Hispanic/Latino",
        "Asian/Pacific Islander", "American Indian/Alaska Native"
    ]

    records = []
    for phase in PHASE_ORDER:
        phase_data = wide[wide["phase"] == phase].copy()
        n_seasons  = len(phase_data)

        if n_seasons == 0:
            continue

        ref_rates = phase_data[RACE_REFERENCE].values if RACE_REFERENCE in phase_data else np.array([])
        if len(ref_rates) == 0 or np.all(np.isnan(ref_rates)):
            log.warning(f"  Phase {phase}: White reference rates all NaN — skipping")
            continue

        denom_versions = phase_data["denom_version"].unique().tolist()
        log.info(f"  Phase {phase}: n_seasons={n_seasons}, "
                 f"denom_versions={denom_versions}")

        records.append({
            "phase"          : PHASE_LABELS[phase],
            "race_group"     : f"{RACE_REFERENCE} (reference)",
            "n_seasons"      : n_seasons,
            "mean_adj_rate"  : round(float(np.nanmean(ref_rates)), 2),
            "rr"             : "1.00",
            "ci_95"          : "Reference",
            "denom_note"     : ", ".join(denom_versions),
        })

        for grp in race_groups:
            if grp not in phase_data.columns:
                log.warning(f"  {grp} not found in pivot — skipping")
                continue

            grp_rates = phase_data[grp].values
            n_avail   = int(np.sum(~np.isnan(grp_rates)))

            if n_avail == 0:
                records.append({
                    "phase"         : PHASE_LABELS[phase],
                    "race_group"    : grp,
                    "n_seasons"     : n_seasons,
                    "mean_adj_rate" : np.nan,
                    "rr"            : "—",
                    "ci_95"         : "No data",
                    "denom_note"    : ", ".join(denom_versions),
                })
                continue

            rr, ci_lo, ci_hi = bootstrap_rr(grp_rates, ref_rates)

            log.info(f"  {phase} | {grp}: RR={rr} (95% CI {ci_lo}–{ci_hi}), "
                     f"n_seasons={n_avail}")

            records.append({
                "phase"          : PHASE_LABELS[phase],
                "race_group"     : grp,
                "n_seasons"      : n_avail,
                "mean_adj_rate"  : round(float(np.nanmean(grp_rates)), 2),
                "rr"             : str(rr),
                "ci_95"          : f"{ci_lo}–{ci_hi}",
                "denom_note"     : ", ".join(denom_versions),
            })

    table2 = pd.DataFrame(records)
    out_path = PATHS["tables"] / "table2_rr_race_phase.csv"
    table2.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  Table 2: {len(table2)} rows → {out_path}")
    log.info(f"\n{table2.to_string(index=False)}")

    export_publication_table2(table2)

    return table2


# =============================================================================
# S05 — SEASON-LEVEL PHASE COMPARISON (OLS + COHEN'S D)
# =============================================================================

def run_phase_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    S05 — Season-level phase comparison: OLS regression + Cohen's d.

    Returns both the OLS results DataFrame and the Cohen's d DataFrame.
    Both are written to CSV and publication-formatted versions exported.

    BUG FIX: Previously returned only result_df; cohens_df was computed but
    never returned, causing phase_cohens_d.csv to log as 0 rows in the final
    summary. Now returns a tuple (result_df, cohens_df) and the final summary
    dict references cohens_df directly.
    """
    section("S05. SEASON-LEVEL PHASE COMPARISON")

    overall_mask = (
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall")
    )
    lmm_df = df[overall_mask].copy()
    lmm_df = lmm_df[
        ~lmm_df["season"].isin(SUPPRESSED_WEEKLY_SEASONS) &
        (lmm_df["partial_season_flag"] == 0)
    ]

    season_level = (
        lmm_df.groupby(["season", "phase", "denom_version",
                         "pandemic_h1n1_flag"])
        .agg(
            mean_weekly = ("weekly_rate", "mean"),
            peak_weekly = ("weekly_rate", "max"),
            cum_rate    = ("cum_rate",    "max"),
            n_weeks     = ("weekly_rate", "count"),
        )
        .reset_index()
    )

    log.info(f"  Season-level dataset: {len(season_level)} seasons")
    log.info(f"  Phase counts: {season_level['phase'].value_counts().to_dict()}")
    log.info(f"\n{season_level[['season','phase','mean_weekly','peak_weekly','cum_rate']].to_string(index=False)}")

    season_level["phase_disruption"]   = (season_level["phase"] == "disruption").astype(float)
    season_level["phase_recovery"]     = (season_level["phase"] == "recovery").astype(float)
    season_level["denom_unbridged"]    = (season_level["denom_version"] == "unbridged").astype(float)
    season_level["pandemic_h1n1_flag"] = season_level["pandemic_h1n1_flag"].astype(float)

    records = []

    for model_label, formula in [
        ("primary",     "mean_weekly ~ phase_disruption + phase_recovery"),
        ("sensitivity", "mean_weekly ~ phase_disruption + phase_recovery "
                        "+ denom_unbridged + pandemic_h1n1_flag"),
    ]:
        try:
            ols = smf.ols(formula, data=season_level).fit()
            log.info(f"\n  [{model_label}] OLS summary:\n{ols.summary()}")

            fe  = ols.params
            ci  = ols.conf_int()
            pv  = ols.pvalues

            for param in fe.index:
                records.append({
                    "model"      : model_label,
                    "parameter"  : param,
                    "estimate"   : round(fe[param], 3),
                    "se"         : round(ols.bse[param], 3),
                    "ci_lower"   : round(ci.loc[param, 0], 3),
                    "ci_upper"   : round(ci.loc[param, 1], 3),
                    "t_stat"     : round(ols.tvalues[param], 3),
                    "p_value"    : round(pv[param], 4),
                    "r_squared"  : round(ols.rsquared, 3),
                    "adj_r2"     : round(ols.rsquared_adj, 3),
                    "n_seasons"  : int(ols.nobs),
                })

        except Exception as e:
            log.error(f"  [{model_label}] OLS failed: {e}")

    # Cohen's d pairwise between phases
    log.info("\n  Pairwise Cohen's d (effect sizes):")
    phase_groups = {
        p: season_level[season_level["phase"] == p]["mean_weekly"].values
        for p in PHASE_ORDER
    }
    pairs = [
        ("pre_pandemic", "disruption"),
        ("pre_pandemic", "recovery"),
        ("disruption",   "recovery"),
    ]
    cohens_records = []
    for a, b in pairs:
        ga, gb = phase_groups.get(a, np.array([])), phase_groups.get(b, np.array([]))
        if len(ga) < 2 or len(gb) < 2:
            continue
        pooled_sd = np.sqrt((np.var(ga, ddof=1) * (len(ga)-1) +
                              np.var(gb, ddof=1) * (len(gb)-1)) /
                             (len(ga) + len(gb) - 2))
        d = (np.mean(gb) - np.mean(ga)) / pooled_sd if pooled_sd > 0 else np.nan
        magnitude = (
            "small"  if abs(d) < 0.5 else
            "medium" if abs(d) < 0.8 else "large"
        )
        log.info(f"    {a} vs {b}: d={d:.2f} ({magnitude}), "
                 f"mean_a={np.mean(ga):.2f}, mean_b={np.mean(gb):.2f}")
        cohens_records.append({
            "comparison" : f"{a} vs {b}",
            "mean_a"     : round(float(np.mean(ga)), 3),
            "mean_b"     : round(float(np.mean(gb)), 3),
            "cohens_d"   : round(float(d), 3),
            "magnitude"  : magnitude,
        })

    result_df = pd.DataFrame(records)
    cohens_df = pd.DataFrame(cohens_records)   # ← was lost before; now returned

    out_path = PATHS["tables"] / "phase_comparison_ols.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    cohens_path = PATHS["tables"] / "phase_cohens_d.csv"
    cohens_df.to_csv(cohens_path, index=False, encoding="utf-8-sig")

    log.info(f"  phase_comparison_ols.csv → {out_path}")
    log.info(f"  phase_cohens_d.csv → {cohens_path} ({len(cohens_df)} rows)")

    # Export publication-formatted S3 tables (OLS only at this stage;
    # LMM passed in after run_lmm completes — see export_publication_supp_s3)
    export_publication_cohens_d(cohens_df)

    return result_df, cohens_df   # ← tuple return (was single df)


# =============================================================================
# S05b — LINEAR MIXED MODEL: PHASE TREND
# =============================================================================

def run_lmm(df: pd.DataFrame) -> pd.DataFrame:
    """
    S05b — Linear Mixed Model: weekly_rate ~ phase + season_week_num + (1|season)

    Season-level random intercept accounts for within-season autocorrelation.
    Phase fixed effects are not estimable (phase is invariant within season;
    ICC = 0 confirms complete confounding). The within-season week coefficient
    is the primary estimable fixed effect and is interpreted descriptively.
    """
    section("S05. LINEAR MIXED MODEL — PHASE TREND")

    overall_mask = (
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall")
    )
    lmm_df = df[overall_mask].copy()

    lmm_df = lmm_df[
        ~lmm_df["season"].isin(SUPPRESSED_WEEKLY_SEASONS) &
        (lmm_df["partial_season_flag"] == 0)
    ]
    lmm_df = lmm_df.dropna(subset=["weekly_rate", "season_week_num"])

    log.info(f"  LMM input: {len(lmm_df):,} rows, "
             f"{lmm_df['season'].nunique()} seasons")

    lmm_df["phase_disruption"] = (lmm_df["phase"] == "disruption").astype(float)
    lmm_df["phase_recovery"]   = (lmm_df["phase"] == "recovery").astype(float)
    lmm_df["season_week_num"]  = lmm_df["season_week_num"].astype(float)

    try:
        model = smf.mixedlm(
            "weekly_rate ~ phase_disruption + phase_recovery + season_week_num",
            data=lmm_df,
            groups=lmm_df["season"]
        )
        result = model.fit(reml=True, method="lbfgs")
        log.info(f"  LMM converged: {result.converged}")
        log.info(f"\n{result.summary()}")

        fe = result.fe_params
        ci = result.conf_int()
        pv = result.pvalues

        fe_table = pd.DataFrame({
            "parameter"  : fe.index,
            "estimate"   : fe.values.round(3),
            "se"         : result.bse[fe.index].values.round(3),
            "ci_lower"   : ci.loc[fe.index, 0].values.round(3),
            "ci_upper"   : ci.loc[fe.index, 1].values.round(3),
            "z_stat"     : result.tvalues[fe.index].values.round(3),
            "p_value"    : pv[fe.index].values.round(4),
        })

        param_labels = {
            "Intercept"          : "Intercept (pre-pandemic, week 1)",
            "phase_disruption"   : "Pandemic disruption vs pre-pandemic",
            "phase_recovery"     : "Post-pandemic recovery vs pre-pandemic",
            "season_week_num"    : "Within-season week (linear trend)",
        }
        fe_table["label"] = fe_table["parameter"].map(param_labels).fillna(
            fe_table["parameter"])

        re_var    = result.cov_re.iloc[0, 0] if hasattr(result, "cov_re") else np.nan
        resid_var = result.scale
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else np.nan
        log.info(f"  Random effects variance (season): {re_var:.4f}")
        log.info(f"  Residual variance: {resid_var:.4f}")
        log.info(f"  ICC (season-level clustering): {icc:.3f}")

        fe_table["random_effect_var_season"] = round(re_var, 4)
        fe_table["residual_var"]             = round(resid_var, 4)
        fe_table["icc_season"]               = round(icc, 3)
        fe_table["n_obs"]                    = len(lmm_df)
        fe_table["n_seasons"]                = lmm_df["season"].nunique()
        fe_table["reml"]                     = True
        fe_table["converged"]                = result.converged

        out_path = PATHS["tables"] / "lmm_phase_results.csv"
        fe_table.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info(f"  LMM results → {out_path}")
        log.info(f"\n{fe_table[['label','estimate','ci_lower','ci_upper','p_value']].to_string(index=False)}")
        return fe_table

    except Exception as e:
        log.error(f"  LMM failed: {e}")
        log.error("  Check that statsmodels >= 0.13 is installed")
        return pd.DataFrame()


# =============================================================================
# PUBLICATION TABLES: S3a (OLS), S3b (LMM), S3c (Cohen's d)
# =============================================================================

def build_publication_supp_s3(ols_df: pd.DataFrame,
                               lmm_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build publication-ready OLS (S3a) and LMM (S3b) tables."""

    PARAM_LABELS = {
        "Intercept"            : "Intercept (pre-pandemic reference)",
        "phase_disruption"     : "Pandemic disruption vs pre-pandemic (DISR)",
        "phase_recovery"       : "Post-pandemic recovery vs pre-pandemic (REC)",
        "denom_unbridged"      : "Denominator version (unbridged vs bridged)",
        "pandemic_h1n1_flag"   : "2009–10 H1N1 pandemic season indicator",
        "season_week_num"      : "Within-season week number (linear trend)",
    }

    # --- OLS ---
    ols = ols_df.copy()
    ols["parameter"] = ols["parameter"].map(PARAM_LABELS).fillna(ols["parameter"])
    ols = ols.rename(columns={
        "model"      : "Model",
        "parameter"  : "Parameter",
        "estimate"   : "Estimate",
        "se"         : "SE",
        "ci_lower"   : "95% CI Lower",
        "ci_upper"   : "95% CI Upper",
        "t_stat"     : "t",
        "p_value"    : "p",
        "r_squared"  : "R²",
        "adj_r2"     : "Adj R²",
        "n_seasons"  : "Seasons (n)",
    })
    ols["Model"] = ols["Model"].map({
        "primary"     : "Primary",
        "sensitivity" : "Sensitivity",
    }).fillna(ols["Model"])
    for col in ["Estimate", "SE", "95% CI Lower", "95% CI Upper"]:
        if col in ols.columns:
            ols[col] = pd.to_numeric(ols[col], errors="coerce").round(3)

    # --- LMM — drop degenerate phase rows, keep only estimable coefficients ---
    lmm = lmm_df.copy()
    degenerate_params = ["phase_disruption", "phase_recovery"]
    lmm = lmm[~lmm["parameter"].isin(degenerate_params)].copy()

    lmm["parameter"] = lmm["parameter"].map(PARAM_LABELS).fillna(lmm["parameter"])

    drop_cols = [c for c in ["random_effect_var_season", "residual_var",
                              "icc_season", "reml", "converged", "label",
                              "z_stat", "p_value", "n_obs"]
                 if c in lmm.columns]
    lmm = lmm.drop(columns=drop_cols)

    lmm = lmm.rename(columns={
        "parameter" : "Parameter",
        "estimate"  : "Estimate",
        "se"        : "SE",
        "ci_lower"  : "95% CI Lower",
        "ci_upper"  : "95% CI Upper",
        "n_seasons" : "Seasons (n)",
    })

    for col in ["Estimate", "SE", "95% CI Lower", "95% CI Upper"]:
        if col in lmm.columns:
            lmm[col] = pd.to_numeric(lmm[col], errors="coerce").round(3)

    return ols, lmm


def build_publication_cohens_d(cohens_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build publication-ready Cohen's d table (Supplementary Table S3c).

    Renames columns to clean manuscript headers, expands comparison labels
    to full phase names, and adds a formatted d (magnitude) column.

    Output: table_s3c_cohens_d_publication.csv
    """
    COMPARISON_LABELS = {
        "pre_pandemic vs disruption" : "Pre-pandemic vs Pandemic disruption (PRE vs DISR)",
        "pre_pandemic vs recovery"   : "Pre-pandemic vs Post-pandemic recovery (PRE vs REC)",
        "disruption vs recovery"     : "Pandemic disruption vs Post-pandemic recovery (DISR vs REC)",
    }

    pub = cohens_df.copy()
    pub["comparison"] = pub["comparison"].map(COMPARISON_LABELS).fillna(pub["comparison"])

    # Format d with magnitude in parentheses for direct manuscript use
    pub["d (magnitude)"] = pub.apply(
        lambda r: f"{r['cohens_d']:.2f} ({r['magnitude']})", axis=1
    )

    pub = pub.rename(columns={
        "comparison" : "Comparison",
        "mean_a"     : "Mean Rate Group A",
        "mean_b"     : "Mean Rate Group B",
        "cohens_d"   : "Cohen's d",
        "magnitude"  : "Magnitude",
    })

    # Reorder for clarity
    pub = pub[["Comparison", "Mean Rate Group A", "Mean Rate Group B",
               "Cohen's d", "Magnitude", "d (magnitude)"]]

    return pub


def export_publication_cohens_d(cohens_df: pd.DataFrame) -> None:
    """Export publication-formatted Cohen's d table (S3c)."""
    if cohens_df.empty:
        log.warning("  Cohen's d df is empty — S3c not written")
        return
    pub = build_publication_cohens_d(cohens_df)
    out_path = PATHS["tables"] / "table_s3c_cohens_d_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table_s3c_cohens_d_publication.csv → {out_path} ({len(pub)} rows)")


def export_publication_supp_s3(ols_df: pd.DataFrame, lmm_df: pd.DataFrame) -> None:
    """Export cleaned OLS and LMM SM tables (S3a, S3b)."""
    ols_pub, lmm_pub = build_publication_supp_s3(ols_df, lmm_df)

    ols_path = PATHS["tables"] / "table_s3a_ols_publication.csv"
    lmm_path = PATHS["tables"] / "table_s3b_lmm_publication.csv"

    ols_pub.to_csv(ols_path, index=False, encoding="utf-8-sig")
    lmm_pub.to_csv(lmm_path, index=False, encoding="utf-8-sig")

    log.info(f"  table_s3a_ols_publication.csv → {ols_path}")
    log.info(f"  table_s3b_lmm_publication.csv → {lmm_path}")


# =============================================================================
# S06 — STL DECOMPOSITION
# =============================================================================

def run_stl(ts_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    S06 — STL decomposition on overall weekly_rate time series.

    Runs separate STL fits on pre-pandemic and post-pandemic recovery series.
    Disruption phase excluded from both fits.

    STL parameters:
      period=52  — annual cycle
      robust=True — downweights outlier weeks in LOESS fitting

    Seasonal amplitude: IQR of seasonal component (robust to outlier weeks).
    Strength of seasonality (Fs): max(0, 1 - Var(R) / Var(S+R))
    Strength of trend (Ft): max(0, 1 - Var(R) / Var(T+R))
    """
    section("S06. STL DECOMPOSITION — PRE VS POST-PANDEMIC SERIES")

    ts = ts_df.sort_values("epiweek_date").copy()

    component_frames  = []
    amplitude_records = []

    for series_label, season_list in [
        ("pre_pandemic",  STL_PRE_SEASONS),
        ("post_pandemic", STL_POST_SEASONS),
    ]:
        sub = ts[ts["season"].isin(season_list)].copy()
        sub = sub.dropna(subset=["weekly_rate"]).sort_values("epiweek_date")

        if len(sub) < STL_PERIOD * 2:
            log.warning(f"  STL [{series_label}]: insufficient observations "
                        f"({len(sub)}) — need at least {STL_PERIOD * 2}")
            continue

        log.info(f"  STL [{series_label}]: n={len(sub)} weeks, "
                 f"seasons={season_list}")

        series = pd.Series(
            sub["weekly_rate"].values,
            index=sub["epiweek_date"],
            name="weekly_rate"
        )

        stl_fit = STL(series, period=STL_PERIOD, robust=STL_ROBUST).fit()

        comp_df = pd.DataFrame({
            "epiweek_date" : sub["epiweek_date"].values,
            "season"       : sub["season"].values,
            "series"       : series_label,
            "observed"     : stl_fit.observed,
            "trend"        : stl_fit.trend,
            "seasonal"     : stl_fit.seasonal,
            "residual"     : stl_fit.resid,
        })
        component_frames.append(comp_df)

        seasonal_amp_iqr   = float(np.percentile(stl_fit.seasonal, 75) -
                                    np.percentile(stl_fit.seasonal, 25))
        seasonal_amp_range = float(stl_fit.seasonal.max() -
                                    stl_fit.seasonal.min())
        trend_range = float(stl_fit.trend.max() - stl_fit.trend.min())
        resid_sd    = float(np.std(stl_fit.resid))

        var_r  = np.var(stl_fit.resid)
        var_sr = np.var(stl_fit.seasonal + stl_fit.resid)
        fs = max(0.0, 1.0 - var_r / var_sr) if var_sr > 0 else np.nan

        var_tr = np.var(stl_fit.trend + stl_fit.resid)
        ft = max(0.0, 1.0 - var_r / var_tr) if var_tr > 0 else np.nan

        log.info(f"  STL [{series_label}] seasonal amplitude IQR: {seasonal_amp_iqr:.3f}")
        log.info(f"  STL [{series_label}] trend range: {trend_range:.3f}")
        log.info(f"  STL [{series_label}] residual SD: {resid_sd:.3f}")
        log.info(f"  STL [{series_label}] strength of seasonality (Fs): {fs:.3f}")
        log.info(f"  STL [{series_label}] strength of trend (Ft): {ft:.3f}")

        amplitude_records.append({
            "series"                     : series_label,
            "n_weeks"                    : len(sub),
            "n_seasons"                  : len(season_list),
            "seasonal_amplitude_iqr"     : round(seasonal_amp_iqr, 3),
            "seasonal_amplitude_range"   : round(seasonal_amp_range, 3),
            "trend_range"                : round(trend_range, 3),
            "residual_sd"                : round(resid_sd, 3),
            "strength_of_seasonality_fs" : round(fs, 3),
            "strength_of_trend_ft"       : round(ft, 3),
            "stl_period"                 : STL_PERIOD,
            "stl_robust"                 : STL_ROBUST,
        })

    if component_frames:
        components = pd.concat(component_frames, ignore_index=True)
        out_comp = PATHS["tables"] / "stl_components.csv"
        components.to_csv(out_comp, index=False, encoding="utf-8-sig")
        log.info(f"  stl_components.csv → {out_comp} ({len(components):,} rows)")
    else:
        components = pd.DataFrame()
        log.warning("  No STL components produced — check series lengths")

    amplitude_df = pd.DataFrame(amplitude_records)
    out_amp = PATHS["tables"] / "stl_amplitude_summary.csv"
    amplitude_df.to_csv(out_amp, index=False, encoding="utf-8-sig")
    log.info(f"  stl_amplitude_summary.csv → {out_amp}")
    log.info(f"\n{amplitude_df.to_string(index=False)}")

    export_publication_stl_amplitude(amplitude_df)

    return components, amplitude_df


def build_publication_stl_amplitude(amplitude_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build publication-ready STL amplitude summary table.

    Renames series labels to display names, rounds all numeric columns,
    and uses clean manuscript-style column headers.

    Output: stl_amplitude_publication.csv
    """
    SERIES_LABELS = {
        "pre_pandemic"  : "Pre-pandemic (2009–10 to 2018–19)",
        "post_pandemic" : "Post-pandemic recovery (2022–23 to 2024–25)",
    }

    COL_RENAME = {
        "series"                     : "Series",
        "n_weeks"                    : "Weeks (n)",
        "n_seasons"                  : "Seasons (n)",
        "seasonal_amplitude_iqr"     : "Seasonal Amplitude IQR",
        "seasonal_amplitude_range"   : "Seasonal Amplitude Range",
        "trend_range"                : "Trend Range",
        "residual_sd"                : "Residual SD",
        "strength_of_seasonality_fs" : "Strength of Seasonality (Fs)",
        "strength_of_trend_ft"       : "Strength of Trend (Ft)",
        "stl_period"                 : "STL Period",
        "stl_robust"                 : "Robust LOESS",
    }

    pub = amplitude_df.copy()
    pub["series"] = pub["series"].map(SERIES_LABELS).fillna(pub["series"])
    pub = pub.rename(columns=COL_RENAME)

    # Round numeric columns for display
    for col in ["Seasonal Amplitude IQR", "Seasonal Amplitude Range",
                "Trend Range", "Residual SD",
                "Strength of Seasonality (Fs)", "Strength of Trend (Ft)"]:
        if col in pub.columns:
            pub[col] = pd.to_numeric(pub[col], errors="coerce").round(3)

    return pub


def export_publication_stl_amplitude(amplitude_df: pd.DataFrame) -> None:
    """Export publication-formatted STL amplitude summary."""
    if amplitude_df.empty:
        log.warning("  STL amplitude df is empty — publication table not written")
        return
    pub = build_publication_stl_amplitude(amplitude_df)
    out_path = PATHS["tables"] / "stl_amplitude_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  stl_amplitude_publication.csv → {out_path} ({len(pub)} rows)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:

    run_start = datetime.now()
    section("03_aim1_stats.py — Descriptive & Inferential Analysis")
    log.info(f"Run started  : {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Project root : {PROJECT_ROOT}")
    log.info(f"Output path  : {PATHS['tables']}")
    log.info(f"Random seed  : {SEED}")
    log.info(f"Bootstrap N  : {N_BOOTSTRAP}")

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------
    section("1. LOAD CLEANED DATA")

    df    = load_cleaned("flusurvnet_cleaned.csv", "flusurvnet_cleaned")
    ts_df = load_cleaned("cleaned_ts_overall.csv", "cleaned_ts_overall")

    log.info(f"  Phases in full dataset: {df['phase'].value_counts().to_dict()}")
    log.info(f"  Seasons in TS panel: {sorted(ts_df['season'].unique().tolist())}")

    # -------------------------------------------------------------------------
    # RUN ANALYSES
    # -------------------------------------------------------------------------

    # S01 — Dispersion check
    dispersion = run_dispersion_check(df)

    # S02 — Table 1: phase-level summary
    table1 = run_table1(df)

    # S03 — Supplementary Table S1: season-level detail
    supp_s1 = run_supp_table_s1(df)

    # S04 — Table 2: race/ethnicity RRs
    table2 = run_table2(df)

    # S05 — Phase comparison OLS + Cohen's d
    # BUG FIX: now unpacks tuple (result_df, cohens_df)
    ols_results, cohens_df = run_phase_comparison(df)

    # S05b — Linear Mixed Model
    lmm_fe = run_lmm(df)

    # SM S3a/S3b — Publication OLS + LMM tables
    export_publication_supp_s3(ols_results, lmm_fe)

    # S06 — STL decomposition
    stl_components, stl_amplitude = run_stl(ts_df)

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    section("FINAL SUMMARY")

    outputs = {
        "dispersion_check.csv"                   : dispersion,
        "table1_phase_summary.csv"               : table1,
        "table1_publication.csv"                 : table1,        # row count via pub builder
        "table1_supp_season_detail.csv"          : supp_s1,
        "table_s2_season_detail_publication.csv" : supp_s1,
        "table2_rr_race_phase.csv"               : table2,
        "table2_publication.csv"                 : table2,
        "phase_comparison_ols.csv"               : ols_results,
        "phase_cohens_d.csv"                     : cohens_df,      # ← FIXED: was pd.DataFrame()
        "table_s3c_cohens_d_publication.csv"     : cohens_df,      # ← NEW
        "lmm_phase_results.csv"                  : lmm_fe,
        "table_s3a_ols_publication.csv"          : ols_results,
        "table_s3b_lmm_publication.csv"          : lmm_fe,
        "stl_components.csv"                     : stl_components,
        "stl_amplitude_summary.csv"              : stl_amplitude,
        "stl_amplitude_publication.csv"          : stl_amplitude,  # ← NEW
    }

    log.info("  Outputs written:")
    for fname, df_out in outputs.items():
        n = len(df_out) if isinstance(df_out, pd.DataFrame) and len(df_out) > 0 else 0
        log.info(f"    {fname}: {n} rows")

    log.info(f"  Log file     : {LOG_FILE}")
    log.info(f"  Run completed: {(datetime.now() - run_start).seconds}s")
    section("DONE — proceed to 04_figures.py")


# =============================================================================
if __name__ == "__main__":
    main()
