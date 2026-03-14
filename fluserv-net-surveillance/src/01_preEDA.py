# =============================================================================
# 01_preEDA.py
# FluSurv-NET Influenza Hospitalization Rates — Pre-EDA Pipeline
#
# Purpose:
#   Ingest, clean, validate, and reshape the FluSurv-NET weekly hospitalization
#   rates dataset in preparation for all three research aims:
#     Aim 1 — Descriptive: pre/pandemic/post-pandemic phase characterization
#     Aim 2 — Inferential: demographic rate ratio and disparity trend analysis
#     Aim 3 — Predictive: interrupted time series + anomaly detection
#
# Data source:
#   CDC FluView Interactive — FluSurv-NET Hospitalization Rates
#   https://gis.cdc.gov/GRASP/Fluview/FluHospRates.html
#   Downloaded: 2026-03-07 | File: FluSurveillance_Custom_Download_Data.csv
#
# Outputs:
#   data/processed/flusurvnet_clean.csv         — full cleaned dataset
#   data/processed/panel_overall.csv           — overall + age-stratified weekly panel
#   data/processed/panel_race.csv               — race/ethnicity weekly panel
#   data/processed/panel_virus.csv              — virus type weekly panel
#   outputs/logs/01_preEDA_log.txt              — full data quality + EDA snapshot log
#
# Author: Hayden
# Seed: 88 (set for any stochastic operations; deterministic in this script)
# =============================================================================

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# -----------------------------------------------------------------------------
# RANDOM SEED
# -----------------------------------------------------------------------------
SEED = 88
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# DYNAMIC PATH RESOLUTION
#   Resolves project root relative to this script's location so the pipeline
#   runs regardless of where it is invoked from. Update RAW_FILENAME if the
#   source file is renamed.
# -----------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent        # .../src/
PROJECT_ROOT = SCRIPT_DIR.parent                       # .../Flu - IDR/
RAW_FILENAME = "FluSurveillance_Custom_Download_Data.csv"

PATHS = {
    "raw"     : PROJECT_ROOT / "data" / "unprocessed" / RAW_FILENAME,
    "proc"    : PROJECT_ROOT / "data" / "processed",
    "tables"  : PROJECT_ROOT / "outputs" / "tables",
    "figures" : PROJECT_ROOT / "outputs" / "figures",
    "logs"    : PROJECT_ROOT / "outputs" / "logs",
}

for p in PATHS.values():
    if p.suffix == "":          # directory
        p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LOGGING
#   Streams to both console and the log file simultaneously.
# -----------------------------------------------------------------------------
LOG_FILE = PATHS["logs"] / "01_preEDA_log.txt"

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
    """Print a visible section divider to the log."""
    bar = "=" * 70
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)

# =============================================================================
# CONSTANTS
# =============================================================================

# CDC uses "X" to denote suppressed/missing values in most columns.
# CI columns (LOWER, MEDIAN, UPPER) use the literal string "null" instead.
# Both are handled uniformly in coerce_rate().
CDC_MISSING_MARKERS = {"X", "null", "NULL", "Null", "NA", "N/A", ""}

# Canonical column names after normalisation.
# The raw FluSurv-NET CSV has two columns literally both named "YEAR":
#   col position 2: season string e.g. "2009-10"
#   col position 3: calendar year integer e.g. 2009
# pandas read_csv deduplicates them automatically as "YEAR" and "YEAR.1".
# The ingestion block verifies this and falls back to positional rename if needed.
COL_MAP = {
    "CATCHMENT"                    : "catchment",
    "NETWORK"                      : "network",
    "YEAR"                         : "season",           # season string "2009-10"
    "YEAR.1"                       : "year",             # calendar year integer
    "WEEK"                         : "week",
    "AGE CATEGORY"                 : "age_cat",
    "SEX CATEGORY"                 : "sex_cat",
    "RACE CATEGORY"                : "race_cat",
    "VIRUS TYPE CATEGORY"          : "virus_cat",
    "CUMULATIVE RATE"              : "cum_rate",
    "WEEKLY RATE"                  : "weekly_rate",
    "AGE ADJUSTED CUMULATIVE RATE" : "age_adj_cum_rate",
    "AGE ADJUSTED WEEKLY RATE"     : "age_adj_weekly_rate",
    " LOWER"                       : "ci_lower",         # leading space in raw header
    " MEDIAN"                      : "ci_median",
    " UPPER"                       : "ci_upper",
    " UPPER "                      : "ci_upper",   # trailing-space variant
}

# Phase classification — seasons are strings like "2009-10"
#   Pre-pandemic baseline  : 2009-10 through 2018-19
#   Pandemic disruption    : 2019-20, 2020-21, 2021-22
#   Post-pandemic recovery : 2022-23 onward
PHASE_MAP = {
    "2009-10": "pre_pandemic",
    "2010-11": "pre_pandemic",
    "2011-12": "pre_pandemic",
    "2012-13": "pre_pandemic",
    "2013-14": "pre_pandemic",
    "2014-15": "pre_pandemic",
    "2015-16": "pre_pandemic",
    "2016-17": "pre_pandemic",
    "2017-18": "pre_pandemic",
    "2018-19": "pre_pandemic",
    "2019-20": "disruption",
    "2020-21": "disruption",
    "2021-22": "disruption",
    "2022-23": "recovery",
    "2023-24": "recovery",    # first year-round surveillance season (52 weeks expected)
    "2024-25": "recovery",
    "2025-26": "recovery",    # in-progress at download (2026-03-07); flagged as partial
}

# Seasons that were incomplete / still in-progress at time of download.
# Retained in descriptive analyses but excluded from seasonality models
# (Prophet, ITS) that require complete seasons.
PARTIAL_SEASONS = {"2025-26"}

# Pandemic H1N1 flag — used for sensitivity analysis annotation only
PANDEMIC_H1N1_SEASON = "2009-10"

# Denominator version — CDC switched from bridged-race to unbridged census
# population estimates starting with the 2020-21 season. This affects rate
# comparability across the boundary and must be addressed in Methods.
DENOM_VERSION_MAP = {s: "bridged" for s in PHASE_MAP if s < "2020-21"}
DENOM_VERSION_MAP.update({s: "unbridged" for s in PHASE_MAP if s >= "2020-21"})

# FluSurv-NET age groups — CDC has expanded groupings over time.
# STANDARD groups present from 2009-10 onward.
# EXTENDED granular groups added in later seasons (e.g. 2012-13+).
# Both sets are valid; the audit warns only on truly unrecognised values.
AGE_GROUPS_STANDARD = {
    "0-4 yr", "5-17 yr", "18-49 yr", "50-64 yr", "65+ yr",
    "65-74 yr", "75-84 yr", "85+", "Overall",
}
AGE_GROUPS_EXTENDED = {
    "0-< 1 yr", "1-4 yr", "5-11  yr", "5-11 yr", "12-17 yr",
    "18-29 yr", "30-39 yr", "40-49 yr",
    "< 18", ">= 18", ">= 65 yr", ">= 75", ">= 85",
}
AGE_GROUPS_ALL = AGE_GROUPS_STANDARD | AGE_GROUPS_EXTENDED

# Race/ethnicity groups available in FluSurv-NET
RACE_GROUPS_STANDARD = {
    "White", "Black", "Hispanic/Latino",
    "Asian/Pacific Islander", "American Indian/Alaska Native", "Overall",
}

# Numeric rate columns for type coercion and QC
RATE_COLS = [
    "weekly_rate", "cum_rate",
    "age_adj_weekly_rate", "age_adj_cum_rate",
    "ci_lower", "ci_median", "ci_upper",
]

# Missingness threshold — strata with > this fraction missing will be flagged
MISS_WARN_THRESHOLD = 0.20


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def coerce_rate(series: pd.Series, col_name: str) -> pd.Series:
    """
    Convert a rate column to float, replacing all CDC missingness markers
    (CDC_MISSING_MARKERS: "X", "null", empty string, etc.) with NaN.
    Logs a warning if unexpected non-numeric values remain after substitution.
    """
    replace_map = {m: np.nan for m in CDC_MISSING_MARKERS}
    cleaned = (
        series.astype(str)
              .str.strip()
              .replace(replace_map)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    n_coerce_fail = cleaned.notna().sum() - numeric.notna().sum()
    if n_coerce_fail > 0:
        # Structurally sparse columns: CI cols (only for age-adj Overall rows)
        # and age_adj cols (only computed for specific strata). Demote to INFO.
        STRUCTURAL_COLS = {"ci_lower", "ci_median", "ci_upper",
                           "age_adj_cum_rate", "age_adj_weekly_rate"}
        if col_name in STRUCTURAL_COLS:
            log.info(f"  {col_name}: {n_coerce_fail:,} null/missing values → NaN (structurally sparse, expected)")
        else:
            log.warning(f"  {col_name}: {n_coerce_fail:,} values could not be coerced to float — set to NaN")
    return numeric


def assign_epiweek_date(year: pd.Series, week: pd.Series) -> pd.Series:
    """
    Construct an approximate ISO date for each MMWR epiweek as the Sunday
    starting that week. Uses pandas week-offset arithmetic.
    Returns a datetime Series. Rows that fail date construction are NaT.
    """
    dates = []
    for y, w in zip(year, week):
        try:
            # MMWR week 1 starts on the first Sunday of January
            # Approximate: Jan 4 is always in week 1 (ISO-aligned)
            jan4 = pd.Timestamp(int(y), 1, 4)
            week1_sun = jan4 - pd.Timedelta(days=jan4.dayofweek + 1)
            d = week1_sun + pd.Timedelta(weeks=int(w) - 1)
            dates.append(d)
        except Exception:
            dates.append(pd.NaT)
    return pd.Series(dates, dtype="datetime64[ns]")


def missingness_report(df: pd.DataFrame, group_cols: list, label: str) -> pd.DataFrame:
    """
    Compute fraction missing for RATE_COLS present in df, grouped by group_cols.
    Returns a tidy DataFrame and logs a summary.
    Flags any group × column combinations exceeding MISS_WARN_THRESHOLD.
    """
    present_rate_cols = [c for c in RATE_COLS if c in df.columns]
    records = []
    for col in present_rate_cols:
        miss = df.groupby(group_cols)[col].apply(lambda x: x.isna().mean()).reset_index()
        miss.columns = group_cols + ["miss_frac"]
        miss["variable"] = col
        records.append(miss)
    if not records:
        return pd.DataFrame()
    result = pd.concat(records, ignore_index=True)
    flagged = result[result["miss_frac"] > MISS_WARN_THRESHOLD]
    log.info(f"  [{label}] Missingness flags (>{MISS_WARN_THRESHOLD*100:.0f}%): {len(flagged)} group×variable combinations")
    for _, row in flagged.iterrows():
        grp_str = " | ".join(f"{c}={row[c]}" for c in group_cols)
        # Suppress structural missingness columns: these are only populated for
        # specific strata in the CDC export; 100% missing elsewhere is expected.
        # ci_lower/ci_median/ci_upper: only for age-adjusted Overall rows.
        # age_adj_cum_rate/age_adj_weekly_rate: only where age adjustment is computed.
        STRUCTURAL_MISS_COLS = {"ci_lower", "ci_median", "ci_upper",
                                "age_adj_cum_rate", "age_adj_weekly_rate"}
        if row["variable"] in STRUCTURAL_MISS_COLS:
            continue
        # Suppress warnings for seasons already known to have structural missingness:
        # 2020-21 (pandemic near-zero) and partial seasons are documented elsewhere.
        season_val = row.get("season", "")
        if str(season_val) == "2020-21" or str(season_val) in PARTIAL_SEASONS:
            continue
        log.warning(f"    FLAG: {row['variable']} — {grp_str} — {row['miss_frac']:.1%} missing")
    return result


def implausibility_check(df: pd.DataFrame) -> None:
    """
    Check for values that are logically implausible in a hospitalization
    rate dataset. Logs warnings; does not drop rows automatically.
      - Negative rates
      - Weekly rate > cumulative rate within same season (indicative of
        a data entry error or reshape issue)
      - Rates implausibly high (> 500 per 100k — extreme outlier threshold)
    """
    if "weekly_rate" in df.columns:
        neg = (df["weekly_rate"] < 0).sum()
        if neg > 0:
            log.warning(f"  IMPLAUSIBILITY: {neg} rows with negative weekly_rate")

        high = (df["weekly_rate"] > 500).sum()
        if high > 0:
            log.warning(f"  IMPLAUSIBILITY: {high} rows with weekly_rate > 500 per 100k")

    if "weekly_rate" in df.columns and "cum_rate" in df.columns:
        both = df[["weekly_rate", "cum_rate"]].dropna()
        exceed = (both["weekly_rate"] > both["cum_rate"]).sum()
        if exceed > 0:
            log.warning(f"  IMPLAUSIBILITY: {exceed} rows where weekly_rate > cum_rate")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:

    run_start = datetime.now()
    section("01_preEDA.py — FluSurv-NET Pre-EDA Pipeline")
    log.info(f"Run started : {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"Raw file    : {PATHS['raw']}")
    log.info(f"Random seed : {SEED}")

    # -------------------------------------------------------------------------
    # SECTION 1: INGESTION
    # -------------------------------------------------------------------------
    section("1. INGESTION")

    if not PATHS["raw"].exists():
        log.error(f"Raw data file not found: {PATHS['raw']}")
        log.error("Check that RAW_FILENAME matches the downloaded file exactly.")
        sys.exit(1)

    # Read with low_memory=False to avoid mixed-type warnings on rate columns
    raw = pd.read_csv(PATHS["raw"], low_memory=False, encoding="utf-8-sig")
    log.info(f"Raw shape       : {raw.shape[0]:,} rows × {raw.shape[1]} columns")
    log.info(f"Raw columns     : {list(raw.columns)}")

    # Detect and handle the duplicate YEAR column (season string vs. int)
    # pandas renames the second "YEAR" to "YEAR.1" automatically.
    # If that didn't happen (e.g. older pandas or tab-separated with BOM),
    # fall back to positional rename using known column order from the CDC export.
    year_cols = [c for c in raw.columns if c.upper().startswith("YEAR")]
    log.info(f"YEAR columns detected: {year_cols}")

    if "YEAR.1" not in raw.columns and len(year_cols) == 2:
        # Positional fallback: rename by index position
        # CDC layout: col 2 = season string, col 3 = calendar year
        col_list = list(raw.columns)
        first_year_idx  = next(i for i, c in enumerate(col_list) if c.upper() == "YEAR")
        second_year_idx = next(i for i, c in enumerate(col_list)
                               if c.upper() == "YEAR" and i != first_year_idx)
        col_list[second_year_idx] = "YEAR.1"
        raw.columns = col_list
        log.info("  Applied positional fallback rename: second YEAR → YEAR.1")
    elif "YEAR.1" in raw.columns:
        log.info("  pandas auto-deduplicated YEAR columns correctly (YEAR / YEAR.1)")

    # -------------------------------------------------------------------------
    # SECTION 2: COLUMN NORMALISATION
    # -------------------------------------------------------------------------
    section("2. COLUMN NORMALISATION")

    # Rename known columns; warn about any unmapped columns
    rename_dict = {k: v for k, v in COL_MAP.items() if k in raw.columns}
    unmapped = [c for c in raw.columns if c not in COL_MAP]
    if unmapped:
        log.warning(f"  Unmapped columns (kept as-is): {unmapped}")

    df = raw.rename(columns=rename_dict)
    log.info(f"  Columns after rename: {list(df.columns)}")

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()

    # -------------------------------------------------------------------------
    # SECTION 3: TYPE COERCION
    # -------------------------------------------------------------------------
    section("3. TYPE COERCION")

    # Integer columns
    for col in ["year", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            log.info(f"  {col}: coerced to Int64 — {df[col].isna().sum()} NaN")

    # Rate columns — replace CDC missingness markers and coerce to float
    for col in RATE_COLS:
        if col in df.columns:
            df[col] = coerce_rate(df[col], col)
            log.info(f"  {col}: float64 — {df[col].isna().sum():,} NaN ({df[col].isna().mean():.1%})")

    # -------------------------------------------------------------------------
    # SECTION 4: SEASON VALIDATION & PHASE ASSIGNMENT
    # -------------------------------------------------------------------------
    section("4. SEASON VALIDATION & PHASE ASSIGNMENT")

    # Drop blank/footer rows before season validation.
    # CDC exports sometimes include trailing metadata rows where season reads as
    # NaN (float) or the literal string "nan". Both are caught here.
    n_before = len(df)
    df = df.dropna(subset=["season"])
    df = df[~df["season"].astype(str).str.strip().str.lower().isin({"", "nan", "none"})]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        log.info(f"  Dropped {n_dropped} blank/footer rows (season NaN, empty, or literal 'nan')")

    # Validate season strings against expected format "YYYY-YY"
    observed_seasons = sorted(df["season"].dropna().unique().tolist())
    log.info(f"  Observed seasons ({len(observed_seasons)}): {observed_seasons}")

    unexpected_seasons = [s for s in observed_seasons if s not in PHASE_MAP]
    if unexpected_seasons:
        log.warning(f"  Seasons not in PHASE_MAP (will be assigned 'unknown'): {unexpected_seasons}")

    df["phase"] = df["season"].map(PHASE_MAP).fillna("unknown")
    df["denom_version"] = df["season"].map(DENOM_VERSION_MAP).fillna("unknown")
    df["pandemic_h1n1_flag"] = (df["season"] == PANDEMIC_H1N1_SEASON).astype(int)
    df["partial_season_flag"] = df["season"].isin(PARTIAL_SEASONS).astype(int)
    n_partial = df["partial_season_flag"].sum()
    log.info(f"  partial_season_flag=1: {n_partial:,} rows ({df['season'][df['partial_season_flag']==1].nunique()} season(s): {sorted(PARTIAL_SEASONS)})")

    # season_year: single integer representing the season start year (e.g. 2022 for "2022-23").
    # Derived from the season string — cleaner than the raw calendar year column for
    # groupby, axis labels, and joinpoint regression which expects an integer time index.
    df["season_year"] = (
        df["season"]
        .str.extract(r"^(\d{4})-", expand=False)
        .astype("Int64")
    )
    log.info(f"  season_year range: {df['season_year'].min()} – {df['season_year'].max()}")

    phase_counts = df["phase"].value_counts()
    log.info(f"  Phase distribution (rows):\n{phase_counts.to_string()}")
    log.info(f"  Denominator version distribution:\n{df['denom_version'].value_counts().to_string()}")

    # -------------------------------------------------------------------------
    # SECTION 5: EPIWEEK DATE INDEX
    # -------------------------------------------------------------------------
    section("5. EPIWEEK DATE INDEX")

    df["epiweek_date"] = assign_epiweek_date(df["year"], df["week"])
    n_nat = df["epiweek_date"].isna().sum()
    log.info(f"  epiweek_date constructed — NaT: {n_nat}")
    log.info(f"  Date range: {df['epiweek_date'].min()} to {df['epiweek_date'].max()}")

    # -------------------------------------------------------------------------
    # SECTION 6: CATCHMENT FILTER
    # -------------------------------------------------------------------------
    section("6. CATCHMENT FILTER")

    catchment_counts = df["catchment"].value_counts()
    log.info(f"  Catchment values:\n{catchment_counts.to_string()}")

    # Retain Entire Network only for primary analysis
    # Sub-network rows (EIP, IHSP individual states) are preserved in the
    # full cleaned file but excluded from the analytic panels
    n_before = len(df)
    df_network = df[df["catchment"].str.upper().str.contains("ENTIRE", na=False)].copy()
    log.info(f"  Rows retained (Entire Network): {len(df_network):,} of {n_before:,}")

    # -------------------------------------------------------------------------
    # SECTION 7: STRATUM AUDIT
    # -------------------------------------------------------------------------
    section("7. STRATUM AUDIT")

    for col, standard, extended in [
        ("age_cat",  AGE_GROUPS_STANDARD, AGE_GROUPS_EXTENDED),
        ("race_cat", RACE_GROUPS_STANDARD, set()),
    ]:
        if col not in df_network.columns:
            continue
        observed = sorted(df_network[col].dropna().unique().tolist())
        all_known = standard | extended
        unexpected = [v for v in observed if v not in all_known]
        extended_present = [v for v in observed if v in extended]
        log.info(f"  {col} observed values ({len(observed)}): {observed}")
        if extended_present:
            log.info(f"    Extended/granular {col} groups present: {extended_present}")
        if unexpected:
            log.warning(f"    Truly unrecognised {col} values: {unexpected}")

    # Sex categories
    if "sex_cat" in df_network.columns:
        log.info(f"  sex_cat values: {sorted(df_network['sex_cat'].dropna().unique().tolist())}")

    # Virus type categories
    if "virus_cat" in df_network.columns:
        log.info(f"  virus_cat values: {sorted(df_network['virus_cat'].dropna().unique().tolist())}")

    # -------------------------------------------------------------------------
    # SECTION 8: IMPLAUSIBILITY CHECKS
    # -------------------------------------------------------------------------
    section("8. IMPLAUSIBILITY CHECKS")

    implausibility_check(df_network)
    log.info(f"  weekly_rate stats:\n{df_network['weekly_rate'].describe().to_string()}")
    log.info(f"  cum_rate stats:\n{df_network['cum_rate'].describe().to_string()}")

    # -------------------------------------------------------------------------
    # SECTION 9: PANDEMIC SEASON STRUCTURE CHECK
    # -------------------------------------------------------------------------
    section("9. PANDEMIC SEASON STRUCTURE CHECK")

    # 2020-21 should show near-zero rates — verify this is data not missingness
    if "2020-21" in df_network["season"].values:
        s2021 = df_network[df_network["season"] == "2020-21"]
        overall_2021 = s2021[
            (s2021["age_cat"] == "Overall") &
            (s2021["sex_cat"] == "Overall") &
            (s2021["race_cat"] == "Overall") &
            (s2021["virus_cat"] == "Overall")
        ]["weekly_rate"]
        log.info(f"  2020-21 overall weekly_rate — n={len(overall_2021)}, "
                 f"mean={overall_2021.mean():.3f}, max={overall_2021.max():.3f}")
        if overall_2021.isna().all():
            log.warning("  2020-21 overall rates are ALL NaN — may not be present in download")
        elif overall_2021.max() < 0.5:
            log.info("  2020-21 confirms near-zero rates (pandemic suppression) — not missing data")

    # -------------------------------------------------------------------------
    # SECTION 10: COMPREHENSIVE MISSINGNESS REPORT
    # -------------------------------------------------------------------------
    section("10. COMPREHENSIVE MISSINGNESS REPORT")

    # By season
    miss_by_season = missingness_report(df_network, ["season"], "by_season")

    # By race × season (most critical for Aim 2 disparity analysis)
    if "race_cat" in df_network.columns:
        miss_race = missingness_report(
            df_network[df_network["race_cat"] != "Overall"],
            ["season", "race_cat"],
            "race_by_season"
        )

    # By age × season
    if "age_cat" in df_network.columns:
        miss_age = missingness_report(
            df_network[df_network["age_cat"] != "Overall"],
            ["season", "age_cat"],
            "age_by_season"
        )

    # Age-adjusted rate availability (only populated for Overall strata)
    if "age_adj_weekly_rate" in df_network.columns:
        adj_avail = df_network["age_adj_weekly_rate"].notna().sum()
        adj_total = len(df_network)
        log.info(f"  age_adj_weekly_rate available: {adj_avail:,} of {adj_total:,} rows ({adj_avail/adj_total:.1%})")

    # -------------------------------------------------------------------------
    # SECTION 11: TEMPORAL COVERAGE COMPLETENESS
    # -------------------------------------------------------------------------
    section("11. TEMPORAL COVERAGE COMPLETENESS")

    # Expected weeks per season (standard flu season: weeks 40–20 = ~33 weeks)
    # Post-2025-26 will be year-round; flag if season length is anomalous
    weeks_per_season = (
        df_network[
            (df_network["age_cat"] == "Overall") &
            (df_network["sex_cat"] == "Overall") &
            (df_network["race_cat"] == "Overall") &
            (df_network["virus_cat"] == "Overall")
        ]
        .groupby("season")["week"]
        .nunique()
        .sort_index()
    )
    log.info(f"  Weeks per season (Overall strata):\n{weeks_per_season.to_string()}")

    short_seasons = weeks_per_season[weeks_per_season < 20]
    if len(short_seasons) > 0:
        log.warning(f"  Seasons with fewer than 20 weeks (may be incomplete): {short_seasons.index.tolist()}")

    # -------------------------------------------------------------------------
    # SECTION 12: RATE RATIO PREVIEW (Aim 2 readiness check)
    # -------------------------------------------------------------------------
    section("12. RATE RATIO PREVIEW — AIM 2 READINESS")

    # Check that White reference group is present and sufficiently complete
    # to compute rate ratios for all other race/ethnicity groups
    if "race_cat" in df_network.columns and "age_adj_cum_rate" in df_network.columns:
        # Use .max() to capture the end-of-season cumulative rate.
        # age_adj_cum_rate is a running cumulative — .first() would return
        # the week-40 partial value (near zero); .max() returns the season total.
        seasonal_race = (
            df_network[
                (df_network["age_cat"] == "Overall") &
                (df_network["sex_cat"] == "Overall") &
                (df_network["virus_cat"] == "Overall")
            ]
            .groupby(["season", "race_cat"])["age_adj_cum_rate"]
            .max()
            .unstack("race_cat")
        )
        log.info(f"  Seasonal age-adjusted cumulative rate by race/ethnicity:")
        log.info(f"\n{seasonal_race.to_string()}")

        if "White" in seasonal_race.columns:
            for grp in [c for c in seasonal_race.columns if c != "White" and c != "Overall"]:
                rr = (seasonal_race[grp] / seasonal_race["White"]).dropna()
                log.info(f"  Rate ratio {grp}/White — n seasons={len(rr)}, "
                         f"mean={rr.mean():.2f}, range=[{rr.min():.2f}, {rr.max():.2f}]")
        else:
            log.warning("  'White' reference group not found in race_cat — rate ratio computation will need adjustment")

    # -------------------------------------------------------------------------
    # SECTION 13: SEASON-LEVEL SUMMARY FOR AIMS 1 & 3
    # -------------------------------------------------------------------------
    section("13. SEASON-LEVEL SUMMARY — AIM 1 & AIM 3 READINESS")

    # Compute peak rate, peak week, cumulative rate per season for Overall strata
    overall_mask = (
        (df_network["age_cat"] == "Overall") &
        (df_network["sex_cat"] == "Overall") &
        (df_network["race_cat"] == "Overall") &
        (df_network["virus_cat"] == "Overall")
    )
    season_summary = (
        df_network[overall_mask]
        .groupby(["season", "phase", "denom_version"])
        .agg(
            n_weeks       = ("week", "nunique"),
            peak_weekly   = ("weekly_rate", "max"),
            peak_week     = ("week", lambda x: x.iloc[df_network.loc[x.index, "weekly_rate"].fillna(0).argmax()]
                             if "weekly_rate" in df_network.columns else np.nan),
            mean_weekly   = ("weekly_rate", "mean"),
            cum_rate_max  = ("cum_rate", "max"),
            age_adj_cum   = ("age_adj_cum_rate", "max"),
        )
        .reset_index()
        .sort_values("season")
    )
    log.info(f"  Season-level summary (Overall strata):\n{season_summary.to_string(index=False)}")

    # -------------------------------------------------------------------------
    # SECTION 14: RESHAPE INTO ANALYTIC PANELS
    # -------------------------------------------------------------------------
    section("14. RESHAPE INTO ANALYTIC PANELS")

    # Panel 1: Overall + age-stratified weekly rates
    panel_overall = df_network[
        (df_network["sex_cat"] == "Overall") &
        (df_network["race_cat"] == "Overall") &
        (df_network["virus_cat"] == "Overall")
    ].copy()
    log.info(f"  panel_overall shape: {panel_overall.shape}")

    # Panel 2: Race/ethnicity panel (Overall age, sex, virus)
    panel_race = df_network[
        (df_network["age_cat"] == "Overall") &
        (df_network["sex_cat"] == "Overall") &
        (df_network["virus_cat"] == "Overall") &
        (df_network["race_cat"] != "Overall")
    ].copy()
    log.info(f"  panel_race shape: {panel_race.shape}")

    # Panel 3: Virus type panel (Overall age, sex, race)
    panel_virus = df_network[
        (df_network["age_cat"] == "Overall") &
        (df_network["sex_cat"] == "Overall") &
        (df_network["race_cat"] == "Overall") &
        (df_network["virus_cat"] != "Overall")
    ].copy()
    log.info(f"  panel_virus shape: {panel_virus.shape}")

    # -------------------------------------------------------------------------
    # SECTION 15: DRIFT & MODELLING PITFALL SUMMARY
    # -------------------------------------------------------------------------
    section("15. DRIFT & MODELLING PITFALL SUMMARY")

    log.info("  Key considerations for downstream modelling:")
    log.info("  [1] DENOMINATOR DISCONTINUITY: Rates before 2020-21 use bridged-race")
    log.info("      population estimates; from 2020-21 onward use unbridged census.")
    log.info("      denom_version column flags this. Recommend sensitivity analysis")
    log.info("      excluding boundary season or using age-adjusted rates for trend work.")
    log.info("  [2] PANDEMIC NEAR-ZERO SEASONS (2020-21): These are real observations,")
    log.info("      not missing data. Include explicitly as disruption phase;")
    log.info("      do NOT impute. Isolation Forest will score these as anomalies.")
    log.info("  [3] 2009-10 H1N1 PANDEMIC: pandemic_h1n1_flag=1 on this season.")
    log.info("      Consider sensitivity analysis excluding from pre-pandemic baseline.")
    log.info("  [4] RACE/ETHNICITY COMPLETENESS: varies by season — check panel_race")
    log.info("      missingness before computing rate ratios for Aim 2.")
    log.info("  [5] TIME SERIES STATIONARITY: STL decomposition and Prophet both")
    log.info("      require the time index to be complete and monotonic — verify no")
    log.info("      duplicate epiweek_date values within Overall strata.")
    log.info("  [6] AGE-ADJUSTED RATES: only available for Overall strata subsets.")
    log.info("      Use age_adj_cum_rate for Aim 2 disparity comparisons; use")
    log.info("      weekly_rate for Aim 3 time series (age-adjusted weekly is sparse).")

    # Check for duplicate epiweek dates in the time series panel.
    # Duplicates arise when two seasons share the same calendar year + week
    # (e.g. a season boundary epiweek that appears in both the trailing
    # and leading season). These must be resolved before ITS/Prophet fitting.
    ts_check = panel_overall[panel_overall["age_cat"] == "Overall"][
        ["epiweek_date", "season", "week", "weekly_rate"]
    ].dropna(subset=["epiweek_date"])
    dupe_mask = ts_check["epiweek_date"].duplicated(keep=False)
    dupes = dupe_mask.sum()
    if dupes > 0:
        log.warning(f"  DRIFT FLAG: {dupes} duplicate epiweek_date values in Overall time series panel")
        log.warning("  Duplicate rows (shown for inspection):")
        dupe_str = ts_check[dupe_mask].sort_values("epiweek_date").to_string(index=False)
        log.warning(f"  {dupe_str}")
        log.warning("  Resolution strategy for 02_EDA.py / 05_aim3_ml.py:")
        log.warning("    Keep the row with the higher weekly_rate when duplicates exist.")
        log.warning("    This is conservative and avoids dropping data from either season.")
    else:
        log.info("  Time series index: no duplicate epiweek_date values in Overall panel — clean for ITS/Prophet")

    # -------------------------------------------------------------------------
    # SECTION 16: WRITE OUTPUTS
    # -------------------------------------------------------------------------
    section("16. WRITE OUTPUTS")

    # Full cleaned dataset (Entire Network, all strata)
    out_csv = PATHS["proc"] / "flusurvnet_clean.csv"
    df_network.to_csv(out_csv, index=False, encoding="utf-8")
    log.info(f"  flusurvnet_clean.csv → {out_csv}")

    # Analytic panels
    for panel, name in [
        (panel_overall, "panel_overall"),
        (panel_race,    "panel_race"),
        (panel_virus,   "panel_virus"),
    ]:
        path = PATHS["proc"] / f"{name}.csv"
        panel.to_csv(path, index=False, encoding="utf-8")
        log.info(f"  {name}.csv → {path}  ({len(panel):,} rows)")

    # Season summary table
    sum_path = PATHS["tables"] / "season_summary_overall.csv"
    season_summary.to_csv(sum_path, index=False)
    log.info(f"  season_summary_overall.csv → {sum_path}")

    # Missingness reports
    if len(miss_by_season) > 0:
        miss_path = PATHS["tables"] / "missingness_by_season.csv"
        miss_by_season.to_csv(miss_path, index=False)
        log.info(f"  missingness_by_season.csv → {miss_path}")

    # -------------------------------------------------------------------------
    # SECTION 17: FINAL SUMMARY
    # -------------------------------------------------------------------------
    section("17. FINAL SUMMARY")

    log.info(f"  Raw rows ingested          : {raw.shape[0]:,}")
    log.info(f"  Entire Network rows        : {len(df_network):,}")
    log.info(f"  Seasons present            : {len(observed_seasons)}")
    log.info(f"  Phase breakdown            :\n{df_network['phase'].value_counts().to_string()}")
    log.info(f"  Analytic panels written    : panel_overall, panel_race, panel_virus")
    log.info(f"  Log file                   : {LOG_FILE}")
    log.info(f"  Run completed in           : {(datetime.now() - run_start).seconds}s")
    section("DONE — proceed to 02_EDA.py")


# =============================================================================
if __name__ == "__main__":
    main()
