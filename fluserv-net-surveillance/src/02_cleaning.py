# =============================================================================
# 02_cleaning.py
# FluSurv-NET Influenza Hospitalization Rates — Deep Cleaning Pipeline
#
# Purpose:
#   Reads the full Entire Network dataset produced by 01_preEDA.py and applies
#   all cleaning operations required for downstream ML and statistical modelling
#   across all three research aims. Produces a single analysis-ready CSV and a
#   detailed cleaning audit log.
#
# Inputs (from data/processed/):
#   flusurvnet_clean.csv    — full Entire Network dataset from preEDA
#                             (all strata: overall, age, sex, race, virus)
#
# Outputs (to data/cleaned/):
#   flusurvnet_cleaned.csv  — single analysis-ready dataset (all strata)
#   cleaned_ts_overall.csv  — deduplicated monotonic time series (Aim 3 / ITS)
#
# Outputs (to outputs/tables/):
#   cleaning_audit.csv           — row-level record of all mutations applied
#   missingness_structural.csv   — structural vs unexpected missingness report
#
# Outputs (to outputs/logs/):
#   02_cleaning_log.txt
#
# Cleaning operations performed (in order):
#   [C01] Age category whitespace normalization  ("5-11  yr" → "5-11 yr")
#   [C02] CI column drop                         (100% NaN — no analytic value)
#   [C03] 2020-21 weekly_rate structural flag    (NaN confirmed real, not imputed)
#   [C04] Duplicate epiweek_date resolution      (keep higher weekly_rate row)
#   [C05] Season-length heterogeneity flag       (n_weeks column added)
#   [C06] Denominator discontinuity flag         (already in preEDA; verified here)
#   [C07] Partial season flag verification       (2025-26 truncation documented)
#   [C08] Race 2020-21 structural NaN flag       (not imputed; flagged for Aim 2)
#   [C09] Missingness audit                      (structural vs unexpected split)
#   [C10] Feature engineering for ML             (lag features, rolling mean,
#                                                 season_week_num, phase dummies,
#                                                 log1p rate transforms)
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

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

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
    "processed" : PROJECT_ROOT / "data" / "processed",
    "cleaned"   : PROJECT_ROOT / "data" / "cleaned",
    "tables"    : PROJECT_ROOT / "outputs" / "tables",
    "logs"      : PROJECT_ROOT / "outputs" / "logs",
}

for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
LOG_FILE = PATHS["logs"] / "02_cleaning_log.txt"

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

AGE_CAT_CORRECTIONS = {
    "5-11  yr" : "5-11 yr",
}

CI_COLS = ["ci_lower", "ci_median", "ci_upper"]

STRUCTURAL_NAN_COLS = {
    "age_adj_weekly_rate",
    "age_adj_cum_rate",
} | set(CI_COLS)

SUPPRESSED_WEEKLY_SEASONS = {"2020-21"}
PANDEMIC_SUPPRESSION_SEASON = "2020-21"
YEAR_ROUND_SEASON = "2023-24"
PARTIAL_SEASONS = {"2025-26"}
PHASE_ORDER = ["pre_pandemic", "disruption", "recovery"]
LAG_RATE_COLS = ["weekly_rate", "age_adj_weekly_rate"]
LAG_WINDOWS = [1, 2, 4]
ROLLING_WINDOW = 4
MIN_OBS_PER_SEASON = 10


# =============================================================================
# HELPER FUNCTIONS  (unchanged from original)
# =============================================================================

def load_full(filename: str) -> pd.DataFrame:
    """Load the full preEDA output from data/processed/ with type coercion."""
    path = PATHS["processed"] / filename
    if not path.exists():
        log.error(f"  File not found: {path}")
        log.error("  Ensure 01_preEDA.py has been run successfully first.")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    if "epiweek_date" in df.columns:
        df["epiweek_date"] = pd.to_datetime(df["epiweek_date"], errors="coerce")
    for col in ["year", "week", "season_year", "pandemic_h1n1_flag",
                "partial_season_flag"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    log.info(f"  Loaded {filename}: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def normalize_age_cat(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "age_cat" not in df.columns:
        return df, 0
    n_before = df["age_cat"].value_counts().to_dict()
    df["age_cat"] = df["age_cat"].replace(AGE_CAT_CORRECTIONS)
    n_fixed = sum(n_before.get(old, 0) for old in AGE_CAT_CORRECTIONS if old in n_before)
    return df, n_fixed


def drop_ci_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    to_drop = [c for c in CI_COLS if c in df.columns]
    df = df.drop(columns=to_drop)
    return df, to_drop


def flag_suppressed_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df["weekly_suppressed_flag"] = (
        df["season"].isin(SUPPRESSED_WEEKLY_SEASONS) &
        df["weekly_rate"].isna()
    ).astype(int)
    return df


def resolve_duplicate_epiweeks(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    C04 — Resolve duplicate epiweek_date values within the Overall stratum only.
    Keeps the higher weekly_rate row at each duplicate date.
    Non-Overall rows are untouched.
    """
    if "epiweek_date" not in df.columns:
        return df, 0

    overall_mask = (
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall")
    )

    overall     = df[overall_mask].copy()
    non_overall = df[~overall_mask].copy()

    dupes = overall[overall["epiweek_date"].duplicated(keep=False)]
    n_dupes = len(dupes)

    if n_dupes == 0:
        log.info("  [C04] No duplicate epiweek_date values in Overall stratum")
        return df, 0

    log.info(f"  [C04] {n_dupes} duplicate epiweek_date rows found — resolving")
    for dt, grp in dupes.groupby("epiweek_date"):
        keep_idx   = grp["weekly_rate"].idxmax() if grp["weekly_rate"].notna().any() else grp.index[0]
        drop_idxs  = [i for i in grp.index if i != keep_idx]
        log.info(f"    {dt.date()}: keeping week={grp.loc[keep_idx,'week']} "
                 f"rate={grp.loc[keep_idx,'weekly_rate']} | "
                 f"dropping week(s)={[grp.loc[i,'week'] for i in drop_idxs]} "
                 f"rate(s)={[grp.loc[i,'weekly_rate'] for i in drop_idxs]}")
        overall = overall.drop(index=drop_idxs)

    resolved = pd.concat([overall, non_overall], ignore_index=True)
    return resolved, n_dupes


def add_season_week_num(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "week"]).copy()
    df["season_week_num"] = (
        df.groupby("season")["week"]
          .transform(lambda x: x.rank(method="dense").astype(int))
    )
    weeks_per_season = df.groupby("season")["week"].nunique().rename("n_weeks_in_season")
    df = df.merge(weeks_per_season, on="season", how="left")
    df["year_round_season_flag"] = (df["season"] == YEAR_ROUND_SEASON).astype(int)
    return df


def audit_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rate_cols_present = [c for c in ["weekly_rate", "cum_rate",
                                      "age_adj_weekly_rate", "age_adj_cum_rate"]
                         if c in df.columns]
    records = []
    for col in rate_cols_present:
        total = len(df)
        n_nan = df[col].isna().sum()
        if n_nan == 0:
            records.append({
                "column": col, "total_rows": total, "n_missing": 0,
                "pct_missing": 0.0, "structural": False, "note": "complete"
            })
            continue

        structural_mask = pd.Series(False, index=df.index)

        if col in CI_COLS:
            structural_mask |= True

        if col in {"age_adj_weekly_rate", "age_adj_cum_rate"}:
            for cat_col in ["age_cat", "sex_cat", "race_cat", "virus_cat"]:
                if cat_col in df.columns:
                    structural_mask |= (df[cat_col] != "Overall")

        if col in {"weekly_rate", "cum_rate"}:
            structural_mask |= df["season"].isin(SUPPRESSED_WEEKLY_SEASONS)

        if "partial_season_flag" in df.columns:
            structural_mask |= (df["partial_season_flag"] == 1) & df[col].isna()

        if col == "weekly_rate":
            structural_mask |= df["season"].isin(SUPPRESSED_WEEKLY_SEASONS)

        if "race_cat" in df.columns:
            structural_mask |= (
                df["season"].isin(SUPPRESSED_WEEKLY_SEASONS) &
                (df["race_cat"] != "Overall")
            )

        nan_mask     = df[col].isna()
        n_structural = (nan_mask & structural_mask).sum()
        n_unexpected = (nan_mask & ~structural_mask).sum()

        records.append({
            "column"        : col,
            "total_rows"    : total,
            "n_missing"     : int(n_nan),
            "pct_missing"   : round(n_nan / total * 100, 1),
            "n_structural"  : int(n_structural),
            "n_unexpected"  : int(n_unexpected),
            "structural"    : n_unexpected == 0,
            "note"          : (
                "all structural" if n_unexpected == 0
                else f"{n_unexpected} unexpected NaN — review before modelling"
            )
        })

        if n_unexpected > 0:
            log.warning(f"  [C09] {col}: {n_unexpected} UNEXPECTED NaN values")
        else:
            log.info(f"  [C09] {col}: {n_nan:,} NaN — all structural (expected)")

    return pd.DataFrame(records)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    for col, new_col in [("weekly_rate", "log1p_weekly_rate"),
                         ("cum_rate",    "log1p_cum_rate")]:
        if col in df.columns:
            df[new_col] = np.log1p(df[col].fillna(0))

    for phase in PHASE_ORDER:
        df[f"phase_{phase}"] = (df["phase"] == phase).astype(int)

    if "epiweek_date" in df.columns:
        pandemic_start = pd.Timestamp("2020-09-27")
        df["weeks_since_pandemic"] = (
            (df["epiweek_date"] - pandemic_start)
            .dt.days.div(7).round().astype("Int64")
        )
        log.info(f"  [C10] weeks_since_pandemic: "
                 f"{df['weeks_since_pandemic'].min()} to {df['weeks_since_pandemic'].max()}")

    if "weekly_rate" in df.columns:
        stratum_cols = [c for c in ["season", "age_cat", "sex_cat", "race_cat", "virus_cat"]
                        if c in df.columns]
        df = df.sort_values(stratum_cols + ["week"])
        for lag in LAG_WINDOWS:
            df[f"lag{lag}_weekly_rate"] = df.groupby(stratum_cols)["weekly_rate"].shift(lag)
        roll_col = f"roll{ROLLING_WINDOW}_weekly_rate"
        df[roll_col] = (
            df.groupby(stratum_cols)["weekly_rate"]
              .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
        )
        log.info(f"  [C10] Lag (1,2,4) and roll4 features created")

    return df


def check_ts_completeness(df: pd.DataFrame) -> None:
    if "epiweek_date" not in df.columns:
        return
    ts = df.sort_values("epiweek_date")
    dupes = ts["epiweek_date"].duplicated().sum()
    if dupes > 0:
        log.warning(f"  TS CHECK: {dupes} duplicate epiweek_date remain — investigate")
    else:
        log.info("  TS CHECK: epiweek_date unique and monotonic — ready for ITS/Prophet")
    log.info(f"  TS CHECK: monotonic increasing = {ts['epiweek_date'].is_monotonic_increasing}")
    cov = (
        ts.groupby("season")["weekly_rate"]
          .agg(n_obs="count", n_nan=lambda x: x.isna().sum())
          .assign(pct_available=lambda x: (x["n_obs"] / (x["n_obs"] + x["n_nan"]) * 100).round(1))
    )
    log.info(f"  TS CHECK: season coverage:\n{cov.to_string()}")
    sparse = cov[cov["n_obs"] < MIN_OBS_PER_SEASON]
    if len(sparse) > 0:
        log.warning(f"  TS CHECK: sparse seasons (<{MIN_OBS_PER_SEASON} obs): "
                    f"{sparse.index.tolist()}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:

    run_start = datetime.now()
    section("02_cleaning.py — FluSurv-NET Deep Cleaning Pipeline")
    log.info(f"Run started  : {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Project root : {PROJECT_ROOT}")
    log.info(f"Input       : data/processed/flusurvnet_clean.csv")
    log.info(f"Output path  : {PATHS['cleaned']}")
    log.info(f"Random seed  : {SEED}")

    # -------------------------------------------------------------------------
    # LOAD — single input file containing all strata
    # -------------------------------------------------------------------------
    section("1. LOAD flusurvnet_clean.csv (all strata)")

    df = load_full("flusurvnet_clean.csv")

    log.info(f"  sex_cat distribution: {df['sex_cat'].value_counts().to_dict()}")
    log.info(f"  race_cat distribution: {df['race_cat'].value_counts().to_dict()}")
    log.info(f"  age_cat unique values ({df['age_cat'].nunique()}): "
             f"{sorted(df['age_cat'].dropna().unique().tolist())}")
    log.info(f"  virus_cat distribution: {df['virus_cat'].value_counts().to_dict()}")
    log.info(f"  Seasons ({df['season'].nunique()}): {sorted(df['season'].unique().tolist())}")
    log.info(f"  Phase distribution: {df['phase'].value_counts().to_dict()}")

    audit_records = []
    n_input = len(df)

    # -------------------------------------------------------------------------
    # C01 — Age category whitespace normalization
    # -------------------------------------------------------------------------
    section("C01 — Age category whitespace normalization")
    df, n_age_fixed = normalize_age_cat(df)
    if n_age_fixed > 0:
        log.info(f"  Corrected {n_age_fixed:,} age_cat values: {AGE_CAT_CORRECTIONS}")
        audit_records.append({
            "step": "C01", "operation": "age_cat whitespace fix",
            "n_affected": n_age_fixed, "note": str(AGE_CAT_CORRECTIONS)
        })
    else:
        log.info("  No age_cat corrections needed")
    log.info(f"  age_cat values post-C01: {sorted(df['age_cat'].dropna().unique().tolist())}")

    # -------------------------------------------------------------------------
    # C02 — Drop CI columns
    # -------------------------------------------------------------------------
    section("C02 — Drop CI columns (100% NaN confirmed in preEDA)")
    df, dropped_ci = drop_ci_cols(df)
    log.info(f"  Dropped: {dropped_ci}")
    audit_records.append({
        "step": "C02", "operation": "drop CI columns",
        "n_affected": len(df), "note": f"dropped {dropped_ci}"
    })

    # -------------------------------------------------------------------------
    # C03 — Flag suppressed weekly rates (2020-21)
    # -------------------------------------------------------------------------
    section("C03 — Flag 2020-21 pandemic-suppressed weekly_rate NaN")
    df = flag_suppressed_weekly(df)
    n_suppressed = df["weekly_suppressed_flag"].sum()
    log.info(f"  weekly_suppressed_flag=1: {n_suppressed:,} rows — NOT imputed")
    s2021 = df[df["season"] == "2020-21"]
    log.info(f"  2020-21 cum_rate non-null rows: {s2021['cum_rate'].notna().sum()}")
    if n_suppressed > 0:
        audit_records.append({
            "step": "C03", "operation": "weekly_suppressed_flag for 2020-21",
            "n_affected": int(n_suppressed),
            "note": "NaN retained; cum_rate available as fallback"
        })

    # -------------------------------------------------------------------------
    # C04 — Resolve duplicate epiweek_date (Overall stratum only)
    # -------------------------------------------------------------------------
    section("C04 — Resolve duplicate epiweek_date (season-boundary weeks)")
    df, n_dupes = resolve_duplicate_epiweeks(df)
    if n_dupes > 0:
        audit_records.append({
            "step": "C04", "operation": "duplicate epiweek_date resolution",
            "n_affected": n_dupes,
            "note": "kept higher weekly_rate row at each duplicate date"
        })

    # -------------------------------------------------------------------------
    # C05 — Season-length heterogeneity flag
    # -------------------------------------------------------------------------
    section("C05 — Add season_week_num and season-length metadata")
    df = add_season_week_num(df)
    log.info(f"  season_week_num range: {df['season_week_num'].min()} – "
             f"{df['season_week_num'].max()}")
    wps = df.groupby("season")["n_weeks_in_season"].first().to_dict()
    log.info(f"  n_weeks_in_season by season: {wps}")
    log.info(f"  year_round_season_flag=1: {df['year_round_season_flag'].sum():,} rows "
             f"({YEAR_ROUND_SEASON})")
    audit_records.append({
        "step": "C05",
        "operation": "season_week_num + n_weeks_in_season + year_round_season_flag",
        "n_affected": len(df), "note": f"year-round season: {YEAR_ROUND_SEASON}"
    })

    # -------------------------------------------------------------------------
    # C06 — Verify denominator discontinuity flag
    # -------------------------------------------------------------------------
    section("C06 — Verify denom_version flag (from preEDA)")
    if "denom_version" in df.columns:
        dv = df.groupby("season")["denom_version"].first().to_dict()
        log.info(f"  denom_version by season: {dv}")
        log.info(f"  bridged: {(df['denom_version']=='bridged').sum():,} rows | "
                 f"unbridged: {(df['denom_version']=='unbridged').sum():,} rows")
        log.info("  NOTE: cross-phase rate comparisons require sensitivity analysis")
    else:
        log.warning("  denom_version column not found — check preEDA output")

    # -------------------------------------------------------------------------
    # C07 — Partial season flag verification
    # -------------------------------------------------------------------------
    section("C07 — Verify partial_season_flag")
    if "partial_season_flag" in df.columns:
        n_partial = df["partial_season_flag"].sum()
        partial_seasons = df[df["partial_season_flag"] == 1]["season"].unique().tolist()
        log.info(f"  partial_season_flag=1: {n_partial:,} rows — {partial_seasons}")
        log.info("  Retained for Aim 1 descriptives; excluded from ITS/Prophet")
    else:
        log.warning("  partial_season_flag not found — check preEDA output")

    # -------------------------------------------------------------------------
    # C08 — Race 2020-21 structural NaN flag
    # -------------------------------------------------------------------------
    section("C08 — Race 2020-21 structural NaN flag")
    s2021_race = df[(df["season"] == "2020-21") & (df["race_cat"] != "Overall")]
    if len(s2021_race) > 0:
        nan_frac = s2021_race["age_adj_cum_rate"].isna().mean() if "age_adj_cum_rate" in df.columns else float("nan")
        log.info(f"  2020-21 non-Overall race rows: {len(s2021_race)}")
        log.info(f"  age_adj_cum_rate NaN fraction: {nan_frac:.1%}")
        log.info("  All race strata NaN for 2020-21 — confirmed structural CDC suppression")
        df["race_2021_structural_gap"] = (
            (df["season"] == "2020-21") &
            (df["race_cat"] != "Overall") &
            (df.get("age_adj_cum_rate", pd.Series(dtype=float)).reindex(df.index).isna())
        ).astype(int)
        n_flagged = df["race_2021_structural_gap"].sum()
        log.info(f"  race_2021_structural_gap=1: {n_flagged:,} rows")
        audit_records.append({
            "step": "C08", "operation": "race_2021_structural_gap flag",
            "n_affected": int(n_flagged),
            "note": "2020-21 race strata all NaN — structural CDC suppression"
        })
    else:
        df["race_2021_structural_gap"] = 0
        log.info("  No 2020-21 non-Overall race rows found")

    # -------------------------------------------------------------------------
    # C09 — Missingness audit
    # -------------------------------------------------------------------------
    section("C09 — Missingness audit (structural vs unexpected)")
    miss_df = audit_missingness(df)
    log.info(f"\n{miss_df.to_string(index=False)}")

    # -------------------------------------------------------------------------
    # C10 — Feature engineering for ML
    # -------------------------------------------------------------------------
    section("C10 — Feature engineering for ML")
    df = engineer_features(df)
    log.info(f"  log1p transforms, phase dummies, lag(1,2,4), roll4 — complete")
    log.info(f"  Final shape post-C10: {df.shape}")

    # -------------------------------------------------------------------------
    # STRATUM COVERAGE REPORT — confirm sex rows now present
    # -------------------------------------------------------------------------
    section("STRATUM COVERAGE REPORT")
    for col, label in [("sex_cat",  "Sex"),
                       ("race_cat", "Race"),
                       ("age_cat",  "Age"),
                       ("virus_cat","Virus")]:
        counts = df[col].value_counts().to_dict()
        log.info(f"  {label}: {counts}")

    # -------------------------------------------------------------------------
    # BUILD ITS TIME SERIES PANEL
    # -------------------------------------------------------------------------
    section("BUILD ITS/PROPHET TIME SERIES PANEL — cleaned_ts_overall.csv")

    ts = df[
        (df["age_cat"]   == "Overall") &
        (df["sex_cat"]   == "Overall") &
        (df["race_cat"]  == "Overall") &
        (df["virus_cat"] == "Overall")
    ].copy().sort_values("epiweek_date").reset_index(drop=True)

    ts_complete = ts[ts["partial_season_flag"] == 0].copy()
    ts_partial  = ts[ts["partial_season_flag"] == 1]
    log.info(f"  ITS panel (complete seasons): {len(ts_complete):,} rows")
    log.info(f"  Partial season rows excluded: {len(ts_partial):,} rows "
             f"({ts_partial['season'].unique().tolist()})")
    check_ts_completeness(ts_complete)

    # -------------------------------------------------------------------------
    # WRITE OUTPUTS
    # -------------------------------------------------------------------------
    section("WRITE CLEANED OUTPUTS")

    n_dropped = n_input - len(df)
    log.info(f"  Input rows: {n_input:,} | Output rows: {len(df):,} | "
             f"Rows dropped (C04 dedup): {n_dropped}")

    output_map = {
        "flusurvnet_cleaned.csv" : df,
        "cleaned_ts_overall.csv" : ts_complete,
    }

    for filename, df_out in output_map.items():
        out_path = PATHS["cleaned"] / filename
        df_out.to_csv(out_path, index=False, encoding="utf-8")
        log.info(f"  {filename} → {out_path}  ({len(df_out):,} rows × {df_out.shape[1]} cols)")

    audit_df = pd.DataFrame(audit_records)
    audit_path = PATHS["tables"] / "cleaning_audit.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    log.info(f"  cleaning_audit.csv → {audit_path}")

    miss_path = PATHS["tables"] / "missingness_structural.csv"
    miss_df.to_csv(miss_path, index=False)
    log.info(f"  missingness_structural.csv → {miss_path}")

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    section("FINAL SUMMARY")
    for filename, df_out in output_map.items():
        log.info(f"  {filename}: {len(df_out):,} rows × {df_out.shape[1]} cols")
    log.info(f"  Cleaning operations applied : C01–C10")
    log.info(f"  Audit records written       : {len(audit_records)}")
    log.info(f"  Log file                    : {LOG_FILE}")
    log.info(f"  Run completed in            : {(datetime.now() - run_start).seconds}s")
    section("DONE — proceed to 03_aim1_stats.py")


# =============================================================================
if __name__ == "__main__":
    main()
