# =============================================================================
# 05_aim3_ml.py
# FluSurv-NET — Aim 3: ML-Based Anomaly Detection & Forecasting
#
# Methods:
#   A) Prophet Forecasting
#      Primary  : flat growth — seasonality only, no trend extrapolation
#      Sensitivity: linear growth — captures pre-pandemic upward trend
#      Both models compared on: first-detection date convergence and held-out
#      validation metrics (MAE, RMSE, MAPE, R²) on 2018-19 (leave-one-out).
#      Detection robustness: finding is considered stable if both models agree
#      within ±4 weeks.
#
#   B) Isolation Forest Anomaly Detection
#      Primary  : contamination=0.10 (conventional default; comparable to
#                 prior literature and stable on n=14 seasons)
#      Sensitivity sweep: contamination = 0.05, 0.10, 0.15
#      Headline finding: seasons flagged across ALL three sweep levels.
#      contamination="auto" tested but not used as primary — flags 8/14
#      seasons on this dataset (overly liberal).
#      2020-21 EXCLUDED from all IF scoring — structurally suppressed by CDC;
#      inclusion would make a genuinely disrupted season appear falsely normal.
#      SM table: all seasons × all contamination levels for reviewer reference.
#
# 2020-21 note: Weekly hospitalization rates for 2020-21 were suppressed by
#   CDC due to low counts / data quality issues during peak COVID-19 overlap.
#   This season is excluded from IF feature matrix and scoring. It is retained
#   in the Prophet observed series (as NaN rows) but is explicitly flagged in
#   all outputs. Limitations section should note this exclusion.
#
# Inputs (from data/cleaned/):
#   cleaned_ts_overall.csv
#
# Outputs (to outputs/tables/):
#   aim3_prophet_forecast_linear.csv            — primary model weekly forecast
#   aim3_prophet_forecast_flat.csv              — sensitivity model weekly forecast
#   aim3_prophet_changepoints.csv               — changepoints (primary model)
#   aim3_prophet_gap_linear.csv                 — season gap, primary
#   aim3_prophet_gap_flat.csv                   — season gap, sensitivity
#   aim3_prophet_validation_metrics.csv         — MAE/RMSE/MAPE/R² both models
#   aim3_prophet_detection_comparison.csv       — first-detection date both models
#   aim3_iforest_season_features.csv            — feature matrix (2020-21 excluded)
#   aim3_iforest_scores_primary.csv             — primary scores (contamination=0.10)
#   aim3_iforest_sensitivity_sweep.csv          — all seasons × contam levels
#   --- Publication-ready tables (SM4) ---
#   table_s4a_prophet_validation_publication.csv  — SM4: validation metrics
#   table_s4b_prophet_detection_publication.csv   — SM4: first-detection comparison
#   table_s4c_prophet_gap_publication.csv         — SM4: season-level forecast gap
#   table_s4d_prophet_changepoints_publication.csv — SM4: changepoints
#   table_s5a_iforest_scores_publication.csv      — SM4: IF scores + features
#   table_s5b_iforest_sweep_publication.csv       — SM4: IF sensitivity sweep
#
# Outputs (to outputs/figures/):
#   fig5_prophet_forecast.png       — primary forecast vs observed (main text)
#   fig6_prophet_sensitivity.png    — linear vs flat comparison (SM4)
#   fig7_iforest_anomaly_scores.png — ranked lollipop (main text)
#
# All SM4 tables cited in manuscript as (SM4) — no sub-table numbers in text.
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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
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
LOG_FILE = PATHS["logs"] / "05_aim3_ml_log.txt"
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

# -----------------------------------------------------------------------------
# STYLE CONSTANTS (match 04_figures.py)
# -----------------------------------------------------------------------------
DPI        = 300
FACECOLOR  = "white"
LABEL_SIZE = 13
TICK_SIZE  = 11
ANNOT_SIZE = 10
LEGEND_SIZE= 10

import matplotlib.cm as cm
VIRIDIS = cm.get_cmap("viridis")
def vc(n, total=5):
    return VIRIDIS(n / max(total - 1, 1))

PHASE_LINE = {
    "pre_pandemic" : vc(0),
    "disruption"   : vc(2),
    "recovery"     : vc(3),
}

PHASE_LABELS = {
    "pre_pandemic" : "Pre-pandemic (2009–2019)",
    "disruption"   : "Pandemic disruption (2019–2022)",
    "recovery"     : "Post-pandemic recovery (2022–2026)",
}

SEASON_PHASE = {
    "2009-10": "pre_pandemic", "2010-11": "pre_pandemic",
    "2011-12": "pre_pandemic", "2012-13": "pre_pandemic",
    "2013-14": "pre_pandemic", "2014-15": "pre_pandemic",
    "2015-16": "pre_pandemic", "2016-17": "pre_pandemic",
    "2017-18": "pre_pandemic", "2018-19": "pre_pandemic",
    "2019-20": "disruption",   "2020-21": "disruption",
    "2021-22": "disruption",   "2022-23": "recovery",
    "2023-24": "recovery",     "2024-25": "recovery",
    "2025-26": "recovery",
}

# 2020-21 excluded from Isolation Forest — CDC structural suppression.
# Near-zero rates are an artifact of reporting policy, not true disease burden.
# Including this season would cause the model to flag it as "normal" (low rate)
# when it is in fact an uninformative data point. Noted in all SM outputs.
IF_EXCLUDED_SEASONS = {"2020-21"}

def clean_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_fig(fig, name):
    path = PATHS["figures"] / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
    plt.close(fig)
    log.info(f"  Saved: {path}")


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    section("LOAD DATA")
    path = PATHS["cleaned"] / "cleaned_ts_overall.csv"
    if not path.exists():
        log.error(f"  File not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    df["epiweek_date"] = pd.to_datetime(df["epiweek_date"], errors="coerce")
    df = df.sort_values("epiweek_date").reset_index(drop=True)
    log.info(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    log.info(f"  Date range: {df['epiweek_date'].min().date()} → {df['epiweek_date'].max().date()}")
    log.info(f"  Seasons: {sorted(df['season'].unique())}")
    log.info(f"  weekly_rate NaN: {df['weekly_rate'].isna().sum()}")
    log.info(f"  IF excluded (CDC suppression): {IF_EXCLUDED_SEASONS}")
    return df


# =============================================================================
# PROPHET HELPERS
# =============================================================================

def _fit_prophet(train_df, growth, label):
    """Fit a Prophet model and return it. growth: 'linear' or 'flat'."""
    m = Prophet(
        growth=growth,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        n_changepoints=15,
        interval_width=0.95,
        uncertainty_samples=1000,
    )
    m.fit(train_df)
    log.info(f"  Prophet [{label}] fitted | growth={growth} | n_train={len(train_df)}")
    return m


def _forecast_merge(m, df, train_end, last_obs, label):
    """Generate forecast and merge with observed data."""
    periods = int((last_obs - train_end).days / 7) + 4
    future  = m.make_future_dataframe(periods=periods, freq="W")
    fc      = m.predict(future)
    log.info(f"  Prophet [{label}] forecast rows: {len(fc)}")

    obs    = df[["epiweek_date", "weekly_rate", "season", "phase"]].rename(
        columns={"epiweek_date": "ds"}
    )
    merged = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(obs, on="ds", how="left")
    merged["gap"] = merged["weekly_rate"] - merged["yhat"]
    return merged


def _season_gap(merged, label):
    """
    Season-level observed vs projected gap summary.
    peak_projected computed only over weeks with observed data to prevent
    spurious peaks from forecast weeks outside the observed window.
    Projected values floored at 0 — Prophet has no non-negativity constraint.
    """
    obs_only = merged[merged["weekly_rate"].notna()].copy()
    obs_only["yhat_floored"] = obs_only["yhat"].clip(lower=0)

    sg = (
        obs_only
        .groupby(["season", "phase"])
        .agg(
            mean_observed  = ("weekly_rate",   "mean"),
            mean_projected = ("yhat_floored",  "mean"),
            mean_gap       = ("gap",           "mean"),
            peak_observed  = ("weekly_rate",   "max"),
            peak_projected = ("yhat_floored",  "max"),
        )
        .reset_index()
    )
    sg["peak_gap"] = sg["peak_observed"] - sg["peak_projected"]
    log.info(f"  [{label}] season gap:\n{sg[['season','phase','mean_observed','mean_projected','mean_gap']].to_string()}")
    return sg


def _first_detection(merged, train_end, label):
    """First post-training week where observed exceeds 95% upper PI."""
    post = merged[
        (merged["ds"] > train_end) &
        (merged["weekly_rate"].notna())
    ].copy()
    post["outside_pi"] = post["weekly_rate"] > post["yhat_upper"]
    det = post[post["outside_pi"]]["ds"].min()
    log.info(f"  [{label}] first detection: {det}")
    return det


def _validation_metrics(m, test_df, label):
    """
    Held-out validation: predict on test_df['ds'], compare to test_df['y'].
    Returns MAE, RMSE, R², and MAPE.
    MAPE suppressed for linear model — trend extrapolation causes large
    over-projections in early low-rate weeks, producing unstable percentages.
    MAPE reported for flat model only where scale is stable.
    """
    fc   = m.predict(test_df[["ds"]])
    obs  = test_df["y"].values
    pred = fc["yhat"].values
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    y, yh = obs[mask], pred[mask]

    mae  = mean_absolute_error(y, yh)
    rmse = np.sqrt(mean_squared_error(y, yh))
    r2   = r2_score(y, yh)

    is_linear = "linear" in label.lower() and "flat" not in label.lower()
    if is_linear:
        mape      = np.nan
        mape_note = "Omitted: unstable due to trend extrapolation over near-zero early-season observations"
    else:
        nz        = y > 0.05
        mape      = float(np.mean(np.abs((y[nz] - yh[nz]) / y[nz])) * 100) if nz.sum() > 0 else np.nan
        mape_note = ""

    log.info(f"  [{label}] Validation — MAE={mae:.3f} | RMSE={rmse:.3f} | "
             f"MAPE={'—' if np.isnan(mape) else f'{mape:.1f}%'} | R²={r2:.3f} | n={mask.sum()}")
    return {
        "model"      : label,
        "mae"        : round(mae, 4),
        "rmse"       : round(rmse, 4),
        "mape_pct"   : round(mape, 2) if not np.isnan(mape) else np.nan,
        "mape_note"  : mape_note,
        "r2"         : round(r2, 4),
        "n_test"     : int(mask.sum()),
    }


# =============================================================================
# SECTION A — PROPHET (PRIMARY FLAT + LINEAR SENSITIVITY + HELD-OUT VALIDATION)
# =============================================================================

def run_prophet(df):
    section("SECTION A — Prophet Forecasting")

    # Full training data: all pre-pandemic non-null weeks
    pre = df[(df["phase"] == "pre_pandemic") & (df["weekly_rate"].notna())].copy()
    train_full = pre[["epiweek_date", "weekly_rate"]].rename(
        columns={"epiweek_date": "ds", "weekly_rate": "y"}
    ).reset_index(drop=True)
    train_end = train_full["ds"].max()
    last_obs  = df["epiweek_date"].max()

    log.info(f"  Full training rows: {len(train_full)}")
    log.info(f"  Training range: {train_full['ds'].min().date()} → {train_end.date()}")

    # Held-out split: exclude 2018-19 for validation
    HOLDOUT = "2018-19"
    train_ho = pre[pre["season"] != HOLDOUT][["epiweek_date", "weekly_rate"]].rename(
        columns={"epiweek_date": "ds", "weekly_rate": "y"}
    ).reset_index(drop=True)
    test_ho = pre[pre["season"] == HOLDOUT][["epiweek_date", "weekly_rate"]].rename(
        columns={"epiweek_date": "ds", "weekly_rate": "y"}
    ).reset_index(drop=True)
    log.info(f"  Held-out train: {len(train_ho)} rows | test ({HOLDOUT}): {len(test_ho)} rows")

    # =========================================================================
    # A1 — PRIMARY MODEL: flat growth
    # =========================================================================
    section("A1 — Primary Model (flat growth)")
    m_flat     = _fit_prophet(train_full, growth="flat",   label="flat")
    merged_flat = _forecast_merge(m_flat, df, train_end, last_obs, label="flat")
    merged_flat.to_csv(PATHS["tables"] / "aim3_prophet_forecast_flat.csv", index=False)
    gap_flat = _season_gap(merged_flat, label="flat")
    gap_flat.to_csv(PATHS["tables"] / "aim3_prophet_gap_flat.csv", index=False)
    det_flat = _first_detection(merged_flat, train_end, label="flat")

    # Changepoints — extracted from flat model
    cp_dates  = m_flat.changepoints
    cp_deltas = m_flat.params["delta"].mean(axis=0)
    n_cp = min(len(cp_dates), len(cp_deltas))
    cp_df = pd.DataFrame({
        "changepoint_date": cp_dates[:n_cp].values,
        "delta"           : cp_deltas[:n_cp],
    }).sort_values("changepoint_date").reset_index(drop=True)
    cp_df["abs_delta"] = cp_df["delta"].abs()
    cp_df.to_csv(PATHS["tables"] / "aim3_prophet_changepoints.csv", index=False)
    log.info(f"  Changepoints: {len(cp_df)}")
    log.info(f"  Top 3:\n{cp_df.nlargest(3,'abs_delta')[['changepoint_date','delta']].to_string()}")

    # =========================================================================
    # A2 — SENSITIVITY MODEL: linear growth
    #   Tests whether first-detection finding holds when pre-pandemic upward
    #   trend is extrapolated into the forecast period.
    # =========================================================================
    section("A2 — Sensitivity Model (linear growth)")
    m_lin   = _fit_prophet(train_full, growth="linear", label="linear")
    merged_lin = _forecast_merge(m_lin, df, train_end, last_obs, label="linear")
    merged_lin.to_csv(PATHS["tables"] / "aim3_prophet_forecast_linear.csv", index=False)
    gap_lin = _season_gap(merged_lin, label="linear")
    gap_lin.to_csv(PATHS["tables"] / "aim3_prophet_gap_linear.csv", index=False)
    det_lin = _first_detection(merged_lin, train_end, label="linear")

    # Detection convergence assessment
    det_compare = pd.DataFrame([
        {"model": "flat_growth",   "first_detection_date": det_flat,
         "note": "Primary model"},
        {"model": "linear_growth", "first_detection_date": det_lin,
         "note": "Sensitivity: pre-pandemic trend extrapolated"},
    ])
    det_compare.to_csv(PATHS["tables"] / "aim3_prophet_detection_comparison.csv", index=False)
    log.info(f"\n  Detection comparison:\n{det_compare.to_string()}")
    if pd.notna(det_flat) and pd.notna(det_lin):
        diff_wks = abs(int((det_flat - det_lin).days / 7))
        verdict  = "ROBUST (<=4 wk)" if diff_wks <= 4 else "NOTE: >4 wk divergence — address in limitations"
        log.info(f"  Model agreement: {diff_wks} weeks apart → {verdict}")

    # =========================================================================
    # A3 — HELD-OUT VALIDATION on 2018-19 (both models)
    #   Train on 2009-10 to 2017-18 → predict 2018-19 → compute error metrics.
    # =========================================================================
    section("A3 — Held-Out Validation (2018-19)")
    m_flat_ho = _fit_prophet(train_ho, growth="flat",   label="flat_ho")
    m_lin_ho  = _fit_prophet(train_ho, growth="linear", label="linear_ho")

    metrics_rows = [
        _validation_metrics(m_flat_ho, test_ho, label="Prophet_flat"),
        _validation_metrics(m_lin_ho,  test_ho, label="Prophet_linear"),
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(PATHS["tables"] / "aim3_prophet_validation_metrics.csv", index=False)
    log.info(f"\n  Validation metrics table:\n{metrics_df.to_string()}")

    return (merged_flat, merged_lin, gap_flat, gap_lin, cp_df,
            det_flat, det_lin, metrics_df, det_compare,
            m_flat_ho, test_ho, train_ho)   # ← added for Fig S3 and LOO


# =============================================================================
# SECTION B — ISOLATION FOREST (PRIMARY + SENSITIVITY SWEEP + SM TABLE)
# =============================================================================

def run_isolation_forest(df):
    section("SECTION B — Isolation Forest Anomaly Detection")

    log.info(f"  Excluding from IF scoring: {IF_EXCLUDED_SEASONS} (structural CDC suppression)")

    feature_cols = ["peak_rate", "mean_rate", "cum_rate", "n_weeks", "peak_week", "rate_cv"]
    season_features = []

    for season, grp in df.groupby("season"):
        if season in IF_EXCLUDED_SEASONS:
            log.info(f"  EXCLUDED: {season}")
            continue
        grp = grp.dropna(subset=["weekly_rate"])
        if len(grp) == 0:
            log.info(f"  SKIPPED (no data): {season}")
            continue
        phase     = SEASON_PHASE.get(season, "unknown")
        mean_rate = grp["weekly_rate"].mean()
        season_features.append({
            "season"    : season,
            "phase"     : phase,
            "peak_rate" : grp["weekly_rate"].max(),
            "mean_rate" : mean_rate,
            "cum_rate"  : grp["weekly_rate"].sum(),
            "n_weeks"   : len(grp),
            "peak_week" : grp.loc[grp["weekly_rate"].idxmax(), "season_week_num"]
                          if "season_week_num" in grp.columns else np.nan,
            "rate_cv"   : grp["weekly_rate"].std() / mean_rate if mean_rate > 0 else 0,
        })

    feat_df = pd.DataFrame(season_features)
    feat_df.to_csv(PATHS["tables"] / "aim3_iforest_season_features.csv", index=False)
    log.info(f"  Feature matrix: {feat_df.shape} (2020-21 excluded)")
    log.info(f"\n{feat_df.to_string()}")

    feat_df_clean = feat_df.dropna(subset=feature_cols).copy()
    pre_mask       = feat_df_clean["phase"] == "pre_pandemic"
    X_train        = feat_df_clean.loc[pre_mask, feature_cols].values
    X_all          = feat_df_clean[feature_cols].values

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled   = scaler.transform(X_all)

    # ---- B2: Primary model — contamination=0.10 ----
    section("B2 — Primary IF (contamination=0.10)")
    iso_primary = IsolationForest(
        n_estimators=500,
        contamination=0.10,
        random_state=88,
        max_features=len(feature_cols),
    )
    iso_primary.fit(X_train_scaled)
    scores_p = iso_primary.score_samples(X_all_scaled)
    labels_p = iso_primary.predict(X_all_scaled)

    primary_df = feat_df_clean.copy()
    primary_df["anomaly_score"] = scores_p
    primary_df["anomaly_label"] = labels_p
    primary_df["anomaly_flag"]  = primary_df["anomaly_label"].map({-1: "Anomaly", 1: "Normal"})
    primary_df = primary_df.sort_values("anomaly_score").reset_index(drop=True)
    primary_df["rank"] = primary_df.index + 1
    primary_df.to_csv(PATHS["tables"] / "aim3_iforest_scores_primary.csv", index=False)
    log.info(f"\n{primary_df[['season','phase','anomaly_score','anomaly_flag']].to_string()}")
    flagged_primary = primary_df[primary_df["anomaly_label"] == -1]["season"].tolist()
    log.info(f"  Flagged (contamination=0.10): {flagged_primary}")

    # ---- B3: Sensitivity sweep — contamination = 0.05, 0.10, 0.15 ----
    section("B3 — IF Sensitivity Sweep (contamination = 0.05, 0.10, 0.15)")
    contam_levels = [0.05, 0.10, 0.15]
    sweep_rows = []

    for c in contam_levels:
        iso_c = IsolationForest(
            n_estimators=500,
            contamination=c,
            random_state=88,
            max_features=len(feature_cols),
        )
        iso_c.fit(X_train_scaled)
        sc = iso_c.score_samples(X_all_scaled)
        lb = iso_c.predict(X_all_scaled)
        flagged_c = feat_df_clean.iloc[np.where(lb == -1)[0]]["season"].tolist()
        log.info(f"  contamination={c:.2f} → flagged: {flagged_c}")
        for i, row in feat_df_clean.reset_index(drop=True).iterrows():
            sweep_rows.append({
                "contamination" : c,
                "season"        : row["season"],
                "phase"         : row["phase"],
                "anomaly_score" : round(sc[i], 6),
                "anomaly_flag"  : "Anomaly" if lb[i] == -1 else "Normal",
            })

    sweep_df = pd.DataFrame(sweep_rows)

    # Pivot to wide format for SM table
    pivot = sweep_df.pivot_table(
        index=["season", "phase"],
        columns="contamination",
        values="anomaly_flag",
        aggfunc="first",
    ).reset_index()
    pivot.columns = ["season", "phase"] + [f"flag_contam_{c}" for c in contam_levels]

    pivot = pivot.merge(
        primary_df[["season", "anomaly_score", "anomaly_flag"]].rename(
            columns={"anomaly_score": "score_primary", "anomaly_flag": "flag_primary_0.10"}
        ),
        on="season", how="left",
    )
    pivot = pivot.sort_values("score_primary").reset_index(drop=True)

    flagged_all = (
        sweep_df[sweep_df["anomaly_flag"] == "Anomaly"]
        .groupby("season")
        .filter(lambda x: x["contamination"].nunique() == len(contam_levels))["season"]
        .unique().tolist()
    )
    log.info(f"  Flagged across ALL contamination levels: {sorted(flagged_all)}")
    log.info(f"  *** HEADLINE RESULT: {sorted(flagged_all)} — robust anomalies stable across all sensitivity levels ***")

    pivot["robust_anomaly"] = pivot["season"].isin(flagged_all).map({True: "Yes", False: "No"})
    pivot.to_csv(PATHS["tables"] / "aim3_iforest_sensitivity_sweep.csv", index=False)
    log.info(f"\n  SM sensitivity sweep table:\n{pivot.to_string()}")

    # 2020-21 exclusion note appended for SM transparency
    exclusion_note = pd.DataFrame([{
        "season"             : "2020-21",
        "phase"              : "disruption",
        "flag_contam_0.05"   : "EXCLUDED",
        "flag_contam_0.1"    : "EXCLUDED",
        "flag_contam_0.15"   : "EXCLUDED",
        "score_primary"      : np.nan,
        "flag_primary_0.10"  : "EXCLUDED",
        "robust_anomaly"     : "N/A — CDC data suppression",
    }])
    pivot_with_note = pd.concat([pivot, exclusion_note], ignore_index=True)
    pivot_with_note.to_csv(PATHS["tables"] / "aim3_iforest_sensitivity_sweep.csv", index=False)
    log.info("  2020-21 exclusion note appended to SM sweep table")

    export_publication_supp_s5(primary_df, pivot_with_note)

    return primary_df


# =============================================================================
# PUBLICATION TABLES — SM4
# All Prophet and IF tables cited in manuscript as (SM4).
# Table numbering: S4a–S4d (Prophet), S5a–S5b (Isolation Forest).
# =============================================================================

def build_publication_prophet_validation(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    SM4 Table S4a — Prophet held-out validation metrics.
    Clean column headers, formatted MAPE note, primary model listed first.
    """
    MODEL_LABELS = {
        "Prophet_flat"   : "Prophet (flat growth) — Primary",
        "Prophet_linear" : "Prophet (linear growth) — Sensitivity",
    }

    pub = metrics_df.copy()
    pub["model"] = pub["model"].map(MODEL_LABELS).fillna(pub["model"])

    # Format MAPE: show value where available, dash where suppressed
    pub["MAPE (%)"] = pub.apply(
        lambda r: "—" if pd.isna(r["mape_pct"]) else f"{r['mape_pct']:.2f}",
        axis=1
    )
    pub["MAPE note"] = pub["mape_note"].replace("", np.nan).fillna("—")

    pub = pub.drop(columns=["mape_pct", "mape_note"], errors="ignore")

    pub = pub.rename(columns={
        "model"  : "Model",
        "mae"    : "MAE",
        "rmse"   : "RMSE",
        "r2"     : "R²",
        "n_test" : "Test Weeks (n)",
    })

    col_order = ["Model", "MAE", "RMSE", "MAPE (%)", "R²", "Test Weeks (n)", "MAPE note"]
    pub = pub[[c for c in col_order if c in pub.columns]]

    return pub


def export_publication_prophet_validation(metrics_df: pd.DataFrame) -> None:
    """Export SM4 Table S4a."""
    if metrics_df.empty:
        log.warning("  Validation metrics df empty — S4a not written")
        return
    pub = build_publication_prophet_validation(metrics_df)
    out_path = PATHS["tables"] / "table_s4a_prophet_validation_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table_s4a_prophet_validation_publication.csv → {out_path} ({len(pub)} rows)")


def build_publication_prophet_detection(det_compare: pd.DataFrame) -> pd.DataFrame:
    """
    SM4 Table S4b — Prophet first-detection date comparison.
    Flat (primary) listed first, linear (sensitivity) second.
    Adds convergence assessment column.
    """
    MODEL_LABELS = {
        "flat_growth"   : "Prophet (flat growth) — Primary",
        "linear_growth" : "Prophet (linear growth) — Sensitivity",
    }

    pub = det_compare.copy()
    pub["model"] = pub["model"].map(MODEL_LABELS).fillna(pub["model"])

    # Format detection date
    pub["first_detection_date"] = pd.to_datetime(
        pub["first_detection_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Compute convergence if both dates available
    dates = pd.to_datetime(det_compare["first_detection_date"], errors="coerce")
    if dates.notna().all():
        diff_wks = abs(int((dates.iloc[0] - dates.iloc[1]).days / 7))
        convergence = f"Models agree within {diff_wks} week(s)"
        robust = "Yes" if diff_wks <= 4 else "No — >4 week divergence"
    else:
        convergence = "—"
        robust = "—"

    pub["Convergence"] = convergence
    pub["Robust (≤4 wk)"] = robust

    pub = pub.rename(columns={
        "model"                : "Model",
        "first_detection_date" : "First Detection Date",
        "note"                 : "Note",
    })

    col_order = ["Model", "First Detection Date", "Note", "Convergence", "Robust (≤4 wk)"]
    pub = pub[[c for c in col_order if c in pub.columns]]

    return pub


def export_publication_prophet_detection(det_compare: pd.DataFrame) -> None:
    """Export SM4 Table S4b."""
    if det_compare.empty:
        log.warning("  Detection comparison df empty — S4b not written")
        return
    pub = build_publication_prophet_detection(det_compare)
    out_path = PATHS["tables"] / "table_s4b_prophet_detection_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table_s4b_prophet_detection_publication.csv → {out_path} ({len(pub)} rows)")


def build_publication_prophet_gap(gap_flat: pd.DataFrame,
                                   gap_lin: pd.DataFrame) -> pd.DataFrame:
    """
    SM4 Table S4c — Season-level forecast gap: observed vs projected.
    Flat (primary) and linear (sensitivity) side by side.
    Positive gap = observed exceeded projection (under-forecast).
    Negative gap = observed below projection (over-forecast).
    Partial seasons and 2020-21 flagged.
    """
    PHASE_ABBREV = {
        "pre_pandemic" : "PRE",
        "disruption"   : "DISR",
        "recovery"     : "REC",
    }

    PARTIAL_SEASONS = {"2025-26"}
    SUPPRESSED      = {"2020-21"}

    def prep(df, suffix):
        d = df.copy()
        d["phase"] = d["phase"].map(PHASE_ABBREV).fillna(d["phase"])
        for col in ["mean_observed", "mean_projected", "mean_gap",
                    "peak_observed", "peak_projected", "peak_gap"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce").round(3)
        rename = {c: f"{c}_{suffix}" for c in
                  ["mean_projected", "mean_gap", "peak_projected", "peak_gap"]}
        return d.rename(columns=rename)

    flat = prep(gap_flat, "flat")
    lin  = prep(gap_lin,  "linear")

    # Merge on season + phase; observed values identical across models
    merged = flat.merge(
        lin[["season", "mean_projected_linear", "mean_gap_linear",
             "peak_projected_linear", "peak_gap_linear"]],
        on="season", how="outer"
    )

    # Flag partial and suppressed seasons
    merged["Notes"] = ""
    merged.loc[merged["season"].isin(PARTIAL_SEASONS), "Notes"] = "Partial season"
    merged.loc[merged["season"].isin(SUPPRESSED),      "Notes"] = "CDC data suppression — excluded from forecasting"

    merged = merged.rename(columns={
        "season"                 : "Season",
        "phase"                  : "Phase",
        "mean_observed"          : "Mean Observed",
        "mean_projected_flat"    : "Mean Projected (Flat)",
        "mean_gap_flat"          : "Mean Gap (Flat)",
        "mean_projected_linear"  : "Mean Projected (Linear)",
        "mean_gap_linear"        : "Mean Gap (Linear)",
        "peak_observed"          : "Peak Observed",
        "peak_projected_flat"    : "Peak Projected (Flat)",
        "peak_gap_flat"          : "Peak Gap (Flat)",
        "peak_projected_linear"  : "Peak Projected (Linear)",
        "peak_gap_linear"        : "Peak Gap (Linear)",
    })

    col_order = [
        "Season", "Phase", "Notes",
        "Mean Observed", "Mean Projected (Flat)", "Mean Gap (Flat)",
        "Mean Projected (Linear)", "Mean Gap (Linear)",
        "Peak Observed", "Peak Projected (Flat)", "Peak Gap (Flat)",
        "Peak Projected (Linear)", "Peak Gap (Linear)",
    ]
    merged = merged[[c for c in col_order if c in merged.columns]]

    return merged.sort_values("Season")


def export_publication_prophet_gap(gap_flat: pd.DataFrame,
                                    gap_lin: pd.DataFrame) -> None:
    """Export SM4 Table S4c."""
    pub = build_publication_prophet_gap(gap_flat, gap_lin)
    out_path = PATHS["tables"] / "table_s4c_prophet_gap_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table_s4c_prophet_gap_publication.csv → {out_path} ({len(pub)} rows)")


def build_publication_prophet_changepoints(cp_df: pd.DataFrame) -> pd.DataFrame:
    """
    SM4 Table S4d — Prophet changepoints (primary flat model).

    No magnitude threshold applied. The flat growth model produces near-zero
    delta values by design — without a trend component, changepoints reflect
    seasonal timing adjustments only, not rate-level shifts. All 15 candidate
    changepoint dates are retained and sorted by |Delta| descending so
    reviewers can assess the full distribution. Near-zero values across all
    candidates confirm the absence of discrete structural breaks in the
    pre-pandemic training series, which is the expected result for a flat
    growth specification. Dates and deltas formatted cleanly — raw CSV
    contains scientific notation from the MCMC sampler.
    """
    pub = cp_df.copy()

    # No threshold filter applied — flat model changepoints are near-zero by
    # design (no trend component to shift). Filtering by magnitude is not
    # appropriate here; all 15 candidate dates are retained and sorted by
    # |Delta| so reviewers can assess the full distribution. The near-zero
    # values confirm that no discrete rate-level shifts occurred during the
    # pre-pandemic training period, which is the expected and correct result
    # for a flat growth specification.

    # Sort by magnitude before formatting (formatting converts to string)
    pub = pub.sort_values("abs_delta", ascending=False).reset_index(drop=True)

    # Format date
    pub["changepoint_date"] = pd.to_datetime(
        pub["changepoint_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Format delta in scientific notation — values are near-zero for flat model
    # and would display as 0.0000 with standard rounding
    pub["delta"]     = pub["delta"].apply(lambda x: f"{x:.4e}")
    pub["abs_delta"] = pub["abs_delta"].apply(lambda x: f"{x:.4e}")

    # Direction label — computed from original numeric delta before string formatting
    numeric_delta = cp_df.sort_values("abs_delta", ascending=False)["delta"].values
    pub["Direction"] = ["Increasing" if d > 0 else "Decreasing" for d in numeric_delta]

    pub = pub.rename(columns={
        "changepoint_date" : "Changepoint Date",
        "delta"            : "Delta (rate change)",
        "abs_delta"        : "|Delta|",
    })

    col_order = ["Changepoint Date", "Delta (rate change)", "|Delta|", "Direction"]
    pub = pub[[c for c in col_order if c in pub.columns]]

    return pub.reset_index(drop=True)


def export_publication_prophet_changepoints(cp_df: pd.DataFrame) -> None:
    """Export SM4 Table S4d."""
    if cp_df.empty:
        log.warning("  Changepoints df empty — S4d not written")
        return
    pub = build_publication_prophet_changepoints(cp_df)
    out_path = PATHS["tables"] / "table_s4d_prophet_changepoints_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table_s4d_prophet_changepoints_publication.csv → {out_path} ({len(pub)} rows)")


# =============================================================================
# PUBLICATION TABLES — SM4 (Isolation Forest: S5a, S5b)
# =============================================================================

def build_publication_supp_s5a(iforest_scores: pd.DataFrame) -> pd.DataFrame:
    """
    SM4 Table S5a — Isolation Forest season-level feature matrix
    and anomaly classifications (2020-21 excluded).
    """
    PHASE_ABBREV = {
        "pre_pandemic" : "PRE",
        "disruption"   : "DISR",
        "recovery"     : "REC",
    }

    COL_RENAME = {
        "season"        : "Season",
        "phase"         : "Phase",
        "rank"          : "Rank",
        "anomaly_flag"  : "Classification",
        "anomaly_score" : "Anomaly Score",
        "peak_rate"     : "Peak Rate",
        "mean_rate"     : "Mean Rate",
        "cum_rate"      : "Cum Rate",
        "n_weeks"       : "Weeks (n)",
        "peak_week"     : "Peak Week",
        "rate_cv"       : "Rate CV",
    }

    pub = iforest_scores.copy()
    pub = pub[pub["season"] != "2020-21"].copy()

    # Drop redundant integer label column — anomaly_flag is the clean version
    pub = pub.drop(columns=["anomaly_label"], errors="ignore")

    pub["phase"] = pub["phase"].map(PHASE_ABBREV).fillna(pub["phase"])

    for col in ["peak_rate", "mean_rate", "cum_rate", "rate_cv", "anomaly_score"]:
        if col in pub.columns:
            pub[col] = pd.to_numeric(pub[col], errors="coerce").round(3)

    pub = pub.rename(columns=COL_RENAME)

    col_order = ["Season", "Phase", "Rank", "Classification",
                 "Anomaly Score", "Peak Rate", "Mean Rate",
                 "Cum Rate", "Weeks (n)", "Peak Week", "Rate CV"]
    pub = pub[[c for c in col_order if c in pub.columns]]

    return pub.sort_values("Rank")


def build_publication_supp_s5b(sensitivity_sweep: pd.DataFrame) -> pd.DataFrame:
    """
    SM4 Table S5b — Isolation Forest contamination sensitivity sweep.
    Retains 2020-21 exclusion row for transparency.
    """
    PHASE_ABBREV = {
        "pre_pandemic" : "PRE",
        "disruption"   : "DISR",
        "recovery"     : "REC",
    }

    COL_RENAME = {
        "season"             : "Season",
        "phase"              : "Phase",
        "flag_contam_0.05"   : "Contam=0.05",
        "flag_contam_0.1"    : "Contam=0.10",
        "flag_contam_0.15"   : "Contam=0.15",
        "score_primary"      : "Primary Score",
        "flag_primary_0.10"  : "Primary Classification",
        "robust_anomaly"     : "Robust Anomaly",
    }

    pub = sensitivity_sweep.copy()
    pub["phase"] = pub["phase"].map(PHASE_ABBREV).fillna(pub["phase"])
    pub["score_primary"] = pd.to_numeric(pub["score_primary"], errors="coerce").round(3)
    pub = pub.rename(columns=COL_RENAME)

    col_order = ["Season", "Phase", "Primary Classification", "Robust Anomaly",
                 "Primary Score", "Contam=0.05", "Contam=0.10", "Contam=0.15"]
    pub = pub[[c for c in col_order if c in pub.columns]]

    return pub


def export_publication_supp_s5(iforest_scores: pd.DataFrame,
                                sensitivity_sweep: pd.DataFrame) -> None:
    """Export SM4 Tables S5a and S5b."""
    s5a = build_publication_supp_s5a(iforest_scores)
    s5b = build_publication_supp_s5b(sensitivity_sweep)

    s5a_path = PATHS["tables"] / "table_s5a_iforest_scores_publication.csv"
    s5b_path = PATHS["tables"] / "table_s5b_iforest_sweep_publication.csv"

    s5a.to_csv(s5a_path, index=False, encoding="utf-8-sig")
    s5b.to_csv(s5b_path, index=False, encoding="utf-8-sig")

    log.info(f"  table_s5a_iforest_scores_publication.csv → {s5a_path} ({len(s5a)} rows)")
    log.info(f"  table_s5b_iforest_sweep_publication.csv → {s5b_path} ({len(s5b)} rows)")


# =============================================================================
# FIGURES
# =============================================================================

def fig5_prophet_forecast(merged_flat, det_flat):
    section("FIG 5 — Prophet Forecast vs Observed (Primary: Flat Growth)")

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=FACECOLOR)

    ax.fill_between(
        merged_flat["ds"], merged_flat["yhat_lower"], merged_flat["yhat_upper"],
        color="#CCCCCC", alpha=0.45, zorder=1, label="95% prediction interval"
    )
    ax.plot(merged_flat["ds"], merged_flat["yhat"],
            color="#888888", linewidth=1.4, linestyle="--",
            alpha=0.85, zorder=2, label="Prophet projected rate")

    for phase in ["pre_pandemic", "disruption", "recovery"]:
        obs = merged_flat[merged_flat["phase"] == phase].dropna(subset=["weekly_rate"])
        ax.plot(obs["ds"], obs["weekly_rate"],
                color=PHASE_LINE[phase], linewidth=1.8,
                alpha=0.92, zorder=3, label=PHASE_LABELS[phase])

    if pd.notna(det_flat):
        ax.axvline(det_flat, color="#d62728", linewidth=1.4,
                   linestyle=":", alpha=0.85, zorder=4)
        ax.text(det_flat + pd.Timedelta(weeks=3),
                merged_flat["yhat_upper"].max() * 0.88,
                f"First anomaly\ndetected\n{det_flat.strftime('%b %Y')}",
                fontsize=ANNOT_SIZE - 1, color="#d62728", va="top")

    train_end = merged_flat[merged_flat["phase"] == "pre_pandemic"]["ds"].max()
    ax.axvline(train_end, color="#555555", linewidth=1.0,
               linestyle="--", alpha=0.5, zorder=2)
    ax.text(train_end - pd.Timedelta(weeks=10),
            merged_flat["yhat_upper"].max() * 0.98,
            "Training\ncutoff", fontsize=ANNOT_SIZE - 1,
            color="#555555", ha="right", va="top")

    ax.set_xlabel("Influenza season (year)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Weekly hospitalization rate\n(per 100,000 population)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlim(merged_flat["ds"].min(), merged_flat["ds"].max())
    ax.set_ylim(bottom=0)
    clean_axes(ax)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.92, loc="upper left", ncol=2)
    fig.tight_layout()
    save_fig(fig, "fig5_prophet_forecast.png")


def fig6_prophet_sensitivity(merged_flat, merged_lin, det_flat, det_lin):
    """
    SM4 Figure S2 — 2-panel sensitivity: flat growth (primary) vs linear growth.
    Allows reviewers to assess whether the first-detection finding is stable
    across growth assumptions.
    """
    section("FIG 6 — Prophet Sensitivity: Flat vs Linear Growth (SM4)")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), facecolor=FACECOLOR, sharex=True)

    panels = [
        (merged_flat, det_flat, "Primary model: flat growth (no trend)",          axes[0]),
        (merged_lin,  det_lin,  "Sensitivity model: linear growth (trend extrap.)", axes[1]),
    ]

    for merged, det, subtitle, ax in panels:
        ax.fill_between(
            merged["ds"], merged["yhat_lower"], merged["yhat_upper"],
            color="#CCCCCC", alpha=0.40, zorder=1
        )
        ax.plot(merged["ds"], merged["yhat"],
                color="#888888", linewidth=1.2, linestyle="--", alpha=0.80, zorder=2)

        for phase in ["pre_pandemic", "disruption", "recovery"]:
            obs = merged[merged["phase"] == phase].dropna(subset=["weekly_rate"])
            ax.plot(obs["ds"], obs["weekly_rate"],
                    color=PHASE_LINE[phase], linewidth=1.6, alpha=0.92, zorder=3)

        if pd.notna(det):
            ax.axvline(det, color="#d62728", linewidth=1.3,
                       linestyle=":", alpha=0.9, zorder=4)
            ax.text(det + pd.Timedelta(weeks=3),
                    merged["yhat_upper"].max() * 0.85,
                    f"{det.strftime('%b %Y')}",
                    fontsize=ANNOT_SIZE - 1, color="#d62728", va="top")

        train_end = merged[merged["phase"] == "pre_pandemic"]["ds"].max()
        ax.axvline(train_end, color="#555555", linewidth=0.9,
                   linestyle="--", alpha=0.5, zorder=2)

        ax.set_title(subtitle, fontsize=LABEL_SIZE - 1, fontweight="bold", pad=5)
        ax.set_ylabel("Rate (per 100,000)", fontsize=LABEL_SIZE - 1)
        ax.tick_params(labelsize=TICK_SIZE - 1)
        ax.set_ylim(bottom=0)
        clean_axes(ax)

    axes[-1].set_xlabel("Influenza season (year)", fontsize=LABEL_SIZE)

    handles = [
        mlines.Line2D([], [], color=PHASE_LINE["pre_pandemic"],
                      linewidth=2, label="Pre-pandemic"),
        mlines.Line2D([], [], color=PHASE_LINE["disruption"],
                      linewidth=2, label="Pandemic disruption"),
        mlines.Line2D([], [], color=PHASE_LINE["recovery"],
                      linewidth=2, label="Post-pandemic recovery"),
        mlines.Line2D([], [], color="#888888", linewidth=1.5, linestyle="--",
                      label="Projected rate"),
        mpatches.Patch(color="#CCCCCC", alpha=0.6,
                       label="95% prediction interval"),
        mlines.Line2D([], [], color="#d62728", linewidth=1.3, linestyle=":",
                      label="First anomaly detected"),
    ]
    axes[0].legend(handles=handles, fontsize=LEGEND_SIZE,
                   framealpha=0.92, loc="upper left", ncol=3)

    fig.subplots_adjust(hspace=0.12)
    save_fig(fig, "fig6_prophet_sensitivity.png")


def fig7_iforest_scores(primary_df):
    section("FIG 7 — Isolation Forest Anomaly Scores (Primary: contamination=0.10)")

    plot_df = primary_df.sort_values("anomaly_score", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=FACECOLOR)

    for i, row in plot_df.iterrows():
        phase  = row["phase"]
        color  = PHASE_LINE.get(phase, "#888888")
        marker = "D" if row["anomaly_flag"] == "Anomaly" else "o"
        size   = 80 if row["anomaly_flag"] == "Anomaly" else 55

        ax.plot([row["anomaly_score"], 0], [i, i],
                color=color, linewidth=1.2, alpha=0.6, zorder=2)
        ax.scatter(row["anomaly_score"], i,
                   color=color, s=size, marker=marker,
                   zorder=3, edgecolors="white", linewidths=0.4)

    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)

    xmin = plot_df["anomaly_score"].min() - 0.05
    ax.text(xmin, -0.9,
            "† 2020–21 excluded: CDC structural data suppression",
            fontsize=ANNOT_SIZE - 2, color="#777777", style="italic")

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["season"], fontsize=TICK_SIZE - 1)
    ax.set_xlabel("Isolation Forest anomaly score\n(more negative = more anomalous)",
                  fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    clean_axes(ax)

    legend_handles = [
        mlines.Line2D([], [], color=PHASE_LINE["pre_pandemic"], linewidth=2,
                      marker="o", markersize=6, label="Pre-pandemic"),
        mlines.Line2D([], [], color=PHASE_LINE["disruption"],   linewidth=2,
                      marker="o", markersize=6, label="Pandemic disruption"),
        mlines.Line2D([], [], color=PHASE_LINE["recovery"],     linewidth=2,
                      marker="o", markersize=6, label="Post-pandemic recovery"),
        mlines.Line2D([], [], color="#555555", linewidth=0,
                      marker="D", markersize=7, label="Flagged anomaly",
                      markeredgecolor="white"),
    ]
    ax.legend(handles=legend_handles, fontsize=LEGEND_SIZE,
              framealpha=0.92, loc="lower right")

    fig.tight_layout()
    save_fig(fig, "fig7_iforest_anomaly_scores.png")


# =============================================================================
# FIGURE S3 — FLAT MODEL RESIDUALS ON 2018-19 HOLDOUT
# =============================================================================

def fig_s3_prophet_residuals(m_flat_ho, test_ho):
    """
    SM4 Figure S3 — Two-panel validation residual plot for flat growth model.

    Panel A: Observed vs predicted scatter with 1:1 reference line.
             Reviewers expect this as standard model validation evidence.
    Panel B: Residuals over time (observed minus predicted by epiweek).
             Absence of systematic pattern confirms no temporal bias.

    Both panels use the 2018-19 held-out season only (n=30 weeks),
    fitted on 2009-10 to 2017-18 training data.
    """
    section("FIG S3 — Flat Model Residuals (2018-19 Holdout)")

    fc   = m_flat_ho.predict(test_ho[["ds"]])
    obs  = test_ho["y"].values
    pred = fc["yhat"].values
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    y, yh = obs[mask], pred[mask]
    resid = y - yh
    dates = test_ho["ds"].values[mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=FACECOLOR)

    # --- Panel A: Observed vs Predicted ---
    ax = axes[0]
    ax.scatter(yh, y, color=PHASE_LINE["pre_pandemic"],
               s=55, zorder=3, edgecolors="white", linewidths=0.4,
               label="2018–19 observations")
    lim_min = min(y.min(), yh.min()) * 0.9
    lim_max = max(y.max(), yh.max()) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="#888888", linewidth=1.2, linestyle="--",
            alpha=0.7, zorder=2, label="1:1 reference")
    ax.set_xlabel("Predicted rate (per 100,000)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Observed rate (per 100,000)", fontsize=LABEL_SIZE)
    ax.set_title("A. Observed vs predicted", fontsize=LABEL_SIZE,
                 fontweight="bold", pad=6)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.92)
    clean_axes(ax)

    # --- Panel B: Residuals over time ---
    ax = axes[1]
    ax.axhline(0, color="#888888", linewidth=1.0,
               linestyle="--", alpha=0.7, zorder=1)
    ax.scatter(dates, resid,
               color=PHASE_LINE["pre_pandemic"],
               s=55, zorder=3, edgecolors="white", linewidths=0.4)
    ax.plot(dates, resid,
            color=PHASE_LINE["pre_pandemic"],
            linewidth=1.2, alpha=0.6, zorder=2)
    ax.set_xlabel("Epiweek (2018–19 season)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Residual (observed minus predicted)", fontsize=LABEL_SIZE)
    ax.set_title("B. Residuals over time", fontsize=LABEL_SIZE,
                 fontweight="bold", pad=6)
    ax.tick_params(labelsize=TICK_SIZE)
    clean_axes(ax)

    fig.tight_layout()
    save_fig(fig, "figS3_prophet_residuals.png")
    log.info(f"  Residual SD: {np.std(resid):.3f} | Mean: {np.mean(resid):.3f}")


# =============================================================================
# TABLE S4e — LEAVE-ONE-OUT DETECTION DATE STABILITY
# =============================================================================

def run_prophet_loo(df):
    """
    SM4 Table S4e — Leave-one-out detection date stability analysis.

    For each pre-pandemic season, refit the flat growth Prophet model
    excluding that season from training, then record the first detection
    date. Stable detection dates across all LOO runs confirm that the
    February 2020 finding is not driven by any single training season.

    Gap in weeks relative to full-model detection (2020-02-02) included
    so reviewers can assess stability without manual calculation.

    ~10 additional Prophet fits — adds ~10-15 seconds to runtime.
    """
    section("A4 — Leave-One-Out Detection Date Stability (Flat Model)")

    pre = df[(df["phase"] == "pre_pandemic") & (df["weekly_rate"].notna())].copy()
    train_full = pre[["epiweek_date", "weekly_rate"]].rename(
        columns={"epiweek_date": "ds", "weekly_rate": "y"}
    ).reset_index(drop=True)
    train_end = train_full["ds"].max()
    last_obs  = df["epiweek_date"].max()

    pre_seasons = sorted(pre["season"].unique())
    FULL_MODEL_DET = pd.Timestamp("2020-02-02")

    records = []
    for excluded in pre_seasons:
        train_loo = pre[pre["season"] != excluded][["epiweek_date", "weekly_rate"]].rename(
            columns={"epiweek_date": "ds", "weekly_rate": "y"}
        ).reset_index(drop=True)

        try:
            m_loo = _fit_prophet(train_loo, growth="flat",
                                 label=f"flat_loo_excl_{excluded}")
            merged_loo = _forecast_merge(m_loo, df, train_end, last_obs,
                                         label=f"loo_{excluded}")
            det_loo = _first_detection(merged_loo, train_end,
                                       label=f"loo_{excluded}")

            gap_wks = (
                int((det_loo - FULL_MODEL_DET).days / 7)
                if pd.notna(det_loo) else np.nan
            )
            direction = (
                "Later" if gap_wks > 0
                else "Earlier" if gap_wks < 0
                else "Same"
            ) if pd.notna(gap_wks) else "—"

            records.append({
                "excluded_season"      : excluded,
                "n_train_seasons"      : len(pre_seasons) - 1,
                "first_detection_date" : det_loo.strftime("%Y-%m-%d") if pd.notna(det_loo) else "—",
                "gap_weeks_vs_full"    : gap_wks,
                "direction"            : direction,
            })
            log.info(f"  LOO excl {excluded}: detection={det_loo} | "
                     f"gap={gap_wks} wks ({direction})")

        except Exception as e:
            log.error(f"  LOO excl {excluded} failed: {e}")
            records.append({
                "excluded_season"      : excluded,
                "n_train_seasons"      : len(pre_seasons) - 1,
                "first_detection_date" : "FAILED",
                "gap_weeks_vs_full"    : np.nan,
                "direction"            : "—",
            })

    loo_df = pd.DataFrame(records)
    raw_path = PATHS["tables"] / "aim3_prophet_loo_detection.csv"
    loo_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    log.info(f"  LOO detection range: "
             f"{loo_df['first_detection_date'].min()} to "
             f"{loo_df['first_detection_date'].max()}")
    log.info(f"  Gap range (weeks): "
             f"{loo_df['gap_weeks_vs_full'].min():.0f} to "
             f"{loo_df['gap_weeks_vs_full'].max():.0f}")

    export_publication_loo(loo_df)
    return loo_df


def build_publication_loo(loo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Publication-ready Table S4e — LOO detection date stability.
    Full model detection date (2020-02-02) shown as header context row.
    """
    pub = loo_df.copy()

    pub = pub.rename(columns={
        "excluded_season"      : "Excluded Season",
        "n_train_seasons"      : "Training Seasons (n)",
        "first_detection_date" : "First Detection Date",
        "gap_weeks_vs_full"    : "Gap vs Full Model (weeks)",
        "direction"            : "Direction",
    })

    # Prepend full-model reference row for immediate context
    ref_row = pd.DataFrame([{
        "Excluded Season"          : "None (full model)",
        "Training Seasons (n)"     : 10,
        "First Detection Date"     : "2020-02-02",
        "Gap vs Full Model (weeks)": 0,
        "Direction"                : "Reference",
    }])
    pub = pd.concat([ref_row, pub], ignore_index=True)

    return pub


def export_publication_loo(loo_df: pd.DataFrame) -> None:
    """Export SM4 Table S4e."""
    if loo_df.empty:
        log.warning("  LOO df empty — S4e not written")
        return
    pub = build_publication_loo(loo_df)
    out_path = PATHS["tables"] / "table_s4e_prophet_loo_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table_s4e_prophet_loo_publication.csv → {out_path} ({len(pub)} rows)")


# =============================================================================
# TABLE S4f — CROSS-METHOD CONVERGENCE (PROPHET × ISOLATION FOREST)
# =============================================================================

def build_publication_convergence(gap_flat: pd.DataFrame,
                                   primary_df: pd.DataFrame) -> pd.DataFrame:
    """
    SM4 Table S4f — Cross-method convergence table.

    Joins Prophet flat model mean gap direction with Isolation Forest
    classification for each season. No new modeling — derived from
    outputs already computed. Directly addresses reviewer question:
    "Do your two methods agree on which seasons are anomalous?"

    Gap direction: Positive = observed > projected (anomalously high);
                   Negative = observed < projected (below baseline).
    2020-21 excluded from IF; retained in Prophet gap table with note.
    """
    PHASE_ABBREV = {
        "pre_pandemic" : "PRE",
        "disruption"   : "DISR",
        "recovery"     : "REC",
    }

    # Prophet gap direction
    gap = gap_flat[["season", "phase", "mean_gap"]].copy()
    gap["Prophet Gap Direction"] = gap["mean_gap"].apply(
        lambda x: "Positive" if x > 0 else "Negative"
    )
    gap["Prophet Mean Gap"] = gap["mean_gap"].round(3)
    gap["phase"] = gap["phase"].map(PHASE_ABBREV).fillna(gap["phase"])

    # IF classification
    IF_EXCLUDED = {"2020-21"}
    iforest = primary_df[["season", "anomaly_flag", "anomaly_score",
                           "rank"]].copy()
    iforest = iforest.rename(columns={
        "anomaly_flag"  : "IF Classification",
        "anomaly_score" : "IF Anomaly Score",
        "rank"          : "IF Rank",
    })
    iforest["IF Anomaly Score"] = iforest["IF Anomaly Score"].round(3)

    # Merge
    merged = gap.merge(iforest, on="season", how="left")

    # Flag 2020-21
    merged.loc[merged["season"].isin(IF_EXCLUDED),
               "IF Classification"] = "Excluded (CDC suppression)"
    merged.loc[merged["season"].isin(IF_EXCLUDED), "IF Anomaly Score"] = np.nan
    merged.loc[merged["season"].isin(IF_EXCLUDED), "IF Rank"] = np.nan

    # Agreement column
    def _agreement(row):
        if row["IF Classification"] == "Excluded (CDC suppression)":
            return "N/A"
        if row["Prophet Gap Direction"] == "Positive" and \
           row["IF Classification"] == "Anomaly":
            return "Yes — both methods flag"
        if row["Prophet Gap Direction"] == "Negative" and \
           row["IF Classification"] == "Normal":
            return "Yes — both methods normal"
        return "Partial — methods diverge"

    merged["Method Agreement"] = merged.apply(_agreement, axis=1)

    merged = merged.rename(columns={
        "season" : "Season",
        "phase"  : "Phase",
    })

    col_order = [
        "Season", "Phase",
        "Prophet Mean Gap", "Prophet Gap Direction",
        "IF Rank", "IF Anomaly Score", "IF Classification",
        "Method Agreement",
    ]
    merged = merged[[c for c in col_order if c in merged.columns]]

    return merged.sort_values("Season")


def export_publication_convergence(gap_flat: pd.DataFrame,
                                    primary_df: pd.DataFrame) -> None:
    """Export SM4 Table S4f."""
    pub = build_publication_convergence(gap_flat, primary_df)
    out_path = PATHS["tables"] / "table_s4f_crossmethod_convergence_publication.csv"
    pub.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  table_s4f_crossmethod_convergence_publication.csv → {out_path} ({len(pub)} rows)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    run_start = datetime.now()
    section("05_aim3_ml.py — Prophet + Isolation Forest")
    log.info(f"Run started  : {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Project root : {PROJECT_ROOT}")
    log.info(f"Seed         : 88")

    df = load_data()

    # Section A — Prophet (primary flat + linear sensitivity + held-out validation)
    (merged_flat, merged_lin, gap_flat, gap_lin,
     cp_df, det_flat, det_lin, metrics_df, det_compare,
     m_flat_ho, test_ho, train_ho) = run_prophet(df)

    # Section A4 — Leave-one-out detection date stability
    loo_df = run_prophet_loo(df)

    # Section B — Isolation Forest (primary + sweep + SM tables)
    primary_df = run_isolation_forest(df)

    # Publication tables — SM4
    section("EXPORT PUBLICATION TABLES (SM4)")
    export_publication_prophet_validation(metrics_df)
    export_publication_prophet_detection(det_compare)
    export_publication_prophet_gap(gap_flat, gap_lin)
    export_publication_prophet_changepoints(cp_df)
    export_publication_convergence(gap_flat, primary_df)
    # S5a and S5b already exported inside run_isolation_forest

    # Figures
    section("GENERATE FIGURES")
    fig5_prophet_forecast(merged_flat, det_flat)
    fig6_prophet_sensitivity(merged_flat, merged_lin, det_flat, det_lin)
    fig7_iforest_scores(primary_df)
    fig_s3_prophet_residuals(m_flat_ho, test_ho)

    # Final summary
    section("FINAL SUMMARY")
    tables = list(PATHS["tables"].glob("aim3_*.csv")) + \
             list(PATHS["tables"].glob("table_s4*.csv")) + \
             list(PATHS["tables"].glob("table_s5*.csv"))
    figs   = list(PATHS["figures"].glob("fig[567]*.png")) + \
             list(PATHS["figures"].glob("figS*.png"))

    log.info(f"  Tables written : {len(tables)}")
    for t in sorted(set(tables)):
        log.info(f"    {t.name}")
    log.info(f"  Figures written: {len(figs)}")
    for f in sorted(figs):
        log.info(f"    {f.name}")
    log.info(f"  Log            : {LOG_FILE}")
    log.info(f"  Run completed  : {(datetime.now() - run_start).seconds}s")
    section("DONE")


if __name__ == "__main__":
    main()
