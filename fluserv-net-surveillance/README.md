# fluserv-net-surveillance

Longitudinal analysis of influenza hospitalization burden across 16 seasons (2009-2025) using FluSurv-NET population-based surveillance data. Characterizes post-pandemic recovery dynamics through descriptive epidemiology, phase-level regression, seasonal decomposition, time-series forecasting, and unsupervised anomaly detection.

**Preprint:** XXX

---

## Data Source

CDC FluSurv-NET Influenza Hospitalization Surveillance Network
https://www.cdc.gov/fluview/overview/influenza-hospitalization-surveillance.html

Weekly age-adjusted hospitalization rates per 100,000 population, stratified by age group, sex, race/ethnicity, and virus type. Covers three epidemiological phases: pre-pandemic baseline (2009-10 to 2018-19), pandemic disruption (2019-20 to 2021-22), and post-pandemic recovery (2022-23 to 2024-25).

---

## Repository Structure

All scripts are located in `/src` and should be run in order.

```
src/
├── 01_preEDA.py
├── 02_cleaning.py
├── 03_aim1_stats.py
├── 04_figures.py
└── 05_aim3_ml.py
```

---

## Scripts

**`01_preEDA.py`**
Preliminary exploratory data analysis. Examines raw FluSurv-NET rate distributions, missingness patterns, and phase-level descriptive summaries. Computes variance-to-mean ratios to assess distributional properties and inform central tendency measure selection.

**`02_cleaning.py`**
Data cleaning and feature engineering. Applies phase classification, handles the 2020-21 CDC data suppression period, standardizes MMWR week indexing, and prepares stratified analytic datasets for downstream analysis.

**`03_aim1_stats.py`**
Phase-level statistical analysis. Computes phase-stratified descriptive statistics, age-adjusted rate ratios by race/ethnicity with bootstrapped confidence intervals, Cohen's d effect sizes for phase comparisons, and season-level OLS regression models with primary and sensitivity specifications.

**`04_figures.py`**
Generates all manuscript figures and supplementary visualizations including weekly rate time series, age-stratified peak rate comparisons, race/ethnicity rate ratio plots, STL decomposition panels, Prophet forecast plots, and Isolation Forest anomaly score figures.

**`05_aim3_ml.py`**
Machine learning analyses. Implements STL seasonal decomposition on pre-pandemic and post-pandemic recovery series, Prophet time-series forecasting with flat and linear growth specifications and leave-one-out validation, and Isolation Forest anomaly detection with contamination parameter sensitivity sweep across season-level feature matrices.

---

## Requirements

```
pandas
numpy
scipy
statsmodels
prophet
scikit-learn
matplotlib
seaborn
```

---

## Reproducibility

Random seed: 904057817. All scripts use dynamic path construction and are compatible with standard Python environments. See SM1 in the manuscript supplementary materials for full reproducibility notes.
