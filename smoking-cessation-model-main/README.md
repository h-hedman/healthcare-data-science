# Smoking Cessation Cohort (Simulated)

This repository contains a simulated analysis of a smoking cessation intervention using logistic regression. The dataset is based on effect sizes and covariate distributions from [Leventhal et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/34850047/), *JNCI*, 114(3), 381–390. The primary goal is to demonstrate a simple but interpretable modeling pipeline for evaluating behavioral health outcomes.

## Project Summary

A synthetic cohort of N = 1,968 adult smokers was generated to evaluate the impact of an intervention program on quit success. Covariates include age, education, and income level. The outcome is a binary indicator of whether the individual successfully quit smoking.

## Key Findings

- Logistic regression results indicate the intervention more than doubles the odds of quitting (OR ≈ 2.1, p < 0.001).
- Education and income levels are positively associated with quit success.
- Age has a small but significant negative association.
- Model fit: Pseudo R² = 0.047, Likelihood Ratio Test p < 1e-23.

## Methodology

### Data Simulation

- Covariate distributions and relationships were derived from published summary statistics in Leventhal et al. (2022).
- The outcome variable was generated using a log-odds function with additive contributions from:
  - Age (continuous)
  - Education level (categorical)
  - Income level (categorical)
  - Intervention participation (binary)

### Model Specification

- Logistic regression was implemented using `statsmodels.Logit`.
- All variables were standardized or dummy encoded where appropriate.
- Coefficient estimates and confidence intervals were derived via maximum likelihood.

## Outputs

- `results/` folder contains summary tables and regression outputs.
- `figures/` folder includes optional visualizations of predicted probabilities and model diagnostics.

## Usage

To run the analysis:

```bash
python simulate_and_model.py
