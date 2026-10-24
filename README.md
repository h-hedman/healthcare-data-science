# healthcare-data-science

This repository contains a curated set of applied healthcare data science projects spanning clinical analytics, public health modeling, and various AI/ML techniques.
Each subfolder is a **standalone project** with its own README and documentation.
Collectively, these projects demonstrate experience working with real-world healthcare data constraints, decision-support pipelines, and analytical methods used in industry and research settings.

---

## Project Index

### Core Projects
- **[Healthify Flu App](healthify-flu-app/)**
  Influenza analysis site and five Shiny applications hosted on Posit Connect Cloud. [Open the live site](https://h-hedman.github.io/healthcare-data-science/healthify-flu-app/).

- **[Summit County Public Health](scph/)**
  Public health analytics, surveillance documentation, and links to community-facing products.

- **`organic-amr-rl/`**
  Reinforcement learning–based simulation framework for exploring antimicrobial resistance dynamics under organic compound exposure, generating experimental results, figures, and tables for a microbiology methods study.

- **`fluserv-net-surveillance/`**
  Longitudinal analysis of 16 seasons of FluSurv-NET influenza hospitalization data characterizing post-pandemic recovery dynamics through descriptive epidemiology, phase-level regression, STL decomposition, Prophet time-series forecasting, and Isolation Forest anomaly detection. Developed in support of a peer-reviewed manuscript submission.

- **`glp1-nlp-demo/`**
  End-to-end NLP pipeline for abstracting structured labels from unstructured clinical notes, demonstrated using synthetic GLP-1 medication data.

- **`rf_asthma_lab/`**
  Machine learning analysis of asthma risk and outcomes developed for an *MDPI Laboratories* submission, including feature engineering, model evaluation, and interpretability.

- **`amr/`**
  Modeling antimicrobial resistance dynamics using simulated datasets, with emphasis on temporal trends, intervention pressure, and system-level behavior.

- **`epidemiological_quarantine_algorithm/`**
  Quarantine triage and intake modeling system combining rule-based logic and machine learning, developed during emergency public health response work.

- **`automated-epidemiological-report/`**
  Automated generation of epidemiological surveillance reports and summary metrics for rapid situational awareness.

- **`smoking-cessation-model-main/`**
  Predictive modeling framework to identify individuals most likely to benefit from smoking cessation interventions using synthetic cohort data.

- **`diabetes-intervention-main/`**
  Scenario-based simulation of a hypothetical diabetes intervention, including population impact and high-level cost modeling.

---

### Exploratory Projects

- **`oral-cancer-spss-analysis/`**
  Legacy statistical analysis of oral cancer trial data using SPSS, retained for completeness.

- **`sql-cms-medicare/`**
  Exploratory SQL analyses of CMS Medicare datasets focused on utilization and chronic condition prevalence.

---

## Notes

- All projects use **de-identified or synthetic data**
- Individual READMEs provide methodological and technical detail

---

## Repository organization and development

Healthify and SCPH are maintained here as ordinary project folders, not nested Git repositories. Their original Git histories were imported on October 24, 2026. The former standalone repositories have been retired; make new changes here. Their imported history remains available in this repository.

Clone `https://github.com/h-hedman/healthcare-data-science.git`, then work within the relevant project folder. Each project documents its own dependencies. For Healthify, run R commands from `healthify-flu-app/`; consult its [deployment guide](healthify-flu-app/deployment/README.md).

GitHub Pages serves the static Healthify site from this repository's `main` branch root. Posit Connect Cloud continues to host the five Shiny apps separately. Existing project paths elsewhere in this repository are unchanged.
