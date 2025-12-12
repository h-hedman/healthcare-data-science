# Asthma Risk Prediction: Occupational Exposure to Chemicals (In Review)

**Preprint available on medRxiv**: https://www.medrxiv.org/content/10.1101/2025.11.25.25341001v1


This repository contains a set of scripts used to model asthma-like symptoms caused by chemical exposure in laboratory cleaning environments. Using synthetic data based on demographic, workplace, and exposure characteristics, machine learning models predict asthma risk for workers exposed to hazardous cleaning agents.

## Study Overview

Exposure to chemical irritants in laboratory and medical environments poses significant health risks, particularly concerning asthma-like symptoms. Routine cleaning practices, which often involve the use of strong chemical agents to maintain hygienic settings, have been shown to contribute to respiratory issues. Laboratories, where chemicals such as hydrochloric acid and ammonia are frequently used, represent an under-explored context in the study of occupational asthma.

This study uses a **simulated cohort** based on key demographic and exposure patterns from the foundational research by **Vizcaya et al. (2011)**. The model assesses the impact of chemical exposure from cleaning products in laboratory environments. Machine learning models, including Decision Trees, Random Forest, Gradient Boosting, and XGBoost, are applied to predict asthma-like symptoms based on the exposure data.

### Key Findings:
- High exposures to hydrochloric acid and ammonia were significantly associated with asthma-like symptoms.
- Workplace type also played a critical role in determining asthma risk.
- The research provides a data-driven framework for assessing and predicting asthma-like symptoms in workers exposed to cleaning agents.

**Estimates and model assumptions were adapted from the study**:
> Vizcaya, D., Mirabelli, M. C., Antó, J. M., Orriols, R., Burgos, F., Arjona, L., & Zock, J. P. (2011). A workforce-based study of occupational exposures and asthma symptoms in cleaning workers. *Occupational and Environmental Medicine*, 68(12), 914-919.  
[DOI: 10.1136/oem.2010.063271](https://doi.org/10.1136/oem.2010.063271)  
[PubMed Link](https://pubmed.ncbi.nlm.nih.gov/21558474/)

## Script Overview

This repository contains the following scripts:

### 1. `01_asthma_data_simulation.py`
This script generates a synthetic dataset simulating occupational cleaning exposure and asthma-like symptoms based on demographic and workplace data similar to **Vizcaya et al. (2011)**. The script includes:

- **Demographic Variables**: Simulates sex, age, smoking status, and foreign-born status.
- **Workplace Type**: Includes common areas, hospitals, and industry-specific environments.
- **Chemical Exposure**: Encodes exposure to chemicals like hydrochloric acid, ammonia, and degreasers.
- **Outcome Variable**: Simulates asthma-like symptoms using a logistic regression model based on exposures and other factors.

The synthetic dataset is saved as `synthetic_cleaner_cohort.csv` for further analysis.

### 2. `02_classification_analysis.py`

This script provides a comprehensive classification pipeline for analyzing the synthetic dataset. Key features of the script include:
- **Dataset Loading & Preprocessing**: Loads the synthetic dataset and applies preprocessing such as one-hot encoding for categorical features and standard scaling for numerical features.
- **Feature Engineering**: Creates interaction features such as "total chemical exposure" and "smoking-workplace interaction."
- **Cross-validation**: Performs k-fold cross-validation with StratifiedKFold for balanced class distribution.
- **Model Training**: Implements classification models (Decision Trees, Random Forest, Gradient Boosting, XGBoost) and evaluates their performance using various metrics (e.g., F1-score, ROC-AUC, Precision-Recall AUC).
- **Calibration and Threshold Analysis**: Includes probability calibration and threshold tuning to optimize the recall and precision of the models.
- **Metrics Reporting**: Generates comprehensive evaluation metrics including clinical metrics (e.g., sensitivity, specificity) and comprehensive classification metrics (e.g., Matthews correlation coefficient, Cohen's Kappa).


