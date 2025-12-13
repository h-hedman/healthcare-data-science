# =========================================================================================================
# Author: Hayden Hedman
# Anchor source paper: Vizcaya et al. 2011, https://doi.org/10.1136/oem.2010.063271
# ENHANCED Classification pipeline with COMPREHENSIVE metrics for MDPI Laboratories
# Includes: dataset characterization, advanced classification metrics, calibration analysis,
# clinical decision metrics, cross-validation, statistical tests, and detailed reporting
# =========================================================================================================
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shap
import os
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score,
                                    cross_validate)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve, classification_report,
                             balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
                             log_loss, brier_score_loss, fbeta_score)
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Statistics
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, ks_2samp

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# SMOTE for resampling
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =========================================================================================================
# Global variables
DEFAULT_SEED = 64
DPI = 300
MAKE_IMPORTANCE = True
CV_FOLDS = 5  # For cross-validation

# =========================================================================================================
# Argument parser
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None,
                   help="Root directory for the project (defaults to script's parent directory)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dataset", type=str, default="synthetic_cleaner_cohort.csv")
    p.add_argument("--smote_ratio", type=float, default=0.7,
                   help="Target ratio for minority class after SMOTE (0.3-0.8 recommended)")
    p.add_argument("--min_recall", type=float, default=0.70,
                   help="Minimum recall constraint for threshold tuning")
    return p.parse_args()

# =========================================================================================================
# Directory management
def ensure_dirs(root: Path):
    data_dir = root / "data"
    fig_dir = root / "figures"
    tab_dir = root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, fig_dir, tab_dir

# =========================================================================================================
# Load data
def load_data(data_path: Path):
    df = pd.read_csv(data_path)
    if "age" in df.columns:
        df["age"] = df["age"].round().astype(int)
    return df

# =========================================================================================================
# COMPREHENSIVE DATASET CHARACTERIZATION
def generate_dataset_report(df: pd.DataFrame, target: str, tab_dir: Path):
    """
    Generate comprehensive dataset characterization including:
    - Overall dataset statistics
    - Feature distributions by outcome
    - Statistical tests for associations
    - Missing data analysis
    - Correlation analysis
    """
    print(f"\n{'='*80}")
    print("DATASET CHARACTERIZATION")
    print(f"{'='*80}\n")
    
    # ===== OVERALL DATASET STATISTICS =====
    overall_stats = {
        'Total Observations': len(df),
        'Number of Features': len(df.columns) - 1,
        'Missing Values': df.isnull().sum().sum(),
        'Missing %': f"{100 * df.isnull().sum().sum() / df.size:.2f}%"
    }
    
    overall_df = pd.DataFrame([overall_stats]).T
    overall_df.columns = ['Value']
    overall_df.to_csv(tab_dir / "dataset_overall_statistics.csv")
    print("Overall Dataset Statistics:")
    print(overall_df)
    print()
    
    # ===== OUTCOME DISTRIBUTION =====
    y = df[target]
    outcome_stats = {
        'Negative (0)': {
            'Count': (y == 0).sum(),
            'Percentage': f"{100 * (y == 0).sum() / len(y):.2f}%"
        },
        'Positive (1)': {
            'Count': (y == 1).sum(),
            'Percentage': f"{100 * (y == 1).sum() / len(y):.2f}%"
        },
        'Imbalance Ratio': {
            'Count': f"{(y == 0).sum() / (y == 1).sum():.2f}:1",
            'Percentage': '-'
        }
    }
    outcome_df = pd.DataFrame(outcome_stats).T
    outcome_df.to_csv(tab_dir / "dataset_outcome_distribution.csv")
    print("Outcome Distribution:")
    print(outcome_df)
    print()
    
    # ===== FEATURE STATISTICS BY OUTCOME =====
    feature_cols = [col for col in df.columns if col != target]
    
    # Separate numeric and categorical features
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Binary features (0/1 or boolean)
    binary_features = []
    for col in numeric_features:
        if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1, True, False}):
            binary_features.append(col)
    
    continuous_features = [f for f in numeric_features if f not in binary_features]
    
    print(f"Feature Types:")
    print(f"  Continuous: {len(continuous_features)}")
    print(f"  Binary: {len(binary_features)}")
    print(f"  Categorical: {len(categorical_features)}\n")
    
    # ===== CONTINUOUS FEATURES ANALYSIS =====
    if continuous_features:
        continuous_stats = []
        for feat in continuous_features:
            neg_vals = df[df[target] == 0][feat].dropna()
            pos_vals = df[df[target] == 1][feat].dropna()
            
            # Descriptive statistics
            stats_dict = {
                'Feature': feat,
                'Overall_Mean': df[feat].mean(),
                'Overall_SD': df[feat].std(),
                'Overall_Median': df[feat].median(),
                'Overall_IQR': df[feat].quantile(0.75) - df[feat].quantile(0.25),
                'Negative_Mean': neg_vals.mean(),
                'Negative_SD': neg_vals.std(),
                'Positive_Mean': pos_vals.mean(),
                'Positive_SD': pos_vals.std(),
                'Missing_Count': df[feat].isnull().sum(),
                'Missing_Pct': f"{100 * df[feat].isnull().sum() / len(df):.2f}%"
            }
            
            # Statistical tests
            # T-test for mean difference
            t_stat, t_pval = ttest_ind(neg_vals, pos_vals, equal_var=False)
            stats_dict['TTest_Statistic'] = t_stat
            stats_dict['TTest_PValue'] = t_pval
            stats_dict['TTest_Significant'] = 'Yes' if t_pval < 0.05 else 'No'
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pval = mannwhitneyu(neg_vals, pos_vals, alternative='two-sided')
            stats_dict['MannWhitney_Statistic'] = u_stat
            stats_dict['MannWhitney_PValue'] = u_pval
            stats_dict['MannWhitney_Significant'] = 'Yes' if u_pval < 0.05 else 'No'
            
            # Effect size (Cohen's d)
            cohens_d = (pos_vals.mean() - neg_vals.mean()) / np.sqrt(
                ((len(pos_vals) - 1) * pos_vals.std()**2 + (len(neg_vals) - 1) * neg_vals.std()**2) / 
                (len(pos_vals) + len(neg_vals) - 2)
            )
            stats_dict['Cohens_D'] = cohens_d
            stats_dict['Effect_Size'] = ('Small' if abs(cohens_d) < 0.5 else 
                                        'Medium' if abs(cohens_d) < 0.8 else 'Large')
            
            continuous_stats.append(stats_dict)
        
        continuous_df = pd.DataFrame(continuous_stats)
        continuous_df.to_csv(tab_dir / "dataset_continuous_features_analysis.csv", index=False)
        print("Continuous Features Analysis (saved to CSV)")
    
    # ===== BINARY/CATEGORICAL FEATURES ANALYSIS =====
    if binary_features or categorical_features:
        categorical_stats = []
        for feat in binary_features + categorical_features:
            # Create contingency table
            contingency = pd.crosstab(df[feat], df[target])
            
            # Chi-square test
            chi2, chi2_pval, dof, expected = chi2_contingency(contingency)
            
            # Cramér's V (effect size for categorical)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            # Calculate proportions
            props = contingency.div(contingency.sum(axis=1), axis=0)
            
            stats_dict = {
                'Feature': feat,
                'N_Categories': df[feat].nunique(),
                'Missing_Count': df[feat].isnull().sum(),
                'Missing_Pct': f"{100 * df[feat].isnull().sum() / len(df):.2f}%",
                'Chi2_Statistic': chi2,
                'Chi2_PValue': chi2_pval,
                'Chi2_Significant': 'Yes' if chi2_pval < 0.05 else 'No',
                'Cramers_V': cramers_v,
                'Effect_Size': ('Small' if cramers_v < 0.1 else 
                               'Medium' if cramers_v < 0.3 else 'Large')
            }
            
            # Add category distributions
            for cat in df[feat].unique():
                if pd.notna(cat):
                    cat_str = str(cat).replace(' ', '_')
                    neg_pct = props.loc[cat, 0] if 0 in props.columns else 0
                    pos_pct = props.loc[cat, 1] if 1 in props.columns else 0
                    stats_dict[f'{cat_str}_Neg_Pct'] = f"{100 * neg_pct:.1f}%"
                    stats_dict[f'{cat_str}_Pos_Pct'] = f"{100 * pos_pct:.1f}%"
            
            categorical_stats.append(stats_dict)
        
        categorical_df = pd.DataFrame(categorical_stats)
        categorical_df.to_csv(tab_dir / "dataset_categorical_features_analysis.csv", index=False)
        print("Categorical/Binary Features Analysis (saved to CSV)")
    
    # ===== CORRELATION ANALYSIS =====
    if numeric_features:
        corr_matrix = df[numeric_features + [target]].corr()
        
        # Save full correlation matrix
        corr_matrix.to_csv(tab_dir / "dataset_correlation_matrix.csv")
        
        # Extract correlations with target
        target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
        target_corr_df = pd.DataFrame({
            'Feature': target_corr.index,
            'Correlation': target_corr.values,
            'Abs_Correlation': np.abs(target_corr.values),
            'Strength': ['Strong' if abs(c) > 0.5 else 'Moderate' if abs(c) > 0.3 else 'Weak' 
                        for c in target_corr.values]
        }).sort_values('Abs_Correlation', ascending=False)
        
        target_corr_df.to_csv(tab_dir / "target_correlations.csv", index=False)
        print("\nFeature Correlations with Target (Top 10):")
        print(target_corr_df.head(10).to_string(index=False))
        print()
    
    # ===== SUMMARY STATISTICS TABLE =====
    summary_stats = {
        'Dataset Size': len(df),
        'Positive Cases': (y == 1).sum(),
        'Negative Cases': (y == 0).sum(),
        'Prevalence': f"{100 * (y == 1).sum() / len(y):.2f}%",
        'N Features': len(feature_cols),
        'N Continuous': len(continuous_features),
        'N Binary': len(binary_features),
        'N Categorical': len(categorical_features),
        'Missing Values': df.isnull().sum().sum(),
        'Complete Cases': (~df.isnull().any(axis=1)).sum(),
        'Complete Cases %': f"{100 * (~df.isnull().any(axis=1)).sum() / len(df):.2f}%"
    }
    
    summary_df = pd.DataFrame([summary_stats]).T
    summary_df.columns = ['Value']
    summary_df.to_csv(tab_dir / "dataset_summary_statistics.csv")
    
    print(f"\n{'='*80}")
    print("Dataset characterization complete. Files saved:")
    print(f"  - dataset_overall_statistics.csv")
    print(f"  - dataset_outcome_distribution.csv")
    print(f"  - dataset_continuous_features_analysis.csv")
    print(f"  - dataset_categorical_features_analysis.csv")
    print(f"  - dataset_correlation_matrix.csv")
    print(f"  - dataset_target_correlations.csv")
    print(f"  - dataset_summary_statistics.csv")
    print(f"{'='*80}\n")
# =========================================================================================================
##Class Imbalance Handling
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE

def handle_imbalance(X_train, y_train, smote_ratio, method="SMOTE"):
    if method == "SMOTE":
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=DEFAULT_SEED)
    elif method == "RandomOverSampling":
        smote = RandomOverSampler(sampling_strategy=smote_ratio, random_state=DEFAULT_SEED)
    elif method == "BorderlineSMOTE":
        smote = BorderlineSMOTE(sampling_strategy=smote_ratio, random_state=DEFAULT_SEED)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Modify perform_cross_validation to handle imbalance methods
def perform_cross_validation(
    model,
    X,
    y,
    preproc,
    cv_folds=5,
    smote_ratio=0.7,
    model_name="Model",
    seed=DEFAULT_SEED
):
    """
    Standard k-fold cross-validation used for primary results.
    Does NOT perform imbalance sensitivity analysis.
    """

    print(f"\n  Performing {cv_folds}-fold cross-validation...")

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed
    )

    cv_scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]

        # Preprocess
        X_fold_train_enc = preproc.fit_transform(X_fold_train)
        X_fold_val_enc = preproc.transform(X_fold_val)

        # Apply SMOTE for non-XGBoost models
        if model_name != "XGBoost":
            smote = SMOTE(
                sampling_strategy=smote_ratio,
                random_state=seed
            )
            X_fold_train_enc, y_fold_train = smote.fit_resample(
                X_fold_train_enc,
                y_fold_train
            )

        # Fit model
        model.fit(X_fold_train_enc, y_fold_train)

        # Predict
        y_pred = model.predict(X_fold_val_enc)
        y_score = model.predict_proba(X_fold_val_enc)[:, 1]

        # Metrics
        cv_scores["accuracy"].append(
            accuracy_score(y_fold_val, y_pred)
        )
        cv_scores["precision"].append(
            precision_score(y_fold_val, y_pred, zero_division=0)
        )
        cv_scores["recall"].append(
            recall_score(y_fold_val, y_pred, zero_division=0)
        )
        cv_scores["f1"].append(
            f1_score(y_fold_val, y_pred, zero_division=0)
        )
        cv_scores["roc_auc"].append(
            roc_auc_score(y_fold_val, y_score)
        )
        cv_scores["pr_auc"].append(
            average_precision_score(y_fold_val, y_score)
        )

    # Aggregate statistics
    cv_stats = {}
    for metric, values in cv_scores.items():
        cv_stats[f"{metric}_mean"] = float(np.mean(values))
        cv_stats[f"{metric}_std"] = float(np.std(values))
        cv_stats[f"{metric}_min"] = float(np.min(values))
        cv_stats[f"{metric}_max"] = float(np.max(values))
        cv_stats[f"{metric}_cv"] = (
            float(np.std(values) / np.mean(values))
            if np.mean(values) > 0 else 0.0
        )

    print(
        f"  CV Summary — ROC-AUC: "
        f"{cv_stats['roc_auc_mean']:.3f} ± {cv_stats['roc_auc_std']:.3f}, "
        f"Recall: {cv_stats['recall_mean']:.3f}"
    )

    return cv_stats

# =========================================================================================================
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler

def run_imbalance_robustness_check(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    preproc,
    seed,
    smote_ratio=0.7
):
    """
    Supplementary-only robustness check for class imbalance.
    This does NOT affect primary results.
    Produces one clean table for reviewer appeasement.
    """

    methods = {
        "ClassWeight_Only": None,
        "SMOTE": SMOTE(sampling_strategy=smote_ratio, random_state=seed),
        "BorderlineSMOTE": BorderlineSMOTE(sampling_strategy=smote_ratio, random_state=seed),
        "RandomOverSampler": RandomOverSampler(sampling_strategy=smote_ratio, random_state=seed),
    }

    rows = []

    for method_name, sampler in methods.items():
        # Preprocess
        Xtr = preproc.fit_transform(X_train)
        Xte = preproc.transform(X_test)
        ytr = y_train.values if hasattr(y_train, "values") else y_train

        # Apply resampling if applicable
        if sampler is not None:
            Xtr, ytr = sampler.fit_resample(Xtr, ytr)

        # Fit model
        model.fit(Xtr, ytr)

        # Predict
        probs = model.predict_proba(Xte)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Collect key metrics only (keep this simple)
        rows.append({
            "method": method_name,
            "roc_auc": roc_auc_score(y_test, probs),
            "pr_auc": average_precision_score(y_test, probs),
            "recall": recall_score(y_test, preds, zero_division=0),
            "precision": precision_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        })

    return pd.DataFrame(rows)
# =========================================================================================================
# SHAP function
def run_shap_analysis(model, X_train, preproc):
    """
    Supplementary SHAP feature importance analysis.
    Produces a single global importance table.
    """

    # Preprocess data
    X_train_enc = preproc.fit_transform(X_train)

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_enc)

    # ---- Handle binary classification + dimensionality ----
    # Case 1: list output (common)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # Case 2: 3D array (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # positive class

    # Now guaranteed: (n_samples, n_features)
    shap_importance = np.abs(shap_values).mean(axis=0)

    # Feature names from preprocessor
    feature_names = preproc.get_feature_names_out()

    # Safety check (prevents silent mismatch)
    if shap_importance.shape[0] != len(feature_names):
        raise ValueError(
            f"SHAP importance length ({shap_importance.shape[0]}) "
            f"does not match number of features ({len(feature_names)})"
        )

    # Output table
    importance_df = (
        pd.DataFrame({
            "Feature": feature_names,
            "SHAP_Importance": shap_importance
        })
        .sort_values("SHAP_Importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df
# =========================================================================================================
# Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed interaction features based on occupational health literature.
    """
    df = df.copy()
    
    # Cumulative Chemical Exposure
    chemical_cols = ["hcl_high", "ammonia_high", "degreaser_high", 
                     "multipurpose_high", "wax_high"]
    df["total_chemical_exposure"] = df[chemical_cols].sum(axis=1)
    df["high_exposure_flag"] = (df["total_chemical_exposure"] >= 3).astype(int)
    
    # Smoking-Workplace Interaction
    if "smoking" in df.columns and "workplace" in df.columns:
        smoking_binary = (df["smoking"] == 1).astype(int) if df["smoking"].dtype in ['int64', 'float64'] else \
                        (df["smoking"].isin(['yes', 'Yes', '1', 1, True])).astype(int)
        workplace_binary = (df["workplace"] == 1).astype(int) if df["workplace"].dtype in ['int64', 'float64'] else \
                          (df["workplace"].isin(['yes', 'Yes', '1', 1, True])).astype(int)
        df["smoking_workplace_risk"] = smoking_binary * workplace_binary
    else:
        df["smoking_workplace_risk"] = 0
    
    # Vulnerable Population Indicator
    if "foreign_born" in df.columns:
        df["vulnerable_population"] = (df["foreign_born"].astype(bool) & 
                                       (df["total_chemical_exposure"] >= 2)).astype(int)
    else:
        df["vulnerable_population"] = 0
    
    return df

# =========================================================================================================
# Data splitting with comprehensive reporting
def split_data(df, seed, tab_dir):
    target = "asthma_like_symptoms" if "asthma_like_symptoms" in df.columns else "respiratory_irritation"
    y = df[target].astype(int)

    # Base features
    feature_cols = [
        "sex", "age", "smoking", "foreign_born", "workplace",
        "hcl_high", "ammonia_high", "degreaser_high", "multipurpose_high", "wax_high"
    ]
    
    # Engineered features
    engineered_cols = [
        "total_chemical_exposure", "high_exposure_flag",
        "smoking_workplace_risk", "vulnerable_population"
    ]
    
    all_features = feature_cols + engineered_cols
    X = df[all_features].copy()

    # Stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    
    # ===== SPLIT DISTRIBUTION REPORT =====
    split_stats = []
    for split_name, y_split in [('Train', y_train), ('Validation', y_valid), ('Test', y_test)]:
        split_stats.append({
            'Split': split_name,
            'Total': len(y_split),
            'Negative': (y_split == 0).sum(),
            'Positive': (y_split == 1).sum(),
            'Prevalence': f"{100 * (y_split == 1).sum() / len(y_split):.2f}%",
            'Imbalance_Ratio': f"{(y_split == 0).sum() / (y_split == 1).sum():.2f}:1"
        })
    
    split_df = pd.DataFrame(split_stats)
    split_df.to_csv(tab_dir / "data_split_distribution.csv", index=False)
    
    print(f"\n{'='*80}")
    print("DATA SPLIT DISTRIBUTION")
    print(f"{'='*80}")
    print(split_df.to_string(index=False))
    print(f"{'='*80}\n")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, all_features, target

# =========================================================================================================
# Preprocessing pipeline
def build_preprocessor(X):
    categorical_features = ["sex", "smoking", "workplace"]
    binary_features = [
        "foreign_born", "hcl_high", "ammonia_high", "degreaser_high", 
        "multipurpose_high", "wax_high", "high_exposure_flag",
        "smoking_workplace_risk", "vulnerable_population"
    ]
    numerical_features = ["age", "total_chemical_exposure"]
    
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_features),
        ("bin", "passthrough", binary_features),
        ("num", StandardScaler(), numerical_features),
    ])

_PRETTY_MAP = {
    "foreign_born": "Foreign-Born",
    "hcl_high": "HCL (High)",
    "ammonia_high": "Ammonia (High)",
    "degreaser_high": "Degreaser (High)",
    "multipurpose_high": "Multipurpose (High)",
    "wax_high": "Wax (High)",
    "age": "Age (years)",
    "total_chemical_exposure": "Total Chemical Exposure",
    "high_exposure_flag": "High Exposure Flag",
    "smoking_workplace_risk": "Smoking × Workplace Risk",
    "vulnerable_population": "Vulnerable Population",
}

def prettify(n): 
    return _PRETTY_MAP.get(n, n.replace("_", " ").title())

# =========================================================================================================
# COMPREHENSIVE METRICS CALCULATION
def calculate_comprehensive_metrics(y_true, y_pred, y_score, threshold=None):
    """
    Calculate ALL relevant classification metrics including:
    - Basic metrics (accuracy, precision, recall, F1)
    - Advanced metrics (MCC, Cohen's kappa, balanced accuracy)
    - Probability-based metrics (ROC-AUC, PR-AUC, Brier score, log loss)
    - Clinical metrics (sensitivity, specificity, PPV, NPV, LR+, LR-)
    - Threshold-dependent metrics
    """
    metrics = {}

    # Confusion matrix components
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    metrics['total'] = len(y_true)

    # ===== BASIC CLASSIFICATION METRICS =====
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['f2_score'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # Emphasizes recall
    metrics['f05_score'] = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)  # Emphasizes precision

    # ===== CLINICAL METRICS =====
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (same as precision)
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

    # Likelihood ratios
    metrics['lr_positive'] = (metrics['sensitivity'] / (1 - metrics['specificity'])) if metrics['specificity'] < 1 else np.inf
    metrics['lr_negative'] = ((1 - metrics['sensitivity']) / metrics['specificity']) if metrics['specificity'] > 0 else np.inf

    # Diagnostic odds ratio
    if metrics['lr_positive'] != np.inf and metrics['lr_negative'] != 0:
        metrics['diagnostic_odds_ratio'] = metrics['lr_positive'] / metrics['lr_negative']
    else:
        metrics['diagnostic_odds_ratio'] = np.inf

    # Youden's J statistic (informedness)
    metrics['youdens_j'] = metrics['sensitivity'] + metrics['specificity'] - 1

    # Markedness
    metrics['markedness'] = metrics['ppv'] + metrics['npv'] - 1

    # ===== AGREEMENT METRICS =====
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)

    # ===== PROBABILITY-BASED METRICS =====
    metrics['roc_auc'] = roc_auc_score(y_true, y_score)
    metrics['pr_auc'] = average_precision_score(y_true, y_score)
    metrics['brier_score'] = brier_score_loss(y_true, y_score)
    metrics['log_loss'] = log_loss(y_true, y_score)

    # ===== PREVALENCE AND RATES =====
    metrics['prevalence'] = (tp + fn) / (tp + tn + fp + fn)
    metrics['true_positive_rate'] = metrics['sensitivity']
    metrics['true_negative_rate'] = metrics['specificity']
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics['false_discovery_rate'] = fp / (fp + tp) if (fp + tp) > 0 else 0
    metrics['false_omission_rate'] = fn / (fn + tn) if (fn + tn) > 0 else 0

    # ===== THRESHOLD =====
    if threshold is not None:
        metrics['threshold'] = threshold

    # ===== GEOMETRIC MEAN =====
    metrics['geometric_mean'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])

    return metrics
# --------------------------------------------------------------------------------------
def tune_threshold_recall_constrained(y_true, y_score, min_recall=0.70, grid=500):
    """Find threshold that maximizes F1 while maintaining minimum recall."""
    thr_vals = np.linspace(0.01, 0.99, grid)
    best_thr, best_f1 = 0.5, -1

    for t in thr_vals:
        y_hat = (y_score >= t).astype(int)
        rec = recall_score(y_true, y_hat, zero_division=0)

        if rec >= min_recall:
            f = f1_score(y_true, y_hat, zero_division=0)
            if f > best_f1:
                best_thr, best_f1 = t, f

    if best_f1 == -1:
        print(f"  Warning: Could not achieve {min_recall:.1%} recall. Using best available.")
        best_recall = 0
        for t in thr_vals:
            y_hat = (y_score >= t).astype(int)
            rec = recall_score(y_true, y_hat, zero_division=0)
            if rec > best_recall:
                best_thr, best_recall = t, rec

    return float(best_thr)
# =========================================================================================================
# CROSS-VALIDATION ANALYSIS
def perform_cross_validation(model, X, y, preproc, cv_folds=5, smote_ratio=0.7, 
                            model_name="Model", seed=DEFAULT_SEED):
    """
    Comprehensive k-fold cross-validation with multiple metrics
    """
    print(f"\n  Performing {cv_folds}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'pr_auc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]
        
        # Preprocess
        X_fold_train_enc = preproc.fit_transform(X_fold_train)
        X_fold_val_enc = preproc.transform(X_fold_val)
        
        # Apply SMOTE (skip for XGBoost)
        if model_name != "XGBoost":
            smote = SMOTE(sampling_strategy=smote_ratio, random_state=seed)
            X_fold_train_enc, y_fold_train = smote.fit_resample(X_fold_train_enc, y_fold_train)
        
        # Fit and predict
        model.fit(X_fold_train_enc, y_fold_train)
        y_pred = model.predict(X_fold_val_enc)
        y_score = model.predict_proba(X_fold_val_enc)[:, 1]
        
        # Calculate metrics
        cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        cv_scores['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
        cv_scores['recall'].append(recall_score(y_fold_val, y_pred, zero_division=0))
        cv_scores['f1'].append(f1_score(y_fold_val, y_pred, zero_division=0))
        cv_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_score))
        cv_scores['pr_auc'].append(average_precision_score(y_fold_val, y_score))
    
    # Calculate statistics
    cv_stats = {}
    for metric, scores in cv_scores.items():
        cv_stats[f'{metric}_mean'] = np.mean(scores)
        cv_stats[f'{metric}_std'] = np.std(scores)
        cv_stats[f'{metric}_min'] = np.min(scores)
        cv_stats[f'{metric}_max'] = np.max(scores)
        cv_stats[f'{metric}_cv'] = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0  # Coefficient of variation
    
    print(f"  Cross-validation complete:")
    print(f"    ROC-AUC: {cv_stats['roc_auc_mean']:.4f} ± {cv_stats['roc_auc_std']:.4f}")
    print(f"    PR-AUC:  {cv_stats['pr_auc_mean']:.4f} ± {cv_stats['pr_auc_std']:.4f}")
    print(f"    Recall:  {cv_stats['recall_mean']:.4f} ± {cv_stats['recall_std']:.4f}")
    
    return cv_stats

# =========================================================================================================
# CALIBRATION ANALYSIS
def plot_calibration_curve(y_true, y_score, model_name, fig_dir):
    """
    Plot calibration curve to assess probability calibration
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_score, n_bins=10, strategy='uniform'
    )
    
    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
           linewidth=2.5, markersize=8, color='#2E86AB', label=model_name)
    
    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect Calibration')
    
    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight='bold')
    ax.set_ylabel("Fraction of Positives", fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='black')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"calibration_curve_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate calibration metrics
    calib_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    
    return calib_error

# =========================================================================================================
# THRESHOLD ANALYSIS
# =========================================================================================================
# THRESHOLD ANALYSIS (DEPLOYMENT-AWARE)
def analyze_threshold_tradeoffs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tab_dir: Path,
    model_name: str,
    thresholds: np.ndarray | None = None,
    per_n: int = 1000
) -> pd.DataFrame:
    """
    Deployment-aware threshold analysis emphasizing false-positive burden.
    Produces a single canonical table for publication and public release.
    """

    if thresholds is None:
        # Conservative range appropriate for rare outcomes
        thresholds = np.linspace(0.05, 0.80, 76)

    rows = []
    n = len(y_true)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    prevalence = pos / n if n > 0 else 0.0

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        m = calculate_comprehensive_metrics(y_true, y_pred, y_score, threshold=thr)

        alerts = int(np.sum(y_pred == 1))
        fp = int(m["fp"])
        tp = int(m["tp"])
        fn = int(m["fn"])
        tn = int(m["tn"])

        # Ensure 'recall' column is populated correctly here
        rows.append({
            "model": model_name,
            "threshold": float(thr),

            # Population context
            "prevalence": prevalence,

            # Core operating characteristics
            "precision_ppv": float(m["precision"]),
            "recall": float(m["recall"]),  # Correct name for recall column
            "specificity": float(m["specificity"]),
            "f1_score": float(m["f1_score"]),
            "f2_score": float(m["f2_score"]),
            "youdens_j": float(m["youdens_j"]),

            # Workload / burden framing
            f"alerts_per_{per_n}": alerts / n * per_n if n > 0 else 0.0,
            f"false_positives_per_{per_n}": fp / n * per_n if n > 0 else 0.0,
            f"true_positives_per_{per_n}": tp / n * per_n if n > 0 else 0.0,

            # Confusion matrix (absolute)
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(
        tab_dir / f"{model_name}_threshold_operating_points.csv",
        index=False
    )

    # Debugging line to ensure recall is present
    print(threshold_df.columns)  # Ensure the 'recall' column is present here

    return threshold_df


# =========================================================================================================
def export_rf_fp_recall_outputs(
    threshold_df,
    output_dir,
    model_name,
    recall_targets=(0.95, 0.85, 0.75, 0.65, 0.55, 0.40, 0.25)
):
    # ---- ensure expected columns exist ----
    required_cols = {"recall", "false_positives_per_1000"}
    missing = required_cols - set(threshold_df.columns)
    if missing:
        raise ValueError(f"Missing columns in threshold_df: {missing}")

    # ---- plot: FP vs Recall (SM) ----
    plt.figure(figsize=(5, 4))
    plt.plot(
        threshold_df["recall"],
        threshold_df["false_positives_per_1000"],
        linewidth=1
    )
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("False Positives per 1,000 Workers")
    plt.tight_layout()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure
    fig_path = os.path.join(output_dir, f"{model_name}_FP_vs_Recall_SM.png")
    try:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved at: {fig_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    plt.close()

    # ---- representative operating points ----
    rows = []
    for r in recall_targets:
        idx = (threshold_df["recall"] - r).abs().idxmin()
        rows.append(threshold_df.loc[idx])

    rep_df = pd.DataFrame(rows).drop_duplicates()

    # Save the representative operating points to CSV
    rep_path = os.path.join(output_dir, f"{model_name}_Representative_Operating_Points.csv")
    try:
        rep_df.to_csv(rep_path, index=False)
        print(f"CSV saved at: {rep_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    return rep_df
# =========================================================================================================
# Plotting functions (keeping your publication-quality versions)
def plot_conf_grid(all_cms, model_names, fig_dir):
    """2x2 confusion matrix grid"""
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5), dpi=300)
    axes = axes.flatten()
    
    panel_labels = ['(A)', '(B)', '(C)', '(D)']
    
    for idx, (cm, model_name, ax, label) in enumerate(zip(all_cms, model_names, axes, panel_labels)):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    square=True, linewidths=1.5, linecolor='gray',
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                    ax=ax, cbar_kws={'shrink': 0.7})
        
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.75, f'({cm_norm[i,j]:.1%})', 
                       ha='center', va='center', fontsize=9, 
                       color='#404040', fontweight='normal')
        
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["No", "Yes"], fontsize=9, fontweight='bold')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["No", "Yes"], fontsize=9, fontweight='bold', rotation=0)
        ax.set_xlabel("Predicted", fontsize=10, fontweight='bold')
        ax.set_ylabel("Actual", fontsize=10, fontweight='bold')
        ax.set_title(f'{label} {model_name}', fontsize=11, fontweight='bold', pad=8)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure2_confusion_matrices_grid.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_conf(cm, out, model_name):
    """Single confusion matrix"""
    plt.figure(figsize=(3.5, 3.2), dpi=300)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1.5, linecolor='gray',
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.75, f'({cm_norm[i,j]:.1%})', 
                    ha='center', va='center', fontsize=11, 
                    color='#404040', fontweight='normal')
    
    plt.xticks([0.5, 1.5], ["No", "Yes"], fontsize=11, fontweight='bold')
    plt.yticks([0.5, 1.5], ["No", "Yes"], fontsize=11, fontweight='bold', rotation=0)
    plt.xlabel("Predicted", fontsize=12, fontweight='bold')
    plt.ylabel("Actual", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(all_metrics, tab_dir, fig_dir):
    """Metrics comparison with extended metrics"""
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(tab_dir / "all_models_metrics_comparison.csv", index=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), dpi=300)
    
    models = metrics_df['model'].values
    x_pos = np.arange(len(models))
    
    panel_labels = ['(A)', '(B)', '(C)', '(D)']
    metrics_to_plot = [
        ('recall', 'Recall (Sensitivity)', axes[0, 0]),
        ('precision', 'Precision', axes[0, 1]),
        ('f1_score', 'F1 Score', axes[1, 0]),
        ('pr_auc', 'PR-AUC', axes[1, 1])
    ]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    for (metric, title, ax), label in zip(metrics_to_plot, panel_labels):
        values = metrics_df[metric].values
        bars = ax.bar(x_pos, values, color=colors, alpha=0.85, 
                     edgecolor='black', linewidth=1.2)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
               fontsize=13, fontweight='bold', va='top', ha='left')
        
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure6_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_compare_roc_pr(results, y_test, fig_dir):
    """ROC and PR curves"""
    colors = cm.viridis(np.linspace(0.15, 0.85, len(results)))

    # ROC Curve
    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
    for (name, prob), col in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, linewidth=2.5, alpha=0.85, color=col, 
               label=f'{name} (AUC={auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.legend(frameon=True, fontsize=10, loc='lower right', framealpha=0.95, 
             edgecolor='black', fancybox=False)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure3_ROC_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # PR Curve
    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
    baseline = y_test.sum() / len(y_test)
    
    for (name, prob), col in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, prob)
        pr_auc = average_precision_score(y_test, prob)
        ax.plot(rec, prec, linewidth=2.5, alpha=0.85, color=col, 
               label=f'{name} (AP={pr_auc:.3f})')
    
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5, 
              alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel("Recall", fontsize=13, fontweight='bold')
    ax.set_ylabel("Precision", fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.legend(frameon=True, fontsize=10, loc='best', framealpha=0.95,
             edgecolor='black', fancybox=False)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure4_PR_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, model_name, fig_dir, top_n=15):
    """Feature importance plot"""
    if not hasattr(model, "feature_importances_"):
        return
    
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    vals = imp[idx][::-1]
    labs = np.array(feature_names)[idx][::-1]
    
    clean_labels = []
    for lbl in labs:
        lbl = lbl.replace("cat__", "").replace("bin__", "").replace("num__", "")
        lbl = lbl.replace("workplace_", "Workplace: ")
        lbl = lbl.replace("smoking_", "Smoking: ")
        lbl = lbl.replace("sex_", "Sex: ")
        clean_labels.append(prettify(lbl))
    
    vir = matplotlib.colormaps["viridis"]
    colors = vir(np.linspace(0.25, 0.85, len(vals)))
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    bars = ax.barh(clean_labels, vals, color=colors, alpha=0.90, 
                   edgecolor='black', linewidth=1.0)
    
    for bar, val in zip(bars, vals):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2., 
               f' {val:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Importance", fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"Figure5_feature_importance_{model_name}.png", 
               dpi=300, bbox_inches='tight')
    plt.close()

# =========================================================================================================
# COMPREHENSIVE FIT AND EVALUATION
def fit_and_eval(name, model, preproc, X_train, y_train, X_valid, y_valid, 
                X_test, y_test, fig_dir, tab_dir, smote_ratio, min_recall, seed):
    """
    Enhanced fitting and evaluation with comprehensive metrics
    """
    print(f"\n{'='*80}")
    print(f"Training {name}")
    print(f"{'='*80}")
    
    # Preprocess
    X_train_encoded = preproc.fit_transform(X_train)
    X_valid_encoded = preproc.transform(X_valid)
    X_test_encoded = preproc.transform(X_test)
    
    # Apply SMOTE conditionally
    if name == "XGBoost":
        print(f"  Skipping SMOTE for XGBoost (using scale_pos_weight={model.scale_pos_weight:.2f})...")
        X_train_resampled = X_train_encoded
        y_train_resampled = y_train.values if hasattr(y_train, 'values') else y_train
    else:
        print(f"  Applying SMOTE with sampling_strategy={smote_ratio}...")
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=seed)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)
    
    print(f"  Training samples: {len(y_train_resampled):,}")
    print(f"    - Positive: {y_train_resampled.sum():,} ({100*y_train_resampled.mean():.1f}%)")
    print(f"    - Negative: {(~y_train_resampled.astype(bool)).sum():,}")
    
    # Fit model
    print(f"  Fitting {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predictions
    proba_train = model.predict_proba(X_train_resampled)[:, 1]
    proba_valid = model.predict_proba(X_valid_encoded)[:, 1]
    proba_test = model.predict_proba(X_test_encoded)[:, 1]
    
    # Threshold tuning
    print(f"  Tuning threshold with min_recall={min_recall}...")
    thr = tune_threshold_recall_constrained(y_valid, proba_valid, min_recall=min_recall)
    print(f"  Optimal threshold: {thr:.4f}")
    
    # Predictions at threshold
    y_pred_train = (proba_train >= thr).astype(int)
    y_pred_valid = (proba_valid >= thr).astype(int)
    y_pred_test = (proba_test >= thr).astype(int)
    
    # ===== COMPREHENSIVE METRICS FOR ALL SPLITS =====
    print(f"\n  Calculating comprehensive metrics...")
    
    m_train = calculate_comprehensive_metrics(y_train_resampled, y_pred_train, proba_train, thr)
    m_train.update({"model": name, "split": "Train (resampled)"})
    
    m_valid = calculate_comprehensive_metrics(y_valid, y_pred_valid, proba_valid, thr)
    m_valid.update({"model": name, "split": "Validation"})
    
    m_test = calculate_comprehensive_metrics(y_test, y_pred_test, proba_test, thr)
    m_test.update({"model": name, "split": "Test"})
    
    # ===== CROSS-VALIDATION =====
    cv_stats = perform_cross_validation(
        model, X_train, y_train, preproc, 
        cv_folds=CV_FOLDS, smote_ratio=smote_ratio,
        model_name=name, seed=seed
    )
    
    # Save CV results
    cv_df = pd.DataFrame([cv_stats])
    cv_df.insert(0, 'model', name)
    cv_df.to_csv(tab_dir / f"{name}_cross_validation.csv", index=False)
    # ---------------------------------------------------------------------------------------
    # ===== THRESHOLD ANALYSIS =====
    print(f"  Performing threshold analysis...")
    threshold_df = analyze_threshold_tradeoffs(
        y_true=y_test, 
        y_score=proba_test, 
        tab_dir=tab_dir, 
        model_name=name
    )

    # Ensure the additional outputs (plot + CSV) are created
    export_rf_fp_recall_outputs(
        threshold_df=threshold_df,
        output_dir=tab_dir,
        model_name=name
    ) 
    # ---------------------------------------------------------------------------------------
    # ===== CALIBRATION ANALYSIS =====
    print(f"  Analyzing probability calibration...")
    calib_error = plot_calibration_curve(y_test, proba_test, name, fig_dir)
    m_test['calibration_error'] = calib_error
    
    # ===== PRINT TEST RESULTS =====
    print(f"\n  Test Set Results:")
    print(f"    Accuracy:           {m_test['accuracy']:.4f}")
    print(f"    Balanced Accuracy:  {m_test['balanced_accuracy']:.4f}")
    print(f"    Precision (PPV):    {m_test['precision']:.4f}")
    print(f"    Recall (Sens):      {m_test['recall']:.4f}")
    print(f"    Specificity:        {m_test['specificity']:.4f}")
    print(f"    F1 Score:           {m_test['f1_score']:.4f}")
    print(f"    F2 Score:           {m_test['f2_score']:.4f}")
    print(f"    ROC-AUC:            {m_test['roc_auc']:.4f}")
    print(f"    PR-AUC:             {m_test['pr_auc']:.4f}")
    print(f"    MCC:                {m_test['matthews_corrcoef']:.4f}")
    print(f"    Cohen's Kappa:      {m_test['cohens_kappa']:.4f}")
    print(f"    NPV:                {m_test['npv']:.4f}")
    print(f"    LR+:                {m_test['lr_positive']:.4f}")
    print(f"    LR-:                {m_test['lr_negative']:.4f}")
    print(f"    Calibration Error:  {m_test['calibration_error']:.4f}")
    
    # ===== SAVE DETAILED METRICS =====
    metrics_df = pd.DataFrame([m_train, m_valid, m_test])
    metrics_df.to_csv(tab_dir / f"{name}_detailed_metrics.csv", index=False)
    
    # Save comprehensive test metrics separately
    test_metrics_detailed = pd.DataFrame([m_test])
    test_metrics_detailed.to_csv(tab_dir / f"{name}_test_comprehensive.csv", index=False)
    
    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(y_test, y_pred_test)
    plot_conf(cm, fig_dir / f"Figure2_confusion_{name}.png", name)
    
    # ===== FEATURE IMPORTANCE =====
    if MAKE_IMPORTANCE and hasattr(model, "feature_importances_"):
        feature_names = list(preproc.get_feature_names_out())
        plot_feature_importance(model, feature_names, name, fig_dir)
    
    return name, proba_test, m_test, cm, cv_stats

# =========================================================================================================
# COMPREHENSIVE JOURNAL SUMMARY
def generate_comprehensive_journal_summary(all_metrics, all_cv_stats, tab_dir):
    """
    Generate comprehensive summary tables for journal publication
    """
    # ===== TABLE 1: PRIMARY PERFORMANCE METRICS =====
    summary_df = pd.DataFrame(all_metrics)
    
    table1_cols = ['model', 'recall', 'specificity', 'precision', 'npv', 
                   'f1_score', 'roc_auc', 'pr_auc', 'matthews_corrcoef']
    table1 = summary_df[table1_cols].copy()
    table1.columns = ['Model', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 
                     'F1-Score', 'ROC-AUC', 'PR-AUC', 'MCC']
    
    # Format
    for col in ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-Score']:
        table1[col] = table1[col].apply(lambda x: f"{x:.1%}")
    for col in ['ROC-AUC', 'PR-AUC', 'MCC']:
        table1[col] = table1[col].apply(lambda x: f"{x:.3f}")
    
    table1.to_csv(tab_dir / "Table1_primary_performance.csv", index=False)
    
    # ===== TABLE 2: EXTENDED METRICS =====
    table2_cols = ['model', 'balanced_accuracy', 'cohens_kappa', 'youdens_j', 
                   'lr_positive', 'lr_negative', 'geometric_mean', 'calibration_error']
    table2 = summary_df[table2_cols].copy()
    table2.columns = ['Model', 'Balanced Acc', 'Cohen κ', "Youden's J", 
                     'LR+', 'LR-', 'G-Mean', 'Calib. Error']
    
    # Format
    for col in ['Balanced Acc', 'Cohen κ', "Youden's J", 'G-Mean', 'Calib. Error']:
        table2[col] = table2[col].apply(lambda x: f"{x:.3f}")
    for col in ['LR+', 'LR-']:
        table2[col] = table2[col].apply(lambda x: f"{x:.2f}" if x != np.inf else "∞")
    
    table2.to_csv(tab_dir / "Table2_extended_metrics.csv", index=False)
    
    # ===== TABLE 3: CROSS-VALIDATION RESULTS =====
    cv_summary = []
    for model_name, cv_stats in zip(summary_df['model'], all_cv_stats):
        cv_row = {
            'Model': model_name,
            'ROC-AUC': f"{cv_stats['roc_auc_mean']:.3f} ± {cv_stats['roc_auc_std']:.3f}",
            'PR-AUC': f"{cv_stats['pr_auc_mean']:.3f} ± {cv_stats['pr_auc_std']:.3f}",
            'Recall': f"{cv_stats['recall_mean']:.3f} ± {cv_stats['recall_std']:.3f}",
            'Precision': f"{cv_stats['precision_mean']:.3f} ± {cv_stats['precision_std']:.3f}",
            'F1': f"{cv_stats['f1_mean']:.3f} ± {cv_stats['f1_mean']:.3f}"
        }
        cv_summary.append(cv_row)
    
    cv_table = pd.DataFrame(cv_summary)
    cv_table.to_csv(tab_dir / "Table3_cross_validation_results.csv", index=False)
    
    # ===== TABLE 4: CONFUSION MATRIX STATISTICS =====
    cm_stats = []
    for _, row in summary_df.iterrows():
        cm_row = {
            'Model': row['model'],
            'TP': int(row['tp']),
            'FP': int(row['fp']),
            'TN': int(row['tn']),
            'FN': int(row['fn']),
            'FPR': f"{row['false_positive_rate']:.1%}",
            'FNR': f"{row['false_negative_rate']:.1%}",
            'FDR': f"{row['false_discovery_rate']:.1%}",
            'FOR': f"{row['false_omission_rate']:.1%}"
        }
        cm_stats.append(cm_row)
    
    cm_table = pd.DataFrame(cm_stats)
    cm_table.to_csv(tab_dir / "Table4_confusion_matrix_statistics.csv", index=False)
    
    # ===== PRINT SUMMARY =====
    print(f"\n{'='*80}")
    print("TABLE 1: Primary Performance Metrics (Test Set)")
    print(f"{'='*80}")
    print(table1.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("TABLE 2: Extended Performance Metrics (Test Set)")
    print(f"{'='*80}")
    print(table2.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"TABLE 3: {CV_FOLDS}-Fold Cross-Validation Results (Mean ± SD)")
    print(f"{'='*80}")
    print(cv_table.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("TABLE 4: Confusion Matrix Statistics (Test Set)")
    print(f"{'='*80}")
    print(cm_table.to_string(index=False))
    print(f"{'='*80}\n")

# =========================================================================================================
# MAIN EXECUTION
def main():
    args = parse_args()
    
    # Setup root directory
    if args.root is None:
        script_dir = Path(__file__).resolve().parent
        root = script_dir.parent
    else:
        root = Path(args.root).expanduser()
    
    print(f"\n{'='*80}")
    print("ENHANCED ASTHMA CLASSIFICATION PIPELINE")
    print("Comprehensive Metrics for MDPI Laboratories")
    print(f"{'='*80}")
    print(f"Root directory: {root}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")
    
    data_dir, fig_dir, tab_dir = ensure_dirs(root)
    
    # ===== LOAD AND CHARACTERIZE DATASET =====
    print("="*80)
    print("PHASE 1: DATA LOADING AND CHARACTERIZATION")
    print("="*80)
    
    df = load_data(data_dir / args.dataset)
    target = "asthma_like_symptoms" if "asthma_like_symptoms" in df.columns else "respiratory_irritation"
    
    # Generate comprehensive dataset report
    generate_dataset_report(df, target, tab_dir)
    
    # ===== FEATURE ENGINEERING =====
    print("="*80)
    print("PHASE 2: FEATURE ENGINEERING")
    print("="*80)
    print("Engineering domain-informed features...")
    df = engineer_features(df)
    print("Feature engineering complete.\n")
    
    # ===== DATA SPLITTING =====
    print("="*80)
    print("PHASE 3: DATA SPLITTING")
    print("="*80)
    X_train, X_valid, X_test, y_train, y_valid, y_test, features, target = split_data(
        df, args.seed, tab_dir
    )
    
    # ===== BUILD PREPROCESSOR =====
    preproc = build_preprocessor(X_train)
    
    # ===== MODEL CONFIGURATION =====
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    print(f"\n{'='*80}")
    print("PHASE 4: MODEL CONFIGURATION")
    print(f"{'='*80}")
    print(f"SMOTE sampling ratio: {args.smote_ratio}")
    print(f"Minimum recall constraint: {args.min_recall}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    print(f"{'='*80}\n")
    
    models = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, 
            min_samples_leaf=50,
            class_weight="balanced", 
            random_state=args.seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, 
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced", 
            n_jobs=-1, 
            random_state=args.seed
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=args.seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            scale_pos_weight=scale_pos_weight * 1.5,
            subsample=0.8,
            colsample_bytree=0.7,
            colsample_bylevel=0.7,
            min_child_weight=3,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=0.5,
            max_delta_step=1,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=args.seed,
            use_label_encoder=False,
            n_jobs=-1
        ) 
    }
    
    # ===== TRAIN AND EVALUATE ALL MODELS =====
    print("="*80)
    print("PHASE 5: MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    results = []
    all_metrics = []
    all_cms = []
    model_names = []
    all_cv_stats = []
    
    
    # -------------------------------------------------------------
    for model_name, model in models.items():
        # Train and evaluate the model
        name, proba, metrics, cm, cv_stats = fit_and_eval(
            model_name, model, preproc, 
            X_train, y_train, X_valid, y_valid, X_test, y_test,
            fig_dir, tab_dir, args.smote_ratio, args.min_recall, args.seed
        )
        
        # Collect results for metrics and confusion matrices
        results.append((name, proba))
        all_metrics.append(metrics)
        all_cms.append(cm)
        model_names.append(name)
        all_cv_stats.append(cv_stats)

        
        # ---------------------------------------------
        # Supplementary: class imbalance robustness (NOT part of main experiment)
        imbalance_df = run_imbalance_robustness_check(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            preproc=preproc,
            seed=args.seed,
            smote_ratio=args.smote_ratio
        )

        imbalance_df.to_csv(
            tab_dir / f"{model_name}_imbalance_robustness.csv",
            index=False
        )
        # ---------------------------------------------

        # SHAP Feature Importance Analysis
        shap_importance_df = run_shap_analysis(model, X_train, preproc)
        shap_importance_df.to_csv(tab_dir / f"{model_name}_shap_importance.csv", index=False)


    # -------------------------------------------------------------
    # ===== COMPARISON VISUALIZATIONS =====
    print(f"\n{'='*80}")
    print("PHASE 6: GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'='*80}")
    
    plot_conf_grid(all_cms, model_names, fig_dir)
    plot_compare_roc_pr(results, y_test, fig_dir)
    plot_metrics_comparison(all_metrics, tab_dir, fig_dir)
    
    print("Visualization generation complete.")
    
    # ===== COMPREHENSIVE JOURNAL SUMMARY =====
    print(f"\n{'='*80}")
    print("PHASE 7: GENERATING COMPREHENSIVE SUMMARY TABLES")
    print(f"{'='*80}")
    
    generate_comprehensive_journal_summary(all_metrics, all_cv_stats, tab_dir)
    
    # ===== FINAL SUMMARY =====
    best_idx = np.argmax([m['pr_auc'] for m in all_metrics])
    best_model = all_metrics[best_idx]
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS FOR JOURNAL PUBLICATION")
    print(f"{'='*80}")
    print(f"\nBest Model (by PR-AUC): {best_model['model']}")
    print(f"\n  Classification Metrics:")
    print(f"    Sensitivity (Recall):     {best_model['recall']:.1%}")
    print(f"    Specificity:              {best_model['specificity']:.1%}")
    print(f"    PPV (Precision):          {best_model['precision']:.1%}")
    print(f"    NPV:                      {best_model['npv']:.1%}")
    print(f"    F1 Score:                 {best_model['f1_score']:.3f}")
    print(f"    ROC-AUC:                  {best_model['roc_auc']:.3f}")
    print(f"    PR-AUC:                   {best_model['pr_auc']:.3f}")
    print(f"\n  Agreement Metrics:")
    print(f"    Matthews Correlation:     {best_model['matthews_corrcoef']:.3f}")
    print(f"    Cohen's Kappa:            {best_model['cohens_kappa']:.3f}")
    print(f"    Youden's J:               {best_model['youdens_j']:.3f}")
    print(f"\n  Clinical Utility:")
    print(f"    LR+ (Positive):           {best_model['lr_positive']:.2f}")
    print(f"    LR- (Negative):           {best_model['lr_negative']:.2f}")
    print(f"    Diagnostic Odds Ratio:    {best_model['diagnostic_odds_ratio']:.2f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives:           {best_model['tp']}")
    print(f"    False Positives:          {best_model['fp']}")
    print(f"    True Negatives:           {best_model['tn']}")
    print(f"    False Negatives:          {best_model['fn']}")
    print(f"\n  Model Calibration:")
    print(f"    Calibration Error:        {best_model['calibration_error']:.4f}")
    print(f"    Brier Score:              {best_model['brier_score']:.4f}")
    
    print(f"\n{'='*80}")
    print("ALL OUTPUTS SAVED TO:")
    print(f"{'='*80}")
    print(f"\nFigures ({fig_dir}):")
    print(f"  - Figure2_confusion_matrices_grid.png")
    print(f"  - Figure3_ROC_comparison.png")
    print(f"  - Figure4_PR_comparison.png")
    print(f"  - Figure5_feature_importance_*.png")
    print(f"  - Figure6_metrics_comparison.png")
    print(f"  - calibration_curve_*.png (per model)")
    
    print(f"\nTables ({tab_dir}):")
    print(f"\n  Dataset Characterization:")
    print(f"    - dataset_overall_statistics.csv")
    print(f"    - dataset_outcome_distribution.csv")
    print(f"    - dataset_continuous_features_analysis.csv")
    print(f"    - dataset_categorical_features_analysis.csv")
    print(f"    - dataset_correlation_matrix.csv")
    print(f"    - dataset_target_correlations.csv")
    print(f"    - dataset_summary_statistics.csv")
    print(f"    - data_split_distribution.csv")
    
    print(f"\n  Model Performance (Journal Tables):")
    print(f"    - Table1_primary_performance.csv")
    print(f"    - Table2_extended_metrics.csv")
    print(f"    - Table3_cross_validation_results.csv")
    print(f"    - Table4_confusion_matrix_statistics.csv")
    
    print(f"\n  Detailed Model Results (per model):")
    print(f"    - *_detailed_metrics.csv (train/val/test)")
    print(f"    - *_test_comprehensive.csv (all test metrics)")
    print(f"    - *_cross_validation.csv (CV statistics)")
    print(f"    - *_threshold_analysis.csv (threshold trade-offs)")
    
    print(f"\n  Comparison Files:")
    print(f"    - all_models_metrics_comparison.csv")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}\n")
# =========================================================================================================
if __name__ == "__main__":
    main()
# =========================================================================================================