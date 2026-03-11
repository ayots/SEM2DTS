"""
=============================================================================
Heart Disease Prediction Predictor
=============================================================================
Author  : Heart Disease DS Analysis
Version : 1.0.0
Dataset : UCI Heart Disease (920 records, 4 clinical centres)


Dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
=============================================================================
"""

# =============================================================================
# SECTION 1: STANDARD LIBRARY IMPORTS
# =============================================================================

import argparse       # For parsing command-line arguments
import json           # For saving the results summary as a JSON file
import logging        # For structured, timestamped log messages
import os             # For directory creation and file path operations
import sys            # For clean exit on fatal errors
import warnings       # For suppressing known non-critical library warnings
from pathlib import Path          # For cross-platform file path handling
from typing import Dict, Optional, Tuple  # For type hints in function signatures

# =============================================================================
# SECTION 2: THIRD-PARTY LIBRARY IMPORTS
# =============================================================================

# Use the non-interactive Agg backend so plots save to file without
# requiring a display -- works on headless servers and CI environments
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt   # For all chart and plot creation
import numpy as np                # For numerical array operations
import pandas as pd               # For tabular data loading and manipulation
import seaborn as sns             # For enhanced statistical visualisations

# Clustering algorithms
from sklearn.cluster import DBSCAN, KMeans

# Dimensionality reduction for 2D cluster visualisation
from sklearn.decomposition import PCA

# Supervised learning classifiers
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Evaluation metrics for classification and clustering
from sklearn.metrics import (
    accuracy_score,       # Fraction of correct predictions
    adjusted_rand_score,  # Clustering similarity to ground truth labels
    confusion_matrix,     # True/False positive/negative breakdown
    f1_score,             # Harmonic mean of precision and recall
    precision_score,      # Positive predictive value
    recall_score,         # Sensitivity / true positive rate
    roc_auc_score,        # Area under the ROC curve
    roc_curve,            # False/True positive rate pairs for ROC plotting
    silhouette_score,     # Cluster cohesion and separation measure
)

# Stratified train/test split utility
from sklearn.model_selection import train_test_split

# XGBoost gradient-boosted classifier
from xgboost import XGBClassifier

# Suppress deprecation and convergence warnings from libraries
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 3: GLOBAL CONFIGURATION
# =============================================================================
# All tunable parameters are centralised here.
# Edit this dictionary instead of hunting through the code for magic numbers.

CONFIG = {
    # --- File paths ---
    # Path to the raw UCI input CSV file
    "input_csv":    "heart_disease_uci.csv",
    # Path where the cleaned, model-ready CSV will be saved
    "cleaned_csv":  "heart_disease_cleaned.csv",
    # Directory where all output plots and reports will be written
    "output_dir":   "outputs",

    # --- Train/test split ---
    # Proportion of data reserved for model testing (0.2 = 20%)
    "test_size":      0.20,
    # Random seed for reproducibility across all stochastic steps
    "random_state":   42,

    # --- Random Forest hyperparameters ---
    "rf_n_estimators": 100,    # Number of trees in the forest

    # --- Gradient Boosting hyperparameters ---
    "gb_n_estimators":  100,   # Number of boosting stages
    "gb_learning_rate": 0.1,   # Shrinkage factor applied to each tree

    # --- XGBoost hyperparameters ---
    "xgb_n_estimators":  100,  # Number of boosting rounds
    "xgb_learning_rate": 0.1,  # Step size shrinkage to prevent overfitting

    # --- K-Means hyperparameters ---
    # Number of clusters (2 = disease / no disease)
    "kmeans_n_clusters": 2,

    # --- DBSCAN hyperparameters ---
    # Maximum distance between two points to be considered neighbours.
    # Set to 3.0 to accommodate the raw (unscaled) feature magnitudes
    # in this dataset (e.g. cholesterol ~200, blood pressure ~130).
    "dbscan_eps":         3.0,
    # Minimum number of points required to form a dense region
    "dbscan_min_samples": 5,

    # --- Categorical feature encoding maps ---
    # Chest pain type: ordinal encoding from least to most severe
    "cp_map": {
        "typical angina":   0,
        "atypical angina":  1,
        "non-anginal":      2,
        "asymptomatic":     3,
    },
    # Resting ECG results
    "restecg_map": {
        "normal":           0,
        "st-t abnormality": 1,
        "lv hypertrophy":   2,
    },
    # ST slope during peak exercise
    "slope_map": {
        "upsloping":   0,
        "flat":        1,
        "downsloping": 2,
    },
    # Thalassemia type
    "thal_map": {
        "normal":            0,
        "fixed defect":      1,
        "reversable defect": 2,
    },

    # --- Columns that carry no predictive signal ---
    # 'id' is a row identifier; 'dataset' is the source clinic name
    "drop_cols": ["id", "dataset"],
}


# =============================================================================
# SECTION 4: LOGGING SETUP
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure and return a module-level logger.

    Produces timestamped messages at the specified severity level
    so the user can follow pipeline progress in the terminal.

    Parameters
    ----------
    level : str
        Logging level string, e.g. "INFO", "DEBUG", "WARNING".

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logging.basicConfig(
        # Convert the string level name to the numeric constant
        level=getattr(logging, level.upper(), logging.INFO),
        # Format: timestamp  [LEVEL]  message
        format="%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# Instantiate the global logger used throughout the module
logger = setup_logging()


# =============================================================================
# SECTION 5: DATA INGESTION AND VALIDATION
# =============================================================================

# The set of column names that must be present in the raw input CSV
REQUIRED_COLUMNS = {
    "id", "age", "sex", "dataset", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalch", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
}


def load_and_validate(path: str) -> pd.DataFrame:
    """
    Load the raw UCI heart disease CSV and validate its structure.

    Checks that:
      - The file exists at the given path.
      - The loaded dataframe is not empty.
      - All required columns are present.

    Parameters
    ----------
    path : str
        Filesystem path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Validated raw dataframe ready for quality assessment.

    Raises
    ------
    FileNotFoundError
        If no file exists at the given path.
    ValueError
        If the dataframe is empty or required columns are absent.
    """
    # Guard: confirm the file actually exists before attempting to read it
    if not Path(path).exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    logger.info("Loading dataset from '%s'", path)
    df = pd.read_csv(path)

    # Guard: reject an empty dataframe early rather than failing later
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")

    # Guard: identify any columns the downstream pipeline depends on
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Log a summary so the user can confirm the correct file was loaded
    logger.info("Dataset loaded successfully -- shape: %s", df.shape)
    logger.info(
        "Target ('num') raw distribution:\n%s",
        df["num"].value_counts().sort_index().to_string(),
    )
    return df


# =============================================================================
# SECTION 6: DATA QUALITY ASSESSMENT
# =============================================================================

def assess_data_quality(df: pd.DataFrame, output_dir: str) -> None:
    """
    Log and visualise three dimensions of data quality:
      1. Missing values  -- per-column count and percentage heatmap.
      2. Duplicate rows  -- total count logged.
      3. Outliers        -- IQR-based count per numeric column plus boxplots.

    Plots are saved to disk rather than displayed interactively so the
    function works safely in non-GUI environments.

    Parameters
    ----------
    df : pd.DataFrame
        The raw (or lightly processed) dataframe to assess.
    output_dir : str
        Directory where PNG plot files will be written.
    """
    # Ensure the output directory exists before attempting to write files
    os.makedirs(output_dir, exist_ok=True)

    # --- Missing values ---
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    # Only log columns that actually have missing data to keep output concise
    logger.info(
        "Missing value percentages (columns with gaps only):\n%s",
        missing_pct[missing_pct > 0].to_string(),
    )

    # Heatmap: bright = present, dark = missing (viridis palette)
    _save_figure(
        plot_fn=lambda: sns.heatmap(df.isnull(), cbar=False, cmap="viridis"),
        title="Missing Value Heatmap",
        path=Path(output_dir) / "missing_values.png",
    )

    # --- Duplicate rows ---
    n_dupes = df.duplicated().sum()
    logger.info("Duplicate rows found: %d", n_dupes)

    # --- Outliers via the IQR fence method ---
    # A value is flagged as an outlier if it falls more than 1.5 * IQR
    # below Q1 or above Q3 (Tukey's fences, standard convention)
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_counts: Dict[str, int] = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outlier_counts[col] = int(
            ((df[col] < lower_fence) | (df[col] > upper_fence)).sum()
        )
    logger.info("Outlier counts per numeric column (IQR method): %s", outlier_counts)

    # Boxplot: shows spread, median, quartiles, and individual outlier points
    _save_figure(
        plot_fn=lambda: df[numeric_cols].boxplot(rot=45),
        title="Boxplot of Numeric Features",
        path=Path(output_dir) / "boxplots.png",
    )


# =============================================================================
# SECTION 7: PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================

def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Transform the raw UCI dataframe into a clean, fully numeric dataset
    suitable for machine learning.

    Processing steps (in order):
      1. Drop non-predictive identifier and source columns.
      2. Create a binary target: 0 = no disease, 1 = disease present.
      3. Encode the sex column as a numeric binary (1 = Male, 0 = Female).
      4. Map boolean-like fbs and exang columns to 0.0 / 1.0 floats.
      5. Ordinally encode chest pain type, ECG results, slope, and thal.
      6. Remove exact duplicate rows.
      7. Impute remaining missing values with column medians.
      8. Force all columns to numeric; drop rows where coercion still fails.

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated dataframe from load_and_validate().
    cfg : dict
        Global CONFIG dictionary containing encoding maps and drop_cols.

    Returns
    -------
    pd.DataFrame
        Cleaned, fully numeric dataframe with a binary 'target' column.
    """
    logger.info("Starting preprocessing...")

    # Step 1: Drop columns that carry no predictive signal.
    # 'id' is a meaningless row counter; 'dataset' is the source clinic name.
    # Using errors='ignore' prevents crashes if a column is already absent.
    cols_to_drop = [c for c in cfg["drop_cols"] if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    logger.info("Dropped non-predictive columns: %s", cols_to_drop)

    # Step 2: Binarise the target variable.
    # The raw 'num' column contains severity grades 0-4.
    # For binary classification: 0 = healthy, 1 = any degree of disease.
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    # Step 3: Encode sex as a binary numeric feature.
    # 1.0 = Male, 0.0 = Female -- sklearn requires numeric input.
    df["sex"] = (df["sex"] == "Male").astype(float)

    # Step 4: Map fasting blood sugar and exercise-induced angina to floats.
    # These arrive as Python booleans or bool-like objects after CSV parsing.
    bool_map = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
    df["fbs"]   = df["fbs"].map(bool_map).astype(float)
    df["exang"] = df["exang"].map(bool_map).astype(float)

    # Step 5: Ordinally encode multi-category string columns.
    # Each map converts a clinical category string to an integer code.
    # Values not in the map become NaN and are handled in step 7.
    df["cp"]      = df["cp"].map(cfg["cp_map"])
    df["restecg"] = df["restecg"].map(cfg["restecg_map"])
    df["slope"]   = df["slope"].map(cfg["slope_map"])
    df["thal"]    = df["thal"].map(cfg["thal_map"])

    # Step 6: Remove exact duplicate rows.
    # Duplicates would bias evaluation by leaking test samples into training.
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows.", n_before - len(df))

    # Step 7: Impute missing values with column medians.
    # Median is preferred over mean because it is robust to the outliers
    # identified in the assessment step (e.g. extreme cholesterol values).
    df = df.fillna(df.median(numeric_only=True))

    # Step 8: Final safety pass -- coerce every column to numeric.
    # Any value that cannot be converted becomes NaN, then those rows are dropped.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rows_before = len(df)
    df = df.dropna()
    if len(df) < rows_before:
        logger.warning(
            "Dropped %d rows where numeric coercion failed.",
            rows_before - len(df),
        )

    logger.info("Preprocessing complete -- final shape: %s", df.shape)
    logger.info(
        "Binary target distribution:\n%s",
        df["target"].value_counts().to_string(),
    )
    return df


def save_cleaned_data(df: pd.DataFrame, path: str) -> None:
    """
    Persist the cleaned dataframe to a CSV file for auditability.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, model-ready dataframe.
    path : str
        Output file path for the CSV.
    """
    df.to_csv(path, index=False)
    logger.info("Cleaned dataset saved to '%s'.", path)


# =============================================================================
# SECTION 8: SUPERVISED MODEL DEFINITIONS
# =============================================================================

def build_supervised_models(cfg: dict) -> Dict:
    """
    Instantiate all supervised classifiers with hyperparameters from CONFIG.

    Keeping model construction in one place makes it easy to add, remove,
    or swap classifiers without touching the training and evaluation loop.

    Parameters
    ----------
    cfg : dict
        Global CONFIG dictionary.

    Returns
    -------
    dict
        Mapping of model name (str) to an unfitted scikit-learn estimator.
    """
    return {
        # Ensemble of independent decision trees using bootstrap aggregation.
        # Each tree votes; the majority class is the final prediction.
        "Random Forest": RandomForestClassifier(
            n_estimators=cfg["rf_n_estimators"],
            random_state=cfg["random_state"],
        ),
        # Sequential ensemble where each new tree corrects the errors of all
        # previous trees. Generally more accurate but slower to train.
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=cfg["gb_n_estimators"],
            learning_rate=cfg["gb_learning_rate"],
            random_state=cfg["random_state"],
        ),
        # Optimised gradient boosting with built-in regularisation,
        # parallel processing, and efficient handling of sparse data.
        # eval_metric is set explicitly to silence a deprecation warning.
        "XGBoost": XGBClassifier(
            n_estimators=cfg["xgb_n_estimators"],
            learning_rate=cfg["xgb_learning_rate"],
            eval_metric="logloss",
            random_state=cfg["random_state"],
        ),
    }


# =============================================================================
# SECTION 9: SUPERVISED MODEL TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate_supervised(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: Dict,
    output_dir: str,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Train each classifier, compute five evaluation metrics, and save plots.

    For each model this function:
      - Fits only on training data to prevent data leakage.
      - Generates hard class predictions and soft probability scores.
      - Computes Accuracy, Precision, Recall, F1-score, and ROC-AUC.
      - Saves a confusion matrix heatmap and a ROC curve plot.
      - Extracts and saves feature importances for tree-based models.

    A grouped bar chart comparing all metrics across all models is also saved.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices for training and testing respectively.
    y_train, y_test : pd.Series
        Binary target labels for training and testing.
    models : dict
        Name-to-estimator mapping from build_supervised_models().
    output_dir : str
        Directory where all plot files will be saved.

    Returns
    -------
    results_df : pd.DataFrame
        One row per model containing all five evaluation metrics.
    feature_importances : dict
        Model name mapped to a pd.Series of sorted feature importance scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []                              # Collects metric dicts for all models
    feature_importances: Dict[str, pd.Series] = {}  # Tree-model importances only

    for name, model in models.items():
        logger.info("Training %s...", name)

        # Fit the model exclusively on the training partition
        model.fit(X_train, y_train)

        # Hard predictions: outputs a 0 or 1 for each test sample
        y_pred = model.predict(X_test)

        # Soft probability scores: probability of the positive class (disease)
        # Used for ROC-AUC which requires a continuous confidence value
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute all five evaluation metrics; round to 4 decimal places
        metrics = {
            "Model":     name,
            # Accuracy: overall fraction of correct predictions
            "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
            # Precision: of all predicted positives, how many are truly positive
            "Precision": round(precision_score(y_test, y_pred), 4),
            # Recall: of all actual positives, how many were correctly identified
            "Recall":    round(recall_score(y_test, y_pred),    4),
            # F1-score: harmonic mean of precision and recall
            "F1-score":  round(f1_score(y_test, y_pred),        4),
            # ROC-AUC: area under the receiver operating characteristic curve
            "ROC-AUC":   round(roc_auc_score(y_test, y_prob),   4),
        }
        results.append(metrics)
        # Log without the 'Model' key to keep the terminal line compact
        logger.info(
            "%s results -- %s",
            name,
            {k: v for k, v in metrics.items() if k != "Model"},
        )

        # --- Confusion matrix heatmap ---
        # Layout: rows = actual class, columns = predicted class.
        # Top-left = true negatives, bottom-right = true positives.
        # Off-diagonal cells represent misclassification errors.
        cm = confusion_matrix(y_test, y_pred)
        _save_figure(
            plot_fn=lambda cm=cm: sns.heatmap(
                cm,
                annot=True,           # Print the count in each cell
                fmt="d",              # Integer formatting (no decimals)
                cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
            ),
            title=f"Confusion Matrix -- {name}",
            path=Path(output_dir) / f"{name.replace(' ', '_')}_confusion_matrix.png",
            xlabel="Predicted Label",
            ylabel="Actual Label",
        )

        # --- ROC curve ---
        # Plots the true positive rate against the false positive rate at every
        # possible classification threshold. A diagonal line represents a random
        # classifier; curves bowed toward the top-left corner are better.
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = metrics["ROC-AUC"]
        _save_figure(
            plot_fn=lambda fpr=fpr, tpr=tpr, name=name, auc_val=auc_val: (
                plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.2f})"),
                plt.plot(
                    [0, 1], [0, 1],
                    linestyle="--",
                    color="grey",
                    label="Random classifier (AUC = 0.50)",
                ),
                plt.legend(loc="lower right"),
            ),
            title=f"ROC Curve -- {name}",
            path=Path(output_dir) / f"{name.replace(' ', '_')}_roc_curve.png",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )

        # --- Feature importances (tree-based models only) ---
        # feature_importances_ contains the mean decrease in impurity (Gini)
        # contributed by each feature across all trees in the ensemble.
        # Higher values indicate greater predictive relevance.
        if hasattr(model, "feature_importances_"):
            feature_importances[name] = pd.Series(
                model.feature_importances_,
                index=X_train.columns,
            ).sort_values(ascending=False)

    # Compile all metric rows into a single DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    logger.info("\nFull supervised results table:\n%s", results_df.to_string(index=False))

    # Save a grouped bar chart comparing all metrics side-by-side across models
    _plot_metric_comparison(results_df, output_dir)

    # Save one horizontal bar chart per model showing each feature's importance
    for name, importance in feature_importances.items():
        _save_figure(
            plot_fn=lambda imp=importance: imp.plot(kind="bar", color="steelblue"),
            title=f"Feature Importance -- {name}",
            path=Path(output_dir) / f"{name.replace(' ', '_')}_feature_importance.png",
            xlabel="Feature",
            ylabel="Importance Score",
        )

    return results_df, feature_importances


def _plot_metric_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Produce a grouped bar chart with one cluster of bars per model.
    Each bar within a cluster represents a different evaluation metric.

    This gives a quick visual summary of which model performs best overall
    and highlights any trade-offs between metrics (e.g. high recall vs precision).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of train_and_evaluate_supervised() containing metric columns.
    output_dir : str
        Directory where the PNG chart will be saved.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    n_models = len(results_df)
    x = np.arange(n_models)   # One integer tick position per model
    bar_width = 0.15           # Width of each individual metric bar

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw one set of bars per metric, offset horizontally to avoid overlap
    for i, metric in enumerate(metrics):
        ax.bar(x + i * bar_width, results_df[metric], width=bar_width, label=metric)

    # Centre tick labels beneath each model's group of bars
    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(results_df["Model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)   # All metrics are proportions bounded in [0, 1]
    ax.set_title("Supervised Model Metric Comparison")
    ax.legend(loc="lower right")

    plt.tight_layout()
    save_path = Path(output_dir) / "supervised_model_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Metric comparison chart saved to '%s'.", save_path)


# =============================================================================
# SECTION 10: UNSUPERVISED CLUSTERING
# =============================================================================

def run_clustering(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    output_dir: str,
) -> Dict:
    """
    Apply K-Means and DBSCAN to discover natural groupings in the feature space,
    then evaluate how well those groupings align with the true disease labels.

    Note: clustering is fully unsupervised during fitting.
    The true labels (y) are used only for post-hoc evaluation via the
    Adjusted Rand Index -- they are never passed to fit() or fit_predict().

    Outputs saved to disk:
      - PCA scatter plot coloured by K-Means cluster assignment
      - PCA scatter plot coloured by DBSCAN cluster assignment
      - Bar chart of silhouette scores for both algorithms
      - Pie chart of K-Means cluster size distribution

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (all rows; no train/test split needed here).
    y : pd.Series
        True binary labels, used only for ARI computation after clustering.
    cfg : dict
        Global CONFIG dictionary.
    output_dir : str
        Directory for saved plots.

    Returns
    -------
    dict
        Clustering evaluation metrics and labels for both algorithms.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Running unsupervised clustering...")

    # Reduce the feature space to 2 dimensions using PCA.
    # This is done ONLY for visualisation -- clustering itself uses all features.
    # PCA finds the two orthogonal directions of maximum variance in the data.
    pca = PCA(n_components=2, random_state=cfg["random_state"])
    X_pca = pca.fit_transform(X)
    logger.info(
        "PCA variance explained by 2 components: %.2f%%",
        pca.explained_variance_ratio_.sum() * 100,
    )

    # --- K-Means clustering ---
    # Iteratively assigns each point to the nearest centroid and updates centroids.
    # n_clusters=2 mirrors the expected binary grouping of disease vs no disease.
    kmeans = KMeans(
        n_clusters=cfg["kmeans_n_clusters"],
        random_state=cfg["random_state"],
    )
    km_labels = kmeans.fit_predict(X)   # Returns integer cluster label per row

    # --- DBSCAN clustering ---
    # Identifies clusters as dense regions of points separated by low-density areas.
    # Does not require specifying the number of clusters in advance.
    # Points in sparse regions receive the label -1 (treated as noise).
    dbscan = DBSCAN(
        eps=cfg["dbscan_eps"],           # Neighbourhood radius
        min_samples=cfg["dbscan_min_samples"],  # Minimum points to form a cluster
    )
    db_labels = dbscan.fit_predict(X)

    # Count the number of real clusters DBSCAN found (exclude noise label -1)
    n_db_clusters = len(set(db_labels) - {-1})
    n_noise = int((db_labels == -1).sum())
    logger.info(
        "DBSCAN result: %d cluster(s) found, %d noise points.",
        n_db_clusters, n_noise,
    )

    # --- PCA scatter plots ---
    # Each dot represents one patient; colour = assigned cluster
    _scatter_pca(
        X_pca, km_labels,
        title="K-Means Clustering (PCA 2D Projection)",
        path=Path(output_dir) / "kmeans_pca.png",
    )
    _scatter_pca(
        X_pca, db_labels,
        title="DBSCAN Clustering (PCA 2D Projection)",
        path=Path(output_dir) / "dbscan_pca.png",
    )

    # --- Adjusted Rand Index (ARI) ---
    # Compares cluster assignments to the true labels corrected for chance.
    # ARI = 1.0 means perfect agreement; ~0.0 means random assignment.
    ari_km = adjusted_rand_score(y, km_labels)
    ari_db = adjusted_rand_score(y, db_labels)

    # --- Silhouette score ---
    # Measures how similar each point is to its own cluster vs other clusters.
    # Range: -1 (wrong cluster) to +1 (perfectly separated clusters).
    # DBSCAN silhouette is only meaningful when more than one cluster was found.
    sil_km = silhouette_score(X, km_labels)
    sil_db = silhouette_score(X, db_labels) if n_db_clusters > 1 else 0.0

    logger.info("K-Means  -- ARI: %.4f | Silhouette: %.4f", ari_km, sil_km)
    logger.info(
        "DBSCAN   -- ARI: %.4f | Silhouette: %.4f | Clusters found: %d",
        ari_db, sil_db, n_db_clusters,
    )

    # --- Silhouette score bar chart ---
    # Allows direct visual comparison of cluster quality between algorithms
    _save_figure(
        plot_fn=lambda: sns.barplot(
            x=["K-Means", "DBSCAN"],
            y=[sil_km, sil_db],
            palette="pastel",
        ),
        title="Clustering Silhouette Scores",
        path=Path(output_dir) / "clustering_silhouette_scores.png",
        ylabel="Silhouette Score",
    )

    # --- K-Means cluster distribution pie chart ---
    # Shows how evenly patients are distributed across the discovered clusters
    cluster_counts = pd.Series(km_labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    cluster_counts.plot(
        kind="pie",
        autopct="%1.1f%%",    # Display percentage inside each slice
        startangle=90,         # Start from the top of the chart
        ax=ax,
        labels=[f"Cluster {i}" for i in cluster_counts.index],
    )
    ax.set_title("K-Means Cluster Size Distribution")
    ax.set_ylabel("")   # Remove the default 'None' y-label pandas adds
    plt.tight_layout()
    pie_path = Path(output_dir) / "kmeans_cluster_distribution.png"
    plt.savefig(pie_path, dpi=150)
    plt.close(fig)
    logger.info("Cluster distribution pie chart saved to '%s'.", pie_path)

    # Return all metrics in a flat dictionary for the reporting step
    return {
        "ari_kmeans":           round(ari_km, 4),
        "ari_dbscan":           round(ari_db, 4),
        "silhouette_kmeans":    round(sil_km, 4),
        "silhouette_dbscan":    round(sil_db, 4),
        "dbscan_clusters":      n_db_clusters,
        "dbscan_noise_points":  n_noise,
        "kmeans_labels":        km_labels.tolist(),   # Stored for downstream use
        "dbscan_labels":        db_labels.tolist(),
    }


def _scatter_pca(
    X_pca: np.ndarray,
    labels,
    title: str,
    path: Path,
) -> None:
    """
    Save a 2D scatter plot of PCA-projected data coloured by cluster label.

    Parameters
    ----------
    X_pca : np.ndarray
        Array of shape (n_samples, 2) produced by PCA.fit_transform().
    labels : array-like
        Integer cluster label for each sample; -1 = noise (DBSCAN).
    title : str
        Chart title displayed above the plot.
    path : Path
        Destination file path for the PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use a qualitative colour palette with enough distinct colours for all labels
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("Set1", n_colors=len(unique_labels))

    sns.scatterplot(
        x=X_pca[:, 0],   # First principal component (x-axis)
        y=X_pca[:, 1],   # Second principal component (y-axis)
        hue=labels,
        palette=palette,
        ax=ax,
        s=40,           # Marker size in points squared
        alpha=0.7,      # Transparency to reveal density in overlapping regions
    )
    ax.set_title(title)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("PCA scatter plot saved to '%s'.", path)


# =============================================================================
# SECTION 11: FINAL EVALUATION AND REPORTING
# =============================================================================

def evaluate_and_report(
    results_df: pd.DataFrame,
    clustering_metrics: Dict,
    output_dir: str,
) -> Dict:
    """
    Select the best model from each paradigm, print a formatted summary
    to the terminal, and write a machine-readable JSON report to disk.

    Selection criteria:
      - Best supervised  : highest F1-score (balances precision and recall).
      - Best unsupervised: highest Adjusted Rand Index.
      - Overall winner   : whichever achieves the higher score on its metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        Supervised evaluation metrics from train_and_evaluate_supervised().
    clustering_metrics : dict
        Clustering metrics from run_clustering().
    output_dir : str
        Directory where the JSON report will be written.

    Returns
    -------
    dict
        Complete results summary suitable for downstream use or logging.
    """
    # Identify the supervised model with the highest F1-score
    best_sup = results_df.sort_values("F1-score", ascending=False).iloc[0]

    # Identify the unsupervised model with the highest ARI
    best_unsup = (
        "K-Means"
        if clustering_metrics["ari_kmeans"] >= clustering_metrics["ari_dbscan"]
        else "DBSCAN"
    )
    best_ari = max(clustering_metrics["ari_kmeans"], clustering_metrics["ari_dbscan"])

    # Compare the best supervised F1 against the best unsupervised ARI
    # to determine which approach should be recommended overall
    if best_sup["F1-score"] > best_ari:
        overall_winner = best_sup["Model"]
        winner_reason  = "best predictive accuracy and clinical interpretability."
    else:
        overall_winner = best_unsup
        winner_reason  = "best clustering alignment with true labels."

    # Build the full summary dictionary.
    # Label arrays are excluded to keep the JSON file a manageable size.
    summary = {
        "best_supervised_model":   best_sup["Model"],
        "best_supervised_f1":      best_sup["F1-score"],
        "best_supervised_roc_auc": best_sup["ROC-AUC"],
        "best_unsupervised_model": best_unsup,
        "best_unsupervised_ari":   best_ari,
        "overall_winner":          overall_winner,
        "winner_reason":           winner_reason,
        # Full table of all supervised model results
        "supervised_results":      results_df.to_dict(orient="records"),
        # Clustering metrics minus the large label arrays
        "clustering_metrics": {
            k: v for k, v in clustering_metrics.items()
            if k not in ("kmeans_labels", "dbscan_labels")
        },
    }

    # Print a human-readable summary using ASCII-safe characters only
    divider = "-" * 60
    print(f"\n{divider}")
    print("  HEART DISEASE PIPELINE -- RESULTS SUMMARY")
    print(divider)

    print("\nSupervised Model Performance:")
    print(results_df.to_string(index=False))

    print(f"\nBest Supervised  : {best_sup['Model']}  "
          f"(F1={best_sup['F1-score']:.4f} | ROC-AUC={best_sup['ROC-AUC']:.4f})")

    print("\nClustering Results:")
    print(f"  K-Means  -- ARI={clustering_metrics['ari_kmeans']:.4f} | "
          f"Silhouette={clustering_metrics['silhouette_kmeans']:.4f}")
    print(f"  DBSCAN   -- ARI={clustering_metrics['ari_dbscan']:.4f} | "
          f"Silhouette={clustering_metrics['silhouette_dbscan']:.4f} | "
          f"Clusters={clustering_metrics['dbscan_clusters']}")

    print(f"\nBest Unsupervised : {best_unsup}  (ARI={best_ari:.4f})")
    print(f"\nOverall Winner    : {overall_winner} -- {winner_reason}")
    print(divider + "\n")

    # Persist the full results dictionary as a JSON file for reproducibility
    report_path = Path(output_dir) / "results_summary.json"
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Results summary saved to '%s'.", report_path)

    return summary


# =============================================================================
# SECTION 12: SHARED PLOTTING UTILITY
# =============================================================================

def _save_figure(
    plot_fn,
    title: str,
    path: Path,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Safely create, annotate, and save a matplotlib figure to disk.

    Wraps the drawing call in a try/except so that a single visualisation
    failure does not abort the entire pipeline. Any error is logged as a
    warning and execution continues with the next step.

    The leading underscore in the name signals that this is an internal
    helper not intended to be called by users directly.

    Parameters
    ----------
    plot_fn : callable
        Zero-argument function that draws onto the current active axes.
        Lambda-with-capture is used throughout the pipeline to defer drawing.
    title : str
        Title string displayed above the chart.
    path : Path
        Destination PNG file path.
    xlabel : str, optional
        Label for the x-axis. Omitted if None.
    ylabel : str, optional
        Label for the y-axis. Omitted if None.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Make ax the active axes so plt.plot() calls inside plot_fn draw here
        plt.sca(ax)
        plot_fn()
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Plot saved: %s", path)
    except Exception as exc:
        # Log the failure but do not re-raise so the pipeline can continue
        logger.warning("Could not save figure '%s': %s", path, exc)
        plt.close("all")   # Release any partially constructed figure objects


# =============================================================================
# SECTION 13: COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    All parameters default to values in CONFIG, so the script can be run
    with zero arguments when the defaults are appropriate. Any flag passed
    on the command line overrides the corresponding CONFIG entry.

    Run with --help to see all available options and their defaults.

    Returns
    -------
    argparse.Namespace
        Parsed argument object with attributes matching each flag name.
    """
    parser = argparse.ArgumentParser(
        description="Heart Disease Prediction Pipeline",
        # Displays default values next to each argument in --help output
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Path to the input data file
    parser.add_argument(
        "--input",
        default=CONFIG["input_csv"],
        help="Path to the raw UCI heart disease CSV file.",
    )
    # Directory for all outputs (created if it does not already exist)
    parser.add_argument(
        "--output_dir",
        default=CONFIG["output_dir"],
        help="Directory where all plots and reports will be written.",
    )
    # Fraction of data held out for evaluating trained models
    parser.add_argument(
        "--test_size",
        type=float,
        default=CONFIG["test_size"],
        help="Proportion of data to hold out for testing (e.g. 0.2 = 20 pct).",
    )
    # Global random seed controlling all stochastic operations
    parser.add_argument(
        "--random_state",
        type=int,
        default=CONFIG["random_state"],
        help="Random seed for reproducibility across all stochastic steps.",
    )
    # Flag to bypass the assessment step when re-running quickly
    parser.add_argument(
        "--skip_assess",
        action="store_true",
        help="If set, skip data quality assessment to save time on re-runs.",
    )

    return parser.parse_args()


# =============================================================================
# SECTION 14: MAIN ORCHESTRATION
# =============================================================================

def main() -> None:
    """
    Orchestrate all pipeline stages in the correct sequence.

    Stage order:
      1. Parse CLI arguments and apply any overrides to CONFIG.
      2. Load and validate the raw CSV file.
      3. Assess data quality (missing values, duplicates, outliers).
      4. Preprocess and encode all features; save the cleaned CSV.
      5. Perform a stratified train/test split.
      6. Train and evaluate all three supervised classifiers.
      7. Run K-Means and DBSCAN unsupervised clustering.
      8. Print and save the final results summary.

    Any fatal error (e.g. file not found, corrupted data) logs a clear
    message and exits with status code 1 so callers can detect failure.
    """
    # Stage 1: Parse command-line arguments and update CONFIG
    args = parse_args()
    CONFIG["input_csv"]    = args.input
    CONFIG["output_dir"]   = args.output_dir
    CONFIG["test_size"]    = args.test_size
    CONFIG["random_state"] = args.random_state

    # Create the top-level output directory immediately so all subsequent
    # steps can write plots and reports without checking again
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Stage 2: Load and validate the raw input CSV
    try:
        raw_df = load_and_validate(CONFIG["input_csv"])
    except (FileNotFoundError, ValueError) as exc:
        # Log descriptively and exit cleanly; avoid a raw Python traceback
        logger.error("Data loading failed: %s", exc)
        sys.exit(1)

    # Stage 3: Data quality assessment
    # Can be skipped with --skip_assess when iterating quickly during development
    if not args.skip_assess:
        assess_data_quality(raw_df, CONFIG["output_dir"])

    # Stage 4: Preprocessing and feature engineering
    try:
        clean_df = preprocess(raw_df, CONFIG)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)

    # Persist the cleaned data so it can be inspected or reused independently
    save_cleaned_data(clean_df, CONFIG["cleaned_csv"])

    # Separate predictive features from the binary target label
    X = clean_df.drop("target", axis=1)   # All feature columns
    y = clean_df["target"].astype(int)    # Binary disease label (0 or 1)

    # Stage 5: Stratified train/test split
    # stratify=y preserves the class ratio in both subsets -- important
    # when classes are imbalanced, as is common in medical datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,
    )
    logger.info(
        "Train/test split complete -- train: %d rows, test: %d rows",
        len(X_train), len(X_test),
    )

    # Stage 6: Supervised learning
    # Build models first so they are easy to inspect or extend
    supervised_models = build_supervised_models(CONFIG)
    try:
        results_df, _ = train_and_evaluate_supervised(
            X_train, X_test, y_train, y_test,
            supervised_models,
            CONFIG["output_dir"],
        )
    except Exception as exc:
        logger.error("Supervised modelling failed: %s", exc)
        sys.exit(1)

    # Stage 7: Unsupervised clustering
    # Uses the FULL dataset (not just the test split) because clustering
    # benefits from seeing the complete data distribution
    try:
        clustering_metrics = run_clustering(X, y, CONFIG, CONFIG["output_dir"])
    except Exception as exc:
        logger.error("Clustering failed: %s", exc)
        sys.exit(1)

    # Stage 8: Final evaluation and report
    evaluate_and_report(results_df, clustering_metrics, CONFIG["output_dir"])

    logger.info(
        "Pipeline complete. All outputs saved to '%s'.",
        CONFIG["output_dir"],
    )


# Run main() only when this file is executed directly,
# not when it is imported as a module by another script or test suite
if __name__ == "__main__":
    main()
