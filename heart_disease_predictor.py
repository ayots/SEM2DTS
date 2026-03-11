"""
=============================================================================
Heart Disease Prediction Pipeline
=============================================================================
Author  : Heart Disease ML Analysis
Version : 1.0.0
Dataset : UCI Heart Disease (920 records, 4 clinical centres)

Description:
    End-to-end modular pipeline for heart disease prediction covering:
      - Data ingestion and validation
      - Preprocessing and feature engineering
      - Supervised learning (Random Forest, Gradient Boosting, XGBoost)
      - Unsupervised clustering (K-Means, DBSCAN)
      - Evaluation, visualisation, and reporting

Usage:
    python heart_disease_pipeline.py --input heart_disease_uci.csv

Dependencies:
    pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# All tunable parameters live here — edit freely
CONFIG = {
    # Paths
    "input_csv":      "heart_disease_uci.csv",
    "cleaned_csv":    "heart_disease_cleaned.csv",
    "output_dir":     "outputs",
    "results_json":   "outputs/results_summary.json",

    # Modelling
    "test_size":      0.20,
    "random_state":   42,

    # Supervised model hyperparameters
    "rf_n_estimators":     100,
    "gb_n_estimators":     100,
    "gb_learning_rate":    0.1,
    "xgb_n_estimators":    100,
    "xgb_learning_rate":   0.1,

    # Clustering
    "kmeans_n_clusters":   2,
    "dbscan_eps":          3.0,
    "dbscan_min_samples":  5,

    # Feature encoding maps
    "cp_map": {
        "typical angina": 0,
        "atypical angina": 1,
        "non-anginal": 2,
        "asymptomatic": 3,
    },
    "restecg_map": {
        "normal": 0,
        "st-t abnormality": 1,
        "lv hypertrophy": 2,
    },
    "slope_map":  {"upsloping": 0, "flat": 1, "downsloping": 2},
    "thal_map":   {"normal": 0, "fixed defect": 1, "reversable defect": 2},

    # Columns to drop before modelling
    "drop_cols": ["id", "dataset"],
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Initialise a module-level logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA INGESTION & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {
    "id", "age", "sex", "dataset", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalch", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
}


def load_and_validate(path: str) -> pd.DataFrame:
    """
    Load the raw UCI heart disease CSV and validate its structure.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Validated raw dataframe.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing or the dataframe is empty.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    logger.info("Loading dataset from '%s'", path)
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    logger.info("Dataset loaded — shape: %s", df.shape)
    logger.info("Target ('num') distribution:\n%s", df["num"].value_counts().sort_index().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

def assess_data_quality(df: pd.DataFrame, output_dir: str) -> None:
    """
    Log and visualise data quality: missing values, duplicates, and outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or semi-processed dataframe.
    output_dir : str
        Directory where diagnostic plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    logger.info("Missing value percentages:\n%s",
                missing_pct[missing_pct > 0].to_string())

    # Missing value heatmap
    _save_figure(
        lambda: sns.heatmap(df.isnull(), cbar=False, cmap="viridis"),
        title="Missing Value Heatmap",
        path=Path(output_dir) / "missing_values.png",
    )

    # Duplicates
    n_dupes = df.duplicated().sum()
    logger.info("Duplicate rows: %d", n_dupes)

    # Outliers via IQR
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_counts: Dict[str, int] = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_counts[col] = int(
            ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        )
    logger.info("Outlier counts (IQR method): %s", outlier_counts)

    # Boxplot
    _save_figure(
        lambda: df[numeric_cols].boxplot(rot=45),
        title="Boxplot of Numeric Features",
        path=Path(output_dir) / "boxplots.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Clean and encode the raw dataset into a model-ready form.

    Steps:
      1. Drop irrelevant columns.
      2. Create binary target variable (num > 0 → 1).
      3. Encode boolean and categorical features.
      4. Remove duplicates.
      5. Impute remaining missing values with column medians.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    cfg : dict
        Global configuration dict.

    Returns
    -------
    pd.DataFrame
        Cleaned, fully numeric dataframe with 'target' column.
    """
    logger.info("Starting preprocessing…")

    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns], errors="ignore")

    # Binary target
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    # Boolean columns
    df["sex"]  = (df["sex"] == "Male").astype(float)
    df["fbs"]  = df["fbs"].map({True: 1.0, False: 0.0}).astype(float)
    df["exang"] = df["exang"].map({True: 1.0, False: 0.0}).astype(float)

    # Categorical ordinal encoding
    df["cp"]      = df["cp"].map(cfg["cp_map"])
    df["restecg"] = df["restecg"].map(cfg["restecg_map"])
    df["slope"]   = df["slope"].map(cfg["slope_map"])
    df["thal"]    = df["thal"].map(cfg["thal_map"])

    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows.", n_before - len(df))

    # Median imputation for remaining NaNs
    df = df.fillna(df.median(numeric_only=True))

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()  # Drop any rows where coercion still failed

    logger.info("Preprocessing complete — final shape: %s", df.shape)
    logger.info("Target distribution:\n%s", df["target"].value_counts().to_string())
    return df


def save_cleaned_data(df: pd.DataFrame, path: str) -> None:
    """Persist the cleaned dataframe to CSV."""
    df.to_csv(path, index=False)
    logger.info("Cleaned dataset saved to '%s'.", path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SUPERVISED MODELLING
# ─────────────────────────────────────────────────────────────────────────────

def build_supervised_models(cfg: dict) -> Dict:
    """
    Instantiate supervised classifiers using CONFIG hyperparameters.

    Returns
    -------
    dict
        Mapping of model name → unfitted estimator.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=cfg["rf_n_estimators"],
            random_state=cfg["random_state"],
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=cfg["gb_n_estimators"],
            learning_rate=cfg["gb_learning_rate"],
            random_state=cfg["random_state"],
        ),
        "XGBoost": XGBClassifier(
            n_estimators=cfg["xgb_n_estimators"],
            learning_rate=cfg["xgb_learning_rate"],
            eval_metric="logloss",
            random_state=cfg["random_state"],
        ),
    }


def train_and_evaluate_supervised(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: Dict,
    output_dir: str,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Train each supervised model, compute evaluation metrics, and produce plots.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Labels.
    models : dict
        Name → estimator mapping.
    output_dir : str
        Directory for saved plots.

    Returns
    -------
    results_df : pd.DataFrame
        One row per model with Accuracy / Precision / Recall / F1 / ROC-AUC.
    feature_importances : dict
        Name → pd.Series of feature importances (tree models only).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    feature_importances: Dict[str, pd.Series] = {}

    for name, model in models.items():
        logger.info("Training %s…", name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall":    round(recall_score(y_test, y_pred),    4),
            "F1-score":  round(f1_score(y_test, y_pred),        4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_prob),   4),
        }
        results.append(metrics)
        logger.info("%s — %s", name, {k: v for k, v in metrics.items() if k != "Model"})

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        _save_figure(
            lambda cm=cm, name=name: sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
            ),
            title=f"Confusion Matrix — {name}",
            path=Path(output_dir) / f"{name.replace(' ', '_')}_confusion_matrix.png",
            xlabel="Predicted", ylabel="Actual",
        )

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = metrics["ROC-AUC"]
        _save_figure(
            lambda fpr=fpr, tpr=tpr, name=name, auc_val=auc_val: (
                plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.2f})"),
                plt.plot([0, 1], [0, 1], linestyle="--", color="grey"),
                plt.legend(),
            ),
            title=f"ROC Curve — {name}",
            path=Path(output_dir) / f"{name.replace(' ', '_')}_roc_curve.png",
            xlabel="False Positive Rate", ylabel="True Positive Rate",
        )

        # Feature importances
        if hasattr(model, "feature_importances_"):
            feature_importances[name] = pd.Series(
                model.feature_importances_, index=X_train.columns
            ).sort_values(ascending=False)

    results_df = pd.DataFrame(results)
    logger.info("\nSupervised Results:\n%s", results_df.to_string(index=False))

    # Grouped bar chart — all metrics at once
    _plot_metric_comparison(results_df, output_dir)

    # Feature importance bars
    for name, importance in feature_importances.items():
        _save_figure(
            lambda imp=importance, name=name: imp.plot(kind="bar", color="steelblue"),
            title=f"Feature Importance — {name}",
            path=Path(output_dir) / f"{name.replace(' ', '_')}_feature_importance.png",
            xlabel="Feature", ylabel="Importance Score",
        )

    return results_df, feature_importances


def _plot_metric_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    """Grouped bar chart comparing all evaluation metrics across models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    n_models = len(results_df)
    x = np.arange(n_models)
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width=width, label=metric)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(results_df["Model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Supervised Model Metric Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "supervised_model_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved metric comparison chart.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. UNSUPERVISED CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    output_dir: str,
) -> Dict:
    """
    Apply K-Means and DBSCAN clustering, evaluate with ARI and Silhouette,
    and save PCA visualisations.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (all rows).
    y : pd.Series
        True labels (used only for ARI — not for fitting).
    cfg : dict
        Global configuration dict.
    output_dir : str
        Directory for saved plots.

    Returns
    -------
    dict
        Clustering metrics and labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Running unsupervised clustering…")

    # PCA for 2D visualisation
    X_pca = PCA(n_components=2, random_state=cfg["random_state"]).fit_transform(X)

    # K-Means
    kmeans = KMeans(
        n_clusters=cfg["kmeans_n_clusters"],
        random_state=cfg["random_state"],
    )
    km_labels = kmeans.fit_predict(X)

    # DBSCAN
    dbscan = DBSCAN(
        eps=cfg["dbscan_eps"],
        min_samples=cfg["dbscan_min_samples"],
    )
    db_labels = dbscan.fit_predict(X)
    n_db_clusters = len(set(db_labels) - {-1})
    n_noise = int((db_labels == -1).sum())
    logger.info("DBSCAN found %d clusters and %d noise points.", n_db_clusters, n_noise)

    # Visualisations
    _scatter_pca(X_pca, km_labels, "K-Means Clustering (PCA)", Path(output_dir) / "kmeans_pca.png")
    _scatter_pca(X_pca, db_labels, "DBSCAN Clustering (PCA)", Path(output_dir) / "dbscan_pca.png")

    # Metrics
    ari_km = adjusted_rand_score(y, km_labels)
    ari_db = adjusted_rand_score(y, db_labels)
    sil_km = silhouette_score(X, km_labels)
    sil_db = silhouette_score(X, db_labels) if n_db_clusters > 1 else 0.0

    logger.info("K-Means  — ARI: %.4f | Silhouette: %.4f", ari_km, sil_km)
    logger.info("DBSCAN   — ARI: %.4f | Silhouette: %.4f | Clusters: %d", ari_db, sil_db, n_db_clusters)

    # Silhouette bar chart
    _save_figure(
        lambda: sns.barplot(
            x=["K-Means", "DBSCAN"],
            y=[sil_km, sil_db],
            palette="pastel",
        ),
        title="Clustering Silhouette Scores",
        path=Path(output_dir) / "clustering_silhouette_scores.png",
        ylabel="Score",
    )

    # K-Means cluster distribution pie chart
    cluster_counts = pd.Series(km_labels).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    cluster_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax)
    ax.set_title("K-Means Cluster Distribution")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "kmeans_cluster_distribution.png", dpi=150)
    plt.close()

    return {
        "ari_kmeans": round(ari_km, 4),
        "ari_dbscan": round(ari_db, 4),
        "silhouette_kmeans": round(sil_km, 4),
        "silhouette_dbscan": round(sil_db, 4),
        "dbscan_clusters": n_db_clusters,
        "dbscan_noise_points": n_noise,
        "kmeans_labels": km_labels.tolist(),
        "dbscan_labels": db_labels.tolist(),
    }


def _scatter_pca(X_pca, labels, title: str, path: Path) -> None:
    """Scatter plot of first two PCA components coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Set1", n_colors=len(np.unique(labels)))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=labels, palette=palette, ax=ax, s=40, alpha=0.7,
    )
    ax.set_title(title)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL EVALUATION & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_report(
    results_df: pd.DataFrame,
    clustering_metrics: Dict,
    output_dir: str,
) -> Dict:
    """
    Determine best models, print a human-readable summary, and save JSON report.

    Parameters
    ----------
    results_df : pd.DataFrame
        Supervised model metrics.
    clustering_metrics : dict
        ARI / silhouette metrics from clustering.
    output_dir : str
        Directory for saved report.

    Returns
    -------
    dict
        Full results summary.
    """
    best_sup = results_df.sort_values("F1-score", ascending=False).iloc[0]
    best_unsup = (
        "K-Means"
        if clustering_metrics["ari_kmeans"] >= clustering_metrics["ari_dbscan"]
        else "DBSCAN"
    )
    best_ari = max(clustering_metrics["ari_kmeans"], clustering_metrics["ari_dbscan"])

    overall_winner = (
        best_sup["Model"]
        if best_sup["F1-score"] > best_ari
        else best_unsup
    )
    winner_reason = (
        "best predictive accuracy and clinical interpretability."
        if best_sup["F1-score"] > best_ari
        else "best clustering alignment with true labels."
    )

    summary = {
        "best_supervised_model": best_sup["Model"],
        "best_supervised_f1": best_sup["F1-score"],
        "best_supervised_roc_auc": best_sup["ROC-AUC"],
        "best_unsupervised_model": best_unsup,
        "best_unsupervised_ari": best_ari,
        "overall_winner": overall_winner,
        "winner_reason": winner_reason,
        "supervised_results": results_df.to_dict(orient="records"),
        "clustering_metrics": {
            k: v for k, v in clustering_metrics.items()
            if k not in ("kmeans_labels", "dbscan_labels")
        },
    }

    # Human-readable summary
    divider = "─" * 60
    print(f"\n{divider}")
    print("  HEART DISEASE PIPELINE — RESULTS SUMMARY")
    print(divider)
    print("\n📊  Supervised Model Performance:")
    print(results_df.to_string(index=False))
    print(f"\n🏆  Best Supervised  : {best_sup['Model']}  (F1={best_sup['F1-score']:.4f})")
    print(f"\n🔵  Clustering Results:")
    print(f"    K-Means  — ARI={clustering_metrics['ari_kmeans']:.4f} | "
          f"Silhouette={clustering_metrics['silhouette_kmeans']:.4f}")
    print(f"    DBSCAN   — ARI={clustering_metrics['ari_dbscan']:.4f} | "
          f"Silhouette={clustering_metrics['silhouette_dbscan']:.4f} | "
          f"Clusters={clustering_metrics['dbscan_clusters']}")
    print(f"\n🏆  Best Unsupervised : {best_unsup}  (ARI={best_ari:.4f})")
    print(f"\n✅  Overall Winner    : {overall_winner} — {winner_reason}")
    print(divider + "\n")

    # Save JSON report
    report_path = Path(output_dir) / "results_summary.json"
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Results summary saved to '%s'.", report_path)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 10. UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_figure(
    plot_fn,
    title: str,
    path: Path,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Safely render and save a matplotlib figure, catching any plotting errors.

    Parameters
    ----------
    plot_fn : callable
        A zero-argument function that produces the plot on the current axes.
    title : str
        Figure title.
    path : Path
        Output file path.
    xlabel, ylabel : str, optional
        Axis labels.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
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
        logger.info("Saved: %s", path)
    except Exception as exc:
        logger.warning("Could not save figure '%s': %s", path, exc)
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# 11. CLI ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, falling back to CONFIG defaults."""
    parser = argparse.ArgumentParser(
        description="Heart Disease ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default=CONFIG["input_csv"],
        help="Path to the raw UCI heart disease CSV.",
    )
    parser.add_argument(
        "--output_dir", default=CONFIG["output_dir"],
        help="Directory for all output plots and reports.",
    )
    parser.add_argument(
        "--test_size", type=float, default=CONFIG["test_size"],
        help="Fraction of data held out for testing.",
    )
    parser.add_argument(
        "--random_state", type=int, default=CONFIG["random_state"],
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip_assess", action="store_true",
        help="Skip the data quality assessment step.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full heart disease prediction pipeline.

    Pipeline steps:
      1. Parse arguments
      2. Load and validate raw data
      3. Assess data quality
      4. Preprocess and clean
      5. Train / evaluate supervised models
      6. Run unsupervised clustering
      7. Report and save results
    """
    args = parse_args()

    # Apply CLI overrides to CONFIG
    CONFIG["input_csv"]    = args.input
    CONFIG["output_dir"]   = args.output_dir
    CONFIG["test_size"]    = args.test_size
    CONFIG["random_state"] = args.random_state

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ── Step 1: Load & Validate ──────────────────────────────────────────────
    try:
        raw_df = load_and_validate(CONFIG["input_csv"])
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Data loading failed: %s", exc)
        sys.exit(1)

    # ── Step 2: Data Quality Assessment ─────────────────────────────────────
    if not args.skip_assess:
        assess_data_quality(raw_df, CONFIG["output_dir"])

    # ── Step 3: Preprocess ───────────────────────────────────────────────────
    try:
        clean_df = preprocess(raw_df, CONFIG)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)

    save_cleaned_data(clean_df, CONFIG["cleaned_csv"])

    X = clean_df.drop("target", axis=1)
    y = clean_df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,
    )
    logger.info(
        "Train/test split — train: %d, test: %d", len(X_train), len(X_test)
    )

    # ── Step 4: Supervised Learning ──────────────────────────────────────────
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

    # ── Step 5: Unsupervised Clustering ──────────────────────────────────────
    try:
        clustering_metrics = run_clustering(X, y, CONFIG, CONFIG["output_dir"])
    except Exception as exc:
        logger.error("Clustering failed: %s", exc)
        sys.exit(1)

    # ── Step 6: Final Report ─────────────────────────────────────────────────
    evaluate_and_report(results_df, clustering_metrics, CONFIG["output_dir"])

    logger.info(
        "Pipeline complete. All outputs saved to '%s'.", CONFIG["output_dir"]
    )


if __name__ == "__main__":
    main()
