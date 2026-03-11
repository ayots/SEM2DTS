# =============================================================================
# COM618 DATA SCIENCE -- HEART DISEASE PREDICTION LAB
# =============================================================================
# Compatible with: PyCharm IDE and Jupyter Notebook
#
# HOW TO RUN:
#   PyCharm : Open this file and press the green Run button (or Shift+F10).
#             All modules execute in order and results print to the console.
#   Jupyter : Paste the whole file into a single cell, or copy individual
#             module functions into separate cells and call main() at the end.
#
# REQUIREMENTS:
#   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
#
# INPUT FILE:
#   Place heart_disease_uci.csv in the same folder as this script.
#   The cleaned dataset is saved to the same folder automatically.
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Saves plots to file -- works everywhere
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from xgboost import XGBClassifier

sns.set_theme(style='whitegrid', palette='muted')

# =============================================================================
# CONFIGURATION
# Change DATA_FILE if your CSV is stored elsewhere.
# OUTPUT_DIR is where all plots and the cleaned CSV will be saved.
# =============================================================================
DATA_FILE  = 'heart_disease_uci.csv'   # Input raw dataset
OUTPUT_DIR = '.'                        # Folder for all outputs (same as script)

# =============================================================================
# MODULE 1 -- LOAD DATA
# =============================================================================
# Reads the raw CSV file and prints a first-look summary so we know
# exactly what we are working with before touching the data.

def module1_load_data(filepath):
    """
    Load the raw UCI heart disease CSV and display a structural summary.

    Returns
    -------
    df : pd.DataFrame  -- raw, unmodified dataset
    """
    _banner('MODULE 1 -- LOAD DATA')

    # Verify the file exists before attempting to read it
    if not os.path.isfile(filepath):
        print(f'ERROR: File not found: {filepath}')
        print('Please place heart_disease_uci.csv in the same folder as this script.')
        sys.exit(1)

    df = pd.read_csv(filepath)

    print(f'\nDataset shape  : {df.shape[0]} rows x {df.shape[1]} columns')
    print('\nFirst 5 rows of the raw dataset:')
    print(df.head().to_string())
    print('\nColumn names and data types:')
    print(df.dtypes.to_string())
    print('\nBasic statistics for numeric columns:')
    print(df.describe().round(2).to_string())

    _done('MODULE 1')
    return df


# =============================================================================
# MODULE 2 -- ASSESS DATA QUALITY
# =============================================================================
# Identifies three types of data quality problem BEFORE any cleaning:
#   (a) Missing values  -- how much data is absent and where
#   (b) Duplicate rows  -- exact row repetitions
#   (c) Outliers        -- extreme values detected via the IQR method
#
# Two plots are saved: a missing-value heatmap and a bar chart of
# missing percentages. A boxplot of numeric columns highlights outliers.

def module2_assess_quality(df, output_dir):
    """
    Assess data quality and save diagnostic plots.

    Parameters
    ----------
    df         : pd.DataFrame  -- raw dataframe from Module 1
    output_dir : str           -- folder for saved plots
    """
    _banner('MODULE 2 -- ASSESS DATA QUALITY')

    # ── (a) Missing values ────────────────────────────────────────────────────
    print('\n-- STEP 2a: Missing Values --')
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)
    missing_df    = pd.DataFrame({'Count': missing_count, 'Percent': missing_pct})
    affected      = missing_df[missing_df['Count'] > 0].sort_values('Percent', ascending=False)

    if affected.empty:
        print('No missing values found.')
    else:
        print(affected.to_string())

    # Plot 1: heatmap showing where data is absent
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Figure 1 -- Missing Value Heatmap\n(Yellow = Missing, Purple = Present)', fontsize=12)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Row Index')
    plt.tight_layout()
    _save(fig, output_dir, 'plot1_missing_heatmap.png')

    # Plot 2: bar chart of missing percentages per column
    if not affected.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        affected['Percent'].sort_values().plot(kind='barh', color='coral', ax=ax)
        ax.set_title('Figure 2 -- Missing Data Percentage per Column', fontsize=12)
        ax.set_xlabel('Missing (%)')
        for i, v in enumerate(affected['Percent'].sort_values()):
            ax.text(v + 0.3, i, f'{v}%', va='center', fontsize=9)
        plt.tight_layout()
        _save(fig, output_dir, 'plot2_missing_bar.png')

    # ── (b) Duplicate rows ────────────────────────────────────────────────────
    print('\n-- STEP 2b: Duplicate Rows --')
    n_dup = df.duplicated().sum()
    print(f'Duplicate rows found: {n_dup}')

    # ── (c) Outliers via IQR ─────────────────────────────────────────────────
    print('\n-- STEP 2c: Outlier Detection (IQR Method) --')
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    outlier_counts = {}
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        outlier_counts[col] = n_out
    for col, n in sorted(outlier_counts.items(), key=lambda x: -x[1]):
        if n > 0:
            print(f'  {col}: {n} outliers')

    # Plot 3: boxplots before cleaning (dots beyond whiskers = outliers)
    useful = [c for c in num_cols if c not in ('id',)]
    n_c, n_r = 3, (len(useful) + 2) // 3
    fig, axes = plt.subplots(n_r, n_c, figsize=(14, n_r * 3))
    axes = axes.flatten()
    for i, col in enumerate(useful):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='lightblue'))
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel('Value')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Figure 3 -- Boxplots Before Cleaning\n(Dots beyond whiskers = outliers)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, 'plot3_boxplots_before.png')

    _done('MODULE 2')


# =============================================================================
# MODULE 3 -- CLEAN DATA
# =============================================================================
# Fixes the problems found in Module 2 in a clearly ordered sequence.
# The cleaned dataset is saved as a CSV to the output directory.
#
# Steps:
#   1. Drop non-predictive columns (id, dataset)
#   2. Binarise the target variable (num -> target)
#   3. Encode categorical and boolean columns as integers
#   4. Remove duplicate rows
#   5. Impute missing values with column medians
#   6. Force all columns to numeric; drop any remaining NaN rows

def module3_clean_data(df, output_dir):
    """
    Clean and prepare the raw dataframe for analysis and modelling.

    Returns
    -------
    df_clean : pd.DataFrame  -- cleaned, fully numeric dataframe
    """
    _banner('MODULE 3 -- CLEAN DATA')

    df_clean = df.copy()   # Always work on a copy; never modify the original

    # Step 1: Drop columns that carry no predictive value
    print('\n-- STEP 1: Drop Non-Predictive Columns --')
    df_clean.drop(columns=['id', 'dataset'], errors='ignore', inplace=True)
    print(f'Columns remaining: {df_clean.columns.tolist()}')

    # Step 2: Create binary target variable
    # Original 'num' = 0 means healthy; 1-4 = varying disease severity.
    # We convert to a simple 0/1 flag for binary classification.
    print('\n-- STEP 2: Create Binary Target (0=No Disease, 1=Disease) --')
    df_clean['target'] = (df_clean['num'] > 0).astype(int)
    df_clean.drop(columns=['num'], inplace=True)
    print(df_clean['target'].value_counts().to_string())

    # Step 3: Encode categorical and boolean columns
    print('\n-- STEP 3: Encode Categorical Columns --')
    df_clean['sex']  = (df_clean['sex'] == 'Male').astype(float)
    bool_map = {True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0}
    df_clean['fbs']   = df_clean['fbs'].map(bool_map)
    df_clean['exang'] = df_clean['exang'].map(bool_map)
    df_clean['cp']      = df_clean['cp'].map({'typical angina': 0, 'atypical angina': 1,
                                               'non-anginal': 2, 'asymptomatic': 3})
    df_clean['restecg'] = df_clean['restecg'].map({'normal': 0, 'st-t abnormality': 1,
                                                    'lv hypertrophy': 2})
    df_clean['slope']   = df_clean['slope'].map({'upsloping': 0, 'flat': 1, 'downsloping': 2})
    df_clean['thal']    = df_clean['thal'].map({'normal': 0, 'fixed defect': 1,
                                                 'reversable defect': 2})
    print('All categorical columns encoded to integers.')

    # Step 4: Remove duplicate rows
    print('\n-- STEP 4: Remove Duplicate Rows --')
    before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    print(f'Removed {before - len(df_clean)} duplicate rows. ({len(df_clean)} remaining)')

    # Step 5: Impute missing values with column medians
    # The median is robust to extreme values (e.g. high cholesterol outliers).
    print('\n-- STEP 5: Impute Missing Values with Column Medians --')
    missing_before = df_clean.isnull().sum().sum()
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    print(f'Missing values: {missing_before} --> {df_clean.isnull().sum().sum()}')

    # Step 6: Force all columns to numeric (safety pass)
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    rows_before = len(df_clean)
    df_clean.dropna(inplace=True)
    if len(df_clean) < rows_before:
        print(f'Dropped {rows_before - len(df_clean)} rows where coercion failed.')

    print(f'\nFinal cleaned shape: {df_clean.shape}')

    # ── Plots: before/after comparison ───────────────────────────────────────
    feat_cols = [c for c in df_clean.columns if c != 'target']
    n_c, n_r = 3, (len(feat_cols) + 2) // 3

    # Plot 4: boxplots after cleaning
    fig, axes = plt.subplots(n_r, n_c, figsize=(14, n_r * 3))
    axes = axes.flatten()
    for i, col in enumerate(feat_cols):
        axes[i].boxplot(df_clean[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='lightgreen'))
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel('Value')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Figure 4 -- Boxplots After Cleaning\n(All missing values imputed)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, 'plot4_boxplots_after.png')

    # Plot 5: class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_clean['target'].value_counts().sort_index()
    bars = ax.bar(['No Disease (0)', 'Disease (1)'], counts.values,
                  color=['steelblue', 'coral'], edgecolor='black')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=11)
    ax.set_title('Figure 5 -- Target Variable Distribution', fontsize=12)
    ax.set_ylabel('Number of Patients')
    ax.set_ylim(0, counts.max() + 60)
    plt.tight_layout()
    _save(fig, output_dir, 'plot5_target_distribution.png')

    # Save cleaned dataset next to the original file
    clean_path = os.path.join(output_dir, 'heart_disease_cleaned.csv')
    df_clean.to_csv(clean_path, index=False)
    print(f'\nCleaned dataset saved to: {clean_path}')

    _done('MODULE 3')
    return df_clean


# =============================================================================
# MODULE 4 -- EXPLORE DATA (EDA)
# =============================================================================
# Exploratory Data Analysis reveals patterns, trends, and relationships
# in the cleaned data before formal modelling.
#
# Outputs: histograms, correlation heatmap, mean difference chart,
#          clinical feature boxplots, chest pain type breakdown.

def module4_explore_data(df_clean, output_dir):
    """
    Perform exploratory data analysis on the cleaned dataset.

    Parameters
    ----------
    df_clean   : pd.DataFrame  -- cleaned dataframe from Module 3
    output_dir : str           -- folder for saved plots
    """
    _banner('MODULE 4 -- EXPLORE DATA (EDA)')

    feat_cols = [c for c in df_clean.columns if c != 'target']

    # Summary statistics
    print('\n-- Summary Statistics (Cleaned Data) --')
    print(df_clean[feat_cols].describe().round(2).to_string())

    # Plot 6: feature histograms
    n_c, n_r = 3, (len(feat_cols) + 2) // 3
    fig, axes = plt.subplots(n_r, n_c, figsize=(14, n_r * 3))
    axes = axes.flatten()
    for i, col in enumerate(feat_cols):
        axes[i].hist(df_clean[col], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Figure 6 -- Feature Distributions (Histograms)', fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, 'plot6_histograms.png')

    # Correlation analysis
    print('\n-- Correlation with Target Variable --')
    corr = df_clean.corr(numeric_only=True)
    target_corr = corr['target'].drop('target').sort_values(key=abs, ascending=False)
    print(target_corr.round(3).to_string())

    # Plot 7: correlation heatmap (lower triangle only)
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                mask=mask, ax=ax, linewidths=0.5, annot_kws={'size': 8})
    ax.set_title('Figure 7 -- Feature Correlation Heatmap\n'
                 '(Positive = red | Negative = blue | 0 = white)', fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, 'plot7_correlation_heatmap.png')

    # Group means: disease vs no-disease
    print('\n-- Mean Feature Values by Disease Status --')
    group_means = df_clean.groupby('target')[feat_cols].mean()
    print(group_means.round(3).to_string())

    # Plot 8: mean difference bar chart
    diff = (group_means.loc[1] - group_means.loc[0]).sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['coral' if v > 0 else 'steelblue' for v in diff.values]
    diff.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Figure 8 -- Mean Feature Difference (Disease minus No Disease)\n'
                 'Red = higher in disease patients | Blue = lower', fontsize=12)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Difference in Mean')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _save(fig, output_dir, 'plot8_feature_mean_diff.png')

    # Plot 9: age and cholesterol by disease status
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_clean.boxplot(column='age',  by='target', ax=axes[0],
                     boxprops=dict(color='steelblue'))
    axes[0].set_title('Age by Disease Status')
    axes[0].set_xlabel('0 = No Disease | 1 = Disease')
    axes[0].set_ylabel('Age (years)')
    df_clean.boxplot(column='chol', by='target', ax=axes[1],
                     boxprops=dict(color='coral'))
    axes[1].set_title('Cholesterol by Disease Status')
    axes[1].set_xlabel('0 = No Disease | 1 = Disease')
    axes[1].set_ylabel('Cholesterol (mg/dL)')
    fig.suptitle('Figure 9 -- Key Clinical Features by Disease Status', fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, 'plot9_clinical_boxplots.png')

    # Plot 10: chest pain type vs disease
    cp_labels = {0: 'Typical Angina', 1: 'Atypical Angina',
                 2: 'Non-Anginal', 3: 'Asymptomatic'}
    cp_counts = df_clean.groupby(['cp', 'target']).size().unstack(fill_value=0)
    cp_counts.index = [cp_labels.get(i, str(i)) for i in cp_counts.index]
    cp_counts.columns = ['No Disease', 'Disease']
    fig, ax = plt.subplots(figsize=(8, 5))
    cp_counts.plot(kind='bar', stacked=False, color=['steelblue', 'coral'],
                   edgecolor='black', ax=ax)
    ax.set_title('Figure 10 -- Chest Pain Type vs Heart Disease', fontsize=12)
    ax.set_xlabel('Chest Pain Type')
    ax.set_ylabel('Number of Patients')
    ax.legend(title='Diagnosis')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    _save(fig, output_dir, 'plot10_chestpain_disease.png')

    _done('MODULE 4')


# =============================================================================
# MODULE 5 -- TRAIN AND EVALUATE MODELS
# =============================================================================
# Trains three supervised classifiers (Random Forest, Gradient Boosting,
# XGBoost) and evaluates each using five metrics: Accuracy, Precision,
# Recall, F1-Score, and ROC-AUC.
#
# Outputs: metric comparison bar chart, confusion matrices, ROC curves,
#          feature importance charts.

def module5_train_and_evaluate(df_clean, output_dir):
    """
    Train and evaluate three classifiers; save all evaluation plots.

    Returns
    -------
    results_df      : pd.DataFrame  -- metric table for all models
    trained_models  : dict          -- fitted model objects
    X, y            : arrays        -- full feature matrix and labels
    X_test, y_test  : arrays        -- held-out test partition
    """
    _banner('MODULE 5 -- TRAIN AND EVALUATE MODELS')

    # Prepare features and target
    X = df_clean.drop('target', axis=1)
    y = df_clean['target'].astype(int)
    print(f'\nFeatures: {X.shape[1]} columns, {X.shape[0]} rows')
    print(f'Feature list: {X.columns.tolist()}')

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f'\nTraining rows : {len(X_train)}')
    print(f'Testing rows  : {len(X_test)}')

    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=100, learning_rate=0.1,
            eval_metric='logloss', random_state=42),
    }

    # Train and evaluate each model
    results        = []
    trained_models = {}
    feat_importance = {}

    print()
    for name, model in models.items():
        print(f'  Training {name}...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        row = {
            'Model':     name,
            'Accuracy':  round(accuracy_score(y_test, y_pred),  4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall':    round(recall_score(y_test, y_pred),    4),
            'F1-Score':  round(f1_score(y_test, y_pred),        4),
            'ROC-AUC':   round(roc_auc_score(y_test, y_prob),   4),
        }
        results.append(row)
        trained_models[name] = model

        # Feature importances (available for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feat_importance[name] = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

        print(f'  Accuracy={row["Accuracy"]}  Precision={row["Precision"]}  '
              f'Recall={row["Recall"]}  F1={row["F1-Score"]}  AUC={row["ROC-AUC"]}')

    results_df = pd.DataFrame(results)
    print('\nFull results table:')
    print(results_df.to_string(index=False))

    # Plot 11: grouped metric comparison
    metrics   = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x         = np.arange(len(results_df))
    bar_width = 0.15
    fig, ax   = plt.subplots(figsize=(13, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * bar_width, results_df[metric], width=bar_width, label=metric)
    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(results_df['Model'], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Figure 11 -- Model Evaluation Metric Comparison\n'
                 '(Higher is better for all metrics)', fontsize=12)
    ax.legend(loc='lower right')
    plt.tight_layout()
    _save(fig, output_dir, 'plot11_model_comparison.png')

    # Plot 12: confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'],
                    annot_kws={'size': 12})
        axes[i].set_title(name, fontsize=11)
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('Actual Label')
    fig.suptitle('Figure 12 -- Confusion Matrices\n'
                 'Diagonal = correct predictions | Off-diagonal = errors',
                 fontsize=12, y=1.03)
    plt.tight_layout()
    _save(fig, output_dir, 'plot12_confusion_matrices.png')

    # Plot 13: ROC curves (all three on one chart)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'coral', 'green']
    for (name, model), color in zip(trained_models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name}  (AUC = {auc_val:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Figure 13 -- ROC Curves for All Models\n'
                 'Higher AUC = better class discrimination', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, output_dir, 'plot13_roc_curves.png')

    # Plot 14: feature importance charts
    for name, importance in feat_importance.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        importance.plot(kind='bar', color='steelblue', edgecolor='black', ax=ax)
        ax.set_title(f'Figure 14 -- Feature Importance: {name}\n'
                     '(Higher bar = more important predictor)', fontsize=12)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance Score')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        fname = f'plot14_importance_{name.replace(" ", "_")}.png'
        _save(fig, output_dir, fname)

    _done('MODULE 5')
    return results_df, trained_models, X, y, X_test, y_test


# =============================================================================
# MODULE 6 -- CLUSTERING (UNSUPERVISED ANALYSIS)
# =============================================================================
# Applies K-Means clustering to discover natural patient groupings
# without using the disease labels during fitting.
#
# PCA reduces the 13 features to 2 dimensions for visualisation.
# The Elbow Method plot helps justify the choice of k = 2.
# ARI and Silhouette scores evaluate cluster quality.

def module6_clustering(X, y, output_dir):
    """
    Apply K-Means clustering and evaluate against true labels.

    Parameters
    ----------
    X          : pd.DataFrame  -- full feature matrix
    y          : pd.Series     -- true labels (used only for ARI)
    output_dir : str           -- folder for saved plots
    """
    _banner('MODULE 6 -- CLUSTERING (UNSUPERVISED ANALYSIS)')

    # Apply K-Means with 2 clusters
    print('\n-- K-Means Clustering (k=2) --')
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f'  Cluster {label}: {count} patients')

    # Evaluate clustering quality
    print('\n-- Clustering Quality Metrics --')
    sil = silhouette_score(X, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)
    print(f'  Silhouette Score   : {sil:.4f}  (1 = perfectly separated clusters)')
    print(f'  Adjusted Rand Index: {ari:.4f}  (1 = perfect match with true labels)')

    # PCA for 2D visualisation
    print('\n-- PCA Dimensionality Reduction --')
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f'  Variance explained by 2 components: '
          f'{pca.explained_variance_ratio_.sum() * 100:.1f}%')

    # Plot 15: cluster assignments vs true labels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                           cmap='Set1', s=25, alpha=0.7)
    axes[0].set_title(f'K-Means Clusters (k=2)\nSilhouette={sil:.3f}  ARI={ari:.3f}')
    axes[0].set_xlabel('PCA Component 1')
    axes[0].set_ylabel('PCA Component 2')
    plt.colorbar(sc1, ax=axes[0], label='Cluster')
    sc2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                           cmap='coolwarm', s=25, alpha=0.7)
    axes[1].set_title('True Disease Labels\n(0 = No Disease, 1 = Disease)')
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')
    plt.colorbar(sc2, ax=axes[1], label='True Label')
    fig.suptitle('Figure 15 -- K-Means Clusters vs True Labels (PCA Projection)',
                 fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, 'plot15_kmeans_pca.png')

    # Plot 16: Elbow method
    inertia = []
    k_range = range(1, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertia.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertia, 'bo-', lw=2, markersize=8)
    ax.axvline(x=2, color='red', linestyle='--', label='Chosen k=2')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    ax.set_title('Figure 16 -- Elbow Method for Optimal k\n'
                 'Red dashed line = chosen k=2', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, output_dir, 'plot16_elbow_method.png')

    _done('MODULE 6')


# =============================================================================
# MODULE 7 -- FINAL SUMMARY
# =============================================================================
# Prints a consolidated summary of all results to the console,
# including the best-performing model on each metric.

def module7_final_summary(results_df):
    """Print a consolidated results summary."""
    _banner('MODULE 7 -- FINAL RESULTS SUMMARY')

    print('\nSupervised Model Performance:')
    print('-' * 62)
    print(results_df.to_string(index=False))
    print('-' * 62)

    print('\nBest model per metric:')
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        best_idx  = results_df[metric].idxmax()
        best_name = results_df.loc[best_idx, 'Model']
        best_val  = results_df.loc[best_idx, metric]
        print(f'  {metric:<12}: {best_name}  ({best_val})')

    best = results_df.sort_values('F1-Score', ascending=False).iloc[0]
    print(f'\nOverall recommended model : {best["Model"]}')
    print(f'  F1-Score  = {best["F1-Score"]}')
    print(f'  ROC-AUC   = {best["ROC-AUC"]}')
    print(f'  Accuracy  = {best["Accuracy"]}')

    print('\nAll output files saved:')
    for fname in sorted(os.listdir('.')):
        if fname.startswith('plot') or fname == 'heart_disease_cleaned.csv':
            print(f'  {fname}')

    _done('MODULE 7')


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _banner(title):
    """Print a section header."""
    print('\n' + '=' * 60)
    print(f'  {title}')
    print('=' * 60)

def _done(title):
    """Print a completion message."""
    print(f'\n[{title} COMPLETE]')

def _save(fig, directory, filename):
    """Save a matplotlib figure and close it."""
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {filename}')


# =============================================================================
# MAIN FUNCTION
# =============================================================================
# Calling main() runs the entire pipeline from start to finish.
# Each module prints its results before the next one begins.

def main():
    """Run the full heart disease analysis pipeline."""

    print('=' * 60)
    print('  COM618 DATA SCIENCE -- HEART DISEASE PREDICTION')
    print('  Starting full pipeline...')
    print('=' * 60)

    # Module 1: Load raw data
    df_raw = module1_load_data(DATA_FILE)

    # Module 2: Assess data quality (before cleaning)
    module2_assess_quality(df_raw, OUTPUT_DIR)

    # Module 3: Clean data and save cleaned CSV
    df_clean = module3_clean_data(df_raw, OUTPUT_DIR)

    # Module 4: Exploratory data analysis
    module4_explore_data(df_clean, OUTPUT_DIR)

    # Module 5: Train models and evaluate
    results_df, trained_models, X, y, X_test, y_test = \
        module5_train_and_evaluate(df_clean, OUTPUT_DIR)

    # Module 6: Unsupervised clustering
    module6_clustering(X, y, OUTPUT_DIR)

    # Module 7: Print final summary
    module7_final_summary(results_df)


# =============================================================================
# ENTRY POINT
# =============================================================================
# This block runs when the script is executed directly (e.g. in PyCharm).
# In Jupyter, you can call main() in any cell after defining the functions.

if __name__ == '__main__':
    main()
