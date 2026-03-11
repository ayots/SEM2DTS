# =============================================================================
# HEART DISEASE DATA ANALYSIS PIPELINE
# Compatible with: PyCharm IDE and Jupyter Notebook
# Dataset: UCI Heart Disease Dataset (heart_disease_uci.csv)
#
# HOW TO RUN IN PYCHARM:
#   - Place heart_disease_uci.csv in the same folder as this script
#   - Run the script -- each module prints results and saves plots
#
# HOW TO RUN IN JUPYTER:
#   - Copy each module into its own cell
#   - Run cells in order from top to bottom
# =============================================================================

# ── INSTALL REQUIRED LIBRARIES (run once if needed) ──────────────────────────
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost

import warnings
warnings.filterwarnings('ignore')   # Keep terminal output clean

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
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

# Set a consistent visual style for all plots
sns.set_theme(style='whitegrid', palette='muted')

print("=" * 60)
print("  HEART DISEASE ANALYSIS PIPELINE")
print("  All libraries loaded successfully.")
print("=" * 60)


# =============================================================================
# MODULE 1 -- LOAD DATA
# PURPOSE : Read the raw CSV file into a pandas DataFrame and display
#           a first look at the data structure, column types, and a
#           sample of rows so we know what we are working with.
# =============================================================================

def module1_load_data(filepath='heart_disease_uci.csv'):
    """
    Load the raw UCI heart disease CSV file.

    Returns
    -------
    df : pd.DataFrame
        Raw, unmodified dataframe straight from the CSV.
    """
    print("\n" + "=" * 60)
    print("  MODULE 1 -- LOAD DATA")
    print("=" * 60)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)

    # Show the number of rows and columns
    print(f"\nDataset shape  : {df.shape[0]} rows x {df.shape[1]} columns")

    # Show the first 5 rows so we can see the raw data
    print("\nFirst 5 rows of the raw dataset:")
    print(df.head())

    # Show column names and their data types
    print("\nColumn names and data types:")
    print(df.dtypes)

    # Show basic statistics for numeric columns
    print("\nBasic statistics for numeric columns:")
    print(df.describe().round(2))

    print("\n[MODULE 1 COMPLETE] Raw data loaded successfully.")
    return df


# =============================================================================
# MODULE 2 -- ASSESS DATA QUALITY  (Part A, Section 2)
# PURPOSE : Identify problems in the raw data before cleaning.
#           We check for: missing values, duplicate rows, outliers.
#           We produce visualisations so the problems are visible.
# =============================================================================

def module2_assess_quality(df):
    """
    Assess the quality of the raw data and produce diagnostic plots.

    Parameters
    ----------
    df : pd.DataFrame  -- raw dataframe from Module 1

    Returns
    -------
    Nothing.  Results are printed and plots are displayed/saved.
    """
    print("\n" + "=" * 60)
    print("  MODULE 2 -- ASSESS DATA QUALITY")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 2a: Count missing values in every column
    # ------------------------------------------------------------------
    print("\n-- STEP 2a: Missing Values --")

    # Count how many values are missing in each column
    missing_count = df.isnull().sum()

    # Calculate what percentage of each column is missing
    missing_pct = (missing_count / len(df) * 100).round(2)

    # Combine into a neat summary table
    missing_summary = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing %': missing_pct
    })

    # Only display columns that actually have missing data
    missing_cols = missing_summary[missing_summary['Missing Count'] > 0]
    if missing_cols.empty:
        print("No missing values found.")
    else:
        print(missing_cols.sort_values('Missing %', ascending=False))

    # ------------------------------------------------------------------
    # PLOT 1: Missing value heatmap
    # A yellow cell = missing; a purple cell = present
    # This makes it easy to see patterns in where data is missing
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('PLOT 1 -- Missing Value Heatmap\n'
                 '(Yellow = Missing, Purple = Present)', fontsize=13, pad=12)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Row Index')
    plt.tight_layout()
    plt.savefig('plot1_missing_heatmap.png', dpi=150)
    plt.show()
    print("Saved: plot1_missing_heatmap.png")

    # ------------------------------------------------------------------
    # PLOT 2: Bar chart showing missing percentage per column
    # Makes it easy to rank columns by how incomplete they are
    # ------------------------------------------------------------------
    if not missing_cols.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing_cols['Missing %'].sort_values().plot(
            kind='barh', color='coral', ax=ax
        )
        ax.set_title('PLOT 2 -- Missing Data Percentage per Column', fontsize=13)
        ax.set_xlabel('Missing (%)')
        for i, v in enumerate(missing_cols['Missing %'].sort_values()):
            ax.text(v + 0.5, i, f'{v}%', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig('plot2_missing_bar.png', dpi=150)
        plt.show()
        print("Saved: plot2_missing_bar.png")

    # ------------------------------------------------------------------
    # STEP 2b: Check for duplicate rows
    # ------------------------------------------------------------------
    print("\n-- STEP 2b: Duplicate Rows --")
    n_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {n_duplicates}")

    # ------------------------------------------------------------------
    # STEP 2c: Detect outliers using the IQR (interquartile range) method
    # A value is an outlier if it is more than 1.5 x IQR below Q1 or above Q3
    # ------------------------------------------------------------------
    print("\n-- STEP 2c: Outlier Detection (IQR Method) --")

    # Select only the numeric columns for outlier checking
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    outlier_summary = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)      # First quartile (25th percentile)
        q3 = df[col].quantile(0.75)      # Third quartile (75th percentile)
        iqr = q3 - q1                    # Interquartile range
        lower = q1 - 1.5 * iqr          # Lower fence
        upper = q3 + 1.5 * iqr          # Upper fence
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = n_outliers

    outlier_df = pd.Series(outlier_summary, name='Outlier Count')
    print(outlier_df[outlier_df > 0].sort_values(ascending=False))

    # ------------------------------------------------------------------
    # PLOT 3: Boxplots showing the distribution of each numeric column
    # The dots outside the whiskers are outliers
    # ------------------------------------------------------------------
    useful_numeric = [c for c in numeric_cols if c not in ('id',)]
    n_cols = 3
    n_rows = (len(useful_numeric) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(useful_numeric):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='lightblue'))
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel('Value')

    # Hide any unused subplot panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('PLOT 3 -- Boxplots of Numeric Features (Before Cleaning)\n'
                 'Dots beyond whiskers = outliers', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('plot3_boxplots_before.png', dpi=150)
    plt.show()
    print("Saved: plot3_boxplots_before.png")

    print("\n[MODULE 2 COMPLETE] Data quality assessment done.")


# =============================================================================
# MODULE 3 -- CLEAN DATA  (Part A, Section 2 continued)
# PURPOSE : Fix the problems identified in Module 2.
#           Steps: drop useless columns, create binary target,
#           encode categories, remove duplicates, fill missing values.
#           Produce before/after comparison plots.
# =============================================================================

def module3_clean_data(df):
    """
    Clean the raw dataframe and prepare it for analysis and modelling.

    Parameters
    ----------
    df : pd.DataFrame  -- raw dataframe from Module 1

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned, fully numeric dataframe ready for modelling.
    """
    print("\n" + "=" * 60)
    print("  MODULE 3 -- CLEAN DATA")
    print("=" * 60)

    # Work on a copy so the original raw data is not modified
    df_clean = df.copy()

    # ------------------------------------------------------------------
    # STEP 3a: Drop columns that add no predictive value
    # 'id'      = just a row number, carries no medical information
    # 'dataset' = name of the clinic that collected the data, not a feature
    # ------------------------------------------------------------------
    print("\n-- STEP 3a: Drop Non-Predictive Columns --")
    df_clean = df_clean.drop(columns=['id', 'dataset'], errors='ignore')
    print("Dropped: 'id', 'dataset'")
    print(f"Columns remaining: {df_clean.columns.tolist()}")

    # ------------------------------------------------------------------
    # STEP 3b: Create a binary target variable
    # The original 'num' column uses grades 0-4 (severity of disease).
    # For binary classification we convert to: 0 = healthy, 1 = disease.
    # ------------------------------------------------------------------
    print("\n-- STEP 3b: Create Binary Target Variable --")
    df_clean['target'] = (df_clean['num'] > 0).astype(int)
    df_clean = df_clean.drop(columns=['num'])
    print("Created 'target' column: 0 = No Disease, 1 = Disease")
    print("Target value counts:")
    print(df_clean['target'].value_counts())

    # ------------------------------------------------------------------
    # STEP 3c: Encode categorical and boolean columns as numbers
    # Machine learning models need numeric input, not strings.
    # ------------------------------------------------------------------
    print("\n-- STEP 3c: Encode Categorical Columns --")

    # Sex: Male = 1, Female = 0
    df_clean['sex'] = (df_clean['sex'] == 'Male').astype(float)
    print("  sex       --> 1=Male, 0=Female")

    # Fasting blood sugar and exercise-induced angina: True=1, False=0
    bool_map = {True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0}
    df_clean['fbs']   = df_clean['fbs'].map(bool_map)
    df_clean['exang'] = df_clean['exang'].map(bool_map)
    print("  fbs, exang --> 1=True, 0=False")

    # Chest pain type: ordered from least (typical) to most severe (asymptomatic)
    cp_map = {'typical angina': 0, 'atypical angina': 1,
              'non-anginal': 2, 'asymptomatic': 3}
    df_clean['cp'] = df_clean['cp'].map(cp_map)
    print("  cp        --> 0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic")

    # Resting ECG results
    restecg_map = {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}
    df_clean['restecg'] = df_clean['restecg'].map(restecg_map)
    print("  restecg   --> 0=normal, 1=st-t abnormality, 2=lv hypertrophy")

    # ST slope during peak exercise
    slope_map = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
    df_clean['slope'] = df_clean['slope'].map(slope_map)
    print("  slope     --> 0=upsloping, 1=flat, 2=downsloping")

    # Thalassemia blood disorder type
    thal_map = {'normal': 0, 'fixed defect': 1, 'reversable defect': 2}
    df_clean['thal'] = df_clean['thal'].map(thal_map)
    print("  thal      --> 0=normal, 1=fixed defect, 2=reversable defect")

    # ------------------------------------------------------------------
    # STEP 3d: Remove duplicate rows
    # Duplicates can cause overfitting by repeating identical examples
    # ------------------------------------------------------------------
    print("\n-- STEP 3d: Remove Duplicate Rows --")
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    print(f"Rows before: {before}  |  Rows after: {after}  |  Removed: {before - after}")

    # ------------------------------------------------------------------
    # STEP 3e: Fill missing values with column medians
    # We use the median rather than the mean because it is not influenced
    # by extreme outlier values (e.g. cholesterol of 564)
    # ------------------------------------------------------------------
    print("\n-- STEP 3e: Fill Missing Values with Column Medians --")
    missing_before = df_clean.isnull().sum().sum()
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values before: {missing_before}  |  After: {missing_after}")

    # ------------------------------------------------------------------
    # STEP 3f: Force all columns to numeric (safety pass)
    # Anything that still cannot be converted to a number is dropped
    # ------------------------------------------------------------------
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()

    print(f"\nFinal cleaned dataset shape: {df_clean.shape}")

    # ------------------------------------------------------------------
    # PLOT 4: Boxplots AFTER cleaning (compare to Plot 3)
    # ------------------------------------------------------------------
    feature_cols = [c for c in df_clean.columns if c != 'target']
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].boxplot(df_clean[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='lightgreen'))
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel('Value')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('PLOT 4 -- Boxplots After Cleaning\n'
                 '(Compare to Plot 3 -- missing values imputed)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('plot4_boxplots_after.png', dpi=150)
    plt.show()
    print("Saved: plot4_boxplots_after.png")

    # ------------------------------------------------------------------
    # PLOT 5: Target variable distribution (how balanced are the classes?)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_clean['target'].value_counts()
    bars = ax.bar(
        ['No Disease (0)', 'Disease (1)'],
        counts.values,
        color=['steelblue', 'coral'],
        edgecolor='black'
    )
    # Print the count on top of each bar
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(int(bar.get_height())),
                ha='center', va='bottom', fontsize=11)
    ax.set_title('PLOT 5 -- Target Variable Distribution\n'
                 '(Class Balance Check)', fontsize=13)
    ax.set_ylabel('Number of Patients')
    ax.set_ylim(0, counts.max() + 60)
    plt.tight_layout()
    plt.savefig('plot5_target_distribution.png', dpi=150)
    plt.show()
    print("Saved: plot5_target_distribution.png")

    # Save the cleaned data to a CSV for reference
    df_clean.to_csv('heart_disease_cleaned.csv', index=False)
    print("\nCleaned dataset saved as 'heart_disease_cleaned.csv'")

    print("\n[MODULE 3 COMPLETE] Data cleaning done.")
    return df_clean


# =============================================================================
# MODULE 4 -- EXPLORE DATA  (Part A, Section 3)
# PURPOSE : Understand the data through statistical summaries and charts.
#           We look at: distributions, correlations, and how each feature
#           relates to the presence of heart disease.
# =============================================================================

def module4_explore_data(df_clean):
    """
    Perform exploratory data analysis (EDA) on the cleaned dataset.

    Parameters
    ----------
    df_clean : pd.DataFrame  -- cleaned dataframe from Module 3
    """
    print("\n" + "=" * 60)
    print("  MODULE 4 -- EXPLORE DATA")
    print("=" * 60)

    # Separate features from the target
    feature_cols = [c for c in df_clean.columns if c != 'target']

    # ------------------------------------------------------------------
    # STEP 4a: Summary statistics for the cleaned data
    # ------------------------------------------------------------------
    print("\n-- STEP 4a: Summary Statistics (Cleaned Data) --")
    print(df_clean[feature_cols].describe().round(2))

    # ------------------------------------------------------------------
    # PLOT 6: Histograms of all features -- understand each distribution
    # ------------------------------------------------------------------
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].hist(df_clean[col], bins=20, color='steelblue',
                     edgecolor='white', alpha=0.8)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('PLOT 6 -- Feature Distributions (Histograms)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('plot6_histograms.png', dpi=150)
    plt.show()
    print("Saved: plot6_histograms.png")

    # ------------------------------------------------------------------
    # PLOT 7: Correlation heatmap
    # Shows how strongly each pair of variables moves together.
    # Values near +1 or -1 mean strong correlation.
    # Values near 0 mean little relationship.
    # ------------------------------------------------------------------
    print("\n-- STEP 4b: Correlation Matrix --")
    corr_matrix = df_clean.corr(numeric_only=True)

    # Print the top features most correlated with the target
    target_corr = corr_matrix['target'].drop('target').sort_values(key=abs, ascending=False)
    print("\nTop features correlated with heart disease (target):")
    print(target_corr.round(3))

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, mask=mask, ax=ax, linewidths=0.5, annot_kws={'size': 8}
    )
    ax.set_title('PLOT 7 -- Feature Correlation Heatmap\n'
                 '(Positive = red, Negative = blue, 0 = white)', fontsize=13)
    plt.tight_layout()
    plt.savefig('plot7_correlation_heatmap.png', dpi=150)
    plt.show()
    print("Saved: plot7_correlation_heatmap.png")

    # ------------------------------------------------------------------
    # PLOT 8: Mean feature values for disease vs no-disease patients
    # A large gap between bars means the feature is a strong predictor
    # ------------------------------------------------------------------
    print("\n-- STEP 4c: Feature Means by Target Group --")

    group_means = df_clean.groupby('target')[feature_cols].mean()
    print(group_means.round(3))

    # Calculate the difference between the two groups
    diff = (group_means.loc[1] - group_means.loc[0]).sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['coral' if v > 0 else 'steelblue' for v in diff.values]
    diff.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('PLOT 8 -- Difference in Feature Means (Disease - No Disease)\n'
                 'Red = higher in disease group, Blue = lower in disease group',
                 fontsize=13)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Mean Difference')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plot8_feature_mean_diff.png', dpi=150)
    plt.show()
    print("Saved: plot8_feature_mean_diff.png")

    # ------------------------------------------------------------------
    # PLOT 9: Age and cholesterol distributions split by disease status
    # Side-by-side boxplots help spot where the two groups separate
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Age by disease status
    df_clean.boxplot(column='age', by='target', ax=axes[0],
                     boxprops=dict(color='steelblue'))
    axes[0].set_title('Age by Disease Status')
    axes[0].set_xlabel('0 = No Disease | 1 = Disease')
    axes[0].set_ylabel('Age (years)')

    # Cholesterol by disease status
    df_clean.boxplot(column='chol', by='target', ax=axes[1],
                     boxprops=dict(color='coral'))
    axes[1].set_title('Cholesterol by Disease Status')
    axes[1].set_xlabel('0 = No Disease | 1 = Disease')
    axes[1].set_ylabel('Cholesterol (mg/dl)')

    fig.suptitle('PLOT 9 -- Key Clinical Features by Disease Status', fontsize=13)
    plt.tight_layout()
    plt.savefig('plot9_clinical_boxplots.png', dpi=150)
    plt.show()
    print("Saved: plot9_clinical_boxplots.png")

    # ------------------------------------------------------------------
    # PLOT 10: Chest pain type vs disease (stacked bar)
    # ------------------------------------------------------------------
    cp_labels = {0: 'Typical Angina', 1: 'Atypical Angina',
                 2: 'Non-Anginal', 3: 'Asymptomatic'}
    cp_counts = df_clean.groupby(['cp', 'target']).size().unstack(fill_value=0)
    cp_counts.index = [cp_labels.get(i, str(i)) for i in cp_counts.index]
    cp_counts.columns = ['No Disease', 'Disease']

    fig, ax = plt.subplots(figsize=(8, 5))
    cp_counts.plot(kind='bar', stacked=False, color=['steelblue', 'coral'],
                   edgecolor='black', ax=ax)
    ax.set_title('PLOT 10 -- Chest Pain Type vs Heart Disease', fontsize=13)
    ax.set_xlabel('Chest Pain Type')
    ax.set_ylabel('Number of Patients')
    ax.legend(title='Diagnosis')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('plot10_chestpain_disease.png', dpi=150)
    plt.show()
    print("Saved: plot10_chestpain_disease.png")

    print("\n[MODULE 4 COMPLETE] Data exploration done.")


# =============================================================================
# MODULE 5 -- TRAIN AND EVALUATE MODELS  (Part B, Sections 4 and 5)
# PURPOSE : Build three supervised classification models, evaluate each one,
#           and compare their performance.
#           Models: Random Forest, Gradient Boosting, XGBoost
# =============================================================================

def module5_train_and_evaluate(df_clean):
    """
    Train three classifiers, evaluate with five metrics, and save result plots.

    Parameters
    ----------
    df_clean : pd.DataFrame  -- cleaned dataframe from Module 3

    Returns
    -------
    results_df : pd.DataFrame
        Table of evaluation metrics for all three models.
    best_model : fitted estimator
        The model with the highest F1-score.
    X_test, y_test : arrays
        Test partition (used again in Module 6 for clustering comparison).
    """
    print("\n" + "=" * 60)
    print("  MODULE 5 -- TRAIN AND EVALUATE MODELS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 5a: Split into features (X) and target (y)
    # ------------------------------------------------------------------
    print("\n-- STEP 5a: Prepare Features and Target --")

    X = df_clean.drop('target', axis=1)   # All columns except 'target'
    y = df_clean['target'].astype(int)    # Only the 'target' column

    print(f"Features (X) shape : {X.shape}")
    print(f"Target  (y) shape  : {y.shape}")
    print(f"Feature columns    : {X.columns.tolist()}")

    # ------------------------------------------------------------------
    # STEP 5b: Train/test split (80% training, 20% testing)
    # stratify=y keeps the same class balance in both splits
    # random_state=42 makes the split reproducible every run
    # ------------------------------------------------------------------
    print("\n-- STEP 5b: Train/Test Split (80% / 20%) --")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")

    # ------------------------------------------------------------------
    # STEP 5c: Define the three models
    # ------------------------------------------------------------------
    print("\n-- STEP 5c: Define Models --")

    models = {
        # Random Forest: builds many independent decision trees and votes.
        # Good baseline, naturally handles non-linear relationships.
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        # Gradient Boosting: builds trees sequentially, each correcting
        # the errors of the last. Often more accurate than random forest.
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        # XGBoost: an optimised implementation of gradient boosting with
        # built-in regularisation to reduce overfitting.
        'XGBoost': XGBClassifier(
            n_estimators=100, learning_rate=0.1,
            eval_metric='logloss', random_state=42
        ),
    }

    print("Models ready:")
    for name in models:
        print(f"  - {name}")

    # ------------------------------------------------------------------
    # STEP 5d: Train each model and record evaluation metrics
    # ------------------------------------------------------------------
    print("\n-- STEP 5d: Train Models and Compute Metrics --")

    results = []           # Will hold one dict of metrics per model
    trained_models = {}    # Will hold the fitted model objects
    feature_importances = {}  # Will hold feature importance arrays

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Train the model on the training partition only
        model.fit(X_train, y_train)

        # Predict hard class labels (0 or 1) for the test set
        y_pred = model.predict(X_test)

        # Predict probability of class 1 (needed for ROC-AUC)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute the five evaluation metrics
        acc  = round(accuracy_score(y_test, y_pred),  4)
        prec = round(precision_score(y_test, y_pred), 4)
        rec  = round(recall_score(y_test, y_pred),    4)
        f1   = round(f1_score(y_test, y_pred),        4)
        auc  = round(roc_auc_score(y_test, y_prob),   4)

        print(f"  Accuracy={acc}  Precision={prec}  Recall={rec}  "
              f"F1={f1}  ROC-AUC={auc}")

        results.append({
            'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc
        })

        trained_models[name] = model

        # Save feature importances (available on tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

    # Compile results into a neat table
    results_df = pd.DataFrame(results)
    print("\nFull results table:")
    print(results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # PLOT 11: Grouped bar chart comparing all metrics for all models
    # ------------------------------------------------------------------
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(results_df))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * bar_width, results_df[metric],
               width=bar_width, label=metric)

    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(results_df['Model'], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('PLOT 11 -- Model Evaluation Metric Comparison\n'
                 '(Higher is better for all metrics)', fontsize=13)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plot11_model_comparison.png', dpi=150)
    plt.show()
    print("Saved: plot11_model_comparison.png")

    # ------------------------------------------------------------------
    # PLOT 12: Confusion matrices -- one per model
    # Layout: true negatives | false positives
    #         false negatives | true positives
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'],
            annot_kws={'size': 12}
        )
        axes[i].set_title(f'{name}', fontsize=11)
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('Actual Label')

    fig.suptitle('PLOT 12 -- Confusion Matrices for All Models\n'
                 'Diagonal = correct predictions | Off-diagonal = errors',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig('plot12_confusion_matrices.png', dpi=150)
    plt.show()
    print("Saved: plot12_confusion_matrices.png")

    # ------------------------------------------------------------------
    # PLOT 13: ROC curves -- all three models on one chart
    # The closer the curve bows to the top-left, the better the model
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['steelblue', 'coral', 'green']
    for (name, model), color in zip(trained_models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name}  (AUC = {auc_val:.2f})')

    # Diagonal dashed line represents a random classifier (AUC = 0.50)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.set_title('PLOT 13 -- ROC Curves for All Models\n'
                 'Higher AUC = better discrimination', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot13_roc_curves.png', dpi=150)
    plt.show()
    print("Saved: plot13_roc_curves.png")

    # ------------------------------------------------------------------
    # PLOT 14: Feature importance charts for each model
    # Shows which features the model relied on most
    # ------------------------------------------------------------------
    for name, importance in feature_importances.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        importance.plot(kind='bar', color='steelblue',
                        edgecolor='black', ax=ax)
        ax.set_title(f'PLOT 14 -- Feature Importance: {name}\n'
                     '(Higher bar = more important feature)', fontsize=12)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance Score')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        fname = f'plot14_importance_{name.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"Saved: {fname}")

    # Identify the best model by F1-Score (most balanced single metric)
    best_row = results_df.sort_values('F1-Score', ascending=False).iloc[0]
    best_name = best_row['Model']
    best_model = trained_models[best_name]

    print(f"\nBest model by F1-Score: {best_name} "
          f"(F1={best_row['F1-Score']}  ROC-AUC={best_row['ROC-AUC']})")

    print("\n[MODULE 5 COMPLETE] Model training and evaluation done.")
    return results_df, best_model, X_test, y_test, X, y


# =============================================================================
# MODULE 6 -- CLUSTERING (Unsupervised Analysis)
# PURPOSE : Use K-Means clustering to discover natural patient groupings
#           without using the disease labels.
#           Then compare those groupings to the actual labels using ARI.
# =============================================================================

def module6_clustering(X, y):
    """
    Apply K-Means clustering and evaluate how well it separates
    patients with and without heart disease.

    Parameters
    ----------
    X : pd.DataFrame  -- full feature matrix (all rows)
    y : pd.Series     -- true binary labels (used only for ARI evaluation)
    """
    print("\n" + "=" * 60)
    print("  MODULE 6 -- CLUSTERING (UNSUPERVISED ANALYSIS)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 6a: Apply K-Means with 2 clusters (disease / no disease)
    # ------------------------------------------------------------------
    print("\n-- STEP 6a: K-Means Clustering (2 clusters) --")

    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Count how many patients landed in each cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Cluster {label}: {count} patients")

    # ------------------------------------------------------------------
    # STEP 6b: Evaluate clustering quality
    # Silhouette score: how well-separated the clusters are (-1 to +1)
    # Adjusted Rand Index: how closely clusters match true labels (0 to 1)
    # ------------------------------------------------------------------
    print("\n-- STEP 6b: Clustering Evaluation Metrics --")

    sil_score = silhouette_score(X, cluster_labels)
    ari_score = adjusted_rand_score(y, cluster_labels)

    print(f"  Silhouette Score  : {sil_score:.4f}  "
          f"(closer to +1 = well-separated clusters)")
    print(f"  Adjusted Rand Index: {ari_score:.4f}  "
          f"(closer to +1 = clusters align with true labels)")

    # ------------------------------------------------------------------
    # STEP 6c: Reduce to 2 dimensions using PCA for visualisation
    # PCA finds the two directions of maximum variance in the data,
    # allowing us to plot high-dimensional data on a 2D scatter plot.
    # ------------------------------------------------------------------
    print("\n-- STEP 6c: PCA Dimensionality Reduction for Visualisation --")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained by 2 PCA components: {variance_explained:.1f}%")

    # ------------------------------------------------------------------
    # PLOT 15: K-Means clusters in PCA space
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: colour by K-Means cluster assignment
    scatter = axes[0].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=cluster_labels, cmap='Set1', s=25, alpha=0.7
    )
    axes[0].set_title('K-Means Cluster Assignments\n(Unsupervised -- no labels used)',
                      fontsize=11)
    axes[0].set_xlabel('PCA Component 1')
    axes[0].set_ylabel('PCA Component 2')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')

    # Right: colour by TRUE disease label (for comparison)
    scatter2 = axes[1].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y, cmap='coolwarm', s=25, alpha=0.7
    )
    axes[1].set_title('True Disease Labels\n(0 = No Disease, 1 = Disease)',
                      fontsize=11)
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')
    plt.colorbar(scatter2, ax=axes[1], label='True Label')

    fig.suptitle(
        f'PLOT 15 -- K-Means Clustering vs True Labels (PCA 2D Projection)\n'
        f'ARI = {ari_score:.4f}  |  Silhouette = {sil_score:.4f}',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig('plot15_kmeans_pca.png', dpi=150)
    plt.show()
    print("Saved: plot15_kmeans_pca.png")

    # ------------------------------------------------------------------
    # PLOT 16: Elbow method -- find the optimal number of clusters
    # We plot inertia (within-cluster variance) for k = 1 to 8.
    # The "elbow" where the curve bends is the suggested best k.
    # ------------------------------------------------------------------
    print("\n-- STEP 6d: Elbow Method (Finding Optimal K) --")

    inertia_values = []
    k_range = range(1, 9)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertia_values.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertia_values, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=2, color='red', linestyle='--', label='Chosen k=2')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    ax.set_title('PLOT 16 -- Elbow Method for Optimal Number of Clusters\n'
                 'Red dashed line = chosen k=2 (disease/no disease)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot16_elbow_method.png', dpi=150)
    plt.show()
    print("Saved: plot16_elbow_method.png")

    print("\n[MODULE 6 COMPLETE] Clustering analysis done.")


# =============================================================================
# MODULE 7 -- FINAL SUMMARY
# PURPOSE : Print a consolidated summary of all results so every
#           section has a clear, collected conclusion.
# =============================================================================

def module7_final_summary(results_df):
    """
    Print a final summary of all model results.

    Parameters
    ----------
    results_df : pd.DataFrame  -- evaluation table from Module 5
    """
    print("\n" + "=" * 60)
    print("  MODULE 7 -- FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\nSupervised Model Performance (all metrics):")
    print("-" * 60)
    print(results_df.to_string(index=False))
    print("-" * 60)

    # Identify the best model on each metric individually
    print("\nBest model per metric:")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        best_idx = results_df[metric].idxmax()
        best_name  = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        print(f"  {metric:<12}: {best_name}  ({best_score})")

    # Overall winner by F1-Score (best single balanced metric)
    best_row = results_df.sort_values('F1-Score', ascending=False).iloc[0]
    print(f"\nOverall recommended model : {best_row['Model']}")
    print(f"  F1-Score  = {best_row['F1-Score']}")
    print(f"  ROC-AUC   = {best_row['ROC-AUC']}")
    print(f"  Accuracy  = {best_row['Accuracy']}")

    print("\nFiles saved in the working directory:")
    plot_files = [
        "plot1_missing_heatmap.png",
        "plot2_missing_bar.png",
        "plot3_boxplots_before.png",
        "plot4_boxplots_after.png",
        "plot5_target_distribution.png",
        "plot6_histograms.png",
        "plot7_correlation_heatmap.png",
        "plot8_feature_mean_diff.png",
        "plot9_clinical_boxplots.png",
        "plot10_chestpain_disease.png",
        "plot11_model_comparison.png",
        "plot12_confusion_matrices.png",
        "plot13_roc_curves.png",
        "plot14_importance_*.png",
        "plot15_kmeans_pca.png",
        "plot16_elbow_method.png",
        "heart_disease_cleaned.csv",
    ]
    for f in plot_files:
        print(f"  {f}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE -- ALL MODULES FINISHED SUCCESSFULLY")
    print("=" * 60)


# =============================================================================
# MAIN -- RUN ALL MODULES IN ORDER
# This block only runs when you execute the file directly.
# In Jupyter, you can run each module function individually in separate cells.
# =============================================================================

if __name__ == '__main__':

    # ── STEP 0: Set the path to your CSV file ────────────────────────────────
    # If the CSV is in the same folder as this script, just use the filename.
    # Otherwise, provide the full path, e.g. 'C:/Users/You/Downloads/data.csv'
    DATA_FILE = 'heart_disease_uci.csv'

    # ── Run all modules in sequence ──────────────────────────────────────────

    # Module 1: Load the raw CSV
    df_raw = module1_load_data(DATA_FILE)

    # Module 2: Assess data quality (before cleaning)
    module2_assess_quality(df_raw)

    # Module 3: Clean and prepare the data
    df_clean = module3_clean_data(df_raw)

    # Module 4: Explore the cleaned data with charts and statistics
    module4_explore_data(df_clean)

    # Module 5: Train models, evaluate, and plot results
    results_df, best_model, X_test, y_test, X, y = module5_train_and_evaluate(df_clean)

    # Module 6: Clustering -- unsupervised analysis
    module6_clustering(X, y)

    # Module 7: Print final summary of all results
    module7_final_summary(results_df)
