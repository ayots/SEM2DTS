"""
Heart Disease – Messy Data Cleaning & EDA (Modular Version)

Dataset: messy_heart_disease.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Data Loading
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the messy heart disease dataset.

    - Treats '?' and 'null' as missing values.
    - Returns a pandas DataFrame.
    """
    df = pd.read_csv(
        filepath,
        na_values=["?", "null", "NULL", "NaN", "nan", ""]
    )
    return df


# ============================================================
# 2. EDA Utilities (Reusable)
# ============================================================

def basic_eda(df: pd.DataFrame, title_prefix: str = "") -> None:
    """
    Perform basic EDA:
    - Shape
    - Info
    - Missing values
    - Basic statistics (numeric)
    """
    print("=" * 60)
    print(f"{title_prefix} BASIC EDA")
    print("=" * 60)
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nNumeric summary:")
    print(df.describe())


def plot_target_distribution(df: pd.DataFrame, title_prefix: str = "") -> None:
    """
    Plot the distribution of the target variable.
    """
    plt.figure()
    df["target"].value_counts().sort_index().plot(kind="bar")
    plt.title(f"{title_prefix}Heart Disease Distribution")
    plt.xlabel("Target (0 = No Disease, 1 = Disease)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_age_distribution(df: pd.DataFrame, title_prefix: str = "") -> None:
    """
    Plot histogram of age (if available).
    """
    if "age" not in df.columns:
        print("Column 'age' not found, skipping age distribution plot.")
        return

    plt.figure()
    df["age"].dropna().astype(float).hist(bins=20)
    plt.title(f"{title_prefix}Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_chol_vs_target(df: pd.DataFrame, title_prefix: str = "") -> None:
    """
    Scatter plot of cholesterol vs target.
    """
    if not {"chol", "target"}.issubset(df.columns):
        print("Columns 'chol' or 'target' not found, skipping scatter plot.")
        return

    plt.figure()
    plt.scatter(df["chol"], df["target"], alpha=0.7)
    plt.title(f"{title_prefix}Cholesterol vs Heart Disease")
    plt.xlabel("Cholesterol")
    plt.ylabel("Heart Disease (Target)")
    plt.tight_layout()
    plt.show()


def plot_cp_vs_target(df: pd.DataFrame, title_prefix: str = "") -> None:
    """
    Bar plot of chest pain type vs mean target (probability of disease).
    """
    if not {"cp", "target"}.issubset(df.columns):
        print("Columns 'cp' or 'target' not found, skipping CP vs target plot.")
        return

    plt.figure()
    df.groupby("cp")["target"].mean().plot(kind="bar")
    plt.title(f"{title_prefix}Chest Pain Type vs Heart Disease Rate")
    plt.xlabel("Chest Pain Type (cp)")
    plt.ylabel("Heart Disease Probability")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, title_prefix: str = "") -> None:
    """
    Heat-style correlation matrix using matplotlib only.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        print("No numeric columns available for correlation matrix.")
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(10, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"{title_prefix}Feature Correlation Matrix")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.show()


def full_eda(df: pd.DataFrame, title_prefix: str = "") -> None:
    """
    Run a full EDA suite:
    - Basic EDA
    - Target distribution
    - Age distribution
    - Cholesterol vs target
    - Chest pain vs target
    - Correlation matrix
    """
    basic_eda(df, title_prefix=title_prefix)
    plot_target_distribution(df, title_prefix=title_prefix)
    plot_age_distribution(df, title_prefix=title_prefix)
    plot_chol_vs_target(df, title_prefix=title_prefix)
    plot_cp_vs_target(df, title_prefix=title_prefix)
    plot_correlation_matrix(df, title_prefix=title_prefix)


# ============================================================
# 3. Cleaning Functions
# ============================================================

def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names:
    - Strip spaces
    - Lowercase
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise key categorical columns:
    - sex
    - chest pain type (cp)
    - thal
    """
    df = df.copy()

    # Standardise 'sex'
    if "sex" in df.columns:
        df["sex"] = (
            df["sex"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "m": "male",
                "male": "male",
                "f": "female",
                "female": "female"
            })
        )

    # Create 'cp' from 'chestpaintype' if needed
    if "cp" not in df.columns and "chestpaintype" in df.columns:
        df["cp"] = df["chestpaintype"]

    # Standardise 'cp'
    if "cp" in df.columns:
        df["cp"] = (
            df["cp"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # Standardise 'thal' if present
    if "thal" in df.columns:
        df["thal"] = (
            df["thal"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert appropriate columns to numeric, coercing errors to NaN.
    """
    df = df.copy()

    numeric_cols = [
        "age", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca",
        "target"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are clearly not useful for modelling/EDA:
    - notes
    - extra_col
    - chestpaintype (if cp exists)
    """
    df = df.copy()
    for col in ["notes", "extra_col"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # If both cp and chestpaintype exist, keep cp only
    if "cp" in df.columns and "chestpaintype" in df.columns:
        df = df.drop(columns=["chestpaintype"])

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Drop rows where target is missing.
    - For key numeric columns, drop rows with missing values (for this small dataset).
    """
    df = df.copy()

    # Drop rows with missing target (cannot use them)
    if "target" in df.columns:
        df = df.dropna(subset=["target"])

    # For this lab, we keep it simple and drop rows with missing in key features
    key_numeric = ["age", "trestbps", "chol", "thalach"]
    existing_keys = [c for c in key_numeric if c in df.columns]
    if existing_keys:
        df = df.dropna(subset=existing_keys)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """
    df = df.copy()
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows.")
    return df


def remove_extreme_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally remove extreme outliers in key numeric columns using simple rules.
    This is intentionally simple for teaching purposes.
    """
    df = df.copy()

    # Example simple rules (can be discussed in class)
    if "chol" in df.columns:
        df = df[df["chol"].isna() | (df["chol"] < 600)]

    if "age" in df.columns:
        df = df[df["age"].isna() | ((df["age"] >= 18) & (df["age"] <= 100))]

    return df


def full_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline in a clear, ordered way.
    """
    print("\n=== Cleaning Step 1: Standardise column names ===")
    df = standardise_column_names(df)

    print("\n=== Cleaning Step 2: Clean categorical values ===")
    df = clean_categorical_values(df)

    print("\n=== Cleaning Step 3: Convert numeric columns ===")
    df = convert_numeric_columns(df)

    print("\n=== Cleaning Step 4: Drop irrelevant columns ===")
    df = drop_irrelevant_columns(df)

    print("\n=== Cleaning Step 5: Handle missing values ===")
    df = handle_missing_values(df)

    print("\n=== Cleaning Step 6: Remove duplicates ===")
    df = remove_duplicates(df)

    print("\n=== Cleaning Step 7: Remove extreme outliers (simple rules) ===")
    df = remove_extreme_outliers(df)

    print("\n=== Cleaning complete ===")
    return df


# ============================================================
# 4. Main Workflow
# ============================================================

def main():
    # --------------------------------------------------------
    # Load messy data
    # --------------------------------------------------------
    filepath = "heart_disease.csv"
    df_raw = load_data(filepath)

    # --------------------------------------------------------
    # EDA BEFORE CLEANING
    # --------------------------------------------------------
    print("\n\n############################")
    print("# EDA ON MESSY DATA (RAW) #")
    print("############################\n")
    full_eda(df_raw, title_prefix="(Before Cleaning) ")

    # --------------------------------------------------------
    # CLEANING PIPELINE
    # --------------------------------------------------------
    df_clean = full_cleaning_pipeline(df_raw)

    # --------------------------------------------------------
    # EDA AFTER CLEANING
    # --------------------------------------------------------
    print("\n\n#############################")
    print("# EDA ON CLEANED DATA      #")
    print("#############################\n")
    full_eda(df_clean, title_prefix="(After Cleaning) ")

    # --------------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------------
    output_path = "cleaned_heart_disease.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
