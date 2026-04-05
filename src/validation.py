"""
Data validation for CERN dielectron dataset.

Runs sanity checks on the raw and cleaned DataFrame before any modelling
begins.  Raises ValueError with a descriptive message if a check fails,
so problems surface immediately rather than silently corrupting results.
"""

import pandas as pd

# Columns that must be present in the raw CSV (after stripping whitespace)
REQUIRED_COLUMNS = [
    'Run', 'Event',
    'E1', 'px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'Q1',
    'E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2', 'phi2', 'Q2',
    'M',
]

# Features that must be strictly positive (used in log-transform)
LOG_FEATURES = ['E1', 'E2', 'pt1', 'pt2']


def validate_raw(df: pd.DataFrame) -> None:
    """Validate the DataFrame immediately after loading from CSV.

    Checks
    ------
    1. All required columns are present (handles trailing-space column names).
    2. The target M is strictly positive wherever it is present.
    3. Log-transform features are strictly positive.
    4. Missing target values are tolerated at this stage and are cleaned later.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by pd.read_csv().

    Raises
    ------
    ValueError
        If any check fails.
    """
    # Strip whitespace from column names to handle trailing-space issue (L1)
    df.columns = df.columns.str.strip()

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns.tolist())}"
        )

    m_non_null = df['M'].dropna()
    if m_non_null.empty:
        raise ValueError("Target column M has no valid values after excluding NaNs.")

    if not (m_non_null > 0).all():
        raise ValueError(
            f"Target M has {(m_non_null <= 0).sum()} non-positive values. "
            "Invariant mass must be strictly positive."
        )

    for feat in LOG_FEATURES:
        n_bad = (df[feat] <= 0).sum()
        if n_bad > 0:
            raise ValueError(
                f"Feature '{feat}' has {n_bad} non-positive values. "
                "Cannot apply log-transform."
            )


def validate_clean(df_clean: pd.DataFrame) -> None:
    """Validate the working DataFrame after identifier columns are dropped.

    Checks
    ------
    1. No NaN values anywhere.
    2. M is strictly positive.
    3. Log-transform features are strictly positive.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned DataFrame (Run/Event columns already removed).

    Raises
    ------
    ValueError
        If any check fails.
    """
    n_nan = df_clean.isnull().sum().sum()
    if n_nan > 0:
        raise ValueError(
            f"Cleaned DataFrame contains {n_nan} NaN values:\n"
            f"{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}"
        )

    if not (df_clean['M'] > 0).all():
        raise ValueError(
            "Target M contains non-positive values after cleaning."
        )

    for feat in LOG_FEATURES:
        if feat in df_clean.columns:
            n_bad = (df_clean[feat] <= 0).sum()
            if n_bad > 0:
                raise ValueError(
                    f"Feature '{feat}' has {n_bad} non-positive values after cleaning."
                )
