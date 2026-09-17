"""
Responsibilities:
- Load raw NSE training data
- Normalize column names
- Validate dataset schema
- Convert date columns
"""

from pathlib import Path
import pandas as pd

# Expected raw NSE dataset columns after cleaning
REQUIRED_COLUMNS = [
    "date",
    "52wk_high",
    "52wk_low",
    "sector",
    "company",
    "isin_code",
    "status",
    "bid",
    "ask",
    "high",
    "low",
    "previous_price",
    "volume",
    "total_no_of_shares_issued",
    "mrt_cap",
    "vwap",
    "10_up",
    "10_down"
]

def load_training_data(file_path: str) -> pd.DataFrame:
    """
    Load historical NSE data used for model training.

    Parameters
    ----------
    file_path : str
        Path to training CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned and validated raw dataframe.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {file_path}"
        )

    # Load raw CSV
    data = pd.read_csv(file_path)

    # Standardize column names
    data = clean_column_names(data)

    # Validate required columns
    validate_schema(data)

    # Convert date column
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # Check invalid dates
    invalid_dates = data["date"].isna().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"{invalid_dates} invalid date values found."
        )

    return data

def load_inference_data(file_path: str) -> pd.DataFrame:
    """
    Load daily NSE data used for anomaly detection.

    Parameters
    ----------
    file_path : str
        Path to inference CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned and validated inference dataframe.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Inference dataset not found: {file_path}"
        )


    # Load raw CSV
    data = pd.read_csv(file_path)


    # Standardize column names
    data = clean_column_names(data)


    # Validate required columns
    validate_schema(data)


    # Convert date column
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )


    # Check invalid dates
    invalid_dates = data["date"].isna().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"{invalid_dates} invalid date values found."
        )


    # Inference should represent one trading day
    unique_dates = data["date"].nunique()

    if unique_dates != 1:

        raise ValueError(
            f"Inference data should contain one trading day. "
            f"Found {unique_dates} dates."
        )


    return data


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe column names.

    Converts:
        '52WK HIGH' -> '52wk_high'
        'ISIN CODE' -> 'isin_code'
        'MRT CAP ' -> 'mrt_cap'

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    data.columns = (
        data.columns
        .str.replace("\xa0", " ", regex=False)  # remove NB spaces
        .str.strip()
        .str.lower()
        .str.replace("%", "", regex=False)
        .str.replace(r"\s+", "_", regex=True)
    )

    return data



def validate_schema(data: pd.DataFrame):
    """
    Validate that all required NSE columns exist.

    Parameters
    ----------
    data : pd.DataFrame

    Raises
    ------
    ValueError
        If columns are missing.
    """

    missing_columns = (
        set(REQUIRED_COLUMNS)
        -
        set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def get_dataset_summary(data: pd.DataFrame) -> dict:
    """
    Provide basic dataset information.

    Used later for training metadata.

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    dict
    """

    summary = {
        "rows": len(data),
        "columns": len(data.columns),
        "start_date": data["date"].min(),
        "end_date": data["date"].max(),
    }

    return summary