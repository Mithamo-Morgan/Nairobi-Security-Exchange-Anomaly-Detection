"""
NSE Market Surveillance System

Model module.

Responsibilities
----------------
- Prepare model input
- Train Isolation Forest
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Columns preserved for reporting/dashboard
RESULT_COLUMNS = [
    "date",
    "company",
    "isin_code",
    "sector",
    "status",
    "previous_price",
    "vwap",
    "volume",
    "total_no_of_shares_issued",
    "mrt_cap"
]


# Prepare training data
def prepare_training_data(data: pd.DataFrame):
    """
    Prepare dataframe for model training.

    Returns:
    - Model input matrix
    - Reporting information
    - Scaler
    - Feature names
    - Numeric feature names
    """

    # Preserve human-readable information

    results_info = data[
        RESULT_COLUMNS
    ].copy()

    # Remove only fields that should not
    # directly enter the model

    model_data = data.drop(
        columns=[
            "date",
            "company",
            "isin_code",
            "previous_price",
            "vwap",
            "volume",
            "total_no_of_shares_issued",
            "mrt_cap"
        ]
    )


    # ------------------------------------
    # Encode categorical variables
    # ------------------------------------

    model_data = pd.get_dummies(
        model_data,
        columns=[
            "sector",
            "status"
        ],
        drop_first=False,
        dtype=int
    )

    # Identify numerical columns

    numeric_features = (
        model_data
        .select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )


    # Keep exact feature order
    feature_columns = (
        model_data
        .columns
        .tolist()
    )


    # ------------------------------------
    # Scale numerical features
    # ------------------------------------

    scaler = StandardScaler()


    model_data[numeric_features] = (
        scaler
        .fit_transform(
            model_data[numeric_features]
        )
    )


    return (
        model_data,
        results_info,
        scaler,
        feature_columns,
        numeric_features
    )


# -------------------------------------------------------
# Train Isolation Forest
# -------------------------------------------------------

def train_model(X):

    model = IsolationForest(

        n_estimators=300,

        contamination=0.01,

        max_samples=1024,

        max_features=0.8,

        random_state=42,

        n_jobs=-1
    )


    model.fit(X)


    return model