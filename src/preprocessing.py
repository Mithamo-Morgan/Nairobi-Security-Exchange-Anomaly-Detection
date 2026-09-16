"""
Preprocessing module.

Responsibilities:
- Convert raw columns into correct data types
- Handle missing values
- Prepare dataframe before feature engineering
"""

import pandas as pd


NUMERIC_COLUMNS = [
    "52wk_high",
    "52wk_low",
    "bid",
    "ask",
    "high",
    "low",
    "previous_price",
    "volume",
    "vwap",
    "total_no_of_shares_issued",
    "mrt_cap",
    "10_up",
    "10_down"
]


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NSE dataframe before feature engineering.

    Parameters
    ----------
    data : pd.DataFrame
        Output from ingest.py

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe
    """

    # Date-based helper columns
    data["year"] = data["date"].dt.year

    # Clean numeric columns

    for column in NUMERIC_COLUMNS:

        data[column] = (
            data[column]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .replace(
                {
                    "-": pd.NA,
                    "": pd.NA
                }
            )
        )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Clean company names

    data["company"] = (
        data["company"]
        .str.strip()
    )

    # Fill missing status

    data["status"] = (
        data["status"]
        .fillna("ord")
    )

    # Fill missing trading limits

    data["10_up"] = (
        data["10_up"]
        .fillna(
            data["previous_price"] * 1.10
        )
    )


    data["10_down"] = (
        data["10_down"]
        .fillna(
            data["previous_price"] * 0.90
        )
    )

    # Sort before historical filling

    data = (
        data
        .sort_values(
            ["company", "date"]
        )
        .reset_index(drop=True)
    )

    # Fill missing previous price

    data["previous_price"] = (
        data["previous_price"]
        .fillna(
            data
            .groupby("company")["vwap"]
            .shift(1)
        )
    )

    # Missing volume means no trading

    data["volume"] = (
        data["volume"]
        .fillna(0)
    )

    # Fill missing shares issued

    data["total_no_of_shares_issued"] = (
        data["total_no_of_shares_issued"]
        .fillna(
            data
            .groupby(
                ["year", "company"]
            )["total_no_of_shares_issued"]
            .transform("max")
        )
    )


    # NSE specific corrections

    data.loc[
        data["company"]
        ==
        "Kenya Power & Lighting Plc 4% Pref 20.00",
        "total_no_of_shares_issued"
    ] = 1800000


    data.loc[
        data["company"]
        ==
        "Kenya Power & Lighting Plc 7% Pref 20.00",
        "total_no_of_shares_issued"
    ] = 350000


    # ----------------------------------
    # Fill missing market cap
    # ----------------------------------

    data["mrt_cap"] = (
        data["mrt_cap"]
        .fillna(
            data["total_no_of_shares_issued"]
            *
            data["vwap"]
        )
    )

    # Drop regulatory limit columns

    data = data.drop(
        columns=[
            "10_up",
            "10_down"
        ]
    )


    return data