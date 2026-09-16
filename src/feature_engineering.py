"""
Feature engineering module.

Responsibilities:
- Create market behaviour features
- Capture unusual price, volume and liquidity movements
"""

import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create NSE anomaly detection features.

    Parameters
    ----------
    data : pd.DataFrame
        Clean dataframe from ingest.py

    Returns
    -------
    pd.DataFrame
        Dataframe containing engineered features
    """

    # Ensure chronological order
    data = (
        data
        .sort_values(["company", "date"])
        .reset_index(drop=True)
    )


    # -----------------------------
    # Price movement features
    # -----------------------------

    # Daily return relative to previous closing price
    data["price_return"] = (
        (data["vwap"] - data["previous_price"])
        /
        data["previous_price"]
    )


    # Intraday volatility/range
    data["intraday_range"] = (
        (data["high"] - data["low"])
        /
        data["previous_price"]
    )


    # Distance from yearly high
    data["distance_to_52wk_high"] = (
        (data["52wk_high"] - data["vwap"])
        /
        data["52wk_high"]
    )


    # Distance from yearly low
    data["distance_to_52wk_low"] = (
        (data["vwap"] - data["52wk_low"])
        /
        data["52wk_low"]
    )


    # -----------------------------
    # Volume behaviour features
    # -----------------------------

    # Trading activity relative to company size
    data["turnover_ratio"] = (
        data["volume"]
        /
        data["total_no_of_shares_issued"]
    )


    # Previous 5-day average volume
    previous_volume_avg = (
        data
        .groupby("company")["volume"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(
                 window=5,
                 min_periods=1
             )
             .mean()
        )
    )


    # Relative volume compared to normal activity
    data["relative_volume"] = (
        data["volume"]
        /
        previous_volume_avg
    )


    # First observations have no history
    data["relative_volume"] = (
        data["relative_volume"]
        .fillna(1)
    )


    # -----------------------------
    # Liquidity features
    # -----------------------------

    # Bid discount
    data["bid_discount"] = (
        (data["vwap"] - data["bid"])
        /
        data["vwap"]
    )


    # Ask premium
    data["ask_premium"] = (
        (data["ask"] - data["vwap"])
        /
        data["vwap"]
    )


    # Bid-ask spread relative to price
    data["relative_spread"] = (
        (data["ask"] - data["bid"])
        /
        data["vwap"]
    )


    return data