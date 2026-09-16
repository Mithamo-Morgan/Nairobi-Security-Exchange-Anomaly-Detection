"""
Responsibilities:
-----------------
- Calculate behavioural thresholds from historical data
- Explain why an observation was flagged
"""


import pandas as pd



def calculate_thresholds(train_df):

    thresholds = {

        "relative_volume":
            train_df["relative_volume"].quantile(0.95),

        "turnover_ratio":
            train_df["turnover_ratio"].quantile(0.95),

        "intraday_range":
            train_df["intraday_range"].quantile(0.95),

        "relative_spread":
            train_df["relative_spread"].quantile(0.95),

        "bid_discount":
            train_df["bid_discount"].quantile(0.95),

        "ask_premium":
            train_df["ask_premium"].quantile(0.95),

        "price_return":
            train_df["price_return"]
            .abs()
            .quantile(0.95)
    }


    return thresholds



def build_explanation(row, thresholds):

    evidence = []
    triggered = []


    if row["relative_volume"] > thresholds["relative_volume"]:

        triggered.append("volume")

        evidence.append(
            f"Trading volume was "
            f"{row['relative_volume']:.1f} times "
            f"above its recent average."
        )


    if abs(row["price_return"]) > thresholds["price_return"]:

        triggered.append("price")

        direction = (
            "increased"
            if row["price_return"] > 0
            else "decreased"
        )

        evidence.append(
            f"Share price {direction} by "
            f"{abs(row['price_return']):.1%}."
        )


    if row["intraday_range"] > thresholds["intraday_range"]:

        triggered.append("volatility")

        evidence.append(
            f"Intraday price range reached "
            f"{row['intraday_range']:.1%}."
        )


    if row["relative_spread"] > thresholds["relative_spread"]:

        triggered.append("liquidity")

        evidence.append(
            "Bid-ask spread widened, indicating reduced liquidity."
        )


    if row["turnover_ratio"] > thresholds["turnover_ratio"]:

        triggered.append("turnover")

        evidence.append(
            "A high proportion of outstanding shares changed hands."
        )


    if len(triggered) >= 2:

        primary = "Multi-factor Trading Anomaly"
        category = "Multi-factor"


    elif "volume" in triggered:

        primary = "Abnormal Trading Volume"
        category = "Trading Activity"


    elif "price" in triggered:

        primary = "Large Price Movement"
        category = "Price Behaviour"


    elif "liquidity" in triggered:

        primary = "Liquidity Deterioration"
        category = "Liquidity"


    elif "volatility" in triggered:

        primary = "High Intraday Volatility"
        category = "Volatility"


    elif "turnover" in triggered:

        primary = "High Market Activity"
        category = "Trading Activity"


    else:

        primary = "Unusual Market Behaviour Pattern"

        category = "Model Detected Pattern"

        evidence.append(
            "Price movement, trading volume and liquidity "
            "indicators were within normal ranges, but the "
            "combined behaviour differed from historical patterns."
        )


    return pd.Series({

        "primary_reason": primary,

        "supporting_evidence":
            " | ".join(evidence),

        "investigation_category":
            category
    })