"""
Production inference pipeline.

Responsibilities:
-----------------
- Load daily NSE inference data
- Generate features
- Load trained artifacts
- Detect anomalies
- Generate explanations
- Save anomaly results
"""

import pandas as pd

from src.ingest import load_inference_data

from src.preprocessing import preprocess

from src.feature_engineering import create_features

from src.model import (
    prepare_inference_data,
    predict_model
)

from src.persistence import (
    load_model_artifacts
)

from src.explanation_engine import (
    build_explanation
)



def main():

    # ----------------------------------
    # Load inference data
    # ----------------------------------

    inference_file = (
        "data/inference/test_data.csv"
    )


    data = load_inference_data(
        inference_file
    )


    print(
        "Inference data shape:",
        data.shape
    )

    # preprocess
    data = preprocess(
        data
    )


    # ----------------------------------
    # Feature engineering
    # ----------------------------------

    data = create_features(
        data
    )


    print(
        "Feature engineering completed"
    )


    # ----------------------------------
    # Load trained artifacts
    # ----------------------------------

    artifacts = load_model_artifacts()


    model = artifacts["model"]

    scaler = artifacts["scaler"]

    feature_columns = artifacts["feature_columns"]

    numeric_features = artifacts["numeric_features"]

    thresholds = artifacts["thresholds"]


    print(
        "Artifacts loaded"
    )


    # ----------------------------------
    # Prepare model input
    # ----------------------------------

    X, results_info = prepare_inference_data(
        data,
        scaler,
        feature_columns,
        numeric_features
    )


    print(
        "Inference features prepared:",
        X.shape
    )


    # ----------------------------------
    # Predict anomalies
    # ----------------------------------

    anomaly, anomaly_score = predict_model(
        model,
        X
    )


    # ----------------------------------
    # Combine predictions with metadata
    # ----------------------------------

    results = results_info.copy()


    results["anomaly"] = anomaly

    results["anomaly_score"] = anomaly_score



    # ----------------------------------
    # Keep only detected anomalies
    # ----------------------------------

    anomalies = results[
        results["anomaly"] == -1
    ].copy()


    print(
        f"Detected anomalies: {len(anomalies)}"
    )



    # ----------------------------------
    # Generate explanations
    # ----------------------------------

    if len(anomalies) > 0:

        explanations = anomalies.apply(
            lambda row:
                build_explanation(
                    row,
                    thresholds
                ),
            axis=1
        )


        anomalies = pd.concat(
            [
                anomalies,
                explanations
            ],
            axis=1
        )



    # ----------------------------------
    # Save results
    # ----------------------------------

    anomalies.to_csv(
        "results/anomaly_report.csv",
        index=False
    )


    print(
        "Anomaly report saved."
    )



if __name__ == "__main__":

    main()