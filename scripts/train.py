from src.ingest import load_training_data
from src.preprocessing import preprocess
from src.feature_engineering import create_features

from src.model import (prepare_training_data, train_model)
from src.explanation_engine import (calculate_thresholds)
from src.persistence import (save_model, save_preprocessing_artifacts, save_thresholds, save_metadata)


def main():

    training_file = (
        "data/training/nse_historical_data.csv"
    )

    # 1
    data = load_training_data(training_file)

    # 2
    data = preprocess(data)

    # 3
    data = create_features(data)

    # Calculate explanation thresholds
    thresholds = calculate_thresholds(data)

    # 4
    (
        X,
        results_info,
        scaler,
        feature_columns,
        numeric_features
    ) = prepare_training_data(data)

    # 5
    model = train_model(X)

    # save artifacts
    save_model(model)

    save_preprocessing_artifacts(scaler, feature_columns, numeric_features)

    save_thresholds(thresholds)

    save_metadata(
        {
            "training_rows": len(X),
            "features": len(feature_columns),
            "model": "IsolationForest",
            "contamination": 0.01
        }
    )

if __name__ == "__main__":
    main()