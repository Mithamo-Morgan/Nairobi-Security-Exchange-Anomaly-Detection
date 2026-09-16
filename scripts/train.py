from src.ingest import load_training_data
from src.preprocessing import preprocess
from src.feature_engineering import create_features

from src.model import (prepare_training_data, train_model)


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

    # 4
    (
        X,
        results_info,
        scaler,
        feature_columns,
        numeric_features
    ) = prepare_training_data(data)

    print("Training matrix:", X.shape)

    # 5
    model = train_model(X)

    print(feature_columns)


if __name__ == "__main__":
    main()