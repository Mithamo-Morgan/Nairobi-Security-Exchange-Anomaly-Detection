from src.ingest import load_training_data
from src.preprocessing import preprocess
from src.feature_engineering import create_features


def main():

    training_file = (
        "data/training/nse_historical_data.csv"
    )


    # 1. Load raw data
    data = load_training_data(training_file)

    print(
        "After ingestion:",
        data.shape
    )


    # 2. Clean and prepare data
    data = preprocess(data)

    print(
        "After preprocessing:",
        data.shape
    )


    # 3. Create anomaly features
    data = create_features(data)

    print(
        "After feature engineering:",
        data.shape
    )


if __name__ == "__main__":
    main()