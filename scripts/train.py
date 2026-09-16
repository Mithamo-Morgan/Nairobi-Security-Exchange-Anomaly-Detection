"""
Training pipeline entry point.
"""

from src.ingest import load_training_data, get_dataset_summary



def main():

    # Location of historical NSE data
    training_file = (
        "data/training/nse_historical_data.csv"
    )

    # Load training dataset
    data = load_training_data(
        training_file
    )


    # Display dataset information
    summary = get_dataset_summary(data)

    print("\nTraining Dataset Loaded Successfully")
    print("-----------------------------------")

    for key, value in summary.items():
        print(f"{key}: {value}")



if __name__ == "__main__":
    main()