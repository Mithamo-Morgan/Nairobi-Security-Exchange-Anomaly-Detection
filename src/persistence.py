"""
Persistence module.

Responsibilities:
-----------------
- Save trained model artifacts
- Save preprocessing artifacts
- Save explanation thresholds
- Save training metadata
- Load artifacts for inference
"""

import os
import json
import joblib
from datetime import datetime



# -------------------------------------------------------
# Generic artifact saving
# -------------------------------------------------------

def save_artifact(obj, path):
    """
    Save Python object using joblib.

    Examples:
    - model
    - scaler
    - feature columns
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    joblib.dump(
        obj,
        path
    )



# -------------------------------------------------------
# Generic artifact loading
# -------------------------------------------------------

def load_artifact(path):
    """
    Load saved joblib object.
    """

    return joblib.load(path)



# -------------------------------------------------------
# Save model
# -------------------------------------------------------

def save_model(model):

    save_artifact(
        model,
        "artifacts/models/isolation_forest.pkl"
    )



# -------------------------------------------------------
# Save preprocessing artifacts
# -------------------------------------------------------

def save_preprocessing_artifacts(
        scaler,
        feature_columns,
        numeric_features
):

    save_artifact(
        scaler,
        "artifacts/preprocessing/scaler.pkl"
    )


    save_artifact(
        feature_columns,
        "artifacts/preprocessing/feature_columns.pkl"
    )


    save_artifact(
        numeric_features,
        "artifacts/preprocessing/numeric_features.pkl"
    )



# -------------------------------------------------------
# Save explanation thresholds
# -------------------------------------------------------

def save_thresholds(thresholds):
    """
    Save feature behaviour thresholds
    used by explanation engine.
    """

    os.makedirs(
        "artifacts/thresholds",
        exist_ok=True
    )


    with open(
        "artifacts/thresholds/feature_thresholds.json",
        "w"
    ) as file:

        json.dump(
            thresholds,
            file,
            indent=4
        )



# -------------------------------------------------------
# Load explanation thresholds
# -------------------------------------------------------

def load_thresholds():
    """
    Load feature behaviour thresholds
    for explanation generation.
    """

    with open(
        "artifacts/thresholds/feature_thresholds.json",
        "r"
    ) as file:

        thresholds = json.load(file)


    return thresholds



# -------------------------------------------------------
# Save metadata
# -------------------------------------------------------

def save_metadata(metadata):

    os.makedirs(
        "artifacts/metadata",
        exist_ok=True
    )


    metadata["created_at"] = (
        datetime.now()
        .isoformat()
    )


    with open(
        "artifacts/metadata/training_metadata.json",
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            default=str
        )



# -------------------------------------------------------
# Load everything needed for inference
# -------------------------------------------------------

def load_model_artifacts():

    artifacts = {}


    # Model

    artifacts["model"] = load_artifact(
        "artifacts/models/isolation_forest.pkl"
    )


    # Preprocessing

    artifacts["scaler"] = load_artifact(
        "artifacts/preprocessing/scaler.pkl"
    )


    artifacts["feature_columns"] = load_artifact(
        "artifacts/preprocessing/feature_columns.pkl"
    )


    artifacts["numeric_features"] = load_artifact(
        "artifacts/preprocessing/numeric_features.pkl"
    )


    # Explanation thresholds

    artifacts["thresholds"] = load_thresholds()


    return artifacts