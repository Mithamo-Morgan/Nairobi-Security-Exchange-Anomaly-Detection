# Anomaly Detection Model for Nairobi Securities Exchange Stocks

## Problem Statement
Monitoring daily trading activity across all listed securities is essential for maintaining a fair, transparent, and orderly securities market. However, manually identifying unusual trading behaviour from large volumes of market data can be time-consuming and challenging.

This notebook develops an intelligent anomaly detection model using historical trading data from the Nairobi Securities Exchange (NSE). The model is intended to support the NSE's market surveillance function by providing an early warning system that flags securities requiring further investigation.

## Objective
To develop an anomaly detection model that identifies securities exhibiting abnormal trading behaviour using historical daily trading data from the Nairobi Securities Exchange (NSE).

## Automation
We'll convert the Jupyter Notebook used for experimentation into python scripts so we can automate the whole process from ingestion to dashboard using Apache Airflow.

## Architecture
NSE/
│
├── data/
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_preprocessing.py
│   ├── anomaly_model.py
│   ├── evaluation.py
│   ├── explanations.py
│   ├── dashboard_data.py
│   └── utils.py
│
├── outputs/
│
├── pipeline.py
└── requirements.txt
