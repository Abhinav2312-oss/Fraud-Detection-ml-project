# Fraud Detection AI

An end-to-end machine learning project for detecting fraudulent financial transactions using the PaySim dataset.

## Features

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Handling highly imbalanced fraud data
- SGD Classifier baseline
- XGBoost model
- PyTorch Deep Learning model
- Model comparison
- Threshold optimization
- SHAP explainability
- Streamlit web application
- Docker deployment

## Dataset

This project uses the PaySim synthetic financial transaction dataset.

The dataset contains approximately 6.3 million transactions, with fraud representing a very small percentage of total transactions.

## Machine Learning Models

The following models were evaluated:

- SGD Classifier
- XGBoost
- PyTorch Neural Network

XGBoost was selected as the final model based on its performance.

## Results

### XGBoost

- ROC-AUC: **0.99976**
- PR-AUC: **0.99846**
- Fraud Precision: **99.51%**
- Fraud Recall: **99.76%**
- Fraud F1-Score: **99.64%**

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- PyTorch
- SHAP
- Streamlit
- Docker
- Git & GitHub

## Project Structure

```text
fraud-detection-ai/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── models/
│   ├── xgb_fraud_model.pkl
│   └── feature_columns.pkl
│
├── src/
│   └── preprocessing.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 03_baseline_ml.ipynb
│   └── 04_deep_learning.ipynb
│
└── README.md