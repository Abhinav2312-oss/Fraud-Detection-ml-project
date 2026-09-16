import numpy as np
import pandas as pd


def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the feature set used to train the fraud model."""
    features = dataframe.copy()
    transaction_types = [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]

    features["Origin_balance_change"] = (
        features["oldbalanceOrg"] - features["newbalanceOrig"]
    )
    features["Destination_balance_change"] = (
        features["oldbalanceDest"] - features["newbalanceDest"]
    )
    features["amount_to_origin_balance_ratio"] = (
        features["amount"] / (features["oldbalanceOrg"] + 1)
    )
    features["amount_to_destination_balance_ratio"] = (
        features["amount"] / (features["oldbalanceDest"] + 1)
    )
    features["origin_balance_error"] = (
        features["oldbalanceOrg"] - features["amount"] - features["newbalanceOrig"]
    )
    features["destination_balance_error"] = (
        features["oldbalanceDest"] + features["amount"] - features["newbalanceDest"]
    )
    features["log_amount"] = np.log1p(features["amount"])

    features = pd.get_dummies(
        features,
        columns=["type"],
        drop_first=False,
        dtype=int,
    )

    for transaction_type in transaction_types:
        column = f"type_{transaction_type}"
        if column not in features:
            features[column] = 0

    return features
