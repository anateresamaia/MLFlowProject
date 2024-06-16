"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """
    # Ensure there are no null values in the data
    assert [col for col in data.columns if data[col].isnull().any()] == []

    # Extract the target variable
    target_var = data[parameters["target_column"]]

    # Drop the target, index, and datetime columns to get the features
    X = data.drop(columns=[parameters["target_column"]], axis=1)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        target_var,
        test_size=parameters["test_fraction"],
        random_state=parameters["random_state"],
        stratify=target_var,
        shuffle=True
    )

    return X_train, X_val, y_train, y_val, X_train.columns



