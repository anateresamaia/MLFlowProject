
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


def additional_preprocessing(X_train_data: pd.DataFrame, y_train_data: pd.DataFrame, ##change y train x val y_train_data: pd.DataFrame
                             X_val_data: pd.DataFrame, y_val_data: pd.DataFrame,
                             categorical_features: list, numerical_features: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ce.TargetEncoder, MinMaxScaler, KNNImputer]:

    # Initialize the TargetEncoder with handle_missing='return_nan'
    encoder = ce.TargetEncoder(handle_missing='return_nan')

    # Iterate through the categorical features
    for column in categorical_features:
        encoder.fit(X_train_data[[column]], y_train_data)

        # Transform the training and validation sets
        X_train_data[column] = encoder.transform(X_train_data[[column]])
        X_val_data[column] = encoder.transform(X_val_data[[column]])

    # Update the numerical_features list since all variables are now numerical
    all_features = X_train_data.columns.tolist()
    numerical_features = all_features

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply MinMaxScaler to the numerical features
    for column in numerical_features:
        scaler.fit(X_train_data[[column]])

        X_train_data[column] = scaler.transform(X_train_data[[column]])
        X_val_data[column] = scaler.transform(X_val_data[[column]])

    # Initialize the KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)

    # Select columns for imputation
    columns_to_impute = ['enrolled_university', 'education_level', 'major_discipline', 'last_new_job']

    # Apply imputation
    for column in columns_to_impute:
        train_data = X_train_data[[column]]
        val_data = X_val_data[[column]]

        # Fit and transform the imputer on the training data
        train_imputed = knn_imputer.fit_transform(train_data)

        # Transform the validation data using the fitted imputer
        val_imputed = knn_imputer.transform(val_data)

        # Update the DataFrames with the imputed values
        X_train_data[column] = train_imputed
        X_val_data[column] = val_imputed

    return X_train_data, y_train_data, X_val_data, y_val_data, encoder, scaler, knn_imputer


