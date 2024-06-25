
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
    # Initialize the TargetEncoder only once
    for column in categorical_features:
        X_train_data[column] = X_train_data[column].astype('category')
        X_val_data[column] = X_val_data[column].astype('category')
    encoder = ce.TargetEncoder(cols=categorical_features, handle_missing='return_nan')
    print(encoder)
    # Fit the encoder using the training data and transform both train and validation datasets
    encoder.fit(X_train_data[categorical_features], y_train_data)
    X_train_data[categorical_features] = encoder.transform(X_train_data[categorical_features])
    X_val_data[categorical_features] = encoder.transform(X_val_data[categorical_features])

    # Update the numerical_features list since all variables are now numerical
    all_features = X_train_data.columns.tolist()
    numerical_features = all_features

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on all numerical features in the training data
    scaler.fit(X_train_data[numerical_features])

    # Transform both training and validation data using the fitted scaler
    X_train_data[numerical_features] = scaler.transform(X_train_data[numerical_features])
    X_val_data[numerical_features] = scaler.transform(X_val_data[numerical_features])

    # Initialize the KNNImputer with the specified number of neighbors
    knn_imputer = KNNImputer(n_neighbors=5)

    # Fit and transform the imputer on the selected columns in the training data
    # Select columns for imputation
    columns_to_impute = ['enrolled_university', 'education_level', 'major_discipline', 'last_new_job']

    knn_imputer.fit(X_train_data[columns_to_impute])
    X_train_data[columns_to_impute] = knn_imputer.transform(X_train_data[columns_to_impute])

    # Transform the validation data using the fitted imputer
    X_val_data[columns_to_impute] = knn_imputer.transform(X_val_data[columns_to_impute])

    return X_train_data, y_train_data, X_val_data, y_val_data, encoder, scaler, knn_imputer


