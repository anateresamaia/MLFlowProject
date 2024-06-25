import logging
from typing import Any, Dict, Tuple
import numpy as np
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import category_encoders as ce
from typing import Tuple, List

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple



def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
        Cleans the data by removing outliers, setting the index, and imputing missing values.

        Args:
            data: Data containing features and target.

        Returns:
            data: Cleaned data
        """
    df_transformed = data.copy()

    # Describe the data before transformation
    describe_to_dict = df_transformed.describe(include='all').to_dict()

    # Remove duplicates based on enrollee_id
    df_transformed.drop_duplicates(subset='enrollee_id', inplace=True)

    # Remove outliers for city_development_index and training_hours
    df_transformed = df_transformed[df_transformed['city_development_index'] >= 0.4]
    df_transformed = df_transformed[df_transformed['training_hours'] <= 350]

    # Set enrollee_id as the index
    df_transformed.set_index('enrollee_id', inplace=True)

    # Impute missing values
    df_transformed['gender'].fillna('Unknown', inplace=True)
    df_transformed['experience'].fillna(0, inplace=True)
    df_transformed['company_size'].fillna('Not Applicable', inplace=True)
    df_transformed['company_type'].fillna('Not Applicable', inplace=True)
    # replace of typo error
    df_transformed['company_size'].replace('10/49', '10-49', inplace=True)

    # Describe the data after transformation
    describe_to_dict_verified = df_transformed.describe(include='all').to_dict()



    return df_transformed, describe_to_dict_verified

def experience_(data):
    def experience_bin(exp):
        if pd.isna(exp):
            return 'NaN'
        elif exp == '>20':
            return '>20'
        elif exp == '<1':
            return '0-5'
        else:
            exp = int(exp)
            if exp <= 5:
                return '0-5'
            elif exp <= 10:
                return '6-10'
            elif exp <= 15:
                return '11-15'
            elif exp <= 20:
                return '16-20'

    data['experience_bin'] = data['experience'].apply(experience_bin)
    return data


def city_development_index_(data):
    city_dev_bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    city_dev_labels = ['<0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.85', '0.85-0.9', '0.9-0.95',
                       '0.95-1.0']

    data['city_development_index_bin'] = pd.cut(data['city_development_index'], bins=city_dev_bins,
                                                labels=city_dev_labels, include_lowest=True)
    return data


def training_hours_(data):
    training_hours_bins = [0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, float('inf')]
    training_hours_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-150', '150-200', '200-250', '250-300',
                             '300-350', '>350']

    data['training_hours_bin'] = pd.cut(data['training_hours'], bins=training_hours_bins, labels=training_hours_labels,
                                        include_lowest=True)
    data['training_hours_bin'] = data['training_hours_bin'].astype('object')

    return data

def feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    # Bin the features
    data = experience_(data)
    data = city_development_index_(data)
    data = training_hours_(data)

    # Drop the original columns used for binning
    data.drop(['experience', 'city_development_index', 'training_hours'], axis=1, inplace=True)
    print("Data Types:\n", data.dtypes)
    print("\nColumn Names:\n", data.columns.tolist())
    #data.drop(["training_hours_bin"], axis=1, inplace=True)
    return data


def additional_preprocessing(data: pd.DataFrame, encoder: ce.TargetEncoder, scaler: MinMaxScaler, knn_imputer: KNNImputer,
                             categorical_features: List[str]) -> pd.DataFrame:

    missing_cols = [col for col in encoder.cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data that were expected for transformation: {missing_cols}")

    data[categorical_features] = encoder.transform(data[categorical_features])

    # Update the numerical_features list since all variables are now numerical
    all_features = data.columns.tolist()
    numerical_features = all_features
    print(numerical_features)
    # Apply MinMaxScaler to the numerical features
    # Applying MinMaxScaler to all numerical features at once
    columns_to_impute = ['enrolled_university', 'education_level', 'major_discipline', 'last_new_job']

    data[numerical_features] = scaler.transform(data[numerical_features])

    # Applying KNN Imputer to the specified columns for imputation

    imputed_data = knn_imputer.transform(data[columns_to_impute])
    data[columns_to_impute] = imputed_data  # Update the data with imputed values

    columns_to_drop = ['gender', 'relevent_experience', 'major_discipline', 'city_development_index_bin']

    # Drop the specified columns from the DataFrame
    data = data.drop(columns=columns_to_drop)

    return data

