"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


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
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


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
    return data


def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
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

    # Describe the data after transformation
    describe_to_dict_verified = df_transformed.describe(include='all').to_dict()

    return df_transformed, describe_to_dict_verified


def feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    # Bin the features
    data = experience_(data)
    data = city_development_index_(data)
    data = training_hours_(data)

    # Drop the original columns used for binning
    data.drop(['experience', 'city_development_index', 'training_hours'], axis=1, inplace=True)

    # Calculate mean balance for each training_hours_bin and assign it to every row in that bin
    data["mean_balance_bin_training"] = data.groupby("training_hours_bin")["balance"].transform("mean")
    # Calculate standard deviation of balance for each training_hours_bin and assign it to every row in that bin
    data["std_balance_bin_training"] = data.groupby("training_hours_bin")["balance"].transform("std")

    # Create lists of numeric and categorical features
    numerical_features = data.select_dtypes(exclude=['object', 'string', 'category']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    return data, categorical_features, numerical_features



