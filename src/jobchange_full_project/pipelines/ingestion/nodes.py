"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

logger = logging.getLogger(__name__)


def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.

    Args:
        expectation_suite_name (str): A dictionary with the feature group name and the respective version.
        feature_group (str): Feature group used to construct the expectations.

    Returns:
        ExpectationSuite: A dictionary containing all the expectations for this particular feature group.
    """

    expectation_suite_job_change = ExpectationSuite(
        expectation_suite_name=expectation_suite_name
    )

    # numerical features
    if feature_group == 'numerical_features':
        # training_hours should be of type int64 and have a minimum value of 0
        expectation_suite_job_change.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "training_hours", "type_": "int64"},
            )
        )
        expectation_suite_job_change.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_min_to_be_between",
                kwargs={"column": "training_hours", "min_value": 0, "strict_min": False},
            )
        )

        # city_development_index should be of type float and have values between 0 and 1
        expectation_suite_job_change.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "city_development_index", "type_": "float64"},
            )
        )
        expectation_suite_job_change.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "city_development_index", "min_value": 0, "max_value": 1},
            )
        )

    # categorical features
    if feature_group == 'categorical_features':
        # Ensure the columns are of type string or object
        categorical_columns = [
            "city", "gender", "relevent_experience", "enrolled_university",
            "education_level", "major_discipline", "experience",
            "company_size", "company_type", "last_new_job"
        ]

        for column in categorical_columns:
            expectation_suite_job_change.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_type_list",
                    kwargs={"column": column, "type_list": ["object", "string"]},
                )
            )

    if feature_group == 'target':
        # Ensure the column values are of type float
        expectation_suite_job_change.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "target", "type_": "float64"},
            )
        )

        # Ensure the column values are within the set [0.0, 1.0]
        expectation_suite_job_change.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "target", "value_set": [0.0, 1.0]},
            )
        )

    return expectation_suite_job_change


import hopsworks


def to_feature_store(
        data: pd.DataFrame,
        group_name: str,
        feature_group_version: int,
        description: str,
        group_description: dict,
        validation_expectation_suite: ExpectationSuite,
        credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group.
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.

    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description=description,
        primary_key=["index"],
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )
    # Upload data.
    object_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions.

    for description in group_description:
        object_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    return object_feature_group


def ingestion(
        df1: pd.DataFrame,
        parameters: Dict[str, Any]
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group.
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.

    Returns:
    """
    logger.info(f"The dataset contains {len(df1.columns)} columns.")

    numerical_features = df1.select_dtypes(exclude=['object', 'string', 'category']).columns.tolist()
    categorical_features = [
        'city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level',
        'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job'
    ]

    numerical_features.remove(parameters["target_column"])
    df1 = df1.reset_index()

    validation_expectation_suite_numerical = build_expectation_suite("numerical_expectations", "numerical_features")
    validation_expectation_suite_categorical = build_expectation_suite("categorical_expectations",
                                                                       "categorical_features")
    validation_expectation_suite_target = build_expectation_suite("target_expectations", "target")

    numerical_feature_descriptions = []
    categorical_feature_descriptions = []
    target_feature_descriptions = []

    df1_numeric = df1[["index"] + numerical_features]
    df1_categorical = df1[["index"] + categorical_features]
    df1_target = df1[["index"] + [parameters["target_column"]]]

    ###########
    df1_categorical = df1_categorical.fillna('null')
    ###########

    if parameters["to_feature_store"]:
        object_fs_numerical_features = to_feature_store(
            df1_numeric, "numerical_features",
            1, "Numerical Features",
            numerical_feature_descriptions,
            validation_expectation_suite_numerical,
            credentials["feature_store"]
        )

        object_fs_categorical_features = to_feature_store(
            df1_categorical, "categorical_features",
            1, "Categorical Features",
            categorical_feature_descriptions,
            validation_expectation_suite_categorical,
            credentials["feature_store"]
        )

        object_fs_target_features = to_feature_store(
            df1_target, "target_features",
            1, "Target Features",
            target_feature_descriptions,
            validation_expectation_suite_target,
            credentials["feature_store"]
        )

    return df1
