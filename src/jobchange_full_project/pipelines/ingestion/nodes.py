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
        # Define the feature and its allowed values
        feature_values = {
            "city": ['city_103', 'city_40', 'city_21', 'city_115', 'city_162', 'city_176', 'city_160',
                     'city_46', 'city_61', 'city_114', 'city_13', 'city_159', 'city_102', 'city_67',
                     'city_100', 'city_16', 'city_71', 'city_104', 'city_64', 'city_101', 'city_83',
                     'city_105', 'city_73', 'city_75', 'city_41', 'city_11', 'city_93', 'city_90',
                     'city_36', 'city_20', 'city_57', 'city_152', 'city_19', 'city_65', 'city_74',
                     'city_173', 'city_136', 'city_98', 'city_97', 'city_50', 'city_138', 'city_82',
                     'city_157', 'city_89', 'city_150', 'city_70', 'city_175', 'city_94', 'city_28',
                     'city_59', 'city_165', 'city_145', 'city_142', 'city_26', 'city_12', 'city_37',
                     'city_43', 'city_116', 'city_23', 'city_99', 'city_149', 'city_10', 'city_45',
                     'city_80', 'city_128', 'city_158', 'city_123', 'city_7', 'city_72', 'city_106',
                     'city_143', 'city_78', 'city_109', 'city_24', 'city_134', 'city_48', 'city_144',
                     'city_91', 'city_146', 'city_133', 'city_126', 'city_118', 'city_9', 'city_167',
                     'city_27', 'city_84', 'city_54', 'city_39', 'city_79', 'city_76', 'city_77',
                     'city_81', 'city_131', 'city_44', 'city_117', 'city_155', 'city_33', 'city_141',
                     'city_127', 'city_62', 'city_53', 'city_25', 'city_2', 'city_69', 'city_120',
                     'city_111', 'city_30', 'city_1', 'city_140', 'city_179', 'city_55', 'city_14',
                     'city_42', 'city_107', 'city_18', 'city_139', 'city_180', 'city_166', 'city_121',
                     'city_129', 'city_8', 'city_31', 'city_171'],
            "gender": ['Male', 'Female', 'Other', np.nan],
            "relevent_experience": ['Has relevent experience', 'No relevent experience'],
            "enrolled_university": ['no_enrollment', 'Full time course', 'Part time course', np.nan],
            "education_level": ['Graduate', 'Masters', 'High School', 'Phd', 'Primary School', np.nan],
            "major_discipline": ['STEM', 'Business Degree', 'Arts', 'Humanities', 'No Major', 'Other', np.nan],
            "experience": ['>20', '15', '5', '<1', '11', '13', '7', '17', '2', '16', '1', '4', '10',
                           '14', '18', '19', '12', '3', '6', '9', '8', '20', np.nan],
            "company_size": ['50-99', '<10', '10000+', '5000-9999', '1000-4999', '10/49', '100-500', '500-999', np.nan],
            "company_type": ['Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 'Other', 'Public Sector', 'NGO', np.nan],
            "last_new_job": ['1', '>4', 'never', '4', '3', '2', np.nan]
        }

        # Add expectations for each feature
        for feature, value_set in feature_values.items():
            expectation_suite_job_change.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_distinct_values_to_be_in_set",
                    kwargs={"column": feature, "value_set": value_set},
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
        event_time="datetime",
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
    df2: pd.DataFrame,
    parameters: Dict[str, Any]):

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

    common_columns= []
    for i in df2.columns.tolist():
        if i in df1.columns.tolist():
            common_columns.append(i)
    
    assert len(common_columns)>0, "Wrong data collected"

    df_full = pd.merge(df1,df2, how = 'left',  on = common_columns  )

    df_full= df_full.drop_duplicates()


    logger.info(f"The dataset contains {len(df_full.columns)} columns.")

    numerical_features = df_full.select_dtypes(exclude=['object','string','category']).columns.tolist()
    categorical_features = df_full.select_dtypes(include=['object','string','category']).columns.tolist()
    categorical_features.remove(parameters["target_column"])

    months_int = {'jan':1, 'feb':2, 'mar':3, 'apr':4,'may':5,'jun':6, 'jul':7 , 'aug':8 , 'sep':9 , 'oct':10, 'nov': 11, 'dec':12 }
    df_full = df_full.reset_index()
    df_full["datetime"]= df_full["month"].map(months_int)

    validation_expectation_suite_numerical = build_expectation_suite("numerical_expectations","numerical_features")
    validation_expectation_suite_categorical = build_expectation_suite("categorical_expectations","categorical_features")
    validation_expectation_suite_target = build_expectation_suite("target_expectations","target")

    numerical_feature_descriptions =[]
    categorical_feature_descriptions =[]
    target_feature_descriptions =[]
    
    df_full_numeric = df_full[["index","datetime"] + numerical_features]
    df_full_categorical = df_full[["index","datetime"] + categorical_features]
    df_full_target = df_full[["index","datetime"] + [parameters["target_column"]]]

    if parameters["to_feature_store"]:

        object_fs_numerical_features = to_feature_store(
            df_full_numeric,"numerical_features",
            1,"Numerical Features",
            numerical_feature_descriptions,
            validation_expectation_suite_numerical,
            credentials["feature_store"]
        )

        object_fs_categorical_features = to_feature_store(
            df_full_categorical,"categorical_features",
            1,"Categorical Features",
            categorical_feature_descriptions,
            validation_expectation_suite_categorical,
            credentials["feature_store"]
        )

        object_fs_taregt_features = to_feature_store(
            df_full_target,"target_features",
            1,"Target Features",
            target_feature_descriptions,
            validation_expectation_suite_target,
            credentials["feature_store"]
        )


    return df_full

#VER SE CREDENCIAIS FICAM ASSIM, PERGUNTAR AO PROF!!!
#Get data from feature Store
project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
    )
fs = project.get_feature_store(name='nrosa_test_featurestore') #nome do projeto
fg_lamp_features = fs.get_feature_group('lamp_features', version=1)
