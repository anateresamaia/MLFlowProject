"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import great_expectations as gx

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

logger = logging.getLogger(__name__)

def get_validation_results(checkpoint_result):
    # Extract validation result
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))
    validation_result_ = validation_result_data.get('validation_result', {})
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')

    df_validation = pd.DataFrame(columns=["Success", "Expectation Type", "Column", "Column Pair", "Max Value",
                                          "Min Value", "Element Count", "Unexpected Count", "Unexpected Percent", "Value Set",
                                          "Unexpected Value", "Observed Value"])

    for result in results:
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')
        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        if isinstance(observed_value, list):
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value = []

        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict([{
            "Success": success, "Expectation Type": expectation_type, "Column": column, "Column Pair": (column_A, column_B),
            "Max Value": max_value, "Min Value": min_value, "Element Count": element_count,
            "Unexpected Count": unexpected_count, "Unexpected Percent": unexpected_percent,
            "Value Set": value_set, "Unexpected Value": unexpected_value, "Observed Value": observed_value
        }])], ignore_index=True)

    return df_validation

def test_data(df):
    cities = ['city_103', 'city_40', 'city_21', 'city_115', 'city_162', 'city_176', 'city_160',
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
              'city_129', 'city_8', 'city_31', 'city_171']
    context = gx.get_context(context_root_dir="//..//..//gx")
    datasource_name = "job_change_datasource"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except Exception as e:
        logger.info(f"Data Source already exists: {str(e)}")
        datasource = context.datasources[datasource_name]

    suite_job_change = context.add_or_update_expectation_suite(expectation_suite_name="Job Change")
    expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "city",
                "value_set": cities
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={"column": "training_hours", "min_value": 0, "strict_min": False},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "city_development_index", "min_value": 0, "max_value": 1},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "experience",
                "value_set": [str(i) for i in range(1, 21)] + [">20", "<1"]
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "gender", "value_set": ["Male", "Female", "Other"]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "relevent_experience", "value_set": ["Has relevent experience", "No relevent experience"]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "enrolled_university", "value_set": ["no_enrollment", "Full time course", "Part time course"]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "education_level", "value_set": ["Primary School", "High School", "Graduate", "Masters", "Phd"]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "company_size", "value_set": ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", ">10000"]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "company_type", "value_set": ["Pvt Ltd", "Funded Startup", "Early Stage Startup", "Other", "Public Sector", "NGO"]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "major_discipline",
                "value_set": ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"]
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "last_new_job", "value_set": ["1", ">4", "never", "4", "3", "2"]}
        ),

    ]

    for expectation in expectations:
        suite_job_change.add_expectation(expectation_configuration=expectation)

    context.add_or_update_expectation_suite(expectation_suite=suite_job_change)

    data_asset_name = "test"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    except Exception as e:
        logger.info(f"The data asset already exists: {str(e)}. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe=df)

    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_bank",
        data_context=context,
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": "Bank",
        }],
    )
    checkpoint_result = checkpoint.run()

    df_validation = get_validation_results(checkpoint_result)

    pd_df_ge = gx.from_pandas(df)

    # Example asserts
    assert pd_df_ge.expect_column_values_to_be_of_type("city_development_index", "float64").success

    log = logging.getLogger(__name__)
    log.info("Data passed the unit data tests")

    return df_validation





