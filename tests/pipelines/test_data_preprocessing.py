from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import os
from src.jobchange_full_project.pipelines.preprocessing_initial.nodes import clean_data, feature_engineer


def test_clean_date_type():
    df = pd.read_csv("./data/02_intermediate/ingested_data.csv")
    df_transformed, describe_to_dict_verified = clean_data(df)
    assert isinstance(describe_to_dict_verified, dict)

def test_clean_date_null():
    df = pd.read_csv("./data/02_intermediate/ingested_data.csv")
    df_transformed, describe_to_dict_verified = clean_data(df)
    assert [col for col in df_transformed.columns if df_transformed[col].isnull().any()] == []
def test_feature_engineer():
    df = pd.read_csv("./data/02_intermediate/ingested_data.csv")
    df_transformed, describe_to_dict_verified = clean_data(df)

    # Feature engineering
    df_final, categorical_features, numerical_features = feature_engineer(df_transformed)
    assert 'experience_bin' in df_final.columns
    assert 'city_development_index_bin' in df_final.columns



