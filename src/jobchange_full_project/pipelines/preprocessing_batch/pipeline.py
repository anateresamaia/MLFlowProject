
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data, feature_engineer, additional_preprocessing

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="aug_test",
                outputs=["cleaned_data", "describe_to_dict_verified"],
                name="clean_data",
            ),
            node(
                func=feature_engineer,
                inputs="cleaned_data",
                outputs="feature_engineered_data",
                name="feature_engineering",
            ),
            node(
                func=additional_preprocessing,
                inputs=["feature_engineered_data", "encoder", "scaler", "knn_imputer", "categorical_features", "numerical_features"],
                outputs="preprocessed_batch_data",
                name="additional_preprocessing",
            ),
        ]
    )
