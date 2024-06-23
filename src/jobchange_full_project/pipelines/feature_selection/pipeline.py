"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_selection,
                inputs=["X_train","X_val","y_train","y_val", "parameters"],
                outputs=["X_train_final", "X_val_final", "y_train_final", "y_val_final"],
                name="model_feature_selection",
            ),
        ]
    )
