from kedro.pipeline import Pipeline, node, pipeline

from .nodes import additional_preprocessing

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=additional_preprocessing,
                inputs=["X_train_data", "y_train_data", "X_val_data", "y_val_data", "categorical_features", "numerical_features"],
                outputs=["X_train", "y_train", "X_val", "y_val", "encoder", "scaler", "knn_imputer"],
                name="additional_preprocessing",
            ),

        ]
    )
