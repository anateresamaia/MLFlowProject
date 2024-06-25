import pandas as pd
import pickle
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def model_predict(batch_data: pd.DataFrame, model: Any, columns: list) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict using the trained model.

    Args:
        batch_data (pd.DataFrame): DataFrame containing the batch data.
        model (Any): Trained model loaded from a pickle file.
        columns (list): List of columns to be used for prediction.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame with new predictions and a summary dictionary.
    """
    # Ensure the batch data contains the necessary columns
    if not all(col in batch_data.columns for col in columns):
        raise ValueError("Batch data is missing required columns for prediction.")

    # Make predictions
    y_pred = model.predict(batch_data[columns])

    # Add predictions to the DataFrame
    batch_data['y_pred'] = y_pred

    # Create a summary dictionary
    describe_servings = batch_data.describe().to_dict()

    logger.info('Service predictions created.')
    logger.info('#servings: %s', len(y_pred))

    return batch_data, describe_servings