import pytest
import pandas as pd
from src.jobchange_full_project.pipelines.split_train_pipeline.nodes import split_data



def test_split_data():
    df = pd.read_csv("./data/03_primary/preprocessed_initial_data.csv")


    # Define the parameters
    parameters = {
        'target_column': 'target',
        'random_state': 42,
        'test_fraction': 0.2
    }

    # Call the split_data function
    X_train, X_val, y_train, y_val = split_data(df, parameters)

    # Assert the existence of the datasets
    assert X_train is not None
    assert X_val is not None
    assert y_train is not None
    assert y_val is not None

    # Assert the shapes of the resulting datasets based on your real data
    assert X_train.shape[0] + X_val.shape[0] == df.shape[0]
    assert y_train.shape[0] + y_val.shape[0] == df.shape[0]


    assert X_train.shape[1] == df.shape[1] - 1  # Assuming one column is the target column
    assert y_train.unique() == [0, 1]





