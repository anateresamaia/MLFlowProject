import numpy as np
import pandas as pd
import pytest
import warnings
warnings.filterwarnings("ignore", category=Warning)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from src.jobchange_full_project.pipelines.model_selection.nodes import model_selection


@pytest.mark.slow
def test_model_selection():

    # Load train and validation data
    X_train = pd.read_csv("data/05_model_input/X_train_final.csv")
    X_val = pd.read_csv("data/05_model_input/X_val_final.csv")
    y_train = pd.read_pickle("data/05_model_input/y_train_final.pkl")
    y_val = pd.read_pickle("data/05_model_input/y_val_final.pkl")

    # Load production model metrics and model
    production_model_metrics = pd.read_json("data/08_reporting/production_model_metrics.json")
    production_model = pd.read_pickle("data/06_models/production_model.pkl")

    # Define the parameters
    parameters = {
        'target_column': 'target',
        'random_state': 42,
        'test_fraction': 0.2
    }

    # Run the model selection function
    champion_model = model_selection(
        X_train,
        X_val,
        y_train,
        y_val,
        production_model_metrics,
        production_model,
        parameters
    )

    # Check that the returned value is a dictionary
    assert isinstance(champion_model, LogisticRegression) or isinstance(champion_model, GradientBoostingClassifier)
    assert isinstance(champion_model.score(X_val, y_val), float)

