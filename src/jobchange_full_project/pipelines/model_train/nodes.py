import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def model_train(X_train_final: pd.DataFrame,
                X_val_final: pd.DataFrame,
                y_train_final: pd.DataFrame,
                y_val_final: pd.DataFrame,
                parameters: Dict[str, Any]):
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.

    Returns:
    --
        model (pickle): Trained models.
        scores (json): Trained model metrics.
    """

    # enable autologging
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    logger.info('Starting first step of model selection : Comparing between modes types')
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)




    # open pickle file with regressors
    try:
        with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
            classifier = pickle.load(f)
    except:
        classifier = LogisticRegression(**parameters['baseline_model_params'])






    results_dict = {}
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        '''
        if parameters["use_feature_selection"]:
            logger.info(f"Using feature selection in model train...")
            X_train = X_train[best_columns]
            X_test = X_test[best_columns]
        '''
        y_train_final = np.ravel(y_train_final)
        model = classifier.fit(X_train_final, y_train_final)
        # making predictions
        y_train_pred = model.predict(X_train_final)
        y_val_pred = model.predict(X_val_final)
        # evaluating model
        acc_train = accuracy_score(y_train_final, y_train_pred)
        acc_val = accuracy_score(y_val_final, y_val_pred)
        f1_train = f1_score(y_train_final, y_train_pred)
        f1_val = f1_score(y_val_final, y_val_pred)
        # saving results in dict
        results_dict['classifier'] = classifier.__class__.__name__
        results_dict['train_score'] = f1_train
        results_dict['val_score'] = f1_val
        # logging in mlflow
        run_id = mlflow.last_active_run().info.run_id
        logger.info(f"Logged train model in run {run_id}")
        logger.info(f"Accuracy is {acc_val}")


    print(acc_train)
    print(acc_val)
    print(f1_train)
    print(f1_val)


    # When we use LR:
    explainer = shap.LinearExplainer(model, X_train_final)
    shap_values = explainer(X_train_final)

    shap.initjs()
    # calculate shap values and plot summary
    shap.summary_plot(shap_values, X_train_final, feature_names=X_train_final.columns, show=False)


    '''
    # When we use RF:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train_final)

    shap.initjs()
    # calculate shap values. This is what we will plot.
    # shap_values[:,:,1] -> since it is a classification problem, I will use SHAP for explaining the outcome of class 1.
    # you can do the same for the class 0 just by using shap_values[:,:,0]
    shap.summary_plot(shap_values[:,:,1], X_train_final,feature_names=X_train_final.columns, show=False)
    '''

    return model, X_train_final.columns, results_dict, plt