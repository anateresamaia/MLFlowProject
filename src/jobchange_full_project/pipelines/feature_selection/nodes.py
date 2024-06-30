
from typing import Any, Dict
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from collections import Counter


def feature_selection(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, parameters: Dict[str, Any]):
    def variance_analysis(X_train):
        variance_values = X_train.var().round(10) * 100
        eliminate = variance_values[variance_values == 0].index.tolist()
        return eliminate

    def correlation_analysis(X_train, y_train):
        # Concatenate X_train and y_train to calculate correlation
        correlation_features = pd.concat([X_train, y_train], axis=1)

        # Calculate Spearman correlation
        cor_spearman = correlation_features.corr(method='spearman')

        # Get correlation matrix for features only (excluding the target)
        feature_corr = cor_spearman.loc[X_train.columns, X_train.columns]

        # List to keep track of features to eliminate
        spearman_eliminate = []

        # Iterate over each column and find highly correlated pairs
        for i in range(len(feature_corr.columns)):
            for j in range(i):
                if abs(feature_corr.iloc[i, j]) > 0.7:  # Check high correlation
                    col_i = feature_corr.columns[i]
                    col_j = feature_corr.columns[j]
                    target_corr_i = abs(cor_spearman.loc[col_i, y_train.target])
                    target_corr_j = abs(cor_spearman.loc[col_j, y_train.target])
                    # Eliminate the feature with lower correlation to the target
                    if target_corr_i < target_corr_j:
                        spearman_eliminate.append(col_i)
                    else:
                        spearman_eliminate.append(col_j)

        # Remove duplicates from the elimination list
        spearman_eliminate = list(set(spearman_eliminate))
        return spearman_eliminate

    def rfe_analysis(X_train, X_val, y_train):
        nof_list = np.arange(2, len(X_train.columns) + 1)
        high_score = 0
        nof = 0
        train_score_list = []
        val_score_list = []

        for n in range(len(nof_list)):
            model = LogisticRegression(random_state=42)
            rfe = RFE(estimator=model, n_features_to_select=nof_list[n])

            X_train_rfe = rfe.fit_transform(X_train, y_train)
            X_val_rfe = rfe.transform(X_val)

            model.fit(X_train_rfe, y_train)

            # Storing results on training data
            train_pred = model.predict(X_train_rfe)
            train_score = f1_score(y_train, train_pred)
            train_score_list.append(train_score)

            # Storing results on validation data
            val_pred = model.predict(X_val_rfe)
            val_score = f1_score(y_val, val_pred)
            val_score_list.append(val_score)

            # Check best score
            if val_score >= high_score:
                high_score = val_score
                nof = nof_list[n]

        # Chose the desired number of features considering score and overfitting
        model = LogisticRegression(random_state=42)
        rfe = RFE(estimator=model, n_features_to_select=5)
        X_rfe = rfe.fit_transform(X_train, y_train)

        # Features to eliminate
        rfe_eliminate = []
        rfe_dataset = pd.Series(rfe.support_, index=X_train.columns)

        for index, value in rfe_dataset.items():
            if not value:
                rfe_eliminate.append(index)
        return rfe_eliminate

    def lasso_analysis(X_train, y_train):
        reg = LassoCV()
        reg.fit(X_train, y_train)

        coef = pd.Series(reg.coef_, index=X_train.columns).round(6)
        # Variables to eliminate using LASSO
        selected_variables = (coef >= -0.02) & (coef <= 0.02)
        lasso_eliminate = coef.index[selected_variables].tolist()

        return lasso_eliminate

    def decision_tree_analysis(X_train, y_train):
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Get feature importances
        feature_importances = clf.feature_importances_

        # List with features to eliminate
        decision_tree_eliminate = []
        importance_threshold = 0.035

        for feature, importance in zip(X_train.columns, feature_importances):
            if importance < importance_threshold:
                decision_tree_eliminate.append(feature)

        return decision_tree_eliminate

    # Collect feature removal lists from different methods
    removal_lists = []

    # Variance analysis
    eliminate_variance = variance_analysis(X_train)
    removal_lists.append(eliminate_variance)

    # Correlation analysis
    eliminate_correlation = correlation_analysis(X_train, y_train)
    removal_lists.append(eliminate_correlation)

    # RFE analysis
    eliminate_rfe = rfe_analysis(X_train, X_val, y_train)
    removal_lists.append(eliminate_rfe)

    # LASSO analysis
    eliminate_lasso = lasso_analysis(X_train, y_train)
    removal_lists.append(eliminate_lasso)

    # Decision Tree analysis
    eliminate_decision_tree = decision_tree_analysis(X_train, y_train)
    removal_lists.append(eliminate_decision_tree)



    # Combine results and count occurrences
    all_removals = [item for sublist in removal_lists for item in sublist]
    removal_counts = Counter(all_removals)

    # Identify features to remove (selected by at least two methods)
    features_to_remove = [feature for feature, count in removal_counts.items() if count >= 2]
    print(features_to_remove)

    # Remove identified features from X_train and X_val
    X_train = X_train.drop(columns=features_to_remove)
    X_val = X_val.drop(columns=features_to_remove)

    return X_train, X_val, y_train, y_val