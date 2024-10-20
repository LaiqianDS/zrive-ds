import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import LogitResults


def three_way_split(
    X: pd.DataFrame,
    Y: pd.Series,
    test_size: float = 0.3,
    random_state: int or None = None,
) -> tuple:
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    # Split the training data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def logit_significance(X: pd.DataFrame, y: pd.Series) -> LogitResults:
    X_sm = sm.add_constant(X)
    model_statmodels = sm.Logit(y, X_sm)
    result_statmodels = model_statmodels.fit()
    return result_statmodels


def get_best_treshold(y_test: np.array, y_pred_prob: np.array) -> float:
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold


def print_metrics(y_test: np.array, y_test_pred: np.array) -> None:
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
