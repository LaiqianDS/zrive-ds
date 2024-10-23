import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import LogitResults

def three_way_split_time(
    X: pd.DataFrame,
    Y: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
) -> tuple:
    """Splits the data by time."""

    # Ordenar X por 'order_date' para garantizar el split temporal
    X = X.sort_values('order_date')
    
    # Reorganizar Y basado en el índice de X
    Y = Y.loc[X.index]

    # Agrupar por fecha y obtener el acumulado de órdenes
    daily_orders = X.groupby('order_date').size()
    cumsum_daily_orders = daily_orders.cumsum() / daily_orders.sum()

    # Obtener los puntos de corte para los splits
    train_val_cutoff = cumsum_daily_orders[cumsum_daily_orders <= train_size].idxmax()
    val_test_cutoff = cumsum_daily_orders[cumsum_daily_orders <= (train_size + val_size)].idxmax()

    # Realizar los splits para X
    X_train = X[X['order_date'] < train_val_cutoff]
    X_val = X[(X['order_date'] >= train_val_cutoff) & (X['order_date'] < val_test_cutoff)]
    X_test = X[X['order_date'] >= val_test_cutoff]

    # Realizar los splits para Y (basado en los mismos índices que X)
    Y_train = Y.loc[X_train.index]
    Y_val = Y.loc[X_val.index]
    Y_test = Y.loc[X_test.index]

    return X_train, X_val, X_test, Y_train.squeeze(), Y_val.squeeze(), Y_test.squeeze()


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
    
def plot_auc_pr_curve(y_test: np.array, y_pred_prob: np.array) -> None:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    average_precision = average_precision_score(y_test, y_pred_prob)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'Precision-Recall curve (area = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.show()
    
def pick_features_for_splits(x_train, x_val, x_test, features: list) -> tuple:
    x_train = x_train[features]
    x_val = x_val[features]
    x_test = x_test[features]
    
    return x_train, x_val, x_test