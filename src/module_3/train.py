import datetime

import joblib
import numpy as np
import pandas as pd
from functions import (
    get_best_treshold,
    logit_significance,
    pick_features_for_splits,
    three_way_split_time,
)
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_PATH = "./src/module_3/models"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Returns a DataFrame according the needs of the company
    ordered = df[df["outcome"] == 1]  # have been ordered
    orders_len = ordered.groupby("order_id").outcome.sum()
    orders_selection = orders_len[orders_len >= 5].index
    df2 = df[df["order_id"].isin(orders_selection)]
    df2["order_date"] = pd.to_datetime(df2["order_date"]).dt.date
    df2["created_at"] = pd.to_datetime(df2["created_at"])
    return df2


def save_model(model: BaseEstimator, name: str, output_path: str) -> None:
    model_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{name}.pkl"
    joblib.dump(model, f"{output_path}/{model_name}")
    return


def evaluate_model(y_test: pd.Series, y_pred: pd.Series) -> dict:
    precission, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precission)
    roc_auc = roc_auc_score(y_test, y_pred)
    return {"pr_auc": pr_auc, "roc_auc": roc_auc}


def feature_label_split(df: pd.DataFrame) -> tuple:
    # Follows the proccess to select the best features acording to Milestone 1
    info_cols = ["variant_id", "order_id", "user_id", "order_date", "created_at"]
    label_cols = ["outcome"]
    features_cols = [col for col in df.columns if col not in info_cols + label_cols]
    categorical_cols = ["product_type", "vendor"]
    binary_cols = [
        "ordered_before",
        "abandoned_before",
        "active_snoozed",
        "set_as_regular",
    ]
    numerical_cols = [
        col for col in features_cols if col not in categorical_cols + binary_cols
    ]

    features = numerical_cols + binary_cols + ["order_date", "created_at"]
    X_train, X_val, _, Y_train, Y_val, _ = three_way_split_time(
        df[features], df["outcome"]
    )
    X_train.drop(columns=["order_date", "created_at"], inplace=True)
    X_val.drop(columns=["order_date", "created_at"], inplace=True)

    return X_train, X_val, Y_train, Y_val


def select_features(df: pd.DataFrame) -> list:
    X_train, X_val, Y_train, Y_val = feature_label_split(df)

    # Train the model
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, Y_train)

    logit_results = logit_significance(X_train, Y_train)
    selected_features = (
        logit_results.pvalues[logit_results.pvalues < 0.05].sort_values().index[:3]
    )

    return list(selected_features)


def train_model(df: pd.DataFrame, output_path: str) -> dict:
    selected_features = select_features(df)
    X_train, X_val, Y_train, Y_val = feature_label_split(df)
    X_train, X_val, _ = pick_features_for_splits(
        X_train, X_val, X_val, selected_features
    )

    model = make_pipeline(StandardScaler(), LogisticRegression())

    model.fit(X_train, Y_train)
    y_val_pred_prob = model.predict_proba(X_val)[:, 1]
    best_threshold = get_best_treshold(Y_val, y_val_pred_prob)
    y_val_pred = np.array([1 if p > best_threshold else 0 for p in y_val_pred_prob])
    evaluation = evaluate_model(Y_val, y_val_pred)

    # Save the model
    save_model(model, "logistic_regression", output_path)
    return evaluation


def main():
    df = pd.read_csv("./data/module2/feature_frame.csv")
    df = clean_dataset(df)

    evaluation = train_model(df, OUTPUT_PATH)
    print(evaluation)


if __name__ == "__main__":
    main()
