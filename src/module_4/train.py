from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from typing import Dict, Tuple


class PushModel:
    MODEL_COLUMNS = [
        "ordered_before",
        "abandoned_before",
        "active_snoozed",
        "set_as_regular",
        "user_order_seq",
        "normalised_price",
        "discount_pct",
        "global_popularity",
        "count_adults",
        "count_children",
        "count_babies",
        "count_pets",
        "people_ex_baby",
        "days_since_purchase_variant_id",
        "avg_days_to_buy_variant_id",
        "std_days_to_buy_variant_id",
        "days_since_purchase_product_type",
        "avg_days_to_buy_product_type",
        "std_days_to_buy_product_type",
    ]

    TARGET_COLUMN = "outcome"

    def __init__(self, classifier_parameters: Dict, threshold: float):
        """
        During training, the best performing parameters for the classifier
        was n_estimators=100.
        """
        self.classifier = RandomForestClassifier(**classifier_parameters)
        self.threshold = threshold

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.MODEL_COLUMNS]

    def _extract_label(self, df: pd.DataFrame) -> pd.Series:
        return df[self.TARGET_COLUMN]

    def feature_label_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return self._extract_features(df), self._extract_label(df)

    def fit(self, df: pd.DataFrame) -> None:
        features, label = self.feature_label_split(df)
        self.classifier.fit(features, label)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self.classifier.predict_proba(self._extract_features(df))[:, 1]
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        features = self._extract_features(df)
        probabilities = self.predict_proba(df[features])
        return (probabilities > self.threshold).astype(int)

    def save(self, path: str) -> None:
        joblib.dump(self.classifier, path)
