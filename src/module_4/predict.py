import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from train import PushModel

DATA_PATH = "../../data/module2/feature_frame.csv"


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def create_model_path(model_folder_path: str, model_name: str = "push.joblib") -> str:
    model_name_without_extension = Path(model_name).stem
    extension = ".joblib"

    model_path = Path(model_folder_path) / model_name_without_extension + extension

    if model_path.exists():
        model_path = (
            Path(model_folder_path) / f"{model_name_without_extension}_{datetime.now()}"
            + extension
        )

    return str(model_path)


def load_model(model_path: str) -> PushModel:
    return joblib.load(model_path)


def handler_fit(event, _):
    model_parametrisation = event["model_parametrisation"]
    threshold = event["threshold"]

    model = PushModel(model_parametrisation, threshold)

    df = load_data()
    df_train, _ = model.feature_label_split(df)

    model.fit(df_train)

    try:
        model_path = create_model_path("push_model")
        model.save(model_path)
    except Exception as e:
        return {
            "statusCode": "500",
            "body": json.dumps(
                {
                    "error": f"An error occurred while saving the model: {str(e)}",
                }
            ),
        }

    return {
        "statusCode": "200",
        "body": json.dumps(
            {
                "model_path": [model_path],
            }
        ),
    }


def handler_predict(event, _):
    data_to_predict = pd.DataFrame.from_dict(json.loads(event["users"]), orient="index")

    try:
        model = load_model(event["model_path"])
    except FileNotFoundError:
        return {
            "statusCode": "500",
            "body": json.dumps(
                {
                    "error": "Model not found",
                }
            ),
        }

    predictions = model.predict(data_to_predict)

    pred_dictionary = {}
    for user, prediction in zip(data_to_predict.index, predictions):
        pred_dictionary[user] = prediction

    return {
        "statusCode": "200",
        "body": json.dumps({"prediction": {pred_dictionary}}),
    }
