import pandas as pd
from joblib import load
from train import OUTPUT_PATH, clean_dataset, evaluate_model


def main():
    model_name = "/20241104193743_logistic_regression.pkl"
    model = load(OUTPUT_PATH + model_name)
    df = pd.read_csv("./data/module2/feature_frame.csv")
    df = clean_dataset(df)
    X, y = (
        df[["global_popularity", "ordered_before", "abandoned_before"]],
        df["outcome"],
    )

    y_pred = model.predict(X)

    print(evaluate_model(y, y_pred))

    return


if __name__ == "__main__":
    main()
