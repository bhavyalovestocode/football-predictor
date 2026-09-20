import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR, PROCESSED_DATA_DIR


TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "baseline_model.joblib"

TARGET_COLUMNS = ("result_after_90", "target")
NON_FEATURE_COLUMNS = {
    "match_id",
    "date",
    "season",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "phase",
    "round",
    "match_type",
}


def resolve_target_column(dataframes):
    for target_column in TARGET_COLUMNS:
        if all(target_column in dataframe.columns for dataframe in dataframes):
            return target_column

    expected = " or ".join(TARGET_COLUMNS)
    raise ValueError(f"All split files must contain the target column {expected}.")


def prepare_features(dataframe, feature_columns):
    missing_columns = set(feature_columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Split is missing feature columns: {sorted(missing_columns)}")

    features = dataframe[feature_columns].apply(pd.to_numeric, errors="raise")
    if features.isna().any().any():
        raise ValueError("Feature columns contain missing values.")

    return features


def evaluate_model(model, features, target, labels):
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)

    return {
        "Log Loss": log_loss(target, probabilities, labels=labels),
        "Accuracy": accuracy_score(target, predictions),
        "F1 Macro": f1_score(target, predictions, average="macro", zero_division=0),
        "F1 Weighted": f1_score(
            target,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def print_metrics(model_name, metrics):
    print(f"\n{model_name} validation metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")


def train_baseline(
    train_file=TRAIN_FILE,
    val_file=VAL_FILE,
    test_file=TEST_FILE,
    model_file=MODEL_FILE,
):
    train = pd.read_csv(train_file)
    validation = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    splits = [train, validation, test]
    target_column = resolve_target_column(splits)

    feature_columns = [
        column
        for column in train.columns
        if column not in NON_FEATURE_COLUMNS
        and column != target_column
        and pd.api.types.is_numeric_dtype(train[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns were found.")

    x_train = prepare_features(train, feature_columns)
    x_validation = prepare_features(validation, feature_columns)
    y_train = train[target_column]
    y_validation = validation[target_column]
    labels = sorted(y_train.unique())

    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(x_train, y_train)
    print_metrics(
        "Majority-class baseline",
        evaluate_model(majority_model, x_validation, y_validation, labels),
    )

    baseline_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )
    baseline_model.fit(x_train, y_train)
    print_metrics(
        "Logistic regression baseline",
        evaluate_model(baseline_model, x_validation, y_validation, labels),
    )

    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": baseline_model,
            "feature_columns": feature_columns,
            "target_column": target_column,
        },
        model_file,
    )

    print(f"\nTraining matches: {len(train)}")
    print(f"Validation matches: {len(validation)}")
    print(f"Test matches available: {len(test)}")
    print(f"Features used: {len(feature_columns)}")
    print(f"Saved model: {model_file}")

    return baseline_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline UCL match classifiers."
    )
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--val", type=Path, default=VAL_FILE)
    parser.add_argument("--test", type=Path, default=TEST_FILE)
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline(
        train_file=args.train,
        val_file=args.val,
        test_file=args.test,
        model_file=args.model,
    )