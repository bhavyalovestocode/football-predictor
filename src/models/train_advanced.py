import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import MODEL_DIR, PROCESSED_DATA_DIR
from src.models.train_baseline import (
    NON_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    prepare_features,
    resolve_target_column,
)
from sklearn.linear_model import LogisticRegression


TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "advanced_model.joblib"


def select_feature_columns(train, target_column):
    feature_columns = [
        column
        for column in train.columns
        if column not in NON_FEATURE_COLUMNS
        and column != target_column
        and pd.api.types.is_numeric_dtype(train[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns were found.")
    return feature_columns


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


def print_comparison(metrics_by_model):
    metric_names = ["Log Loss", "Accuracy", "F1 Macro", "F1 Weighted"]
    print("\nValidation metrics comparison:")
    print(f"{'Model':<28} " + " ".join(f"{name:>14}" for name in metric_names))
    print("-" * 88)
    for model_name, metrics in metrics_by_model.items():
        values = " ".join(f"{metrics[name]:14.4f}" for name in metric_names)
        print(f"{model_name:<28} {values}")


def build_logistic_baseline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )


def build_calibrated_xgboost(n_classes):
    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    return CalibratedClassifierCV(
        estimator=classifier,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
    )


def train_advanced(
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
    feature_columns = select_feature_columns(train, target_column)

    x_train = prepare_features(train, feature_columns)
    x_validation = prepare_features(validation, feature_columns)
    x_test = prepare_features(test, feature_columns)
    y_train = train[target_column]
    y_validation = validation[target_column]
    y_test = test[target_column]
    labels = sorted(y_train.unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    encoded_y_train = y_train.map(label_to_index)

    logistic_baseline = build_logistic_baseline()
    logistic_baseline.fit(x_train, y_train)
    logistic_metrics = evaluate_model(
        logistic_baseline,
        x_validation,
        y_validation,
        labels,
    )

    calibrated_xgboost = build_calibrated_xgboost(len(labels))
    calibrated_xgboost.fit(x_train, encoded_y_train)
    xgboost_metrics = evaluate_model(
        calibrated_xgboost,
        x_validation,
        y_validation.map(label_to_index),
        list(range(len(labels))),
    )

    print_comparison(
        {
            "Logistic Regression": logistic_metrics,
            "Calibrated XGBoost": xgboost_metrics,
        }
    )

    candidates = {
        "Logistic Regression": (logistic_baseline, logistic_metrics),
        "Calibrated XGBoost": (calibrated_xgboost, xgboost_metrics),
    }
    best_name, (best_model, best_validation_metrics) = min(
        candidates.items(),
        key=lambda item: item[1][1]["Log Loss"],
    )

    test_target = y_test if best_name == "Logistic Regression" else y_test.map(label_to_index)
    test_labels = labels if best_name == "Logistic Regression" else list(range(len(labels)))
    test_metrics = evaluate_model(best_model, x_test, test_target, test_labels)

    print(f"\nSelected model: {best_name}")
    print("Validation metrics for selected model:")
    for metric_name, value in best_validation_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print("\nHeld-out test metrics:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "classes": labels,
            "validation_metrics": best_validation_metrics,
            "test_metrics": test_metrics,
        },
        model_file,
    )

    print(f"\nSaved model: {model_file}")
    return best_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train calibrated gradient-boosting and baseline classifiers."
    )
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--val", type=Path, default=VAL_FILE)
    parser.add_argument("--test", type=Path, default=TEST_FILE)
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_advanced(
        train_file=args.train,
        val_file=args.val,
        test_file=args.test,
        model_file=args.model,
    )