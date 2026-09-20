import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, f1_score, log_loss, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import MODEL_DIR, PROCESSED_DATA_DIR
from src.models.train_baseline import (
    NON_FEATURE_COLUMNS,
    prepare_features,
    resolve_target_column,
)


TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
V1_MODEL_FILE = MODEL_DIR / "baseline_model.joblib"
MODEL_FILE = MODEL_DIR / "v2_model.joblib"
DISPLAY_LABELS = ["HOME_WIN", "DRAW", "AWAY_WIN"]


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


def encode_target(target, labels):
    label_to_index = {label: index for index, label in enumerate(labels)}
    return target.map(label_to_index), label_to_index


def decode_predictions(predictions, model_labels, labels):
    label_map = dict(zip(model_labels, labels))
    return pd.Series(predictions).map(label_map).tolist()


def model_labels(model, labels):
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model[-1].classes_
    if all(isinstance(value, int) for value in classes):
        return [labels[value] for value in classes]
    return list(classes)


def evaluate(model, features, target, labels, encoded=False):
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)
    if encoded:
        model_label_values = model_labels(model, labels)
        prediction_labels = decode_predictions(predictions, model_label_values, labels)
        probability_labels = list(range(len(labels)))
        encoded_target = target.map({label: index for index, label in enumerate(labels)})
        matrix_target = encoded_target
        matrix_predictions = predictions
        matrix_labels = [labels.index(label) for label in DISPLAY_LABELS]
    else:
        prediction_labels = predictions
        probability_labels = labels
        matrix_target = target
        matrix_predictions = prediction_labels
        matrix_labels = DISPLAY_LABELS

    return {
        "log_loss": log_loss(matrix_target, probabilities, labels=probability_labels),
        "draw_recall": recall_score(
            matrix_target,
            matrix_predictions,
            labels=[labels.index("DRAW") if encoded else "DRAW"],
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            matrix_target,
            matrix_predictions,
            average="macro",
            zero_division=0,
        ),
        "predictions": prediction_labels,
        "confusion_matrix": confusion_matrix(
            matrix_target,
            matrix_predictions,
            labels=matrix_labels,
        ),
    }


def print_confusion_matrix(title, metrics, labels):
    print(f"\n{title}:")
    print(f"{'Actual / Predicted':<20}" + "".join(f"{label:>12}" for label in labels))
    for label, row in zip(labels, metrics["confusion_matrix"]):
        print(f"{label:<20}" + "".join(f"{value:12d}" for value in row))


def build_logistic():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost(n_classes):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def calibrate(model):
    return CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
    )


def print_comparison(results):
    print("\nV2 validation comparison:")
    print(f"{'Model':<32}{'Log Loss':>12}{'DRAW Recall':>14}")
    print("-" * 58)
    for name, metrics in results.items():
        print(
            f"{name:<32}{metrics['log_loss']:12.4f}"
            f"{metrics['draw_recall']:14.4f}"
        )


def train_v2(
    train_file=TRAIN_FILE,
    val_file=VAL_FILE,
    test_file=TEST_FILE,
    v1_model_file=V1_MODEL_FILE,
    model_file=MODEL_FILE,
):
    train = pd.read_csv(train_file)
    validation = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    target_column = resolve_target_column([train, validation, test])
    feature_columns = select_feature_columns(train, target_column)

    x_train = prepare_features(train, feature_columns)
    x_validation = prepare_features(validation, feature_columns)
    y_train = train[target_column]
    y_validation = validation[target_column]
    labels = sorted(y_train.unique())
    encoded_y_train, _ = encode_target(y_train, labels)

    raw_logistic = build_logistic()
    raw_logistic.fit(x_train, y_train)
    calibrated_logistic = calibrate(build_logistic())
    calibrated_logistic.fit(x_train, y_train)

    raw_xgboost = build_xgboost(len(labels))
    raw_xgboost.fit(x_train, encoded_y_train)
    calibrated_xgboost = calibrate(build_xgboost(len(labels)))
    calibrated_xgboost.fit(x_train, encoded_y_train)

    results = {
        "Balanced Logistic Regression (before calibration)": evaluate(
            raw_logistic, x_validation, y_validation, labels
        ),
        "Balanced Logistic Regression (calibrated)": evaluate(
            calibrated_logistic, x_validation, y_validation, labels
        ),
        "XGBoost (before calibration)": evaluate(
            raw_xgboost, x_validation, y_validation, labels, encoded=True
        ),
        "XGBoost (calibrated)": evaluate(
            calibrated_xgboost, x_validation, y_validation, labels, encoded=True
        ),
    }

    print_comparison(results)
    for name, metrics in results.items():
        print_confusion_matrix(f"{name} confusion matrix", metrics, DISPLAY_LABELS)

    v1_artifact = joblib.load(v1_model_file)
    v1_features = prepare_features(validation, v1_artifact["feature_columns"])
    v1_target = validation[v1_artifact["target_column"]]
    v1_model = v1_artifact["model"]
    v1_predictions = v1_model.predict(v1_features)
    v1_labels = list(v1_model.classes_)
    v1_draw_recall = recall_score(
        v1_target,
        v1_predictions,
        labels=["DRAW"],
        average="macro",
        zero_division=0,
    )
    print("\nComparison against v1 baseline:")
    print(f"  v1 validation DRAW recall: {v1_draw_recall:.4f}")
    for name, metrics in results.items():
        change = metrics["draw_recall"] - v1_draw_recall
        print(
            f"  {name} DRAW recall: {metrics['draw_recall']:.4f} "
            f"(change: {change:+.4f})"
        )

    calibrated_results = {
        name: metrics
        for name, metrics in results.items()
        if "calibrated" in name.lower()
    }
    best_name = min(
        calibrated_results,
        key=lambda name: calibrated_results[name]["log_loss"],
    )
    best_model = {
        "Balanced Logistic Regression (calibrated)": calibrated_logistic,
        "XGBoost (calibrated)": calibrated_xgboost,
    }[best_name]

    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "classes": labels,
            "validation_metrics": calibrated_results[best_name],
        },
        model_file,
    )
    print(f"\nSelected calibrated model: {best_name}")
    print(f"Saved model: {model_file}")
    return best_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train calibrated v2 UCL models.")
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--val", type=Path, default=VAL_FILE)
    parser.add_argument("--test", type=Path, default=TEST_FILE)
    parser.add_argument("--v1-model", type=Path, default=V1_MODEL_FILE)
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_v2(
        train_file=args.train,
        val_file=args.val,
        test_file=args.test,
        v1_model_file=args.v1_model,
        model_file=args.model,
    )