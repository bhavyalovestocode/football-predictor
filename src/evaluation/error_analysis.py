import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import MODEL_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.models.train_baseline import prepare_features, resolve_target_column


MODEL_FILE = MODEL_DIR / "advanced_model.joblib"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "confusion_matrix.txt"
CLASS_LABELS = ["HOME_WIN", "DRAW", "AWAY_WIN"]


def model_classes(model, artifact_classes):
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model[-1], "classes_", None)

    if classes is None:
        return list(artifact_classes)

    classes = list(classes)
    if all(isinstance(value, int) for value in classes):
        return [artifact_classes[value] for value in classes]
    return classes


def feature_influence(model, feature_columns):
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model[-1]

    if hasattr(estimator, "coef_"):
        coefficients = estimator.coef_
        scores = abs(coefficients).mean(axis=0)
        return pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": scores,
            }
        ).sort_values("importance", ascending=False)

    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": estimator.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    if hasattr(estimator, "calibrated_classifiers_"):
        importances = []
        for calibrated in estimator.calibrated_classifiers_:
            base_estimator = calibrated.estimator
            if hasattr(base_estimator, "feature_importances_"):
                importances.append(base_estimator.feature_importances_)
        if importances:
            return pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": pd.DataFrame(importances).mean(axis=0),
                }
            ).sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["feature", "importance"])


def decode_predictions(predictions, model_labels, artifact_classes):
    label_map = dict(zip(model_labels, artifact_classes))
    return pd.Series(predictions).map(label_map).fillna(pd.Series(predictions)).tolist()


def build_report(
    model_name,
    influence,
    matrix,
    labels,
    confident_errors,
):
    lines = [
        "UCL MODEL ERROR ANALYSIS",
        "=" * 70,
        f"Model: {model_name}",
        f"Test matches: {matrix.sum()}",
        "",
        "TOP 15 FEATURE INFLUENCE",
        "-" * 70,
    ]

    if influence.empty:
        lines.append("Feature importance is not available for this model.")
    else:
        for rank, row in enumerate(influence.head(15).itertuples(), start=1):
            lines.append(
                f"{rank:2}. {row.feature}: {row.importance:.6f}"
            )

    lines.extend(
        [
            "",
            "CONFUSION MATRIX (actual rows, predicted columns)",
            "-" * 70,
            "".join(f"{label:>14}" for label in ["Actual"] + labels),
        ]
    )
    for label, row in zip(labels, matrix):
        lines.append(f"{label:>14}" + "".join(f"{value:14d}" for value in row))

    lines.extend(
        [
            "",
            "HIGH-CONFIDENCE INCORRECT PREDICTIONS (probability > 0.60)",
            "-" * 70,
            f"Count: {len(confident_errors)}",
        ]
    )
    if not confident_errors.empty:
        distribution = (
            confident_errors.groupby(["actual", "predicted"])
            .size()
            .sort_values(ascending=False)
        )
        lines.append("Error distribution:")
        for (actual, predicted), count in distribution.items():
            lines.append(f"  Actual {actual}, predicted {predicted}: {count}")

    return "\n".join(lines) + "\n"


def run_error_analysis(
    model_file=MODEL_FILE,
    test_file=TEST_FILE,
    report_file=REPORT_FILE,
):
    artifact = joblib.load(model_file)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    artifact_classes = artifact.get("classes", CLASS_LABELS)
    test = pd.read_csv(test_file)
    target_column = resolve_target_column([test])
    features = prepare_features(test, feature_columns)
    actual = test[target_column].tolist()

    probabilities = model.predict_proba(features)
    raw_predictions = model.predict(features)
    model_labels = model_classes(model, artifact_classes)
    predicted = decode_predictions(raw_predictions, model_labels, artifact_classes)
    confidence = probabilities.max(axis=1)
    confident_errors = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "confidence": confidence,
        }
    )
    confident_errors = confident_errors[
        (confident_errors["actual"] != confident_errors["predicted"])
        & (confident_errors["confidence"] > 0.60)
    ]

    labels = [label for label in CLASS_LABELS if label in artifact_classes]
    labels.extend(label for label in artifact_classes if label not in labels)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    influence = feature_influence(model, feature_columns)
    report = build_report(
        artifact.get("model_name", type(model).__name__),
        influence,
        matrix,
        labels,
        confident_errors,
    )

    print(report)
    report_file = Path(report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report, encoding="utf-8")
    print(f"Saved report: {report_file}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze model predictions and errors.")
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    parser.add_argument("--test", type=Path, default=TEST_FILE)
    parser.add_argument("--report", type=Path, default=REPORT_FILE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_error_analysis(
        model_file=args.model,
        test_file=args.test,
        report_file=args.report,
    )