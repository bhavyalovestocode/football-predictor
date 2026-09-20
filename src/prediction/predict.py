import argparse
from pathlib import Path
from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

from src.config import MODEL_DIR, PROCESSED_DATA_DIR
from src.models.train_baseline import prepare_features, resolve_target_column


MODEL_FILE = MODEL_DIR / "v2_model.joblib"
FEATURES_FILE = PROCESSED_DATA_DIR / "ucl_features_v2.csv"
VALIDATION_FILE = PROCESSED_DATA_DIR / "val.csv"
OUTCOME_ORDER = ["HOME_WIN", "DRAW", "AWAY_WIN"]
DIFFERENTIAL_FEATURES = {
    "ppg_diff",
    "overall_ppg_diff",
    "goal_diff_pg",
    "expected_goal_margin",
}


def load_artifact(model_file=MODEL_FILE):
    return joblib.load(model_file)


def get_available_teams(features_df=None):
    if features_df is None:
        features_df = pd.read_csv(FEATURES_FILE)
    return sorted(
        set(features_df["home_team"].dropna())
        | set(features_df["away_team"].dropna())
    )


def _latest_team_row(features_df, team_name, role=None):
    if role == "home":
        rows = features_df[features_df["home_team"] == team_name]
    elif role == "away":
        rows = features_df[features_df["away_team"] == team_name]
    else:
        rows = features_df[
            (features_df["home_team"] == team_name)
            | (features_df["away_team"] == team_name)
        ]

    if rows.empty:
        role_description = f" as {role}" if role else ""
        raise ValueError(f"Team not found{role_description}: {team_name}")

    rows = rows.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce", utc=True)
    rows = rows.dropna(subset=["date"])
    if rows.empty:
        raise ValueError(f"Team has no dated matches: {team_name}")
    sort_columns = ["date"]
    if "match_id" in rows.columns:
        sort_columns.append("match_id")
    return rows.sort_values(sort_columns).iloc[-1]


def get_latest_team_features(
    home_team,
    away_team,
    features_df=None,
    feature_columns=None,
):
    """Build a pre-match feature row from each team's latest available state."""
    if home_team == away_team:
        raise ValueError("Home and away teams must be different.")
    if features_df is None:
        features_df = pd.read_csv(FEATURES_FILE)
    if feature_columns is None:
        feature_columns = load_artifact()["feature_columns"]

    home_latest = _latest_team_row(features_df, home_team)
    away_latest = _latest_team_row(features_df, away_team)
    home_role_latest = _latest_team_row(features_df, home_team, role="home")
    away_role_latest = _latest_team_row(features_df, away_team, role="away")

    def value_for_team(column, latest, role_latest, source_side):
        if column.startswith("home_team_home_"):
            return role_latest[column]
        if column.startswith("away_team_away_"):
            return role_latest[column]

        prefix = f"{source_side}_"
        if column.startswith("home_"):
            source_column = prefix + column[len("home_"):]
        elif column.startswith("away_"):
            source_column = prefix + column[len("away_"):]
        else:
            raise ValueError(f"Unsupported feature column: {column}")
        return latest[source_column]

    values = {}
    home_source_side = "home" if home_latest["home_team"] == home_team else "away"
    away_source_side = "home" if away_latest["home_team"] == away_team else "away"
    for column in feature_columns:
        if column in DIFFERENTIAL_FEATURES:
            continue
        if column.startswith("home_"):
            values[column] = value_for_team(
                column, home_latest, home_role_latest, home_source_side
            )
        elif column.startswith("away_"):
            values[column] = value_for_team(
                column, away_latest, away_role_latest, away_source_side
            )
        else:
            raise ValueError(f"Unsupported feature column: {column}")

    values["ppg_diff"] = values["home_form_ppg"] - values["away_form_ppg"]
    values["overall_ppg_diff"] = (
        values["home_overall_ppg"] - values["away_overall_ppg"]
    )
    values["goal_diff_pg"] = (
        values["home_form_goals_for_pg"]
        - values["away_form_goals_against_pg"]
    )
    values["expected_goal_margin"] = (
        values["home_form_goals_for_pg"]
        - values["away_form_goals_against_pg"]
        - values["away_form_goals_for_pg"]
        + values["home_form_goals_against_pg"]
    )
    return pd.DataFrame([values], columns=feature_columns)


def _model_class_labels(model, artifact_classes):
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model[-1].classes_

    if classes is None:
        return list(artifact_classes)

    labels = []
    for value in classes:
        if isinstance(value, (int, np.integer)):
            labels.append(artifact_classes[int(value)])
        else:
            labels.append(value)
    return labels


def _prepare_model_features(features_df, feature_columns):
    return prepare_features(features_df, feature_columns)


def predict_match_probabilities(features_df):
    """Return calibrated Home/Draw/Away probabilities for feature rows."""
    artifact = load_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    features = _prepare_model_features(features_df, feature_columns)
    raw_probabilities = model.predict_proba(features)
    model_labels = _model_class_labels(model, artifact["classes"])

    probabilities = np.zeros((len(features), len(OUTCOME_ORDER)))
    for model_index, label in enumerate(model_labels):
        if label in OUTCOME_ORDER:
            outcome_index = OUTCOME_ORDER.index(label)
            probabilities[:, outcome_index] = raw_probabilities[:, model_index]

    return pd.DataFrame(
        probabilities,
        columns=[f"P_{outcome}" for outcome in OUTCOME_ORDER],
        index=features_df.index,
    )


def predict_match_outcome(probs, draw_threshold=0.26):
    """Choose Draw when its probability clears the threshold, else Home/Away."""
    if not 0.0 <= draw_threshold <= 1.0:
        raise ValueError("draw_threshold must be between 0 and 1.")

    if isinstance(probs, Mapping) or isinstance(probs, pd.Series):
        values = [probs[key] for key in [f"P_{outcome}" for outcome in OUTCOME_ORDER]]
    else:
        values = list(probs)
    if len(values) != 3:
        raise ValueError("probs must contain Home, Draw, and Away probabilities.")

    home_probability, draw_probability, away_probability = values
    if draw_probability >= draw_threshold:
        return "DRAW"
    return "HOME_WIN" if home_probability >= away_probability else "AWAY_WIN"


def implied_odds(probabilities):
    return probabilities.map(
        lambda probability: np.inf if probability <= 0 else 1.0 / probability
    )


def evaluate_draw_threshold(
    validation_file=VALIDATION_FILE,
    draw_threshold=0.26,
):
    validation = pd.read_csv(validation_file)
    artifact = load_artifact()
    target_column = resolve_target_column([validation])
    probabilities = predict_match_probabilities(validation)
    actual = validation[target_column]
    standard_predictions = probabilities.idxmax(axis=1).str.removeprefix("P_")
    threshold_predictions = probabilities.apply(
        lambda row: predict_match_outcome(row, draw_threshold),
        axis=1,
    )

    standard_accuracy = accuracy_score(actual, standard_predictions)
    threshold_accuracy = accuracy_score(actual, threshold_predictions)
    standard_draw_recall = recall_score(
        actual,
        standard_predictions,
        labels=["DRAW"],
        average="macro",
        zero_division=0,
    )
    threshold_draw_recall = recall_score(
        actual,
        threshold_predictions,
        labels=["DRAW"],
        average="macro",
        zero_division=0,
    )

    candidate_thresholds = np.arange(0.20, 0.41, 0.001)
    accuracy_preserving = []
    for candidate in candidate_thresholds:
        candidate_predictions = probabilities.apply(
            lambda row: predict_match_outcome(row, candidate),
            axis=1,
        )
        candidate_accuracy = accuracy_score(actual, candidate_predictions)
        if candidate_accuracy >= standard_accuracy:
            accuracy_preserving.append(
                (
                    recall_score(
                        actual,
                        candidate_predictions,
                        labels=["DRAW"],
                        average="macro",
                        zero_division=0,
                    ),
                    candidate_accuracy,
                    candidate,
                )
            )

    best_threshold_result = max(accuracy_preserving)

    print("\nDraw-threshold validation evaluation:")
    print(f"  Model: {artifact['model_name']}")
    print(f"  Draw threshold: {draw_threshold:.2f}")
    print(f"  Standard argmax accuracy: {standard_accuracy:.4f}")
    print(f"  Threshold accuracy: {threshold_accuracy:.4f}")
    print(f"  Standard argmax DRAW recall: {standard_draw_recall:.4f}")
    print(f"  Threshold DRAW recall: {threshold_draw_recall:.4f}")
    print(
        "  Accuracy preserved or improved: "
        f"{threshold_accuracy >= standard_accuracy}"
    )
    print(
        "  Best accuracy-preserving threshold: "
        f"{best_threshold_result[2]:.3f}"
    )
    print(
        "  Best threshold accuracy / DRAW recall: "
        f"{best_threshold_result[1]:.4f} / {best_threshold_result[0]:.4f}"
    )

    return {
        "standard_accuracy": standard_accuracy,
        "threshold_accuracy": threshold_accuracy,
        "standard_draw_recall": standard_draw_recall,
        "threshold_draw_recall": threshold_draw_recall,
        "best_threshold": best_threshold_result[2],
        "best_threshold_accuracy": best_threshold_result[1],
        "best_threshold_draw_recall": best_threshold_result[0],
    }


def _load_cli_rows(features_file, match_ids, data_file):
    if features_file is not None:
        return pd.read_csv(features_file)

    data = pd.read_csv(data_file)
    if "match_id" not in data.columns:
        raise ValueError("The match lookup data must contain 'match_id'.")
    rows = data[data["match_id"].astype(str).isin(match_ids)].copy()
    missing = sorted(set(match_ids) - set(rows["match_id"].astype(str)))
    if missing:
        raise ValueError(f"Match IDs were not found: {', '.join(missing)}")
    return rows


def print_predictions(rows, probabilities, draw_threshold):
    odds = implied_odds(probabilities)
    for row_number, (row_index, probability_row) in enumerate(
        probabilities.iterrows(), start=1
    ):
        outcome = predict_match_outcome(probability_row, draw_threshold)
        match_id = rows.loc[row_index, "match_id"] if "match_id" in rows else row_number
        print(f"\nMatch {match_id}:")
        print(f"  P(Home): {probability_row['P_HOME_WIN']:.4f}")
        print(f"  P(Draw): {probability_row['P_DRAW']:.4f}")
        print(f"  P(Away): {probability_row['P_AWAY_WIN']:.4f}")
        print(f"  Estimated outcome: {outcome}")
        print(f"  Implied odds Home: {odds.loc[row_index, 'P_HOME_WIN']:.2f}")
        print(f"  Implied odds Draw: {odds.loc[row_index, 'P_DRAW']:.2f}")
        print(f"  Implied odds Away: {odds.loc[row_index, 'P_AWAY_WIN']:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict calibrated UCL match probabilities and implied odds."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--features",
        type=Path,
        help="CSV containing one or more feature rows.",
    )
    input_group.add_argument(
        "--match-id",
        nargs="+",
        help="One or more match IDs to look up in the data file.",
    )
    parser.add_argument("--data", type=Path, default=VALIDATION_FILE)
    parser.add_argument("--draw-threshold", type=float, default=0.26)
    parser.add_argument("--evaluate-threshold", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.evaluate_threshold:
        evaluate_draw_threshold(
            validation_file=args.data,
            draw_threshold=args.draw_threshold,
        )
    elif args.features is not None or args.match_id:
        rows = _load_cli_rows(args.features, args.match_id, args.data)
        probabilities = predict_match_probabilities(rows)
        print_predictions(rows, probabilities, args.draw_threshold)
    else:
        raise SystemExit("Provide --features, --match-id, or --evaluate-threshold.")