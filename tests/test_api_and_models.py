from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "v2_model.joblib"
VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(MODEL_FILE)


@pytest.fixture(scope="module")
def sample_features(artifact):
    validation = pd.read_csv(VALIDATION_FILE, nrows=1)
    return validation[artifact["feature_columns"]].iloc[0].to_dict()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def assert_prediction_response(response):
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"probabilities", "implied_odds", "predicted_outcome"}
    assert set(body["probabilities"]) == {"home_win", "draw", "away_win"}
    assert set(body["implied_odds"]) == {"home_win", "draw", "away_win"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["predicted_outcome"] in {"HOME_WIN", "DRAW", "AWAY_WIN"}
    for outcome, probability in body["probabilities"].items():
        assert 0.0 <= probability <= 1.0
        assert body["implied_odds"][outcome] == pytest.approx(1.0 / probability)
    return body


def test_health_endpoint_reports_ready_model(client, artifact):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": artifact["model_name"],
        "feature_count": len(artifact["feature_columns"]),
        "target_column": artifact["target_column"],
    }


def test_predict_accepts_valid_sample_features(client, sample_features):
    response = client.post(
        "/predict",
        json={"features": sample_features, "draw_threshold": 0.26},
    )

    assert_prediction_response(response)


def test_predict_rejects_missing_and_unexpected_features(client, sample_features):
    missing_features = dict(sample_features)
    missing_features.pop(next(iter(missing_features)))
    missing_response = client.post(
        "/predict",
        json={"features": missing_features},
    )

    unexpected_features = dict(sample_features)
    unexpected_features["not_a_model_feature"] = 0.0
    unexpected_response = client.post(
        "/predict",
        json={"features": unexpected_features},
    )

    assert missing_response.status_code == 422
    assert "missing features" in missing_response.json()["detail"]
    assert unexpected_response.status_code == 422
    assert "unexpected features" in unexpected_response.json()["detail"]


def test_predict_accepts_zero_initialized_team_stats(client, artifact):
    zero_features = {column: 0.0 for column in artifact["feature_columns"]}

    response = client.post("/predict", json={"features": zero_features})

    assert_prediction_response(response)


def test_team_lookup_and_prediction_endpoints(client):
    teams_response = client.get("/teams")
    assert teams_response.status_code == 200
    teams = teams_response.json()
    assert len(teams) > 0
    assert teams == sorted(teams)

    prediction_response = client.post(
        "/predict/teams",
        json={
            "home_team": teams[0],
            "away_team": teams[1],
        },
    )

    assert_prediction_response(prediction_response)


def test_team_prediction_rejects_unknown_team(client):
    response = client.post(
        "/predict/teams",
        json={
            "home_team": "Not a UCL team",
            "away_team": "Also not a UCL team",
        },
    )

    assert response.status_code == 422
    assert "Team not found" in response.json()["detail"]