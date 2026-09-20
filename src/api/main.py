from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODEL_DIR
from src.models.train_baseline import prepare_features
from src.prediction.predict import (
    OUTCOME_ORDER,
    _model_class_labels,
    predict_match_outcome,
)
from src.api.ui import dashboard_html


MODEL_FILE = MODEL_DIR / "v2_model.joblib"


class MatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: Dict[str, float] = Field(
        ...,
        description="Pre-match numeric features keyed by the trained model feature names.",
    )
    draw_threshold: float = Field(default=0.26, ge=0.0, le=1.0)


class MatchPredictionResponse(BaseModel):
    probabilities: Dict[str, float]
    implied_odds: Dict[str, float]
    predicted_outcome: str


def _load_model_artifact(model_file=MODEL_FILE):
    return joblib.load(model_file)


def _predict_with_artifact(artifact, feature_values):
    feature_columns = artifact["feature_columns"]
    missing = sorted(set(feature_columns) - set(feature_values))
    unexpected = sorted(set(feature_values) - set(feature_columns))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing features: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected features: {', '.join(unexpected)}")
        raise ValueError("; ".join(details))

    feature_frame = pd.DataFrame([feature_values], columns=feature_columns)
    prepared_features = prepare_features(feature_frame, feature_columns)
    model = artifact["model"]
    raw_probabilities = model.predict_proba(prepared_features)[0]
    model_labels = _model_class_labels(model, artifact["classes"])

    probabilities = {outcome: 0.0 for outcome in OUTCOME_ORDER}
    for index, label in enumerate(model_labels):
        if label in probabilities:
            probabilities[label] = float(raw_probabilities[index])

    odds = {
        outcome: 1.0 / max(probability, np.finfo(float).tiny)
        for outcome, probability in probabilities.items()
    }
    return probabilities, odds


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path(MODEL_FILE).exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_FILE}")
    app.state.model_artifact = _load_model_artifact()
    yield


app = FastAPI(
    title="UCL Match Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    artifact = request.app.state.model_artifact
    return HTMLResponse(dashboard_html(artifact["feature_columns"]))


@app.get("/health")
def health(request: Request):
    artifact = request.app.state.model_artifact
    return {
        "status": "ok",
        "model": artifact.get("model_name", "unknown"),
        "feature_count": len(artifact["feature_columns"]),
        "target_column": artifact.get("target_column"),
    }


@app.post("/predict", response_model=MatchPredictionResponse)
def predict(request: MatchPredictionRequest, app_request: Request):
    try:
        probabilities, implied_odds = _predict_with_artifact(
            app_request.app.state.model_artifact,
            request.features,
        )
        predicted_outcome = predict_match_outcome(
            {f"P_{outcome}": probability for outcome, probability in probabilities.items()},
            draw_threshold=request.draw_threshold,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return MatchPredictionResponse(
        probabilities={
            "home_win": probabilities["HOME_WIN"],
            "draw": probabilities["DRAW"],
            "away_win": probabilities["AWAY_WIN"],
        },
        implied_odds={
            "home_win": implied_odds["HOME_WIN"],
            "draw": implied_odds["DRAW"],
            "away_win": implied_odds["AWAY_WIN"],
        },
        predicted_outcome=predicted_outcome,
    )