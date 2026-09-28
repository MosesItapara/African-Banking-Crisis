import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from flask import Flask, jsonify, request

from src.pipelines.feature_eng_pipeline import FeatureEngineeringPipeline

# ---------- Loaded once at startup ----------
with open("config/api.yaml") as f:
    CFG = yaml.safe_load(f)
with open(CFG["feature_config"]) as f:
    FE_CFG = yaml.safe_load(f)

MODEL_DIR = Path(CFG["model_dir"])
with open(MODEL_DIR / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)
FEATURE_COLUMNS = json.loads((MODEL_DIR / "feature_columns.json").read_text())
METADATA = json.loads((MODEL_DIR / "metadata.json").read_text())

HISTORY = pd.read_csv(CFG["history_path"])
LAST_YEAR = HISTORY.groupby("cc3")["year"].max().to_dict()
# Training capped inflation at its 99th percentile; the cleaned history's max is that cap
INFLATION_CAP = HISTORY["inflation_annual_cpi"].max()

REQUIRED = CFG["required_fields"]
BINARY = CFG["binary_fields"]

app = Flask(__name__)


def validate(payload: dict) -> list:
    """Return a list of problems with the request (empty if valid)"""
    errors = [f"missing field '{field}'" for field in ["cc3", "year"] + REQUIRED if field not in payload]
    if errors:
        return errors

    cc3 = payload["cc3"]
    if cc3 not in LAST_YEAR:
        return [f"unknown country '{cc3}', expected one of {sorted(LAST_YEAR)}"]

    if not isinstance(payload["year"], int) or isinstance(payload["year"], bool):
        errors.append("'year' must be an integer")
    elif payload["year"] > LAST_YEAR[cc3] + 1:
        errors.append(f"'year' can be at most {LAST_YEAR[cc3] + 1} (history for {cc3} ends in {LAST_YEAR[cc3]})")
    elif (HISTORY[HISTORY["cc3"] == cc3]["year"] < payload["year"]).sum() == 0:
        errors.append(f"no history for {cc3} before {payload['year']}")

    for field in REQUIRED:
        value = payload[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"'{field}' must be a number")
        elif field in BINARY and value not in (0, 1):
            errors.append(f"'{field}' must be 0 or 1")

    return errors


def model_feature_names() -> np.ndarray:
    """Names of the columns that reach the final estimator"""
    names = np.array(FEATURE_COLUMNS)
    for _, step in MODEL.steps[:-1]:
        if hasattr(step, "get_support"):
            names = names[step.get_support()]
    return names


def top_contributions(X: pd.DataFrame, n: int) -> list:
    """Per-row contribution of each feature to this prediction's log-odds"""
    estimator = MODEL[-1]
    X_model = MODEL[:-1].transform(X)

    if hasattr(estimator, "coef_"):
        # Logistic regression: contribution = coefficient * (scaled) value
        contributions = X_model[0] * estimator.coef_[0]
    elif isinstance(estimator, xgb.XGBClassifier):
        contributions = estimator.get_booster().predict(xgb.DMatrix(X_model), pred_contribs=True)[0, :-1]
    else:
        return []

    names = model_feature_names()
    order = np.argsort(-np.abs(contributions))[:n]
    return [{"feature": str(names[i]), "contribution": round(float(contributions[i]), 3)} for i in order]


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": METADATA})


@app.get("/countries")
def countries():
    return jsonify({cc3: {"last_year": int(year), "next_year": int(year) + 1}
                    for cc3, year in sorted(LAST_YEAR.items())})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"errors": ["request body must be a JSON object"]}), 400

    errors = validate(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    cc3, year = payload["cc3"], payload["year"]

    # New year's raw indicators, cleaned the same way as training
    new_row = {field: float(payload[field]) for field in REQUIRED}
    new_row.update({"cc3": cc3, "year": year})
    new_row["inflation_annual_cpi"] = min(new_row["inflation_annual_cpi"], INFLATION_CAP)

    # History + new row -> same feature steps as training -> keep the new row
    history = HISTORY[(HISTORY["cc3"] == cc3) & (HISTORY["year"] < year)]
    combined = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    features = FeatureEngineeringPipeline(FE_CFG).transform(combined)

    # Only the country dummies may be absent (one country per request); anything else is a config mismatch
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns and not c.startswith("cc3_")]
    if missing:
        return jsonify({"errors": [f"model expects columns the API can't build: {missing}"]}), 500

    X = features.iloc[[-1]].reindex(columns=FEATURE_COLUMNS, fill_value=0).astype("float64")
    probability = float(MODEL.predict_proba(X)[0, 1])

    return jsonify({
        "cc3": cc3,
        "year": year,
        "crisis_probability": round(probability, 4),
        "predicted_crisis": int(probability >= CFG["threshold"]),
        "threshold": CFG["threshold"],
        "top_features": top_contributions(X, CFG["n_top_features"]),
        "model_version": METADATA["version"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)