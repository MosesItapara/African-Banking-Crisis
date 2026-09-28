from pathlib import Path

import pytest
import yaml

from src.pipelines.preprocessing_pipeline import PreprocessingPipeline

# The API loads the cleaned history at import time. On a fresh checkout (e.g. CI)
# it doesn't exist yet, so build it first.
if not Path("data/interim/crisis_cleaned.csv").exists():
    with open("config/preprocessing.yaml") as f:
        PreprocessingPipeline(yaml.safe_load(f)).run()

from src.api.app import app  # noqa: E402


@pytest.fixture
def client():
    return app.test_client()


def kenya_2015(**overrides):
    payload = {
        "cc3": "KEN", "year": 2015, "exch_usd": 98.2,
        "domestic_debt_in_default": 0, "sovereign_external_debt_default": 0,
        "gdp_weighted_default": 0, "inflation_annual_cpi": 6.6,
        "independence": 1, "currency_crises": 0, "inflation_crises": 0,
    }
    payload.update(overrides)
    return payload


# ---------- happy path ----------

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
    assert "version" in response.get_json()["model"]


def test_countries_lists_next_year(client):
    countries = client.get("/countries").get_json()
    assert countries["KEN"]["next_year"] == countries["KEN"]["last_year"] + 1


def test_predict_returns_valid_prediction(client):
    response = client.post("/predict", json=kenya_2015())
    assert response.status_code == 200
    body = response.get_json()
    assert 0 <= body["crisis_probability"] <= 1
    assert body["predicted_crisis"] in (0, 1)
    assert len(body["top_features"]) == 5


def test_stress_raises_crisis_probability(client):
    calm = client.post("/predict", json=kenya_2015()).get_json()
    stressed = client.post("/predict", json=kenya_2015(
        exch_usd=180.0, domestic_debt_in_default=1, sovereign_external_debt_default=1,
        gdp_weighted_default=0.1, inflation_annual_cpi=45.0, currency_crises=1, inflation_crises=1,
    )).get_json()
    assert stressed["crisis_probability"] > calm["crisis_probability"]


def test_predictions_are_deterministic(client):
    first = client.post("/predict", json=kenya_2015()).get_json()
    second = client.post("/predict", json=kenya_2015()).get_json()
    assert first == second


# ---------- validation ----------

def test_missing_fields_rejected(client):
    response = client.post("/predict", json={"cc3": "KEN", "year": 2015})
    assert response.status_code == 400
    assert any("missing field" in e for e in response.get_json()["errors"])


def test_unknown_country_rejected(client):
    response = client.post("/predict", json=kenya_2015(cc3="XYZ"))
    assert response.status_code == 400
    assert "unknown country" in response.get_json()["errors"][0]


def test_year_too_far_ahead_rejected(client):
    response = client.post("/predict", json=kenya_2015(year=2020))
    assert response.status_code == 400


def test_binary_flag_must_be_0_or_1(client):
    response = client.post("/predict", json=kenya_2015(currency_crises=2))
    assert response.status_code == 400


def test_boolean_is_not_a_number(client):
    response = client.post("/predict", json=kenya_2015(exch_usd=True))
    assert response.status_code == 400


def test_non_json_body_rejected(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400