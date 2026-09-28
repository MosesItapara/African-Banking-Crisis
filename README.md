# African Banking Crisis Prediction

![CI](https://github.com/MosesItapara/African-Banking-Crisis/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps project that estimates the risk of a **banking crisis** in 13 African countries from macroeconomic indicators: from a raw CSV to a tested, versioned model served by a public API and deployed automatically on every push.

**Live API:** https://moise19-african-banking-crisis-api.hf.space  ·  [Hugging Face Space](https://huggingface.co/spaces/Moise19/african-banking-crisis-api)

> The free Space sleeps when idle. The first request after a pause can take about a minute.

---

## Highlights

- **Config-driven pipelines**: preprocessing, feature engineering, training and inference, each driven by its own YAML config
- **Honest evaluation**: time-based train/validation/test split, a target-leakage fix, and a documented trade-off between a *detection* model and a *forecast* model
- **Experiment tracking and model registry**: every run is logged to MLflow, and the deployed model is the version with the `champion` alias
- **Serving**: a Flask API with input validation and per-prediction explanations, running on gunicorn in Docker
- **CI/CD**: GitHub Actions runs the pipeline, a model quality gate, 27 tests, image builds and a container smoke test, then deploys to Hugging Face Spaces

---

## Architecture

```
 data/african_econ_crises.csv
            │
   ┌────────▼────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌────────────────┐
   │  preprocessing  │──▶│ feature          │──▶│    training     │──▶│   inference    │
   │ clean, encode   │   │ engineering      │   │ CV-selected     │   │ batch scoring  │
   │ target, cap     │   │ lags, rolling,   │   │ sklearn Pipeline│   │ + explanations │
   │ outliers        │   │ diffs, one-hot   │   └────────┬────────┘   └────────────────┘
   └─────────────────┘   └──────────────────┘            │ logs runs, registers model
                                                ┌────────▼────────┐
                                                │  MLflow registry│  alias: champion
                                                └────────┬────────┘
                                                         │ scripts/export_champion.py
                                                ┌────────▼────────┐
  GitHub push ─▶ GitHub Actions ─▶ tests ─▶     │  Flask API      │ ─▶ Hugging Face Space
                 quality gate ─▶ Docker ─▶      │  (Docker,       │    (public URL)
                 smoke test ─▶ deploy           │   gunicorn)     │
                                                └─────────────────┘
```

---

## Data

[Africa Economic, Banking and Systemic Crisis Data](https://www.kaggle.com/datasets/chirin/africa-economic-banking-and-systemic-crisis-data): **1,059 country-years** for **13 countries** (Algeria, Angola, Central African Republic, Côte d'Ivoire, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia, Zimbabwe), **1860–2014**.

The target is `banking_crisis` (crisis / no crisis). Crises make up about **9%** of rows, so the classes are imbalanced.

---

## Results

Time-based split: models are compared by training on years before 1985 and **validating** on 1985–1994. The chosen model is then refit on **all years before 1995** and **tested** once on 1995–2014 (258 rows, **40 crises**).

| Setup | Model (chosen on validation) | Test F1 | Recall | Precision | ROC AUC |
|---|---|---|---|---|---|
| Detection: includes `systemic_crisis` | XGBoost | **0.905** | 0.95 | 0.86 | 0.97 |
| **Forecast: excludes it (deployed)** | Logistic regression | **0.667** | **0.85** | 0.55 | 0.90 |

**Why the lower-scoring model is the one deployed:** `systemic_crisis` coincides with a banking crisis in the same year almost every time. A model that uses it mainly recognises a crisis that is already under way, and it drew over half its feature importance from that one column. The forecast model works from macro indicators alone. It catches **34 of 40** crises, at the cost of more false alarms, which is usually the right trade-off for an early-warning system.

### Key decisions

- **Target leakage removed.** An interaction feature originally multiplied the target itself into a feature. It now uses *last year's* crisis instead.
- **Exchange-rate outliers kept.** Per-country IQR capping flattened Zimbabwe's and Angola's currency collapses, which are the crisis signal itself, so it was removed.
- **The validation method matters.** Shuffled CV picked the wrong model once `systemic_crisis` was removed. Validating on the most recent pre-test years (1985–1994) reflects how the model is actually used.
- **Model selection never touches the test set.** The test set is used once, for reporting.

---

## API

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status and the served model version |
| `GET` | `/countries` | Supported countries and the next year each can predict |
| `POST` | `/predict` | Crisis probability for a country's new year of indicators |

### Example

```bash
curl -X POST https://moise19-african-banking-crisis-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"cc3": "KEN", "year": 2015, "exch_usd": 98.2,
       "domestic_debt_in_default": 0, "sovereign_external_debt_default": 0,
       "gdp_weighted_default": 0, "inflation_annual_cpi": 6.6,
       "independence": 1, "currency_crises": 0, "inflation_crises": 0}'
```

```json
{
  "cc3": "KEN",
  "year": 2015,
  "crisis_probability": 0.2257,
  "predicted_crisis": 0,
  "threshold": 0.5,
  "top_features": [
    {"feature": "exch_usd_lag1", "contribution": -0.775},
    {"feature": "exch_usd", "contribution": 0.705},
    {"feature": "exch_usd_roll3_max", "contribution": -0.567}
  ],
  "model_version": 4
}
```

**How it works:** lags and 3-year rolling features need history, so the API keeps each country's cleaned history. It appends the new year, runs the **same feature-engineering code as training**, and scores only the new row. Invalid input returns `400` with a list of errors: an unknown country, a non-numeric value, a flag that isn't 0 or 1, or a year that isn't directly after the country's history.

---

## Project structure

```
├── config/                  # one YAML per stage + api.yaml
├── data/african_econ_crises.csv
├── deploy/huggingface/      # Space README (settings header)
├── models/champion/         # exported champion model served by the API
├── notebook/EDA.ipynb
├── scripts/
│   ├── export_champion.py   # MLflow registry -> models/champion/
│   └── deploy_to_hf.py      # uploads the API to the Hugging Face Space
├── src/
│   ├── pipelines/           # preprocessing, feature_eng, training, inference
│   └── api/app.py           # Flask API
├── tests/                   # 27 pytest tests
├── main.py                  # runs the pipeline end to end, or one stage
├── Dockerfile               # pipeline image
├── Dockerfile.api           # API image (gunicorn)
├── requirements*.txt        # core / api / train / dev, pinned
└── .github/workflows/ci.yml
```

---

## Run it locally

Requires Python 3.9. The Docker images pin the same versions.

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements-dev.txt
```

```bash
python main.py                    # all stages
python main.py --stage train      # one stage
python -m pytest tests -v         # 27 tests
mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment history at http://127.0.0.1:5000
```

Run the API:

```bash
python scripts/export_champion.py   # after choosing a champion in MLflow
python -m src.api.app               # http://127.0.0.1:8000
```

With Docker:

```bash
docker build -t african-crisis .                          # pipeline
docker run --rm african-crisis

docker build -f Dockerfile.api -t african-crisis-api .    # API
docker run --rm -p 8000:8000 african-crisis-api
```

---

## How a new model reaches production

1. Change features or config, then train. The run is logged to MLflow and registered as a new version.
2. Compare runs in the MLflow UI, and move the **`champion`** alias to the better version.
3. Export it: `python scripts/export_champion.py`, then commit `models/champion/` and push.
4. **GitHub Actions** runs the pipeline and the **quality gate** (test F1 ≥ 0.60), runs all tests, builds both images and smoke-tests the API container.
5. Only if every step passes does it **deploy** to Hugging Face Spaces. Pull requests are tested but never deployed.

---

## Limitations and future work

- **Forecast performance is modest (F1 0.667).** Without a same-year crisis signal, the indicators carry limited information.
- **Exchange-rate features are collinear levels.** Replacing them with year-on-year **% change**, which is comparable across currencies, is the most promising next step, together with multi-year trends and regional contagion features.
- **Data quality:** some values look erroneous (for example Tunisia's exchange rate reaching 350), and the data ends in 2014.
- **Monitoring:** the next MLOps step is logging prediction inputs and adding drift reports (e.g. Evidently).
- **Python 3.9 is end-of-life.** Upgrade the venv and both images together.
