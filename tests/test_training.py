import json
import pickle
import numpy as np
import pandas as pd
import pytest
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline

@pytest.fixture
def features_path(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for cc3 in ["AAA", "BBB", "CCC", "DDD"]:
        for year in range(1960, 2010):
            x1, x2 = rng.normal(), rng.normal()
            rows.append({"cc3": cc3, "year": year, "x1": x1, "x2": x2,
                         "target": int(x1 + rng.normal(scale=0.5) > 1.2)})
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

@pytest.fixture
def train_config(tmp_path, features_path):
    return {
        "file_path": str(features_path),
        "target": "target",
        "drop_columns": ["cc3", "year"],
        "split_strategy": "time_based",
        "split_year": 2000,
        "random_state": 42,
        "scaling": "standard",
        "feature_selection": {"method": "none"},
        "models": {
            "logistic_regression": {"enabled": True, "hyperparams": {"max_iter": 200}},
            "xgboost": {"enabled": True, "hyperparams": {"n_estimators": 20, "max_depth": 2}},
        },
        "cv_folds": 3,
        "cv_strategy": "stratified",
        "selection_metric": "f1",
        "class_weight": "balanced",
        "model_path": str(tmp_path / "models" / "best_model.pkl"),
        "features_path": str(tmp_path / "models" / "feature_columns.json"),
        "results_path": str(tmp_path / "results"/ "training_results.json"),
    }

def test_training_saves_artifacts(train_config):
    TrainingPipeline(train_config).run()
    for key in ["model_path", "features_path", "results_path"]:
        assert pd.io.common.file_exists(train_config[key])

def test_id_columns_are_not_features(train_config):
    TrainingPipeline(train_config).run()
    with open(train_config["features_path"]) as f:
        columns = json.load(f)
    assert "cc3" not in columns and "year" not in columns and "target" not in columns

def test_unknown_model_raises(train_config):
    train_config["models"] = {"svm": {"enabled": True}}
    with pytest.raises(ValueError, match="Unknown model"):
        TrainingPipeline(train_config).run()

def test_inference_scores_only_unseen_years(tmp_path, train_config, features_path):
    TrainingPipeline(train_config).run()
    infer_config = {
        "model_path": train_config["model_path"],
        "features_path": train_config["features_path"],
        "input_data": str(features_path),
        "score_from_year": 2000,
        "target": "target",
        "threshold": 0.5,
        "output_path": str(tmp_path / "results" / "predictions.csv"),
        "explain_predictions": True,
        "n_top_features": 2,
    }
    preds = InferencePipeline(infer_config).run()
    assert preds["year"].min() >= 2000
    assert preds["crisis_probability"].between(0, 1).all()
    assert {"cc3", "predicted_crisis", "actual_crisis"} <= set(preds.columns)

def test_saved_model_is_full_pipeline(train_config):
    TrainingPipeline(train_config).run()
    with open(train_config["model_path"], "rb") as f:
        model = pickle.load(f)
    assert "scaler" in model.named_steps