import numpy as np
import pandas as pd
import yaml
from src.pipelines.feature_eng_pipeline import FeatureEngineeringPipeline

def make_config(tmp_path):
    cleaned = pd.DataFrame({
        "cc3": ["AAA"] * 3 + ["BBB"] * 3,
        "year": [2000, 2001, 2002] * 2,
        "inflation": [1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
        "currency_crises": [0, 1, 1, 0, 0, 1],
        "banking_crisis_encoded": [1, 0, 1, 0, 1, 1],
    })
    path = tmp_path / "cleaned.csv"
    cleaned.to_csv(path, index=False)
    return {
        "file_path": str(path),
        "group_by": "cc3",
        "lag_cols": ["inflation", "banking_crisis_encoded"],
        "lag_periods": [1],
        "agg_cols": ["inflation"],
        "agg_funcs": ["mean", "min", "max"],
        "agg_window": 3,
        "diff_pairs": [["inflation", "inflation_lag1", "inflation_change"]],
        "ratio_pairs": [],
        "crisis_pairs": [["banking_crisis_encoded_lag1", "currency_crises", "stress"]],
        "categorical_columns": ["cc3"],
        "output_path": str(tmp_path / "processed" / "features.csv"),
    }

def test_lags_do_not_cross_countries(tmp_path):
    pipe = FeatureEngineeringPipeline(make_config(tmp_path))
    pipe.read_data().compute_lags()
    first_bbb = pipe.df[pipe.df["cc3"] == "BBB"].index[0]
    assert np.isnan(pipe.df.loc[first_bbb, "inflation_lag1"])

def test_rolling_features_have_distinct_names(tmp_path):
    df = FeatureEngineeringPipeline(make_config(tmp_path)).run()
    for func in ["mean", "min", "max"]:
        assert f"inflation_roll3_{func}" in df.columns

def test_no_missing_values(tmp_path):
    df = FeatureEngineeringPipeline(make_config(tmp_path)).run()
    assert df.select_dtypes("number").isnull().sum().sum() == 0

def test_one_hot_keeps_country_code(tmp_path):
    df = FeatureEngineeringPipeline(make_config(tmp_path)).run()
    assert {"cc3_AAA", "cc3_BBB", "cc3"} <= set(df.columns)

def test_real_config_does_not_leak_target():
    """Regression test: interactions must use the lagged target, never the target itself."""
    with open("config/feature_engineering.yaml") as f:
        cfg = yaml.safe_load(f)
    for a, b, _ in cfg.get("crisis_pairs", []):
        assert "banking_crisis_encoded" not in (a, b)