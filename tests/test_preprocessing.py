import pandas as pd
from src.pipelines.preprocessing_pipeline import PreprocessingPipeline

def make_config(tmp_path, raw_df):
    raw_path = tmp_path / "raw.csv"
    raw_df.to_csv(raw_path, index=False)
    return {
        "file_path": str(raw_path),
        "drop_columns": ["case", "country"],
        "target": {"source": "banking_crisis", "name": "banking_crisis_encoded",
                   "mapping": {"crisis": 1, "no_crisis": 0}},
        "binary_columns": ["currency_crises"],
        "numeric_columns": ["exch_usd"],
        "group_by": "cc3",
        "outliers_rules": {},
        "drop_duplicates": True,
        "subset": ["cc3", "year"],
        "sort_by": ["cc3", "year"],
        "output_path": str(tmp_path / "interim" / "cleaned.csv"),
    }

def make_raw():
    return pd.DataFrame({
        "case": [1, 1, 2, 2],
        "cc3": ["BBB", "BBB", "AAA", "AAA"],
        "country": ["B", "B", "A", "A"],
        "year": [2001, 2000, 2000, 2001],
        "exch_usd": [1.0, None, 2.0, 3.0],
        "currency_crises": [0, 2, 1, 0],
        "banking_crisis": ["crisis", "no_crisis", "no_crisis", "crisis"],
    })

def test_target_is_encoded_and_source_dropped(tmp_path):
    df = PreprocessingPipeline(make_config(tmp_path, make_raw())).run()
    assert set(df["banking_crisis_encoded"]) <= {0, 1}
    assert "banking_crisis" not in df.columns

def test_id_columns_dropped(tmp_path):
    df = PreprocessingPipeline(make_config(tmp_path, make_raw())).run()
    assert "case" not in df.columns and "country" not in df.columns
    assert "cc3" in df.columns

def test_binary_flags_clipped(tmp_path):
    df = PreprocessingPipeline(make_config(tmp_path, make_raw())).run()
    assert df["currency_crises"].max() <= 1

def test_no_missing_values(tmp_path):
    df = PreprocessingPipeline(make_config(tmp_path, make_raw())).run()
    assert df.isnull().sum().sum() == 0

def test_row_sorted_by_country_then_year(tmp_path):
    df = PreprocessingPipeline(make_config(tmp_path, make_raw())).run()
    assert df[["cc3", "year"]].values.tolist() == [["AAA", 2000], ["AAA", 2001], ["BBB", 2000], ["BBB", 2001]]

def test_unknown_label_raises(tmp_path):
    raw = make_raw()
    raw.loc[0, "banking_crisis"] = "typo"
    try:
        PreprocessingPipeline(make_config(tmp_path, raw)).run()
        assert False, "expected ValueError"
    except ValueError:
        pass