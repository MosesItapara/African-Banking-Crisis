import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class FeatureEngineeringPipeline:
    """Feature engineering: read -> lags -> rolling aggs -> diffs -> ratios -> interactions -> NaNs -> encode -> save"""
    def __init__(self, config: dict):
        self.cfg = config
        self.df = None

    def read_data(self):
        """Load cleaned data from disk based on file extension."""
        readers = {".csv": pd.read_csv}
        self.df = readers[Path(self.cfg["file_path"]).suffix](self.cfg["file_path"])
        print(f"  Loaded: {self.df.shape}")
        return self

    def compute_lags(self):
        """Add lag features grouped by country"""
        group_col = self.cfg.get("group_by", "cc3")
        for col in self.cfg.get("lag_cols", []):
            for period in self.cfg.get("lag_periods", [1]):
                lag_name = f"{col}_lag{period}"
                self.df[lag_name] = self.df.groupby(group_col)[col].shift(period)
        return self

    def compute_rolling_aggs(self):
        """Add rolling aggregations grouped by country"""
        group_col = self.cfg.get("group_by", "cc3")
        window = self.cfg.get("agg_window", 3)

        for col in self.cfg.get("agg_cols", []):
            for func in self.cfg.get("agg_funcs", ["mean", "std"]):
                col_name = f"{col}_roll{window}_{func}"
                self.df[col_name] = self.df.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).agg(func)
                )
        return self

    def compute_diffs(self):
        """Add difference features (a - b)"""
        for a, b, name in self.cfg.get("diff_pairs", []):
            self.df[name] = self.df[a] - self.df[b]
        return self

    def compute_ratios(self):
        """Add ratio features (numerator / denominator), 0 where the denominator is 0"""
        for num, den, name in self.cfg.get("ratio_pairs", []):
            ratio = self.df[num] / self.df[den].replace(0, np.nan)
            self.df[name] = ratio.fillna(0)
        return self

    def compute_interactions(self):
        """Add interaction features (a * b)"""
        for a, b, name in self.cfg.get("crisis_pairs", []):
            self.df[name] = self.df[a] * self.df[b]
        return self

    def handle_missing_values(self):
        """Fill NaNs (first years of lags / rolling std): median by country, then overall median"""
        group_col = self.cfg.get("group_by", "cc3")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                # Fill with country median
                self.df[col] = self.df[col].fillna(
                    self.df.groupby(group_col)[col].transform("median")
                )

                # Fill remaining with global median
                self.df[col] = self.df[col].fillna(self.df[col].median())
        return self

    def encode_categorical(self):
        """One-hot encode categorical columns, keeping the original as an identifier"""
        for col in self.cfg.get("categorical_columns", []):
            dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
            self.df = pd.concat([self.df, dummies], axis=1)
        return self

    def save(self):
        """Save engineered features to CSV"""
        output_path = Path(self.cfg["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"\n Saved: {output_path}")
        return self

    def run(self) -> pd.DataFrame:
        """Execute full pipeline"""
        (self.read_data()
         .compute_lags()
         .compute_rolling_aggs()
         .compute_diffs()
         .compute_ratios()
         .compute_interactions()
         .handle_missing_values()
         .encode_categorical()
         .save())
        return self.df

if __name__ == "__main__":
    with open("config/feature_engineering.yaml") as f:
        config = yaml.safe_load(f)
    features = FeatureEngineeringPipeline(config).run()
    print(f"Feature Engineering Complete. Shape: {features.shape}")
