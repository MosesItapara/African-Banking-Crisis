import pandas as pd
import numpy as np
import yaml
from pathlib import Path

class PreprocessingPipeline:
    """Data cleaning: read -> rename -> drop -> encode target -> fix flags -> NaNs -> duplicates -> outliers -> sort -> save"""

    def __init__(self, config: dict):
        self.cfg = config
        self.df = None
        self.outlier_indices = []

    def read_data(self):
        """Load raw data"""
        self.df = pd.read_csv(self.cfg["file_path"])
        print(f"  Loaded: {self.df.shape}")
        return self

    def rename_columns(self):
        """Rename columns to snake_case"""
        rename_map = self.cfg.get("rename_map", {})
        self.df = self.df.rename(columns=rename_map)
        return self

    def drop_columns(self):
        """Drop identifier columns that are not features"""
        cols = [c for c in self.cfg.get("drop_columns", []) if c in self.df.columns]
        self.df = self.df.drop(columns=cols)
        return self

    def encode_target(self):
        """Map the string target to 0/1 and drop the original column"""
        target = self.cfg["target"]
        encoded = self.df[target["source"]].map(target["mapping"])

        unmapped = encoded.isnull().sum()
        if unmapped > 0:
            raise ValueError(f"{unmapped} rows in '{target['source']}' have labels not in the mapping")

        self.df[target["name"]] = encoded.astype(int)
        self.df = self.df.drop(columns=[target["source"]])
        return self

    def fix_binary_flags(self):
        """Clip 0/1 flag columns to [0, 1]"""
        for col in self.cfg.get("binary_columns", []):
            invalid = (self.df[col] > 1).sum()
            if invalid > 0:
                print(f"  {col}: clipped {invalid} values > 1")
            self.df[col] = self.df[col].clip(0, 1)
        return self

    def handle_missing_values(self):
        """Fill NaNs with the country's mean/median, then the global one"""
        strategy = self.cfg.get("nan_strategy", "median")
        group_col = self.cfg.get("group_by")

        for col in self.cfg.get("numeric_columns", []):
            if self.df[col].isnull().sum() > 0:
                if group_col:
                    self.df[col] = self.df[col].fillna(
                        self.df.groupby(group_col)[col].transform(strategy)
                    )
                self.df[col] = self.df[col].fillna(self.df[col].agg(strategy))

        return self

    def remove_duplicates(self):
        """Drop duplicate rows on the configured key columns"""
        if self.cfg.get("drop_duplicates", False):
            before = len(self.df)
            self.df = self.df.drop_duplicates(subset=self.cfg.get("subset"))
            print(f"  Duplicates removed: {before - len(self.df)}")
        return self

    def detect_outliers_iqr(self):
        """Report outliers using the IQR method (reporting only, nothing is changed)"""
        q1, q3 = self.cfg.get("outlier_quantiles", [0.25, 0.75])
        multiplier = self.cfg.get("iqr_multiplier", 1.5)
        numeric_cols = self.cfg.get("numeric_columns", [])

        outliers = pd.DataFrame(index=self.df.index)
        for col in numeric_cols:
            Q1 = self.df[col].quantile(q1)
            Q3 = self.df[col].quantile(q3)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outliers[col] = mask

        self.outlier_indices = outliers.any(axis=1)
        outlier_count = self.outlier_indices.sum()
        print(f"  Outliers detected: {outlier_count} rows ({outlier_count/len(self.df)*100:.2f}%)")

        return self

    def treat_outliers(self):
        """Cap outliers per column using the rules in outliers_rules"""
        q1, q3 = self.cfg.get("outlier_quantiles", [0.25, 0.75])
        group_col = self.cfg.get("group_by")

        for col, rule in self.cfg.get("outliers_rules", {}).items():
            before = self.df[col].copy()

            if rule["method"] == "clip":
                upper = self.df[col].quantile(rule["percentile"] / 100)
                self.df[col] = self.df[col].clip(upper=upper)

            elif rule["method"] == "iqr":
                multiplier = rule.get("multiplier", self.cfg.get("iqr_multiplier", 1.5))
                grouped = self.df.groupby(group_col)[col] if group_col else self.df[col]
                Q1 = grouped.transform(lambda x: x.quantile(q1))
                Q3 = grouped.transform(lambda x: x.quantile(q3))
                IQR = Q3 - Q1
                self.df[col] = self.df[col].clip(lower=Q1 - multiplier * IQR, upper=Q3 + multiplier * IQR)

            else:
                raise ValueError(f"Unknown outlier method '{rule['method']}' for column '{col}'")

            print(f"  {col}: capped {(before != self.df[col]).sum()} values ({rule['method']})")

        return self

    def sort_rows(self):
        """Sort rows so time-based features downstream are computed in order"""
        sort_by = self.cfg.get("sort_by")
        if sort_by:
            self.df = self.df.sort_values(sort_by).reset_index(drop=True)
        return self

    def save(self):
        """Save the cleaned data to disk"""
        output_path = Path(self.cfg["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"\n Saved: {output_path}")
        return self

    def run(self) -> pd.DataFrame:
        """Execute full pipeline"""
        (self.read_data()
         .rename_columns()
         .drop_columns()
         .encode_target()
         .fix_binary_flags()
         .handle_missing_values()
         .remove_duplicates()
         .detect_outliers_iqr()
         .treat_outliers()
         .sort_rows()
         .save())

        return self.df

if __name__ == "__main__":
    with open("config/preprocessing.yaml") as f:
        config = yaml.safe_load(f)
    cleaned_df = PreprocessingPipeline(config).run()
    print(f"Preprocessing complete. Shape: {cleaned_df.shape}")
