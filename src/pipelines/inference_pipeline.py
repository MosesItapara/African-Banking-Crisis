import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import yaml
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class InferencePipeline:
    """Inference: load model -> load data -> predict -> explain -> evaluate -> save"""

    def __init__(self, config: dict):
        self.cfg = config
        self.model = None
        self.feature_columns = []
        self.df = None
        self.X = None
        self.predictions = None

    def load_model(self):
        """Load the trained pipeline (scaler -> selector -> model) and its input columns"""
        with open(self.cfg["model_path"], "rb") as f:
            self.model = pickle.load(f)

        self.feature_columns = json.loads(Path(self.cfg["features_path"]).read_text())
        print(f"  Model: {type(self.model[-1]).__name__}, expects {len(self.feature_columns)} columns")
        return self

    def load_data(self):
        """Load data to score, keeping only rows the model was not trained on"""
        self.df = pd.read_csv(self.cfg["input_data"])

        from_year = self.cfg.get("score_from_year")
        if from_year is not None:
            self.df = self.df[self.df["year"] >= from_year].reset_index(drop=True)

        missing = [c for c in self.feature_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} training columns, e.g. {missing[:5]}")

        # Exactly the training columns, in the training order
        self.X = self.df[self.feature_columns]
        print(f"  Scoring: {len(self.df)} rows")
        return self

    def predict(self):
        """Generate predictions"""
        y_pred_proba = self.model.predict_proba(self.X)[:, 1]

        threshold = self.cfg.get("threshold", 0.5)
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        self.predictions = pd.DataFrame({
            "cc3": self.df["cc3"] if "cc3" in self.df else "unknown",
            "year": self.df["year"] if "year" in self.df else np.nan,
            "crisis_probability": y_pred_proba.round(4),
            "predicted_crisis": y_pred_binary,
            "confidence": np.maximum(y_pred_proba, 1 - y_pred_proba).round(4),
        })

        return self

    def model_feature_names(self) -> np.ndarray:
        """Names of the columns that reach the final model, after every filtering step"""
        names = np.array(self.feature_columns)
        for _, step in self.model.steps[:-1]:
            if hasattr(step, "get_support"):
                names = names[step.get_support()]
        return names

    def add_explanations(self):
        """Add top contributing features (per row for XGBoost, global otherwise)"""
        if not self.cfg.get("explain_predictions", False):
            return self

        estimator = self.model[-1]
        names = self.model_feature_names()
        top_n = self.cfg.get("n_top_features", 5)

        if isinstance(estimator, xgb.XGBClassifier):
            # Per-row contributions to the log-odds; the last column is the bias term
            X_model = self.model[:-1].transform(self.X)
            contribs = estimator.get_booster().predict(xgb.DMatrix(X_model), pred_contribs=True)[:, :-1]
            self.predictions["top_features"] = [
                ", ".join(f"{names[i]} ({row[i]:+.2f})" for i in np.argsort(-row)[:top_n])
                for row in contribs
            ]

        else:
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                importances = np.abs(estimator.coef_[0])
            else:
                print("  No explanations: model has neither feature_importances_ nor coef_")
                return self

            top = ", ".join(names[np.argsort(importances)[::-1][:top_n]])
            self.predictions["model_top_features"] = top

        return self

    def evaluate(self):
        """Print metrics when the true labels are available"""
        target = self.cfg.get("target")
        if not target or target not in self.df.columns:
            return self

        y_true = self.df[target]
        y_pred = self.predictions["predicted_crisis"]
        self.predictions["actual_crisis"] = y_true

        print(f"\nMetrics vs actual ({y_true.sum()} crises):")
        print(f"  • Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"  • Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  • Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  • F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
        if y_true.nunique() > 1:
            print(f"  • ROC AUC:   {roc_auc_score(y_true, self.predictions['crisis_probability']):.4f}")
        print(f"  • Confusion matrix [[TN, FP], [FN, TP]]: {confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}")
        return self

    def save_predictions(self):
        """Save predictions to CSV"""
        output_path = Path(self.cfg["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.predictions.to_csv(output_path, index=False)

        # Print summary
        crisis_pct = (self.predictions["predicted_crisis"].sum() / len(self.predictions)) * 100
        print(f"\nPredictions Summary:")
        print(f"  • Total: {len(self.predictions)}")
        print(f"  • Predicted crises: {self.predictions['predicted_crisis'].sum()} ({crisis_pct:.1f}%)")
        print(f"\n Saved: {output_path}")

        return self

    def run(self) -> pd.DataFrame:
        """Execute full inference pipeline"""
        (self.load_model()
         .load_data()
         .predict()
         .add_explanations()
         .evaluate()
         .save_predictions())

        return self.predictions


if __name__ == "__main__":
    with open("config/inference.yaml") as f:
        config = yaml.safe_load(f)
    results = InferencePipeline(config).run()
    print(f"Inference complete. {len(results)} rows scored.")
