import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import yaml
import json

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}

class TrainingPipeline:
    """Training: load -> split -> build pipelines -> cross-validate -> fit -> evaluate -> save"""
    def __init__(self, config: dict):
        self.cfg = config
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.feature_columns = []
        self.pipelines = {}
        self.cv_scores = {}
        self.results = {}
        self.best_model_name = None

    def load_data(self):
        """Load engineered features, sorted by year so time-series CV folds move forward in time"""
        self.df = pd.read_csv(self.cfg["file_path"])
        self.df = self.df.sort_values("year", kind="stable").reset_index(drop=True)
        print(f"  Loaded: {self.df.shape}")
        return self

    def split_data(self):
        """Split by time or stratified"""
        target = self.cfg["target"]
        X = self.df.drop(columns=[target] + self.cfg.get("drop_columns", []))
        y = self.df[target]
        self.feature_columns = X.columns.tolist()

        if self.cfg["split_strategy"] == "time_based":
            split_year = self.cfg.get("split_year", 1995)
            train_mask = self.df["year"] < split_year

            self.X_train = X[train_mask]
            self.X_test = X[~train_mask]
            self.y_train = y[train_mask]
            self.y_test = y[~train_mask]

        elif self.cfg["split_strategy"] == "random":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.cfg["test_size"],
                stratify=y,
                random_state=self.cfg.get("random_state", 42)
            )

        else:
            raise ValueError(f"Unknown split_strategy '{self.cfg['split_strategy']}'")

        print(f"  Train: {self.X_train.shape} ({self.y_train.sum()} crises), "
              f"Test: {self.X_test.shape} ({self.y_test.sum()} crises)")
        return self

    def build_model(self, model_name, hyperparams):
        """Create an unfitted classifier from its config entry"""
        class_weight = self.cfg.get("class_weight")
        random_state = self.cfg.get("random_state", 42)

        if model_name == "logistic_regression":
            return LogisticRegression(**hyperparams, class_weight=class_weight)
        elif model_name == "random_forest":
            return RandomForestClassifier(**hyperparams, class_weight=class_weight, random_state=random_state)
        elif model_name == "xgboost":
            # XGBoost has no class_weight: weight positives by the class ratio instead
            scale_pos_weight = 1.0
            if class_weight == "balanced":
                scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            return xgb.XGBClassifier(**hyperparams, scale_pos_weight=scale_pos_weight,
                                     random_state=random_state, eval_metric="logloss")
        else:
            raise ValueError(f"Unknown model '{model_name}'")

    def build_pipelines(self):
        """Wrap scaler -> selector -> model per enabled model, so all three are saved together"""
        scaler_type = self.cfg.get("scaling", "standard")
        selection = self.cfg.get("feature_selection", {})

        for model_name, config in self.cfg.get("models", {}).items():
            if not config.get("enabled", False):
                continue

            # Drop columns that are constant in training (e.g. interactions that never fire before 1995)
            steps = [("drop_constant", VarianceThreshold(0.0))]
            if scaler_type != "none":
                steps.append(("scaler", SCALERS[scaler_type]()))
            if selection.get("method", "none") == "selectkbest":
                k = min(selection.get("n_features", 20), len(self.feature_columns))
                steps.append(("selector", SelectKBest(f_classif, k=k)))
            steps.append(("model", self.build_model(model_name, config.get("hyperparams", {}))))

            self.pipelines[model_name] = Pipeline(steps)
        return self

    def cross_validate(self):
        """Score each pipeline with CV on the training set only - the test set stays unseen"""
        n_folds = self.cfg.get("cv_folds", 5)
        metric = self.cfg.get("selection_metric", "f1")

        if self.cfg.get("cv_strategy", "time_series") == "time_series":
            folds = TimeSeriesSplit(n_splits=n_folds).split(self.X_train)
        else:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                    random_state=self.cfg.get("random_state", 42)).split(self.X_train, self.y_train)
        folds = list(folds)

        print(f"\n  {n_folds}-fold CV ({metric}):")
        for model_name, pipeline in self.pipelines.items():
            scores = []
            for train_idx, val_idx in folds:
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                # Early time folds can hold a single class - nothing to learn or score there
                if y_tr.nunique() < 2 or y_val.nunique() < 2:
                    continue
                fold_pipe = clone(pipeline).fit(self.X_train.iloc[train_idx], y_tr)
                scores.append(self.compute_metrics(y_val, fold_pipe.predict(self.X_train.iloc[val_idx]),
                                                   fold_pipe.predict_proba(self.X_train.iloc[val_idx])[:, 1])[metric])

            self.cv_scores[model_name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
                                          "folds_used": len(scores)}
            print(f"    {model_name}: {np.mean(scores):.4f} +/- {np.std(scores):.4f} ({len(scores)} folds)")

        self.best_model_name = max(self.cv_scores, key=lambda x: self.cv_scores[x]["mean"])
        print(f"  Best by CV: {self.best_model_name}")
        return self

    def train_models(self):
        """Fit every pipeline on the full training set"""
        for model_name, pipeline in self.pipelines.items():
            pipeline.fit(self.X_train, self.y_train)
            print(f"  Trained: {model_name}")
        return self

    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba) -> dict:
        """Binary classification metrics for the crisis class"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            # ROC AUC is undefined when only one class is present
            "roc_auc": roc_auc_score(y_true, y_proba) if pd.Series(y_true).nunique() > 1 else None,
        }

    def evaluate(self):
        """Evaluate all models on the held-out test set"""
        for model_name, pipeline in self.pipelines.items():
            y_pred = pipeline.predict(self.X_test)
            y_proba = pipeline.predict_proba(self.X_test)[:, 1]

            metrics = self.compute_metrics(self.y_test, y_pred, y_proba)
            metrics["confusion_matrix"] = confusion_matrix(self.y_test, y_pred, labels=[0, 1]).tolist()
            self.results[model_name] = metrics

            print(f"\n  {model_name}{' (best)' if model_name == self.best_model_name else ''}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}" if isinstance(value, float) else f"    {metric}: {value}")

        return self

    def save_best_model(self):
        """Save the best pipeline, its input columns, and all results"""
        best_pipeline = self.pipelines[self.best_model_name]

        model_path = Path(self.cfg["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(best_pipeline, f)

        features_path = Path(self.cfg["features_path"])
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features_path.write_text(json.dumps(self.feature_columns, indent=2))

        results_path = Path(self.cfg["results_path"])
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps({
            "best_model": self.best_model_name,
            "cv_scores": self.cv_scores,
            "test_metrics": self.results,
        }, indent=2))

        print(f"\n Saved: {model_path}, {features_path}, {results_path}")
        return self

    def run(self) -> dict:
        """Execute full training pipeline"""
        (self.load_data()
         .split_data()
         .build_pipelines()
         .cross_validate()
         .train_models()
         .evaluate()
         .save_best_model())

        return self.results

### USAGE
if __name__ == "__main__":
    with open("config/training.yaml") as f:
        config = yaml.safe_load(f)
    trainer = TrainingPipeline(config)
    metrics = trainer.run()
    print(f"\nTraining complete. Best model (by CV): {trainer.best_model_name}")
