"""
Export the MLFlow 'champion' model to models/champion/ for the API.
"""

import json
import pickle
from pathlib import Path
import mlflow
import mlflow.sklearn
import yaml
from mlflow import MlflowClient

OUTPUT_DIR = Path("models/champion")

with open("config/training.yaml") as f:
    ml_cfg = yaml.safe_load(f)["mlflow"]

mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
name = ml_cfg["registered_model_name"]

version = MlflowClient().get_model_version_by_alias(name, "champion")
model = mlflow.sklearn.load_model(f"models:/{name}@champion")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)


# The exact input columns this version was trained on
mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path="feature_columns.json",
                                    dst_path=str(OUTPUT_DIR))


(OUTPUT_DIR / "metadata.json").write_text(json.dumps({
    "name": name,
    "version": version.version,
    "alias": "champion",
    "run_id": version.run_id,
}, indent=2))

print(f"Exported {name} v{version.version} (run {version.run_id}) to {OUTPUT_DIR}")