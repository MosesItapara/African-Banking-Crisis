"""
Run the Pipeline end to end
"""
import argparse
import yaml

from src.pipelines.preprocessing_pipeline import PreprocessingPipeline
from src.pipelines.feature_eng_pipeline import FeatureEngineeringPipeline
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline

STAGES = {
    "preprocess": (PreprocessingPipeline, "config/preprocessing.yaml"),
    "features": (FeatureEngineeringPipeline, "config/feature_engineering.yaml"),
    "train": (TrainingPipeline, "config/training.yaml"),
    "predict": (InferencePipeline, "config/inference.yaml")
}

def run_stage(name):
    pipeline_class, config_path = STAGES[name]
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"\n====== {name} =======")
    pipeline_class(config).run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="African banking crisis pipeline")
    parser.add_argument("--stage", choices=list(STAGES), help="run a single stage (default:all)")
    args = parser.parse_args()

    for name in ([args.stage] if args.stage else STAGES):
        run_stage(name)