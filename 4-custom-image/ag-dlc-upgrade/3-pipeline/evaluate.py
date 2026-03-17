"""
Evaluation script for SageMaker Pipelines.

Loads AutoGluon model and test data, computes evaluation metrics,
and writes them to evaluation.json for conditional pipeline steps.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor


def get_input_path(path: str) -> str:
    """Return the first file found in the given directory."""
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No files found in {path}")
    if len(files) > 1:
        print(f"WARN: multiple files found in {path}, using first: {files[0]}")
    filename = os.path.join(path, files[0])
    print(f"Using {filename}")
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-dir", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}")
    predictor = TabularPredictor.load(args.model_dir)

    print(f"Loading test data from {args.test_dir}")
    test_file = get_input_path(args.test_dir)
    test_data = pd.read_csv(test_file) if test_file.endswith(".csv") else pd.read_parquet(test_file)

    print("Running evaluation")
    perf = predictor.evaluate(test_data)

    # Handle both dict and scalar returns
    if isinstance(perf, dict):
        metrics = {"metrics": {k: abs(v) for k, v in perf.items()}}
    else:
        eval_metric = predictor.eval_metric.name if hasattr(predictor.eval_metric, "name") else "metric"
        metrics = {"metrics": {eval_metric: abs(perf)}}

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "evaluation.json")

    print(f"Writing metrics to {output_path}")
    print(json.dumps(metrics, indent=2))

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
