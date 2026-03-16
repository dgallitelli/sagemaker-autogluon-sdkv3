"""
AutoGluon Multimodal inference script for SageMaker endpoints.

Accepts CSV input with columns matching the trained feature set
(numerical + categorical + text features, no header for batch).

Based on: churn_prediction_multimodality_of_text_and_tabular/
          containers/autogluon_multimodal_fusion/inference.py
"""
import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from sagemaker_inference import encoder


def model_fn(model_dir: str) -> MultiModalPredictor:
    """Load MultiModalPredictor from model directory."""
    try:
        model = MultiModalPredictor.load(model_dir)

        # Extract column names from data processors
        col_names = []
        if hasattr(model, "_data_processors"):
            if "numerical" in model._data_processors:
                col_names += model._data_processors["numerical"][0].numerical_column_names
            if "categorical" in model._data_processors:
                col_names += model._data_processors["categorical"][0].categorical_column_names
            if "text" in model._data_processors:
                col_names += model._data_processors["text"][0].text_column_names

        globals()["column_names"] = col_names
        logging.warning(f"Column names: {col_names}")
        return model
    except Exception:
        logging.exception("Failed to load model")
        raise


def transform_fn(task: MultiModalPredictor, input_data: Any, content_type: str, accept: str) -> np.ndarray:
    """Predict and return serialized probabilities."""
    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    if content_type == "text/csv":
        data = pd.read_csv(io.StringIO(input_data), sep=",", header=None)
        data.columns = column_names

        try:
            model_output = task.predict_proba(data).values
            output = {"probabilities": model_output.tolist()}
            return encoder.encode(output, accept)
        except Exception:
            logging.exception("Failed to predict")
            raise

    elif content_type == "application/json":
        data = pd.read_json(io.StringIO(input_data))
        model_output = task.predict_proba(data).values
        output = {"probabilities": model_output.tolist()}
        return encoder.encode(output, accept)

    elif content_type == "application/jsonl":
        data = pd.read_json(io.StringIO(input_data), orient="records", lines=True)
        model_output = task.predict_proba(data).values
        output = {"probabilities": model_output.tolist()}
        return encoder.encode(output, accept)

    raise ValueError(f"{content_type} content type not supported")
