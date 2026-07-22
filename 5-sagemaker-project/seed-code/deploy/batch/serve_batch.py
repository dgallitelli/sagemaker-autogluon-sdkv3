"""Pluggable AutoGluon batch transform inference handler.

Accepts headerless CSV (column order must match training data for tabular).
Adapted from 1-tabular-classification/2-inference/serve_batch.py.
"""
from io import StringIO

import pandas as pd


def model_fn(model_dir):
    from autogluon.tabular import TabularPredictor

    model = TabularPredictor.load(model_dir)
    globals()["column_names"] = model.feature_metadata_in.get_features()
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    if input_content_type != "text/csv":
        raise ValueError(f"{input_content_type} content type not supported")

    body_str = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    data = pd.read_csv(StringIO(body_str), header=None)
    if len(data.columns) != len(column_names):
        raise ValueError(f"Input has {len(data.columns)} columns but model expects {len(column_names)}")
    data.columns = column_names

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1)
    return prediction.to_json(), output_content_type
