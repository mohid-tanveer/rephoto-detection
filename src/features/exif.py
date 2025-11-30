from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "image_width",
    "image_height",
    "focal_length_35mm_eq",
    "f_number",
    "exposure_time",
    "shutter_speed_value",
    "iso",
]

CATEGORICAL_COLUMNS = [
    "device_make",
    "device_model",
    "screen_type",
    "screen_source",
    "metering_mode",
]


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        features[col] = _safe_numeric(df[col])

    features["aspect_ratio"] = features["image_width"] / (
        features["image_height"] + 1e-6
    )
    features["log_iso"] = np.log1p(features["iso"])
    features["exposure_value"] = np.log2(
        (features["f_number"] ** 2) / (features["exposure_time"] + 1e-6) + 1e-6
    )
    features["focal_per_aperture"] = features["focal_length_35mm_eq"] / (
        features["f_number"] + 1e-6
    )

    available_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if available_cols:
        cat_df = df[available_cols].fillna("unknown")
        cat_features = pd.get_dummies(cat_df)
        features = pd.concat([features, cat_features], axis=1)
    return features


def build_exif_feature_table(index_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    construct a clean dataframe of engineered exif features and return the column order.
    """
    feature_df = engineer_features(index_df)
    columns = sorted(feature_df.columns.tolist())
    feature_df = feature_df.reindex(columns=columns, fill_value=0.0)
    return feature_df, columns


def transform_single_exif(
    raw: Dict[str, object], feature_columns: List[str]
) -> np.ndarray:
    """
    convert a raw metadata dict into a feature vector aligned with training columns.
    """
    df = pd.DataFrame([raw])
    features = engineer_features(df)
    features = features.reindex(columns=feature_columns, fill_value=0.0)
    return features.values.astype(np.float32)[0]
