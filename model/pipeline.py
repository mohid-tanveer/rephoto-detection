from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import ExifTags, Image
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from model.data.indexer import IndexConfig, load_or_build_index
from model.features.exif import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_exif_feature_table,
    transform_single_exif,
)
from model.features.moire import MoireTileMetrics, extract_moire_features
from model.features.subpixel import SubpixelTileMetrics, extract_subpixel_features
from model.models.exif_prior import ExifPriorModel
from model.models.hybrid_classifier import HybridConfig, HybridFusionModel
from model.models.logistic_head import LogisticConfig as SubpixelLogisticConfig
from model.models.logistic_head import LogisticHead
from model.models.spectral_net import SpectralNetConfig, SpectralNetModel
from model.utils.image import load_image

logger = logging.getLogger(__name__)

from tqdm.auto import tqdm

_EXIF_TAG_MAP = {tag_id: tag_name for tag_id, tag_name in ExifTags.TAGS.items()}


def _rational_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            den = float(value.denominator)
            if den == 0:
                return None
            return float(value.numerator) / den
        if isinstance(value, (tuple, list)) and len(value) == 2:
            num, den = value
            den = float(den)
            if den == 0:
                return None
            return float(num) / den
        return float(value)
    except Exception:
        return None


def _extract_raw_exif(image_path: Path | str) -> Dict[str, Any]:
    """
    pull core exif fields from an arbitrary jpeg so we can engineer features on the fly.
    """
    base: Dict[str, Any] = {}
    for col in NUMERIC_COLUMNS:
        base[col] = 0.0
    for col in CATEGORICAL_COLUMNS:
        base[col] = "unknown"
    base["screen_type"] = "none"
    base["screen_source"] = "none"

    with Image.open(image_path) as img:
        exif_raw = img._getexif() or {}
        exif_dict = {
            _EXIF_TAG_MAP.get(tag_id, str(tag_id)): value
            for tag_id, value in exif_raw.items()
        }
        width, height = img.size
        base["image_width"] = width
        base["image_height"] = height
        base["focal_length_35mm_eq"] = exif_dict.get(
            "FocalLengthIn35mmFilm", base["focal_length_35mm_eq"]
        )
        base["f_number"] = (
            _rational_to_float(exif_dict.get("FNumber")) or base["f_number"]
        )
        base["exposure_time"] = (
            _rational_to_float(exif_dict.get("ExposureTime")) or base["exposure_time"]
        )
        base["shutter_speed_value"] = (
            _rational_to_float(exif_dict.get("ShutterSpeedValue"))
            or base["shutter_speed_value"]
        )

        iso_val = exif_dict.get(
            "PhotographicSensitivity", exif_dict.get("ISOSpeedRatings")
        )
        if isinstance(iso_val, (list, tuple)):
            iso_val = iso_val[0] if iso_val else None
        base["iso"] = iso_val if iso_val is not None else base["iso"]

        base["metering_mode"] = exif_dict.get("MeteringMode", base["metering_mode"])
        base["device_model"] = exif_dict.get("Model", base["device_model"])
        base["device_make"] = exif_dict.get("Make", base["device_make"])
        base["screen_type"] = base.get("screen_type", "none")
        base["screen_source"] = base.get("screen_source", "none")

    return base


@dataclass
class PipelineConfig:
    data_dir: Path
    exif_csv: Path
    artifacts_dir: Path
    tile_size: int = 256
    tile_stride: int = 128
    max_tiles_per_image: int = 16
    force_feature_recompute: bool = False
    force_index_recompute: bool = False
    device: str | None = None
    exif_holdout_fraction: float = 0.2


@dataclass
class FeatureStore:
    table: pd.DataFrame
    feature_groups: Dict[str, List[str]]

    def subset(self, mask: pd.Series) -> "FeatureStore":
        return FeatureStore(
            table=self.table.loc[mask].reset_index(drop=True),
            feature_groups=self.feature_groups,
        )


@dataclass
class ModelBundle:
    moire_model: SpectralNetModel
    subpixel_model: object
    exif_model: ExifPriorModel
    fusion_model: HybridFusionModel
    feature_groups: Dict[str, List[str]]


def _external_exif_cache_dir(config: PipelineConfig) -> Path:
    cache_dir = config.artifacts_dir / "external_exif"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _external_exif_cache_path(config: PipelineConfig, image_path: Path | str) -> Path:
    cache_dir = _external_exif_cache_dir(config)
    key = hashlib.sha1(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _load_cached_exif(
    config: PipelineConfig,
    image_path: Path | str,
    feature_columns: List[str],
) -> Optional[np.ndarray]:
    cache_file = _external_exif_cache_path(config, image_path)
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text())
        if payload.get("feature_columns") != feature_columns:
            return None
        values = payload.get("feature_values")
        if values is None:
            return None
        return np.asarray(values, dtype=np.float32)
    except Exception:  # pragma: no cover
        logger.warning("failed to read external exif cache for %s", image_path)
        return None


def _save_cached_exif(
    config: PipelineConfig,
    image_path: Path | str,
    raw_exif: Dict[str, Any],
    feature_vector: np.ndarray,
    feature_columns: List[str],
) -> None:
    cache_file = _external_exif_cache_path(config, image_path)
    payload = {
        "image_path": str(Path(image_path).resolve()),
        "feature_columns": feature_columns,
        "feature_values": feature_vector.astype(float).tolist(),
        "raw_exif": raw_exif,
    }
    cache_file.write_text(json.dumps(payload, indent=2))


def _feature_cache_path(config: PipelineConfig, name: str) -> Path:
    feature_dir = config.artifacts_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    return feature_dir / name


def _model_path(config: PipelineConfig, name: str) -> Path:
    model_dir = config.artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / name


def build_feature_store(config: PipelineConfig) -> FeatureStore:
    index_cfg = IndexConfig(
        data_dir=config.data_dir,
        exif_csv=config.exif_csv,
        artifacts_dir=config.artifacts_dir,
    )
    index_df = load_or_build_index(
        index_cfg, force_recompute=config.force_index_recompute
    )

    moire_cache = _feature_cache_path(
        config,
        f"moire_t{config.tile_size}_s{config.tile_stride}.pkl",
    )
    subpixel_cache = _feature_cache_path(
        config,
        f"subpixel_t{config.tile_size}_s{config.tile_stride}.pkl",
    )

    if moire_cache.exists() and not config.force_feature_recompute:
        moire_df = pd.read_pickle(moire_cache)
    else:
        moire_records = []
        for row in tqdm(
            index_df.itertuples(), total=len(index_df), desc="moire features"
        ):
            try:
                img = load_image(row.abs_path, mode="gray")
                summary, _ = extract_moire_features(
                    img,
                    tile_size=config.tile_size,
                    stride=config.tile_stride,
                    max_tiles=config.max_tiles_per_image,
                )
                summary["image_id"] = row.image_id
                moire_records.append(summary)
            except Exception:  # pragma: no cover
                logger.exception(
                    "failed to compute moire features for %s", row.abs_path
                )
                continue
        moire_df = pd.DataFrame(moire_records)
        moire_df.to_pickle(moire_cache)

    if subpixel_cache.exists() and not config.force_feature_recompute:
        subpixel_df = pd.read_pickle(subpixel_cache)
    else:
        subpixel_records = []
        for row in tqdm(
            index_df.itertuples(), total=len(index_df), desc="subpixel features"
        ):
            try:
                img = load_image(row.abs_path, mode="rgb")
                summary, _ = extract_subpixel_features(
                    img,
                    tile_size=config.tile_size,
                    stride=config.tile_stride,
                    max_tiles=config.max_tiles_per_image,
                )
                summary["image_id"] = row.image_id
                subpixel_records.append(summary)
            except Exception:  # pragma: no cover
                logger.exception(
                    "failed to compute subpixel features for %s", row.abs_path
                )
                continue
        subpixel_df = pd.DataFrame(subpixel_records)
        subpixel_df.to_pickle(subpixel_cache)

    exif_features, exif_columns = build_exif_feature_table(index_df)
    exif_features = exif_features.reset_index(drop=True)
    exif_features["image_id"] = index_df["image_id"].values

    feature_df = index_df[
        [
            "image_id",
            "abs_path",
            "label_binary",
            "screen_type",
            "screen_group",
            "camera_body",
        ]
    ].copy()
    feature_df = feature_df.merge(moire_df, on="image_id", how="left")
    feature_df = feature_df.merge(subpixel_df, on="image_id", how="left")
    feature_df = feature_df.merge(exif_features, on="image_id", how="left")
    feature_df = feature_df.fillna(0.0)

    feature_groups = {
        "moire": [c for c in feature_df.columns if c.startswith("moire_")],
        "subpixel": [c for c in feature_df.columns if c.startswith("subpixel_")],
        "exif": exif_columns,
    }

    return FeatureStore(table=feature_df, feature_groups=feature_groups)


def _fit_model(model, X_train, y_train, X_val=None, y_val=None):
    if X_val is not None and y_val is not None:
        try:
            model.fit(X_train, y_train, val_data=(X_val, y_val))
            return
        except TypeError:
            pass
    model.fit(X_train, y_train)


def _train_signal_model(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    folds: int = 5,
) -> Tuple[object, np.ndarray]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.zeros_like(y, dtype=np.float32)
    for train_idx, val_idx in skf.split(X, y):
        model = model_factory()
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        _fit_model(model, X_train, y_train, X_val, y_val)
        oof[val_idx] = model.predict_proba(X_val)

    final_model = model_factory()
    _fit_model(final_model, X, y)
    return final_model, oof


def train_models(
    store: FeatureStore,
    config: PipelineConfig,
    *,
    save_artifacts: bool = True,
) -> ModelBundle:
    table = store.table
    y = table["label_binary"].values.astype(np.float32)

    moire_cols = store.feature_groups["moire"]
    subpixel_cols = store.feature_groups["subpixel"]
    exif_cols = store.feature_groups["exif"]

    moire_X = table[moire_cols].values.astype(np.float32)
    subpixel_X = table[subpixel_cols].values.astype(np.float32)
    exif_X = table[exif_cols].values.astype(np.float32)

    moire_factory = lambda: SpectralNetModel(
        SpectralNetConfig(
            input_dim=moire_X.shape[1],
            device=config.device,
        )
    )
    subpixel_factory = lambda: LogisticHead(
        feature_names=subpixel_cols,
        config=SubpixelLogisticConfig(),
    )
    exif_factory = lambda: ExifPriorModel(feature_names=exif_cols)

    moire_model, moire_oof = _train_signal_model(moire_factory, moire_X, y)
    subpixel_model, subpixel_oof = _train_signal_model(subpixel_factory, subpixel_X, y)

    exif_indices = np.arange(len(y))
    exif_train_idx, exif_holdout_idx = train_test_split(
        exif_indices,
        test_size=config.exif_holdout_fraction,
        stratify=y,
        random_state=42,
    )
    exif_train_X, exif_train_y = exif_X[exif_train_idx], y[exif_train_idx]
    exif_holdout_X = exif_X[exif_holdout_idx]

    exif_model, exif_train_oof = _train_signal_model(
        exif_factory, exif_train_X, exif_train_y
    )
    exif_oof = np.zeros_like(y, dtype=np.float32)
    exif_oof[exif_train_idx] = exif_train_oof
    exif_oof[exif_holdout_idx] = exif_model.predict_proba(exif_holdout_X)

    fusion_inputs = np.stack([moire_oof, subpixel_oof, exif_oof], axis=1)
    fusion_train_X, fusion_val_X, fusion_train_y, fusion_val_y = train_test_split(
        fusion_inputs,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    fusion_model = HybridFusionModel(HybridConfig(input_dim=3, device=config.device))
    fusion_model.fit(
        fusion_train_X, fusion_train_y, val_data=(fusion_val_X, fusion_val_y)
    )

    bundle = ModelBundle(
        moire_model=moire_model,
        subpixel_model=subpixel_model,
        exif_model=exif_model,
        fusion_model=fusion_model,
        feature_groups=store.feature_groups,
    )

    if save_artifacts:
        save_model_bundle(bundle, config)
        _save_metadata(store, config)

    return bundle


def save_model_bundle(bundle: ModelBundle, config: PipelineConfig) -> None:
    bundle.moire_model.save(_model_path(config, "moire.pt"))
    bundle.subpixel_model.save(_model_path(config, "subpixel.joblib"))
    bundle.exif_model.save(_model_path(config, "exif.joblib"))
    bundle.fusion_model.save(_model_path(config, "fusion.pt"))


def load_model_bundle(
    config: PipelineConfig, feature_groups: Dict[str, List[str]]
) -> ModelBundle:
    moire_model = SpectralNetModel.load(_model_path(config, "moire.pt"))
    subpixel_model = LogisticHead.load(_model_path(config, "subpixel.joblib"))
    exif_model = ExifPriorModel.load(_model_path(config, "exif.joblib"))
    fusion_model = HybridFusionModel.load(_model_path(config, "fusion.pt"))
    return ModelBundle(
        moire_model=moire_model,
        subpixel_model=subpixel_model,
        exif_model=exif_model,
        fusion_model=fusion_model,
        feature_groups=feature_groups,
    )


def _metadata_path(config: PipelineConfig) -> Path:
    return config.artifacts_dir / "models" / "metadata.json"


def _save_metadata(store: FeatureStore, config: PipelineConfig) -> None:
    payload = {
        "feature_groups": store.feature_groups,
        "tile_size": config.tile_size,
        "tile_stride": config.tile_stride,
        "max_tiles_per_image": config.max_tiles_per_image,
    }
    _metadata_path(config).write_text(json.dumps(payload, indent=2))


def load_metadata(config: PipelineConfig) -> Dict:
    return json.loads(_metadata_path(config).read_text())


def predict_with_bundle(store: FeatureStore, bundle: ModelBundle) -> pd.DataFrame:
    table = store.table
    moire_cols = bundle.feature_groups["moire"]
    subpixel_cols = bundle.feature_groups["subpixel"]
    exif_cols = bundle.feature_groups["exif"]

    moire_probs = bundle.moire_model.predict_proba(
        table[moire_cols].values.astype(np.float32)
    )
    subpixel_probs = bundle.subpixel_model.predict_proba(
        table[subpixel_cols].values.astype(np.float32)
    )
    exif_probs = bundle.exif_model.predict_proba(
        table[exif_cols].values.astype(np.float32)
    )
    fusion_inputs = np.stack([moire_probs, subpixel_probs, exif_probs], axis=1)
    hybrid_probs = bundle.fusion_model.predict_proba(fusion_inputs)

    result = table[
        [
            "image_id",
            "label_binary",
            "screen_type",
            "screen_group",
            "camera_body",
            "abs_path",
        ]
    ].copy()
    result["moire_prob"] = moire_probs
    result["subpixel_prob"] = subpixel_probs
    result["exif_prob"] = exif_probs
    result["hybrid_prob"] = hybrid_probs
    return result


def _fpr_at_tpr(
    y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95
) -> float | None:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ValueError:
        return None
    if np.max(tpr) < target_tpr:
        return float(fpr[-1])
    idx = np.searchsorted(tpr, target_tpr)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def summarize_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    y = pred_df["label_binary"].values
    metrics = []
    for name in ["moire_prob", "subpixel_prob", "exif_prob", "hybrid_prob"]:
        scores = pred_df[name].values
        try:
            auc = roc_auc_score(y, scores)
        except ValueError:
            auc = None
        fpr = _fpr_at_tpr(y, scores, target_tpr=0.95)
        metrics.append(
            {
                "signal": name.replace("_prob", ""),
                "auc": auc,
                "fpr_at_95_tpr": fpr,
            }
        )
    return pd.DataFrame(metrics)


def evaluate_leave_one(
    store: FeatureStore,
    config: PipelineConfig,
    split_column: str,
) -> pd.DataFrame:
    results = []
    for value in sorted(store.table[split_column].dropna().unique()):
        if value in ("none", "unknown"):
            continue
        df = store.table

        if split_column == "screen_type":
            # hold out all rephotos for this screen type, and sample a subset of
            # authentic images as controls so the test set has both classes.
            rephoto_mask = df["label_binary"] == 1
            authentic_mask = df["label_binary"] == 0
            target_rephotos = rephoto_mask & (df[split_column] == value)

            if target_rephotos.sum() < 5:
                continue
            if authentic_mask.sum() == 0:
                continue

            authentic_indices = df.index[authentic_mask]
            _, auth_test_idx = train_test_split(
                authentic_indices,
                test_size=0.3,
                random_state=42,
            )
            test_mask = df.index.isin(auth_test_idx) | target_rephotos
        else:
            test_mask = df[split_column] == value
            if test_mask.sum() < 5:
                continue

        train_mask = ~test_mask
        train_store = store.subset(train_mask)
        test_store = store.subset(test_mask)

        if train_store.table["label_binary"].nunique() < 2:
            logger.info(
                "skipping split %s=%s because train set lacks both classes",
                split_column,
                value,
            )
            continue
        if test_store.table["label_binary"].nunique() < 2:
            logger.info(
                "skipping split %s=%s because test set lacks both classes",
                split_column,
                value,
            )
            continue
        bundle = train_models(train_store, config, save_artifacts=False)
        preds = predict_with_bundle(test_store, bundle)
        summary = summarize_metrics(preds)
        summary["held_out_value"] = value
        summary["split_column"] = split_column
        results.append(summary)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def run_full_pipeline(
    config: PipelineConfig,
) -> Tuple[FeatureStore, ModelBundle, pd.DataFrame]:
    """
    convenience helper used by the notebook to build features, train models, and capture metrics.
    """
    store = build_feature_store(config)
    bundle = train_models(store, config, save_artifacts=True)
    predictions = predict_with_bundle(store, bundle)
    metrics = summarize_metrics(predictions)
    return store, bundle, metrics


def build_exif_vector(store: FeatureStore, image_id: str) -> np.ndarray:
    row = store.table.loc[store.table["image_id"] == image_id]
    if row.empty:
        raise ValueError(f"image_id {image_id} not found in store")
    cols = store.feature_groups["exif"]
    return row[cols].values.astype(np.float32)


def score_image(
    image_path: Path | str,
    bundle: ModelBundle,
    config: PipelineConfig,
    *,
    exif_vector: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], List[MoireTileMetrics], List[SubpixelTileMetrics]]:
    """
    evaluate a single image path and return probabilities along with tile diagnostics.
    """
    img_path = Path(image_path)
    gray = load_image(img_path, mode="gray")
    rgb = load_image(img_path, mode="rgb")

    moire_summary, moire_tiles = extract_moire_features(
        gray,
        tile_size=config.tile_size,
        stride=config.tile_stride,
        max_tiles=config.max_tiles_per_image,
    )
    subpixel_summary, subpixel_tiles = extract_subpixel_features(
        rgb,
        tile_size=config.tile_size,
        stride=config.tile_stride,
        max_tiles=config.max_tiles_per_image,
    )

    moire_vec = np.array(
        [moire_summary.get(col, 0.0) for col in bundle.feature_groups["moire"]],
        dtype=np.float32,
    ).reshape(1, -1)
    subpixel_vec = np.array(
        [subpixel_summary.get(col, 0.0) for col in bundle.feature_groups["subpixel"]],
        dtype=np.float32,
    ).reshape(1, -1)

    feature_cols = bundle.feature_groups["exif"]
    if exif_vector is not None:
        exif_vec = exif_vector.reshape(1, -1)
    else:
        cached = _load_cached_exif(config, img_path, feature_cols)
        if cached is not None:
            exif_arr = cached
        else:
            try:
                raw_exif = _extract_raw_exif(img_path)
                exif_arr = transform_single_exif(raw_exif, feature_cols)
                _save_cached_exif(config, img_path, raw_exif, exif_arr, feature_cols)
            except Exception:
                logger.exception(
                    "failed to parse exif for %s, falling back to zeros", img_path
                )
                exif_arr = np.zeros(len(feature_cols), dtype=np.float32)
        exif_vec = exif_arr.reshape(1, -1)

    moire_prob = float(bundle.moire_model.predict_proba(moire_vec)[0])
    subpixel_prob = float(bundle.subpixel_model.predict_proba(subpixel_vec)[0])
    exif_prob = float(bundle.exif_model.predict_proba(exif_vec)[0])
    fusion_input = np.array([[moire_prob, subpixel_prob, exif_prob]], dtype=np.float32)
    hybrid_prob = float(bundle.fusion_model.predict_proba(fusion_input)[0])

    scores = {
        "moire_prob": moire_prob,
        "subpixel_prob": subpixel_prob,
        "exif_prob": exif_prob,
        "hybrid_prob": hybrid_prob,
    }
    return scores, moire_tiles, subpixel_tiles
