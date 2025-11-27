from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class IndexConfig:
    data_dir: Path
    exif_csv: Path
    artifacts_dir: Path
    cache_name: str = "index.pkl"

    @property
    def cache_path(self) -> Path:
        return self.artifacts_dir / self.cache_name


def _prepare_dataframe(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    df = df.copy()

    def resolve_path(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((base_dir / path).resolve())

    df["abs_path"] = df["filepath"].apply(resolve_path)
    df["image_id"] = df["filename"].str.replace(".", "_", regex=False)
    df["label_binary"] = (df["label"].str.lower() != "authentic").astype(int)
    df["screen_group"] = (
        df["screen_type"].fillna("none") + "__" + df["screen_source"].fillna("none")
    )
    df["camera_body"] = (
        df["device_make"].fillna("unknown")
        + "__"
        + df["device_model"].fillna("unknown")
    )
    return df


def load_or_build_index(
    config: IndexConfig, *, force_recompute: bool = False
) -> pd.DataFrame:
    """
    load cached index dataframe or rebuild it from the csv.
    """
    cache_path = config.cache_path
    if cache_path.exists() and not force_recompute:
        return pd.read_pickle(cache_path)

    df = pd.read_csv(config.exif_csv)
    repo_root = config.data_dir.parent
    df = _prepare_dataframe(df, repo_root)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    return df
