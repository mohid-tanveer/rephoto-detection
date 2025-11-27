from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ExifPriorConfig:
    n_estimators: int = 400
    max_depth: int | None = None
    min_samples_leaf: int = 2
    random_state: int = 42


class ExifPriorModel:
    def __init__(self, feature_names: List[str], config: ExifPriorConfig | None = None):
        self.feature_names = feature_names
        self.config = config or ExifPriorConfig()
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path | str) -> None:
        payload = {
            "model": self.model,
            "config": asdict(self.config),
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> "ExifPriorModel":
        payload = joblib.load(Path(path))
        config = ExifPriorConfig(**payload["config"])
        model = cls(feature_names=payload["feature_names"], config=config)
        model.model = payload["model"]
        return model
