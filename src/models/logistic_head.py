from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LogisticConfig:
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    random_state: int = 42


class LogisticHead:
    def __init__(self, feature_names: List[str], config: LogisticConfig | None = None):
        self.feature_names = feature_names
        self.config = config or LogisticConfig()
        self.model: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.config.C,
                        max_iter=self.config.max_iter,
                        class_weight=self.config.class_weight,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
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
    def load(cls, path: Path | str) -> "LogisticHead":
        payload = joblib.load(Path(path))
        config = LogisticConfig(**payload["config"])
        model = cls(feature_names=payload["feature_names"], config=config)
        model.model = payload["model"]
        return model
