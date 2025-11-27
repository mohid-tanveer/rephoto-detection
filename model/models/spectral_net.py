from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SpectralNetConfig:
    input_dim: int
    hidden_dim: int = 128
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 40
    patience: int = 5
    weight_decay: float = 1e-4
    device: str | None = None


class SpectralNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SpectralNetModel:
    def __init__(self, config: SpectralNetConfig):
        self.config = config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.net = SpectralNet(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(self.device)

    def _build_loader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        loader = self._build_loader(X, y)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        best_state = None
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.net.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.net(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(loader.dataset)
            val_loss = (
                self._evaluate_loss(val_data, criterion) if val_data else epoch_loss
            )

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _evaluate_loss(
        self,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]],
        criterion: nn.Module,
    ) -> float:
        if val_data is None:
            return float("inf")
        X_val, y_val = val_data
        self.net.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
            targets = torch.from_numpy(y_val.astype(np.float32)).to(self.device)
            logits = self.net(inputs)
            loss = criterion(logits, targets).item()
        return loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X.astype(np.float32)).to(self.device)
            logits = self.net(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def save(self, path: Path | str) -> None:
        payload = {
            "state_dict": self.net.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> "SpectralNetModel":
        payload = torch.load(Path(path), map_location="cpu")
        config = SpectralNetConfig(**payload["config"])
        model = cls(config=config)
        model.net.load_state_dict(payload["state_dict"])
        return model
