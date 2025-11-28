from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class MoireWaveletConfig:
    wavelet_channels: int = 8
    spatial_channels: int = 3
    lr: float = 1e-3
    batch_size: int = 12
    epochs: int = 50
    patience: int = 8
    weight_decay: float = 1e-4
    device: str | None = None


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualBranchNet(nn.Module):
    def __init__(self, wavelet_channels: int, spatial_channels: int):
        super().__init__()
        self.wavelet_branch = nn.Sequential(
            _ConvBlock(wavelet_channels, 32),
            _ConvBlock(32, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.spatial_branch = nn.Sequential(
            _ConvBlock(spatial_channels, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128, pool=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, wavelet: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        w = self.wavelet_branch(wavelet).view(wavelet.size(0), -1)
        s = self.spatial_branch(spatial).view(spatial.size(0), -1)
        fused = torch.cat([w, s], dim=1)
        return self.classifier(fused).squeeze(-1)


class MoireWaveletModel:
    def __init__(self, config: MoireWaveletConfig):
        self.config = config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.net = DualBranchNet(
            wavelet_channels=config.wavelet_channels,
            spatial_channels=config.spatial_channels,
        ).to(self.device)

    def _loader(
        self,
        wavelet: np.ndarray,
        spatial: np.ndarray,
        labels: Optional[np.ndarray],
        shuffle: bool,
    ) -> DataLoader:
        wavelet_tensor = torch.from_numpy(wavelet.astype(np.float32))
        spatial_tensor = torch.from_numpy(spatial.astype(np.float32))
        if labels is not None:
            label_tensor = torch.from_numpy(labels.astype(np.float32))
            dataset = TensorDataset(wavelet_tensor, spatial_tensor, label_tensor)
        else:
            dataset = TensorDataset(wavelet_tensor, spatial_tensor)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def fit(
        self,
        wavelet: np.ndarray,
        spatial: np.ndarray,
        labels: np.ndarray,
        *,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> None:
        train_loader = self._loader(wavelet, spatial, labels, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        best_state = None
        best_loss = float("inf")
        patience_counter = 0

        for _ in range(self.config.epochs):
            self.net.train()
            for wave_batch, spatial_batch, batch_y in train_loader:
                wave_batch = wave_batch.to(self.device)
                spatial_batch = spatial_batch.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.net(wave_batch, spatial_batch)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            val_loss = (
                self._evaluate_loss(val_data, criterion) if val_data else loss.item()
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
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        criterion: nn.Module,
    ) -> float:
        if val_data is None:
            return float("inf")
        wavelet_val, spatial_val, y_val = val_data
        loader = self._loader(wavelet_val, spatial_val, y_val, shuffle=False)
        self.net.eval()
        losses = []
        with torch.no_grad():
            for wave_batch, spatial_batch, batch_y in loader:
                wave_batch = wave_batch.to(self.device)
                spatial_batch = spatial_batch.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.net(wave_batch, spatial_batch)
                loss = criterion(logits, batch_y)
                losses.append(loss.item() * wave_batch.size(0))
        return float(np.sum(losses) / len(loader.dataset))

    def predict_proba(self, wavelet: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        loader = self._loader(wavelet, spatial, labels=None, shuffle=False)
        preds = []
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                wave_batch, spatial_batch = batch
                wave_batch = wave_batch.to(self.device)
                spatial_batch = spatial_batch.to(self.device)
                logits = self.net(wave_batch, spatial_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
        return np.concatenate(preds, axis=0)

    def save(self, path: Path | str) -> None:
        payload = {
            "state_dict": self.net.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> "MoireWaveletModel":
        payload = torch.load(Path(path), map_location="cpu")
        config = MoireWaveletConfig(**payload["config"])
        model = cls(config=config)
        model.net.load_state_dict(payload["state_dict"])
        return model
