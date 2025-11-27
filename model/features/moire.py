from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from model.features.tiling import (
    Tile,
    center_frequency_grid,
    evenly_sample_tiles,
    sliding_windows,
    summarize_tiles,
)


@dataclass
class MoireTileMetrics:
    x: int
    y: int
    score: float
    high_freq_ratio: float
    peak_ratio: float
    angle_var: float
    spectral_entropy: float


def _hann2d(size: int) -> np.ndarray:
    window_1d = np.hanning(size)
    outer = np.outer(window_1d, window_1d)
    return outer / (outer.sum() + 1e-8)


def _entropy(values: np.ndarray, eps: float = 1e-8) -> float:
    probs = values / (values.sum() + eps)
    probs = np.clip(probs, eps, 1.0)
    return float(-(probs * np.log(probs)).sum())


def _analyze_tile(tile: Tile, eps: float = 1e-8) -> MoireTileMetrics:
    data = tile.data.astype(np.float32)
    data = data - data.mean()
    window = _hann2d(data.shape[0])
    windowed = data * window
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(windowed)))
    total_energy = spectrum.sum() + eps

    radius, angle, _ = center_frequency_grid(spectrum.shape)
    high_freq_mask = radius > 0.35
    high_freq_energy = float(spectrum[high_freq_mask].sum() / total_energy)

    mean_mag = float(spectrum.mean() + eps)
    peak_mag = float(np.percentile(spectrum, 99))
    peak_ratio = peak_mag / mean_mag

    flat = spectrum.flatten()
    idx = np.argpartition(flat, -50)[-50:]
    selected_angles = angle.flatten()[idx]
    sin_var = np.var(np.sin(selected_angles))
    cos_var = np.var(np.cos(selected_angles))
    angle_var = float(sin_var + cos_var + eps)

    entropy = _entropy(spectrum)

    score = float(high_freq_energy * peak_ratio / (angle_var + 0.15))

    return MoireTileMetrics(
        x=tile.x,
        y=tile.y,
        score=score,
        high_freq_ratio=high_freq_energy,
        peak_ratio=peak_ratio,
        angle_var=angle_var,
        spectral_entropy=entropy,
    )


def _aggregate(records: List[MoireTileMetrics]) -> Dict[str, float]:
    if not records:
        return {
            "moire_score_mean": 0.0,
            "moire_score_std": 0.0,
            "moire_score_max": 0.0,
            "moire_high_freq_mean": 0.0,
            "moire_peak_ratio_mean": 0.0,
            "moire_angle_var_mean": 0.0,
            "moire_entropy_mean": 0.0,
        }

    scores = np.array([r.score for r in records], dtype=np.float32)
    high_freq = np.array([r.high_freq_ratio for r in records], dtype=np.float32)
    peaks = np.array([r.peak_ratio for r in records], dtype=np.float32)
    angle_var = np.array([r.angle_var for r in records], dtype=np.float32)
    entropy = np.array([r.spectral_entropy for r in records], dtype=np.float32)

    return {
        "moire_score_mean": float(scores.mean()),
        "moire_score_std": float(scores.std()),
        "moire_score_max": float(scores.max()),
        "moire_high_freq_mean": float(high_freq.mean()),
        "moire_high_freq_std": float(high_freq.std()),
        "moire_peak_ratio_mean": float(peaks.mean()),
        "moire_peak_ratio_std": float(peaks.std()),
        "moire_angle_var_mean": float(angle_var.mean()),
        "moire_angle_var_std": float(angle_var.std()),
        "moire_entropy_mean": float(entropy.mean()),
    }


def extract_moire_features(
    gray_image: np.ndarray,
    *,
    tile_size: int = 256,
    stride: int = 128,
    max_tiles: int | None = None,
    pad_mode: str = "reflect",
) -> Tuple[Dict[str, float], List[MoireTileMetrics]]:
    """
    compute per-image moire descriptors and keep per-tile diagnostics for visualization.
    """
    records: List[MoireTileMetrics] = []
    all_tiles = list(
        sliding_windows(
            gray_image,
            tile_size=tile_size,
            stride=stride,
            pad_mode=pad_mode,
        )
    )
    tiles = evenly_sample_tiles(all_tiles, max_tiles)
    for tile in tiles:
        if tile.data.shape[0] != tile_size or tile.data.shape[1] != tile_size:
            continue
        records.append(_analyze_tile(tile))

    summary = _aggregate(records)
    return summary, records
