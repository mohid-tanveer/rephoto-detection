from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import convolve2d

from src.features.tiling import (
    Tile,
    center_frequency_grid,
    evenly_sample_tiles,
    sliding_windows,
)


@dataclass
class SubpixelTileMetrics:
    x: int
    y: int
    score: float
    rg_peak_ratio: float
    gb_peak_ratio: float
    rb_peak_ratio: float
    rg_period: float
    gb_period: float
    rb_period: float
    period_consistency: float
    edge_strength: float


def _hann2d(size: int) -> np.ndarray:
    win = np.hanning(size)
    outer = np.outer(win, win)
    return outer / (outer.sum() + 1e-8)


def _channel_fft(channel: np.ndarray, window: np.ndarray) -> np.ndarray:
    centered = channel - channel.mean()
    return np.fft.fftshift(np.fft.fft2(centered * window))


def _cross_metrics(
    spec_a: np.ndarray, spec_b: np.ndarray, radius_grid: np.ndarray
) -> Tuple[float, float]:
    cross_power = np.abs(spec_a * np.conjugate(spec_b))
    mean_val = float(cross_power.mean() + 1e-8)
    peak_idx = np.argmax(cross_power)
    peak_ratio = float(cross_power.flat[peak_idx] / mean_val)
    dominant_period = float(radius_grid.flat[peak_idx])
    return peak_ratio, dominant_period


def _edge_strength(tile: np.ndarray) -> float:
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = sobel_x.T
    gray = tile.mean(axis=2)
    gx = convolve2d(gray, sobel_x, mode="same", boundary="symm")
    gy = convolve2d(gray, sobel_y, mode="same", boundary="symm")
    magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(magnitude))


def _period_consistency(periods: np.ndarray) -> float:
    return float(1.0 / (np.std(periods) + 0.05))


def _score_tile(metrics: SubpixelTileMetrics) -> float:
    ratios = np.array(
        [metrics.rg_peak_ratio, metrics.gb_peak_ratio, metrics.rb_peak_ratio],
        dtype=np.float32,
    )
    edge_weight = float(np.tanh(metrics.edge_strength * 5.0))
    return float(ratios.mean() * metrics.period_consistency * (0.5 + 0.5 * edge_weight))


def _analyze_tile(tile: Tile) -> SubpixelTileMetrics:
    data = tile.data.astype(np.float32)
    window = _hann2d(data.shape[0])
    radius, _, _ = center_frequency_grid((data.shape[0], data.shape[1]))

    r = _channel_fft(data[..., 0], window)
    g = _channel_fft(data[..., 1], window)
    b = _channel_fft(data[..., 2], window)

    rg_ratio, rg_period = _cross_metrics(r, g, radius)
    gb_ratio, gb_period = _cross_metrics(g, b, radius)
    rb_ratio, rb_period = _cross_metrics(r, b, radius)
    edge_strength = _edge_strength(data)
    period_consistency = _period_consistency(
        np.array([rg_period, gb_period, rb_period], dtype=np.float32)
    )

    metrics = SubpixelTileMetrics(
        x=tile.x,
        y=tile.y,
        score=0.0,
        rg_peak_ratio=rg_ratio,
        gb_peak_ratio=gb_ratio,
        rb_peak_ratio=rb_ratio,
        rg_period=rg_period,
        gb_period=gb_period,
        rb_period=rb_period,
        period_consistency=period_consistency,
        edge_strength=edge_strength,
    )
    metrics.score = _score_tile(metrics)
    return metrics


def _stat_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p75": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
    }


def _aggregate(records: List[SubpixelTileMetrics]) -> Dict[str, float]:
    if not records:
        scores = np.zeros(1, dtype=np.float32)
        edges = np.zeros(1, dtype=np.float32)
        ratios = np.zeros((1, 3), dtype=np.float32)
        periods = np.zeros((1, 3), dtype=np.float32)
        consistency = np.zeros(1, dtype=np.float32)
    else:
        scores = np.array([r.score for r in records], dtype=np.float32)
        edges = np.array([r.edge_strength for r in records], dtype=np.float32)
        ratios = np.array(
            [[r.rg_peak_ratio, r.gb_peak_ratio, r.rb_peak_ratio] for r in records],
            dtype=np.float32,
        )
        periods = np.array(
            [[r.rg_period, r.gb_period, r.rb_period] for r in records],
            dtype=np.float32,
        )
        consistency = np.array(
            [r.period_consistency for r in records],
            dtype=np.float32,
        )

    features: Dict[str, float] = {}
    features.update(_stat_summary(scores, "subpixel_score"))
    features.update(_stat_summary(edges, "subpixel_edge"))
    features.update(_stat_summary(consistency, "subpixel_period_consistency"))

    top_k = min(5, scores.size)
    features["subpixel_score_topk_mean"] = float(
        scores[np.argsort(scores)[-top_k:]].mean()
    )

    for idx, name in enumerate(["rg", "gb", "rb"]):
        features.update(_stat_summary(ratios[:, idx], f"subpixel_{name}_ratio"))
        features[f"subpixel_{name}_ratio_frac_gt_1_5"] = float(
            np.mean(ratios[:, idx] > 1.5)
        )
        features.update(_stat_summary(periods[:, idx], f"subpixel_{name}_period"))

    features["subpixel_period_global_std"] = float(periods.std())
    features["subpixel_ratio_global_std"] = float(ratios.std())
    return features


def extract_subpixel_features(
    rgb_image: np.ndarray,
    *,
    tile_size: int = 256,
    stride: int = 128,
    max_tiles: int | None = None,
    pad_mode: str = "reflect",
) -> Tuple[Dict[str, float], List[SubpixelTileMetrics]]:
    records: List[SubpixelTileMetrics] = []
    all_tiles = list(
        sliding_windows(
            rgb_image,
            tile_size=tile_size,
            stride=stride,
            pad_mode=pad_mode,
        )
    )
    tiles = evenly_sample_tiles(all_tiles, max_tiles)
    for tile in tiles:
        if tile.data.shape[2] != 3:
            raise ValueError("rgb image expected for subpixel analysis")
        if tile.data.shape[0] != tile_size or tile.data.shape[1] != tile_size:
            continue
        records.append(_analyze_tile(tile))

    summary = _aggregate(records)
    return summary, records
