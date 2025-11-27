from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from PIL import Image

from model.features.moire import MoireTileMetrics
from model.features.subpixel import SubpixelTileMetrics
from model.utils.image import ensure_three_channel, to_uint8


def _normalize_scores(scores: Iterable[float]) -> float:
    arr = np.asarray(list(scores), dtype=np.float32)
    if arr.size == 0:
        return 1.0
    max_val = float(arr.max())
    return max(max_val, 1e-3)


def _apply_overlay(
    base: np.ndarray,
    metrics: Iterable,
    tile_size: int,
    color: Tuple[float, float, float],
) -> np.ndarray:
    canvas = base.copy()
    metric_list = list(metrics)
    denom = _normalize_scores([m.score for m in metric_list])
    for m in metric_list:
        weight = min(m.score / denom, 1.0)
        y0, x0 = m.y, m.x
        y1, x1 = y0 + tile_size, x0 + tile_size
        canvas[y0:y1, x0:x1, 0] = (
            canvas[y0:y1, x0:x1, 0] * (1 - weight) + weight * color[0]
        )
        canvas[y0:y1, x0:x1, 1] = (
            canvas[y0:y1, x0:x1, 1] * (1 - weight) + weight * color[1]
        )
        canvas[y0:y1, x0:x1, 2] = (
            canvas[y0:y1, x0:x1, 2] * (1 - weight) + weight * color[2]
        )
    return canvas


def render_moire_overlay(
    image: np.ndarray, metrics: Iterable[MoireTileMetrics], tile_size: int
) -> Image.Image:
    """
    render a heatmap showing moire-heavy regions in reddish hues.
    """
    rgb = ensure_three_channel(image)
    overlay_space = np.clip(rgb.copy(), 0.0, 1.0)
    colored = _apply_overlay(overlay_space, metrics, tile_size, (1.0, 0.2, 0.2))
    return Image.fromarray(to_uint8(colored))


def render_subpixel_overlay(
    image: np.ndarray, metrics: Iterable[SubpixelTileMetrics], tile_size: int
) -> Image.Image:
    """
    render a heatmap showing subpixel cues in bluish hues.
    """
    rgb = ensure_three_channel(image)
    overlay_space = np.clip(rgb.copy(), 0.0, 1.0)
    colored = _apply_overlay(overlay_space, metrics, tile_size, (0.2, 0.2, 1.0))
    return Image.fromarray(to_uint8(colored))


def render_combined_overlay(
    image: np.ndarray,
    moire_metrics: Iterable[MoireTileMetrics],
    subpixel_metrics: Iterable[SubpixelTileMetrics],
    tile_size: int,
) -> Image.Image:
    """
    overlay moire (red) and subpixel (blue) cues for manual inspection.
    """
    rgb = ensure_three_channel(image)
    overlay_space = np.clip(rgb.copy(), 0.0, 1.0)
    overlay_space = _apply_overlay(
        overlay_space, moire_metrics, tile_size, (1.0, 0.2, 0.2)
    )
    overlay_space = _apply_overlay(
        overlay_space, subpixel_metrics, tile_size, (0.2, 0.2, 1.0)
    )
    return Image.fromarray(to_uint8(overlay_space))
