from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    index: int
    data: np.ndarray


def sliding_windows(
    image: np.ndarray,
    tile_size: int,
    stride: int,
    max_tiles: int | None = None,
    pad_mode: str = "reflect",
) -> Iterator[Tile]:
    """
    yield tiles along with their top-left coordinates.
    """
    h, w = image.shape[:2]
    tiles_returned = 0
    tile_index = 0

    padded = image
    if h < tile_size or w < tile_size:
        pad_y = max(tile_size - h, 0)
        pad_x = max(tile_size - w, 0)
        pad_args = [
            (0, pad_y),
            (0, pad_x),
        ]
        if image.ndim == 3:
            pad_args.append((0, 0))
        padded = np.pad(image, pad_args, mode=pad_mode)
        h, w = padded.shape[:2]

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = padded[y : y + tile_size, x : x + tile_size]
            yield Tile(x=x, y=y, index=tile_index, data=tile)
            tiles_returned += 1
            tile_index += 1
            if max_tiles is not None and tiles_returned >= max_tiles:
                return


def evenly_sample_tiles(tiles: List[Tile], max_tiles: int | None) -> List[Tile]:
    if max_tiles is None or max_tiles <= 0 or len(tiles) <= max_tiles:
        return tiles
    indices = np.linspace(0, len(tiles) - 1, max_tiles, dtype=int)
    seen = set()
    sampled = []
    for idx in indices:
        if idx in seen:
            continue
        sampled.append(tiles[idx])
        seen.add(idx)
    if len(sampled) < max_tiles:
        for tile in tiles:
            if tile.index in seen:
                continue
            sampled.append(tile)
            seen.add(tile.index)
            if len(sampled) >= max_tiles:
                break
    return sampled


def center_frequency_grid(
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    build normalized radius and angle grids for fft domains.
    """
    h, w = size
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = yy - cy
    x = xx - cx
    radius = np.sqrt(x**2 + y**2)
    max_radius = radius.max() or 1.0
    radius /= max_radius
    angle = np.arctan2(y, x)
    return radius, angle, max_radius
