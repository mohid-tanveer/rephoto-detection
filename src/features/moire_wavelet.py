from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pywt
from PIL import Image

from src.utils.image import load_image


def _resize_image(gray: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    if gray.shape == target_size:
        return gray
    pil_img = Image.fromarray(np.uint8(np.clip(gray * 255.0, 0, 255)))
    pil_img = pil_img.resize(target_size[::-1], Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return arr


def _resize_coeff(coeff: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if coeff.shape == target_shape:
        return coeff
    pil_img = Image.fromarray(coeff.astype(np.float32), mode="F")
    pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.asarray(pil_img, dtype=np.float32)


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    return (channel - channel.mean()) / (channel.std() + 1e-6)


def compute_wavelet_and_spatial(
    image_path: Path | str,
    *,
    wavelet_size: int = 512,
    spatial_size: int = 512,
    wavelet: str = "db2",
) -> Tuple[np.ndarray, np.ndarray]:
    gray = load_image(image_path, mode="gray")
    gray = _resize_image(gray, (wavelet_size, wavelet_size))
    rgb = load_image(image_path, mode="rgb")
    spatial_img = Image.fromarray(np.uint8(np.clip(rgb * 255.0, 0, 255))).resize(
        (spatial_size, spatial_size), Image.BICUBIC
    )
    spatial = np.asarray(spatial_img, dtype=np.float32) / 255.0
    spatial = np.transpose(spatial, (2, 0, 1))  # C, H, W

    cA1, (cH1, cV1, cD1) = pywt.dwt2(gray, wavelet)
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, wavelet)

    target_shape = cH1.shape

    def prep_channel(arr):
        resized = _resize_coeff(arr, target_shape)
        return _normalize_channel(resized)

    level1 = [_normalize_channel(ch) for ch in (cA1, cH1, cV1, cD1)]
    level2 = [prep_channel(ch) for ch in (cA2, cH2, cV2, cD2)]
    wavelet_stack = np.stack(level1 + level2, axis=0).astype(np.float32)
    return wavelet_stack, spatial.astype(np.float32)


def build_wavelet_dataset(
    index_df,
    *,
    target_size: int,
    wavelet: str,
    cache_path: Path,
    force_recompute: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path.exists() and not force_recompute:
        data = np.load(cache_path, allow_pickle=True)
        tensors = data["wavelet_tensors"]
        spatial = data["spatial_tensors"]
        cached_ids = data["image_ids"]
        if np.array_equal(cached_ids, index_df["image_id"].values):
            return tensors, spatial

    wavelet_tensors = []
    spatial_tensors = []
    for row in index_df.itertuples():
        wavelet_tensor, spatial_tensor = compute_wavelet_and_spatial(
            row.abs_path,
            wavelet_size=target_size,
            spatial_size=target_size,
            wavelet=wavelet,
        )
        wavelet_tensors.append(wavelet_tensor)
        spatial_tensors.append(spatial_tensor)
    wavelet_np = np.stack(wavelet_tensors, axis=0)
    spatial_np = np.stack(spatial_tensors, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        wavelet_tensors=wavelet_np,
        spatial_tensors=spatial_np,
        image_ids=index_df["image_id"].values,
    )
    return wavelet_np, spatial_np
