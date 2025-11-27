from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from PIL import Image

ArrayLike = np.ndarray


def load_image(path: Path | str, mode: Literal["rgb", "gray"] = "rgb") -> ArrayLike:
    """
    load an image as a float32 numpy array normalized to [0, 1].
    """
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")

    with Image.open(img_path) as img:
        if mode == "rgb":
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        elif mode == "gray":
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        else:
            raise ValueError(f"unsupported mode: {mode}")

    return arr


def ensure_three_channel(gray_array: ArrayLike) -> ArrayLike:
    """
    expand a grayscale array to rgb if needed for visualization.
    """
    if gray_array.ndim == 2:
        return np.stack([gray_array, gray_array, gray_array], axis=-1)
    return gray_array


def to_uint8(image: ArrayLike) -> ArrayLike:
    """
    convert float image in [0, 1] to uint8.
    """
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def resize_if_needed(
    image: ArrayLike, max_size: Tuple[int, int] | None = None
) -> ArrayLike:
    """
    optionally resize while preserving aspect ratio to keep processing manageable.
    """
    if max_size is None:
        return image

    max_h, max_w = max_size
    h, w = image.shape[:2]
    scale = min(max_h / h, max_w / w, 1.0)
    if scale == 1.0:
        return image

    new_h = int(h * scale)
    new_w = int(w * scale)
    pil_img = Image.fromarray(to_uint8(image if image.dtype != np.uint8 else image))
    resized = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr
