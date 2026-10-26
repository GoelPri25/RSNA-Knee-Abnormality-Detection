"""Inference-time implementation of the Fit160 cache transformation."""

from __future__ import annotations

import numpy as np


def physical_fit_with_support(
    image: np.ndarray,
    support: np.ndarray,
    spacing: tuple[float, float] | None,
    minimum_fov_mm: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the complete slice into a square physical canvas, then resize.

    The square canvas is at least ``minimum_fov_mm`` on each axis and expands
    when either acquired dimension is physically larger. The acquired slice is
    centred without cropping. Intensity interpolation is bilinear; support is
    resized with nearest-neighbour so real zero intensities remain distinct
    from padding.
    """
    from PIL import Image

    image = np.asarray(image, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    if image.ndim != 2 or support.shape != image.shape:
        raise ValueError("image and support must be same-size 2D arrays")
    rows, columns = image.shape
    if spacing is None:
        # Match the existing cache's no-spacing fallback: preserve the entire
        # image and pad its shorter dimension to a square before resizing.
        canvas_rows = canvas_columns = max(rows, columns)
    else:
        row_mm, column_mm = map(float, spacing)
        if not (np.isfinite(row_mm) and np.isfinite(column_mm)
                and row_mm > 0 and column_mm > 0):
            raise ValueError(f"Invalid pixel spacing: {spacing}")
        canvas_fov_mm = max(
            float(minimum_fov_mm), rows * row_mm, columns * column_mm
        )
        canvas_rows = max(rows, int(round(canvas_fov_mm / row_mm)))
        canvas_columns = max(columns, int(round(canvas_fov_mm / column_mm)))

    top = (canvas_rows - rows) // 2
    left = (canvas_columns - columns) // 2
    canvas = np.zeros((canvas_rows, canvas_columns), dtype=np.float32)
    mask = np.zeros((canvas_rows, canvas_columns), dtype=np.uint8)
    canvas[top:top + rows, left:left + columns] = image
    mask[top:top + rows, left:left + columns] = support
    resized = np.asarray(
        Image.fromarray(canvas).resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    )
    resized_support = np.asarray(
        Image.fromarray(mask * 255).resize((size, size), Image.NEAREST),
        dtype=np.uint8,
    )
    return resized, resized_support
