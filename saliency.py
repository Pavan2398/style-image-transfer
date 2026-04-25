from __future__ import annotations

import numpy as np


def _normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(x.min())
    return x / (float(x.max()) + eps)


def get_saliency_map(image: np.ndarray) -> np.ndarray:
    """
    Compute a saliency mask in [0, 1] for an RGB image.

    Returns a 2D float32 array (H, W) where higher values indicate more "salient"
    regions (faces/objects/strong structure), intended to be preserved.

    Implementation:
    - Prefer OpenCV Spectral Residual saliency if `cv2` is available.
    - Fallback to an edge/structure-based saliency using simple gradients.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("get_saliency_map expects an RGB image of shape (H, W, 3)")

    # Try OpenCV spectral residual (fast + decent without ML models).
    try:
        import cv2  # type: ignore

        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape

        # Spectral residual saliency (Hou & Zhang style)
        fft = np.fft.fft2(gray)
        log_ampl = np.log(np.abs(fft) + 1e-8)
        phase = np.angle(fft)

        # Average filter in frequency domain (spatial average on log amplitude)
        log_ampl_blur = cv2.blur(log_ampl, (3, 3))
        residual = log_ampl - log_ampl_blur

        sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
        sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), sigmaX=2.5, sigmaY=2.5)
        sal = _normalize_01(sal)

        # Slightly sharpen salient foreground
        sal = np.clip(sal ** 0.8, 0.0, 1.0)
        if sal.shape != (h, w):
            sal = sal.reshape(h, w)
        return sal.astype(np.float32)
    except Exception:
        pass

    # Fallback: edge/structure magnitude from simple gradients.
    img = image.astype(np.float32) / 255.0
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # Sobel-ish gradients (finite differences)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    mag = np.sqrt(gx * gx + gy * gy)

    # Blur-ish smoothing using a small box filter (no external deps)
    # Pad and average 3x3 neighborhood
    padded = np.pad(mag, ((1, 1), (1, 1)), mode="reflect")
    sm = (
        padded[0:-2, 0:-2]
        + padded[0:-2, 1:-1]
        + padded[0:-2, 2:]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, 0:-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0

    sal = _normalize_01(sm)
    sal = np.clip(sal ** 0.9, 0.0, 1.0)
    return sal.astype(np.float32)

