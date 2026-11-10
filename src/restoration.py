"""Restoration module for Brain MRI restoration project.

Implements four traditional image restoration filters:
  1. Median Filter      — spatial nonlinear rank filter
  2. NLM               — Non-Local Means patch-similarity denoiser
  3. Wiener Filter      — adaptive minimum-MSE linear filter
  4. Lanczos Sharpening — unsharp masking with Lanczos-inspired parameters

All functions operate on normalized float32 arrays in [0.0, 1.0].
Fixed parameters are chosen from standard literature defaults and documented
here for full academic reproducibility.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from scipy.signal import wiener as scipy_wiener
from skimage.filters import unsharp_mask

from .preprocessing import normalize_image, save_image
from .metrics import calculate_mse, calculate_psnr, calculate_ssim


# ============================================================
# Method Registry
# ============================================================
RESTORATION_METHODS: List[str] = [
    'median_filter',
    'nlm',
    'wiener',
    'lanczos_sharpening',
]

# ============================================================
# Fixed Parameters (same for all degradation types / severities)
# ============================================================

# -- Median Filter -------------------------------------------
# 3×3 spatial median: removes noise spikes while preserving edges
MEDIAN_KERNEL_SIZE: int = 3

# -- Non-Local Means (NLM) -----------------------------------
# cv2.fastNlMeansDenoising default values (uint8 grayscale)
# h        — filter strength: higher values smooth more but lose detail
# template — patch size for similarity measure (pixels)
# search   — candidate region size (pixels)
NLM_H: float = 10.0
NLM_TEMPLATE_WINDOW: int = 7
NLM_SEARCH_WINDOW: int = 21

# -- Wiener Filter -------------------------------------------
# scipy.signal.wiener: local mean/variance estimation window
# mysize=5 → 5×5 local estimation window
WIENER_SIZE: int = 5

# -- Lanczos + Sharpening ------------------------------------
# skimage.filters.unsharp_mask
# radius — Gaussian blur radius (pixels) for residual extraction
# amount — gain applied to the high-frequency residual
LANCZOS_RADIUS: float = 1.0
LANCZOS_AMOUNT: float = 1.5


# ============================================================
# Restoration Functions
# ============================================================

def apply_median_filter(image_array: np.ndarray,
                        kernel_size: int = MEDIAN_KERNEL_SIZE) -> np.ndarray:
    """Applies a 2-D spatial median filter.

    Process: float32→uint8 → cv2.medianBlur → float32

    Args:
        image_array: 2D float32 array in [0.0, 1.0].
        kernel_size: Odd integer kernel size (default 3).

    Returns:
        Restored float32 array in [0.0, 1.0].
    """
    uint8_img = np.round(np.clip(image_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    restored = cv2.medianBlur(uint8_img, kernel_size)
    return restored.astype(np.float32) / 255.0


def apply_nlm(image_array: np.ndarray,
              h: float = NLM_H,
              template_window_size: int = NLM_TEMPLATE_WINDOW,
              search_window_size: int = NLM_SEARCH_WINDOW) -> np.ndarray:
    """Applies Non-Local Means denoising (cv2.fastNlMeansDenoising).

    Process: float32→uint8 → NLM → float32

    Args:
        image_array: 2D float32 array in [0.0, 1.0].
        h: Filter strength (higher = smoother, default 10).
        template_window_size: Template patch radius in pixels (default 7).
        search_window_size: Search window size in pixels (default 21).

    Returns:
        Restored float32 array in [0.0, 1.0].
    """
    uint8_img = np.round(np.clip(image_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    restored = cv2.fastNlMeansDenoising(
        uint8_img,
        h=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )
    return restored.astype(np.float32) / 255.0


def apply_wiener_filter(image_array: np.ndarray,
                        size: int = WIENER_SIZE) -> np.ndarray:
    """Applies adaptive Wiener filter (scipy.signal.wiener).

    The Wiener filter minimises local MSE by estimating local mean and variance.
    Operates directly on float64, then clips result to [0.0, 1.0].

    Args:
        image_array: 2D float32 array in [0.0, 1.0].
        size: Local estimation window size (default 5 → 5×5 window).

    Returns:
        Restored float32 array clipped to [0.0, 1.0].
    """
    restored = scipy_wiener(image_array.astype(np.float64), mysize=size)
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def apply_lanczos_sharpening(image_array: np.ndarray,
                              radius: float = LANCZOS_RADIUS,
                              amount: float = LANCZOS_AMOUNT) -> np.ndarray:
    """Applies Lanczos-inspired edge sharpening via unsharp masking.

    Unsharp mask: I_sharp = I + amount × (I − GaussianBlur(I, radius))
    This recovers high-frequency detail lost to blurring/downsampling.

    Args:
        image_array: 2D float32 array in [0.0, 1.0].
        radius: Gaussian blur radius in pixels (default 1.0).
        amount: Sharpening gain (default 1.5).

    Returns:
        Sharpened float32 array clipped to [0.0, 1.0].
    """
    restored = unsharp_mask(
        image_array.astype(np.float64),
        radius=radius,
        amount=amount,
        channel_axis=None,
    )
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def apply_all_methods(image_array: np.ndarray) -> Dict[str, np.ndarray]:
    """Applies all four restoration methods to one image.

    Args:
        image_array: 2D float32 array in [0.0, 1.0].

    Returns:
        Dict mapping method name → restored float32 array.
    """
    return {
        'median_filter':      apply_median_filter(image_array),
        'nlm':                apply_nlm(image_array),
        'wiener':             apply_wiener_filter(image_array),
        'lanczos_sharpening': apply_lanczos_sharpening(image_array),
    }


# ============================================================
# Batch / Multiprocessing Worker
# ============================================================

def process_single_image_restore(
        degraded_path_str: str,
        clean_path_str: str,
        degradation: str,
        category: str,
        severity: int,
        image_idx: int,
        sample_output_dir_str: Optional[str] = None) -> List[Dict]:
    """Worker function: load one degraded image, apply all 4 methods, compute metrics.

    Compares each restored image against the corresponding clean original using
    MSE, PSNR, and SSIM.  Optionally saves a sample restored image to disk
    (only for image_idx == 0 to limit disk usage).

    Args:
        degraded_path_str: Path to the degraded input image.
        clean_path_str:    Path to the clean reference image.
        degradation:       Degradation type label (e.g. 'gaussian').
        category:          MRI category label (e.g. 'glioma').
        severity:          Severity percentage (10, 20, … 70).
        image_idx:         Index of this image within the current batch.
        sample_output_dir_str: Base dir for sample image saves (or None).

    Returns:
        List of 4 dicts (one per method) each containing metric results.
    """
    degraded_p = Path(degraded_path_str)
    clean_p = Path(clean_path_str)
    results: List[Dict] = []

    try:
        with Image.open(degraded_p) as img:
            degraded_arr = normalize_image(img, target_size=(512, 512))
        with Image.open(clean_p) as img:
            clean_arr = normalize_image(img, target_size=(512, 512))

        restored_dict = apply_all_methods(degraded_arr)

        for method_name, restored_arr in restored_dict.items():
            mse_val  = float(calculate_mse(clean_arr, restored_arr))
            psnr_val = float(calculate_psnr(clean_arr, restored_arr))
            ssim_val = float(calculate_ssim(clean_arr, restored_arr))

            # Save sample image (first image only per combination)
            if sample_output_dir_str is not None and image_idx == 0:
                sample_dir = (
                    Path(sample_output_dir_str)
                    / method_name / degradation / category / str(severity)
                )
                sample_dir.mkdir(parents=True, exist_ok=True)
                stem = degraded_p.stem
                save_image(
                    restored_arr,
                    sample_dir / f"{stem}_restored_{method_name}.png",
                    format="PNG",
                    compress_level=1,
                )

            results.append({
                'degradation': degradation,
                'category':    category,
                'severity':    severity,
                'method':      method_name,
                'filename':    degraded_p.name,
                'mse':         mse_val,
                'psnr':        psnr_val,
                'ssim':        ssim_val,
            })

    except Exception as exc:  # noqa: BLE001
        for method_name in RESTORATION_METHODS:
            results.append({
                'degradation': degradation,
                'category':    category,
                'severity':    severity,
                'method':      method_name,
                'filename':    degraded_p.name,
                'mse':         float('nan'),
                'psnr':        float('nan'),
                'ssim':        float('nan'),
            })

    return results


def run_worker_restore(item: tuple) -> List[Dict]:
    """Top-level multiprocessing wrapper (must be picklable on Windows)."""
    deg_p, clean_p, degradation, category, severity, idx, sample_dir = item
    return process_single_image_restore(
        degraded_path_str=deg_p,
        clean_path_str=clean_p,
        degradation=degradation,
        category=category,
        severity=severity,
        image_idx=idx,
        sample_output_dir_str=sample_dir,
    )
