"""Metrics module for Brain MRI restoration project.
Provides MSE, PSNR, SSIM, and experiment logging tools.
"""

from pathlib import Path
from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_mse(clean: np.ndarray, degraded: np.ndarray) -> float:
    """Calculates Mean Squared Error between clean reference and degraded/restored image.
    
    MSE = (1 / (M * N)) * sum((clean - degraded)^2)
    """
    diff = clean.astype(np.float64) - degraded.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_psnr(clean: np.ndarray, degraded: np.ndarray, max_val: float = 1.0) -> float:
    """Calculates Peak Signal-to-Noise Ratio (dB) between clean and degraded/restored image.
    
    PSNR = 10 * log10(max_val^2 / MSE)
    """
    mse = calculate_mse(clean, degraded)
    if mse == 0.0:
        return 99.0  # Convention for identical images
    return float(peak_signal_noise_ratio(clean, degraded, data_range=max_val))


def calculate_ssim(clean: np.ndarray, degraded: np.ndarray, max_val: float = 1.0) -> float:
    """Calculates Structural Similarity Index (SSIM) between clean and degraded/restored image.
    """
    return float(structural_similarity(clean, degraded, data_range=max_val))


def save_metrics(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Saves metrics dataframe to CSV format."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
