"""Phase 10 Worker Module for Multiprocessing Image Restoration.
Provides top-level picklable worker functions for parallel execution on Windows.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image

import sys
PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import normalize_image, save_image
from src.metrics import calculate_mse, calculate_psnr, calculate_ssim
from src.restoration import apply_all_methods, RESTORATION_METHODS


def process_phase10_image(item: tuple) -> List[Dict]:
    """Worker function for single combined degraded MRI restoration.
    Loads degraded image and corresponding clean original, applies 4 traditional filters,
    calculates MSE, PSNR, SSIM, optionally saves representative sample, and returns standardized dicts.

    Args:
        item: Tuple of (deg_path_str, clean_path_str, category, dataset, severity, idx, sample_dir_str)

    Returns:
        List of 4 dicts matching the standardized 11-column evaluation schema.
    """
    deg_path_str, clean_path_str, category, dataset, severity, idx, sample_dir_str = item
    deg_p = Path(deg_path_str)
    clean_p = Path(clean_path_str)
    results = []

    try:
        with Image.open(deg_p) as img:
            deg_arr = normalize_image(img, target_size=(512, 512))
        with Image.open(clean_p) as img:
            clean_arr = normalize_image(img, target_size=(512, 512))

        restored_dict = apply_all_methods(deg_arr)

        for method_name, rest_arr in restored_dict.items():
            mse_val = float(calculate_mse(clean_arr, rest_arr))
            psnr_val = float(calculate_psnr(clean_arr, rest_arr))
            ssim_val = float(calculate_ssim(clean_arr, rest_arr))

            # Save representative sample image (first image of combination only)
            if sample_dir_str is not None and idx == 0:
                out_sample = (
                    Path(sample_dir_str)
                    / method_name / category / str(severity)
                    / f"{deg_p.stem}_restored_{method_name}.png"
                )
                save_image(rest_arr, out_sample, format="PNG", compress_level=1)

            results.append({
                'image_name': deg_p.name,
                'category': category,
                'dataset': dataset,
                'degradation_type': 'combined',
                'severity': severity,
                'restoration_method': method_name,
                'MSE': mse_val,
                'PSNR': psnr_val,
                'SSIM': ssim_val,
                'status': 'SUCCESS',
                'error_message': ''
            })

    except Exception as e:
        for method_name in RESTORATION_METHODS:
            results.append({
                'image_name': deg_p.name,
                'category': category,
                'dataset': dataset,
                'degradation_type': 'combined',
                'severity': severity,
                'restoration_method': method_name,
                'MSE': float('nan'),
                'PSNR': float('nan'),
                'SSIM': float('nan'),
                'status': 'ERROR',
                'error_message': str(e)
            })

    return results
