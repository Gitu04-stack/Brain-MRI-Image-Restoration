"""Phase 8: Complete Testing Evaluation on 1,600 Unseen Test Images.

Processes 1,600 clean test MRI images across 4 categories (glioma, meningioma, notumor, pituitary)
x 4 degradation types (gaussian, rician, low_resolution, motion_blur)
x 7 severity levels (10% to 70%)
x 4 restoration filters (Median, NLM, Wiener, Lanczos+Sharpening)
= 179,200 total evaluations on the unseen test dataset.

Key Features:
1. Permanent degraded test dataset creation under dataset/degraded_test/ for future phases (Phase 11, 13).
2. Incremental flush to outputs/metrics/phase8_testing_metrics.csv after each batch.
3. Resumability: Skips batches already present in the metrics CSV.
4. Comprehensive verification of outputs, sample images, and summary statistics.
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import normalize_image, save_image
from src.degradation import (
    add_gaussian_noise,
    add_rician_noise,
    apply_low_resolution,
    apply_motion_blur,
    GAUSSIAN_SIGMA_MAP,
    RICIAN_SIGMA_MAP,
    LOW_RES_SCALE_MAP,
    MOTION_BLUR_KERNEL_MAP,
)
from src.restoration import apply_all_methods, RESTORATION_METHODS
from src.metrics import calculate_mse, calculate_psnr, calculate_ssim

TEST_BASE_DIR = PROJECT_ROOT / "dataset" / "archive" / "Testing"
DEG_TEST_BASE = PROJECT_ROOT / "dataset" / "degraded_test"
METRICS_DIR   = PROJECT_ROOT / "outputs" / "metrics"
SAMPLE_DIR    = PROJECT_ROOT / "outputs" / "samples" / "test_restored"
OUTPUT_CSV    = METRICS_DIR / "phase8_testing_metrics.csv"
SUMMARY_CSV   = METRICS_DIR / "phase8_testing_summary.csv"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES      = ["glioma", "meningioma", "notumor", "pituitary"]
DEGRADATIONS    = ["gaussian", "rician", "low_resolution", "motion_blur"]
SEVERITY_LEVELS = [10, 20, 30, 40, 50, 60, 70]
NUM_WORKERS     = 10
CHUNK_SIZE      = 20
BASE_TEST_SEED  = 9999

EXPECTED_PER_BATCH = 400 * len(RESTORATION_METHODS)  # 1,600 rows per (deg, cat, sev)
TOTAL_EXPECTED     = len(DEGRADATIONS) * len(CATEGORIES) * len(SEVERITY_LEVELS) * EXPECTED_PER_BATCH  # 179,200 rows


def process_single_test_image(item: tuple) -> list:
    """Worker: loads 1 test image, applies 1 degradation at 1 severity, restores with 4 filters, computes metrics."""
    clean_path_str, deg_type, cat, sev, img_idx, out_deg_dir_str, out_sample_dir_str = item
    clean_p = Path(clean_path_str)
    stem    = clean_p.stem
    results = []

    try:
        with Image.open(clean_p) as img:
            clean_arr = normalize_image(img, target_size=(512, 512))

        # Generate degradation
        if deg_type == "gaussian":
            sigma = GAUSSIAN_SIGMA_MAP[sev]
            seed = (BASE_TEST_SEED * 100000 + img_idx * 100 + sev) % (2**31 - 1)
            deg_arr = add_gaussian_noise(clean_arr, sigma=sigma, seed=seed)
            deg_filename = f"{stem}_gaussian_{sev}.png"
        elif deg_type == "rician":
            sigma = RICIAN_SIGMA_MAP[sev]
            seed = (BASE_TEST_SEED * 200000 + img_idx * 100 + sev) % (2**31 - 1)
            deg_arr = add_rician_noise(clean_arr, sigma=sigma, seed=seed)
            deg_filename = f"{stem}_rician_{sev}.png"
        elif deg_type == "low_resolution":
            scale = LOW_RES_SCALE_MAP[sev]
            deg_arr = apply_low_resolution(clean_arr, scale_factor=scale, target_size=(512, 512))
            deg_filename = f"{stem}_lowres_{sev}.png"
        elif deg_type == "motion_blur":
            k_len = MOTION_BLUR_KERNEL_MAP[sev]
            deg_arr = apply_motion_blur(clean_arr, kernel_length=k_len, angle_deg=45.0)
            deg_filename = f"{stem}_motion_{sev}.png"
        else:
            raise ValueError(f"Unknown degradation: {deg_type}")

        # Save degraded test image
        out_deg_dir = Path(out_deg_dir_str)
        out_deg_dir.mkdir(parents=True, exist_ok=True)
        deg_file_path = out_deg_dir / deg_filename
        if not deg_file_path.exists():
            save_image(deg_arr, deg_file_path, format="PNG", compress_level=1)

        # Apply all 4 traditional restoration methods
        restored_dict = apply_all_methods(deg_arr)

        for m_name, restored_arr in restored_dict.items():
            mse_val  = float(calculate_mse(clean_arr, restored_arr))
            psnr_val = float(calculate_psnr(clean_arr, restored_arr))
            ssim_val = float(calculate_ssim(clean_arr, restored_arr))

            # Save sample restored image for the first image of each combination
            if img_idx == 0:
                s_dir = Path(out_sample_dir_str) / m_name / deg_type / cat / str(sev)
                s_dir.mkdir(parents=True, exist_ok=True)
                s_file = s_dir / f"{stem}_restored_{m_name}.png"
                if not s_file.exists():
                    save_image(restored_arr, s_file, format="PNG", compress_level=1)

            results.append({
                'image_name':         deg_filename,
                'category':           cat,
                'dataset':            'Testing',
                'degradation_type':   deg_type,
                'severity':           sev,
                'restoration_method': m_name,
                'MSE':                mse_val,
                'PSNR':               psnr_val,
                'SSIM':               ssim_val,
                'status':             'SUCCESS',
                'error_message':      ''
            })

    except Exception as exc:
        for m_name in RESTORATION_METHODS:
            results.append({
                'image_name':         f"{stem}_{deg_type}_{sev}.png",
                'category':           cat,
                'dataset':            'Testing',
                'degradation_type':   deg_type,
                'severity':           sev,
                'restoration_method': m_name,
                'MSE':                float('nan'),
                'PSNR':               float('nan'),
                'SSIM':               float('nan'),
                'status':             'FAILED',
                'error_message':      str(exc)
            })

    return results


def get_completed_batches() -> set:
    """Returns set of (degradation, category, severity) already completed."""
    if not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(OUTPUT_CSV)
        if df.empty or 'degradation_type' not in df.columns:
            return set()
        counts = df.groupby(['degradation_type', 'category', 'severity']).size()
        completed = {
            combo for combo, cnt in counts.items()
            if cnt == EXPECTED_PER_BATCH
        }
        return completed
    except Exception:
        return set()


def main():
    print("=" * 70)
    print("PHASE 8: UNSEEN TESTING EVALUATION RUNNER")
    print("=" * 70)
    print(f"Output CSV:      {OUTPUT_CSV}")
    print(f"Degraded Test:   {DEG_TEST_BASE}")
    print(f"Categories (4):  {CATEGORIES}")
    print(f"Degradations (4):{DEGRADATIONS}")
    print(f"Severities (7):  {SEVERITY_LEVELS}")
    print(f"Methods (4):     {RESTORATION_METHODS}")
    print(f"Total Expected:  {TOTAL_EXPECTED} evaluation records (112 batches x 1,600)")
    print("-" * 70)

    completed_batches = get_completed_batches()
    if completed_batches:
        print(f"Found {len(completed_batches)} / 112 batches already completed in CSV.")
    else:
        print("Starting fresh Phase 8 testing evaluation.")

    start_total_time = time.time()
    batches_run = 0

    for deg_type in DEGRADATIONS:
        for cat in CATEGORIES:
            clean_paths = sorted(list((TEST_BASE_DIR / cat).glob("*.jpg")))
            assert len(clean_paths) == 400, f"Expected 400 test images in {cat}, found {len(clean_paths)}"

            for sev in SEVERITY_LEVELS:
                batch_key = (deg_type, cat, sev)
                if batch_key in completed_batches:
                    continue

                t_b_start = time.time()
                out_deg_dir = DEG_TEST_BASE / deg_type / cat / str(sev)

                # Build 400 work items
                work_items = [
                    (str(clean_p), deg_type, cat, sev, idx, str(out_deg_dir), str(SAMPLE_DIR))
                    for idx, clean_p in enumerate(clean_paths)
                ]

                batch_results = []
                with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    for res_list in executor.map(process_single_test_image, work_items, chunksize=CHUNK_SIZE):
                        batch_results.extend(res_list)

                assert len(batch_results) == EXPECTED_PER_BATCH, f"Mismatch: expected {EXPECTED_PER_BATCH}, got {len(batch_results)}"

                df_batch = pd.DataFrame(batch_results)
                file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
                df_batch.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)

                batches_run += 1
                t_elapsed = time.time() - t_b_start
                curr_rows = len(pd.read_csv(OUTPUT_CSV))
                print(f"[{deg_type.upper():<14}] {cat:<10} @ {sev:>2}% -> {len(df_batch)} rows in {t_elapsed:>5.1f}s | Total: {curr_rows:,} / {TOTAL_EXPECTED:,} ({curr_rows/TOTAL_EXPECTED*100:.1f}%)")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"ALL 112 TESTING BATCHES PROCESSED in {total_time/60:.1f} minutes.")
    print("=" * 70)

    # 1. Final Dataset Verification
    print("\n--- FINAL CSV VERIFICATION ---")
    df_all = pd.read_csv(OUTPUT_CSV)
    print(f"Total Rows:     {len(df_all):,} (Expected: {TOTAL_EXPECTED:,})")
    assert len(df_all) == TOTAL_EXPECTED, f"Row count mismatch! {len(df_all)} != {TOTAL_EXPECTED}"
    print("[PASS] Row count matches exactly 179,200.")

    nulls = df_all[['MSE', 'PSNR', 'SSIM']].isnull().sum()
    print(f"Nulls in core metrics:\n{nulls.to_string()}")
    assert nulls.sum() == 0, "Null values detected in core metrics!"
    print("[PASS] Zero nulls or NaNs in core metrics.")

    # 2. Compute Summary Statistics
    print("\nComputing statistical summary table (448 combinations) ...")
    df_summary = (
        df_all.groupby(['degradation_type', 'category', 'severity', 'restoration_method'])
        .agg(
            Mean_MSE =('MSE', 'mean'),
            Std_MSE  =('MSE', 'std'),
            Mean_PSNR=('PSNR', 'mean'),
            Std_PSNR =('PSNR', 'std'),
            Mean_SSIM=('SSIM', 'mean'),
            Std_SSIM =('SSIM', 'std'),
            Count    =('MSE', 'count')
        )
        .round(4)
        .reset_index()
    )
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary table saved: {SUMMARY_CSV.name} (Rows: {len(df_summary)})")
    assert len(df_summary) == 448, f"Expected 448 summary rows, got {len(df_summary)}"

    # 3. Degraded test image count verification
    deg_test_files = list(DEG_TEST_BASE.rglob("*.png"))
    print(f"\nDegraded Test Images Generated on Disk: {len(deg_test_files):,} / 44,800 expected")
    assert len(deg_test_files) == 44800, f"Expected 44,800 files, found {len(deg_test_files)}"
    print("[PASS] All 44,800 degraded test images verified on disk.")

    # 4. Sample test restored images verification
    sample_files = list(SAMPLE_DIR.rglob("*.png"))
    print(f"Sample Test Restored Images on Disk:   {len(sample_files)} / 448 expected")
    assert len(sample_files) == 448, f"Expected 448 sample files, found {len(sample_files)}"
    print("[PASS] All 448 sample restored test images present and verified.")

    print("\nPhase 8 Testing Evaluation is 100% COMPLETE and VERIFIED.")


if __name__ == "__main__":
    main()
