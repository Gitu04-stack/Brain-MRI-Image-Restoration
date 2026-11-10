"""Phase 10: Combined Traditional MRI Restoration Runner (Training & Testing Sets).

Applies 4 traditional image restoration filters:
  1. Median Filter (3x3 spatial rank filter)
  2. Non-Local Means (NLM) (patch-similarity denoiser)
  3. Wiener Filter (adaptive local minimum-MSE filter, 5x5)
  4. Lanczos Sharpening (unsharp masking with Lanczos parameters)

Evaluates on:
  - Training Combined Dataset: 39,200 images x 4 methods = 156,800 evaluation rows
  - Testing Combined Dataset:  11,200 images x 4 methods = 44,800 evaluation rows
  - Total Evaluations: 201,600 records

Key Architecture & Safety:
  1. Full Resumability: Checks completed (category, severity) buckets and skips them.
  2. Incremental Flush: Appends and flushes to disk after every single batch.
  3. Multiprocessing: Parallel chunked worker execution.
  4. Non-Destructive: Original clean scans and Phase 1-9 files are read-only and preserved.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase10_worker import process_phase10_image
from src.restoration import RESTORATION_METHODS

CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
SEVERITY_LEVELS = [10, 20, 30, 40, 50, 60, 70]
CLEAN_EXTS = {".jpg", ".jpeg", ".png"}

TRAIN_CLEAN_DIR = PROJECT_ROOT / "dataset" / "archive" / "Training"
TEST_CLEAN_DIR  = PROJECT_ROOT / "dataset" / "archive" / "Testing"

TRAIN_DEG_DIR = PROJECT_ROOT / "dataset" / "degraded" / "combined"
TEST_DEG_DIR  = PROJECT_ROOT / "dataset" / "degraded_test" / "combined"

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_METRICS_CSV = METRICS_DIR / "phase10_combined_training_metrics.csv"
TRAIN_SUMMARY_CSV = METRICS_DIR / "phase10_combined_training_summary.csv"
TRAIN_SAMPLE_DIR  = PROJECT_ROOT / "outputs" / "samples" / "phase10_restored"
TRAIN_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

TEST_METRICS_CSV = METRICS_DIR / "phase10_combined_testing_metrics.csv"
TEST_SUMMARY_CSV = METRICS_DIR / "phase10_combined_testing_summary.csv"
TEST_SAMPLE_DIR  = PROJECT_ROOT / "outputs" / "samples" / "phase10_test_restored"
TEST_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def get_completed_buckets(csv_path: Path) -> dict:
    """Returns dict of (category, severity) -> row_count for completed batches."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'category' not in df.columns or 'severity' not in df.columns:
            return {}
        return df.groupby(['category', 'severity']).size().to_dict()
    except Exception as e:
        print(f"[WARN] Error reading {csv_path.name}: {e}")
        return {}


def append_records_to_csv(records: list, csv_path: Path):
    """Appends records safely to CSV and flushes to disk."""
    if not records:
        return
    df = pd.DataFrame(records)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        df.to_csv(f, header=write_header, index=False)
        f.flush()
        os.fsync(f.fileno())


def generate_summary(metrics_csv: Path, summary_csv: Path):
    """Generates mean MSE, PSNR, SSIM summary grouped by category, severity, method."""
    print(f"Generating summary from {metrics_csv.name}...")
    df = pd.read_csv(metrics_csv)
    summary = (
        df.groupby(['category', 'severity', 'restoration_method'])
        .agg(
            mean_mse=('MSE', 'mean'),
            std_mse=('MSE', 'std'),
            mean_psnr=('PSNR', 'mean'),
            std_psnr=('PSNR', 'std'),
            mean_ssim=('SSIM', 'mean'),
            std_ssim=('SSIM', 'std'),
            count=('MSE', 'count')
        )
        .reset_index()
    )
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary ({len(summary)} rows) to: {summary_csv}")


def run_restoration_dataset(clean_base: Path, deg_base: Path, metrics_csv: Path,
                            summary_csv: Path, sample_dir: Path, dataset_name: str,
                            expected_clean_per_cat: int):
    expected_rows_per_bucket = expected_clean_per_cat * len(RESTORATION_METHODS)
    total_expected = len(CATEGORIES) * len(SEVERITY_LEVELS) * expected_rows_per_bucket

    print("\n" + "=" * 75)
    print(f"PHASE 10: COMBINED RESTORATION — {dataset_name.upper()} DATASET")
    print("=" * 75)
    print(f"Degraded Input Base: {deg_base}")
    print(f"Clean Reference Base: {clean_base}")
    print(f"Output Metrics CSV:   {metrics_csv}")
    print(f"Output Samples:       {sample_dir}")
    print(f"Total Expected Rows:  {total_expected:,}")
    print("-" * 75)

    completed_counts = get_completed_buckets(metrics_csv)
    fully_done = {
        combo for combo, cnt in completed_counts.items()
        if cnt == expected_rows_per_bucket
    }

    if fully_done:
        print(f"Found {len(fully_done)} / 28 buckets already completed in {metrics_csv.name}.")
        for cat, sev in sorted(fully_done):
            print(f"  [ALREADY DONE] {cat} @ {sev}% ({completed_counts[(cat, sev)]:,} rows)")
    else:
        print(f"Starting {dataset_name} restoration from initial combination.")

    num_workers = min(12, os.cpu_count() or 4)
    print(f"Using {num_workers} parallel workers with chunksize=25.")

    total_t0 = time.time()
    total_processed_combos = len(fully_done)

    for category in CATEGORIES:
        clean_cat_dir = clean_base / category
        clean_files = sorted([p for p in clean_cat_dir.iterdir() if p.suffix.lower() in CLEAN_EXTS])

        if len(clean_files) != expected_clean_per_cat:
            print(f"[ERROR] Expected {expected_clean_per_cat} images in {clean_cat_dir}, found {len(clean_files)}!")
            sys.exit(1)

        for severity in SEVERITY_LEVELS:
            if (category, severity) in fully_done:
                continue

            t_bucket0 = time.time()
            deg_cat_sev_dir = deg_base / category / str(severity)
            tasks = []

            for idx, clean_p in enumerate(clean_files):
                stem = clean_p.stem
                deg_p = deg_cat_sev_dir / f"{stem}_combined_{severity}.png"
                if not deg_p.exists():
                    print(f"[ERROR] Missing degraded file: {deg_p}")
                    sys.exit(1)
                tasks.append((str(deg_p), str(clean_p), category, dataset_name, severity, idx, str(sample_dir)))

            bucket_records = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for res_list in executor.map(process_phase10_image, tasks, chunksize=25):
                    bucket_records.extend(res_list)

            # Flush to disk
            append_records_to_csv(bucket_records, metrics_csv)
            t_bucket_elapsed = time.time() - t_bucket0
            total_processed_combos += 1

            # Summary stats for this bucket
            df_bucket = pd.DataFrame(bucket_records)
            psnr_means = df_bucket.groupby('restoration_method')['PSNR'].mean().to_dict()
            psnr_str = " | ".join([f"{m[:6]}:{v:.2f}dB" for m, v in sorted(psnr_means.items())])

            print(
                f"[{total_processed_combos:02d}/28] COMPLETED {category:<11} @ {severity}% "
                f"({len(bucket_records)} rows in {t_bucket_elapsed:.1f}s) -> Mean PSNR: {psnr_str}"
            )

    total_time = time.time() - total_t0
    print(f"\n{dataset_name} Restoration Completed in {total_time:.2f}s ({total_time/60:.2f} min)")

    # Audit CSV
    df_final = pd.read_csv(metrics_csv)
    print(f"Final {dataset_name} CSV Row Count: {len(df_final):,} / {total_expected:,} expected")
    if len(df_final) == total_expected:
        print(f"[VERIFIED] All {total_expected:,} rows present with 0 errors!")
    else:
        print(f"[WARNING] Row count mismatch! Expected {total_expected:,}, found {len(df_final):,}")

    # Generate summary CSV
    generate_summary(metrics_csv, summary_csv)


def main():
    print("Initializing Phase 10: Combined Traditional MRI Restoration...")
    
    # 1. Process Training Combined Dataset (156,800 rows)
    run_restoration_dataset(
        clean_base=TRAIN_CLEAN_DIR,
        deg_base=TRAIN_DEG_DIR,
        metrics_csv=TRAIN_METRICS_CSV,
        summary_csv=TRAIN_SUMMARY_CSV,
        sample_dir=TRAIN_SAMPLE_DIR,
        dataset_name="Training",
        expected_clean_per_cat=1400
    )

    # 2. Process Testing Combined Dataset (44,800 rows)
    run_restoration_dataset(
        clean_base=TEST_CLEAN_DIR,
        deg_base=TEST_DEG_DIR,
        metrics_csv=TEST_METRICS_CSV,
        summary_csv=TEST_SUMMARY_CSV,
        sample_dir=TEST_SAMPLE_DIR,
        dataset_name="Testing",
        expected_clean_per_cat=400
    )

    print("\n=======================================================")
    print("PHASE 10 RESTORATION EXECUTION FINISHED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
