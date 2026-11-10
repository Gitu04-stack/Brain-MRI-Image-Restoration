"""Resilient and Resumable Motion Blur Restoration Runner for Phase 6.

Processes Motion Blur degraded images (39,200 images total across 4 categories x 7 severities)
using 4 restoration methods (Median, NLM, Wiener, Lanczos+Sharpening) = 156,800 evaluations.

Key Resilience Features:
1. Resumability: Automatically detects already completed (category, severity) combinations
   in outputs/metrics/phase6_motion_blur_metrics.csv and skips them.
2. Incremental Flush: Appends and flushes metrics to disk after each (category, severity) batch.
3. Multi-worker Optimization: Uses chunked multiprocessing to eliminate IPC serialization bottlenecks.
4. Data Integrity: Verifies row counts, checks for nulls/errors, and confirms sample images.
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# Setup project root
PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.restoration import run_worker_restore, RESTORATION_METHODS

TRAIN_BASE_DIR = PROJECT_ROOT / "dataset" / "archive" / "Training"
MOTION_DEG_DIR = PROJECT_ROOT / "dataset" / "degraded" / "motion_blur"
METRICS_DIR    = PROJECT_ROOT / "outputs" / "metrics"
SAMPLE_DIR     = PROJECT_ROOT / "outputs" / "samples" / "restored"
OUTPUT_CSV     = METRICS_DIR / "phase6_motion_blur_metrics.csv"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES       = ["glioma", "meningioma", "notumor", "pituitary"]
SEVERITY_LEVELS  = [10, 20, 30, 40, 50, 60, 70]
CLEAN_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NUM_WORKERS      = 10
CHUNK_SIZE       = 25

EXPECTED_PER_COMBO = 1400 * len(RESTORATION_METHODS)  # 5,600 rows per (category, severity)
TOTAL_EXPECTED     = len(CATEGORIES) * len(SEVERITY_LEVELS) * EXPECTED_PER_COMBO  # 156,800 rows


def get_completed_combos() -> dict:
    """Returns dict of (category, severity) -> row_count for completed batches."""
    if not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(OUTPUT_CSV)
        if df.empty or 'category' not in df.columns or 'severity' not in df.columns:
            return {}
        counts = df.groupby(['category', 'severity']).size().to_dict()
        return counts
    except Exception as e:
        print(f"[WARN] Error reading existing CSV: {e}")
        return {}


def main():
    print("=" * 70)
    print("PHASE 6: RESUMABLE MOTION BLUR RESTORATION RUNNER")
    print("=" * 70)
    print(f"Output CSV:     {OUTPUT_CSV}")
    print(f"Workers:        {NUM_WORKERS}")
    print(f"Categories:     {CATEGORIES}")
    print(f"Severities:     {SEVERITY_LEVELS}")
    print(f"Methods:        {RESTORATION_METHODS}")
    print(f"Total Expected: {TOTAL_EXPECTED} evaluation rows")
    print("-" * 70)

    completed_counts = get_completed_combos()
    fully_done_combos = {
        combo for combo, count in completed_counts.items()
        if count == EXPECTED_PER_COMBO
    }
    
    if fully_done_combos:
        print(f"Found {len(fully_done_combos)} / 28 combinations already fully completed in CSV.")
        for cat, sev in sorted(fully_done_combos):
            print(f"  [ALREADY DONE] {cat} @ {sev}% ({completed_counts[(cat, sev)]} rows)")
    else:
        print("Starting Motion Blur restoration from initial combination.")

    start_total_time = time.time()
    combos_processed = 0

    for category in CATEGORIES:
        clean_dir = TRAIN_BASE_DIR / category
        clean_paths = sorted([
            p for p in clean_dir.iterdir()
            if p.suffix.lower() in CLEAN_EXTENSIONS
        ])
        
        if len(clean_paths) != 1400:
            print(f"[ERROR] Expected 1,400 clean images in {clean_dir}, found {len(clean_paths)}!")
            sys.exit(1)

        for severity in SEVERITY_LEVELS:
            if (category, severity) in fully_done_combos:
                continue

            print(f"\n>>> Processing: Category = {category.upper()} | Severity = {severity}% ...")
            t_combo_start = time.time()

            # Build 1,400 work items for this combination
            deg_dir = MOTION_DEG_DIR / category / str(severity)
            work_items = []
            for idx, clean_p in enumerate(clean_paths):
                deg_path = deg_dir / f"{clean_p.stem}_motion_{severity}.png"
                if not deg_path.exists():
                    print(f"[ERROR] Missing degraded file: {deg_path}")
                    sys.exit(1)
                work_items.append((
                    str(deg_path),
                    str(clean_p),
                    "motion_blur",
                    category,
                    severity,
                    idx,
                    str(SAMPLE_DIR),
                ))

            combo_results = []
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                for res_list in executor.map(run_worker_restore, work_items, chunksize=CHUNK_SIZE):
                    combo_results.extend(res_list)

            # Verification of current combo batch
            if len(combo_results) != EXPECTED_PER_COMBO:
                print(f"[ERROR] Expected {EXPECTED_PER_COMBO} results for {category} {severity}%, got {len(combo_results)}")
                sys.exit(1)

            df_combo = pd.DataFrame(combo_results)
            
            # Check for NaN / computation errors
            nan_count = df_combo[['mse', 'psnr', 'ssim']].isna().sum().sum()
            if nan_count > 0:
                print(f"[WARN] {nan_count} NaN values encountered in {category} {severity}% metrics!")

            # Append incrementally to disk and flush
            file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
            df_combo.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)

            combos_processed += 1
            t_combo_elapsed = time.time() - t_combo_start
            
            # Current cumulative count
            current_total = len(pd.read_csv(OUTPUT_CSV))
            print(f"  [SAVED] {category} @ {severity}% -> {len(df_combo)} rows appended in {t_combo_elapsed:.1f}s ({t_combo_elapsed/60:.2f} min).")
            print(f"  Cumulative CSV Rows: {current_total} / {TOTAL_EXPECTED} ({current_total/TOTAL_EXPECTED*100:.1f}%)")

    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"EXECUTION FINISHED in {total_elapsed/60:.1f} minutes.")
    print("=" * 70)

    # Final Comprehensive Verification
    print("\n--- FINAL VERIFICATION OF OUTPUT CSV ---")
    df_final = pd.read_csv(OUTPUT_CSV)
    print(f"Total Rows:     {len(df_final)} (Expected: {TOTAL_EXPECTED})")
    
    assert len(df_final) == TOTAL_EXPECTED, f"Row count mismatch! Expected {TOTAL_EXPECTED}, got {len(df_final)}"
    print("[PASS] Row count matches exactly 156,800.")

    null_counts = df_final.isnull().sum()
    print("Null value counts per column:")
    print(null_counts.to_string())
    assert null_counts.sum() == 0, "Null values detected in metrics CSV!"
    print("[PASS] Zero null or NaN values in CSV.")

    print("\nCounts by Category:")
    print(df_final['category'].value_counts().to_string())

    print("\nCounts by Severity:")
    print(df_final['severity'].value_counts().to_string())

    print("\nCounts by Method:")
    print(df_final['method'].value_counts().to_string())

    # Sample images check (checking by directory path)
    sample_files = list(SAMPLE_DIR.glob("*/motion_blur/*/*/*.png"))
    print(f"\nMotion Blur Sample Images on Disk: {len(sample_files)} / 112 expected")
    assert len(sample_files) == 112, f"Expected 112 sample files, found {len(sample_files)}"
    print("[PASS] All 112 Motion Blur sample images present and verified.")

    print("\nPhase 6 Motion Blur Restoration is 100% COMPLETE and VERIFIED.")


if __name__ == "__main__":
    main()
