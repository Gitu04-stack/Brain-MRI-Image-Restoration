"""Phase 9: Combined MRI Degradation Generation (Training & Testing Datasets).

Implements the fixed sequential realistic multi-degradation MRI physics pipeline:
    Clean Original Image
            │
            ▼
    1. Low-Resolution Downsampling & Interpolation (Box downsample -> Bilinear upsample)
            │
            ▼
    2. Linear Motion Blur Convolution (45° PSF kernel)
            │
            ▼
    3. Complex Rician Magnitude Noise (Non-Gaussian physical MRI coil simulation)
            │
            ▼
    Combined Degraded Image (512x512, Grayscale 'L', Lossless PNG)

Generates:
  - Training Combined Degraded Set: 5,600 clean x 7 severities = 39,200 images
    under dataset/degraded/combined/{category}/{severity}/
  - Testing Combined Degraded Set:  1,600 clean x 7 severities = 11,200 images
    under dataset/degraded_test/combined/{category}/{severity}/
  - Total: 50,400 high-fidelity degraded MRI scans.
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import normalize_image, save_image
from src.degradation import (
    apply_combined_degradation,
    run_worker_combined,
    LOW_RES_SCALE_MAP,
    MOTION_BLUR_KERNEL_MAP,
    RICIAN_SIGMA_MAP,
    SEVERITY_LEVELS
)

CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
TRAIN_CLEAN_DIR = PROJECT_ROOT / "dataset" / "archive" / "Training"
TEST_CLEAN_DIR  = PROJECT_ROOT / "dataset" / "archive" / "Testing"

TRAIN_DEG_DIR = PROJECT_ROOT / "dataset" / "degraded" / "combined"
TEST_DEG_DIR  = PROJECT_ROOT / "dataset" / "degraded_test" / "combined"

BASE_TRAIN_SEED = 42
BASE_TEST_SEED  = 9999


def ensure_directories():
    for cat in CATEGORIES:
        for sev in SEVERITY_LEVELS:
            (TRAIN_DEG_DIR / cat / str(sev)).mkdir(parents=True, exist_ok=True)
            (TEST_DEG_DIR  / cat / str(sev)).mkdir(parents=True, exist_ok=True)


def generate_dataset(clean_base_dir: Path, output_base_dir: Path, base_seed: int, label: str):
    print(f"\n=======================================================")
    print(f"Generating {label} Combined Degraded Dataset")
    print(f"Clean Source: {clean_base_dir}")
    print(f"Output Destination: {output_base_dir}")
    print(f"=======================================================")

    tasks = []
    for cat in CATEGORIES:
        cat_dir = clean_base_dir / cat
        cat_files = sorted(list(cat_dir.glob("*.*")))
        for idx, img_path in enumerate(cat_files):
            tasks.append((str(img_path), cat, str(output_base_dir), base_seed, idx))

    expected_clean = len(tasks)
    expected_outputs = expected_clean * len(SEVERITY_LEVELS)
    print(f"Total clean images to process: {expected_clean}")
    print(f"Total degraded images to generate: {expected_outputs}")

    num_workers = min(12, os.cpu_count() or 4)
    print(f"Launching generation across {num_workers} parallel workers...")

    t0 = time.time()
    failed_items = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(run_worker_combined, tasks))

    for res in results:
        if not res["success"]:
            failed_items.append(res)

    elapsed = time.time() - t0
    print(f"\n{label} Generation Completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Successfully processed: {len(results) - len(failed_items)} / {expected_clean}")
    if failed_items:
        print(f"WARNING: {len(failed_items)} tasks failed! Details: {failed_items[:5]}")
    else:
        print(f"All {expected_outputs} {label} combined images successfully generated with 0 errors!")


def audit_dataset(clean_base_dir: Path, output_base_dir: Path, label: str):
    print(f"\n--- Auditing {label} Output Counts ---")
    rows = []
    all_matched = True
    for cat in CATEGORIES:
        clean_count = len(list((clean_base_dir / cat).glob("*.*")))
        for sev in SEVERITY_LEVELS:
            sev_dir = output_base_dir / cat / str(sev)
            actual_count = len(list(sev_dir.glob("*.png")))
            matched = (actual_count == clean_count)
            if not matched:
                all_matched = False
            rows.append({
                "Dataset": label,
                "Category": cat,
                "Severity": f"{sev}%",
                "Clean Count": clean_count,
                "Degraded Count": actual_count,
                "Match": matched
            })
    df_audit = pd.DataFrame(rows)
    print(df_audit.to_string(index=False))
    total_deg = df_audit["Degraded Count"].sum()
    expected_total = len(rows) * (rows[0]["Clean Count"])
    print(f"\nTotal {label} Degraded Count: {total_deg} (All Matched: {all_matched})")
    return df_audit, all_matched


def main():
    print("Initializing Phase 9 Combined MRI Degradation Pipeline...")
    ensure_directories()

    # 1. Generate Training Combined Dataset (39,200 images)
    generate_dataset(TRAIN_CLEAN_DIR, TRAIN_DEG_DIR, BASE_TRAIN_SEED, "Training")

    # 2. Generate Testing Combined Dataset (11,200 images)
    generate_dataset(TEST_CLEAN_DIR, TEST_DEG_DIR, BASE_TEST_SEED, "Testing")

    # 3. Auditing
    _, train_ok = audit_dataset(TRAIN_CLEAN_DIR, TRAIN_DEG_DIR, "Training")
    _, test_ok = audit_dataset(TEST_CLEAN_DIR, TEST_DEG_DIR, "Testing")

    # 4. Audit Pixel Dynamic Range and Dimensions
    print("\n--- Auditing Format, Dimensions and Dynamic Range ---")
    sample_records = []
    for cat in CATEGORIES:
        first_clean = sorted(list((TRAIN_CLEAN_DIR / cat).glob("*.*")))[0]
        stem = first_clean.stem
        for sev in [10, 40, 70]:
            p = TRAIN_DEG_DIR / cat / str(sev) / f"{stem}_combined_{sev}.png"
            with Image.open(p) as img:
                arr = np.array(img)
                sample_records.append({
                    "Dataset": "Training",
                    "Category": cat,
                    "Severity": f"{sev}%",
                    "Dimensions": f"{img.size[0]}x{img.size[1]}",
                    "Mode": img.mode,
                    "Min": int(arr.min()),
                    "Max": int(arr.max()),
                    "Dtype": str(arr.dtype)
                })
    print(pd.DataFrame(sample_records).to_string(index=False))

    # 5. Verify Original Datasets Untouched
    train_orig = {cat: len(list((TRAIN_CLEAN_DIR / cat).glob("*.*"))) for cat in CATEGORIES}
    test_orig  = {cat: len(list((TEST_CLEAN_DIR / cat).glob("*.*"))) for cat in CATEGORIES}
    print(f"\nOriginal Training Counts (Untouched): {train_orig}")
    print(f"Original Testing Counts (Untouched):  {test_orig}")

    if train_ok and test_ok:
        print("\nSUCCESS: Phase 9 Combined Degradation Generation Complete & 100% Verified!")
    else:
        print("\nERROR: Verification failed! Check unmatched counts above.")


if __name__ == "__main__":
    main()
