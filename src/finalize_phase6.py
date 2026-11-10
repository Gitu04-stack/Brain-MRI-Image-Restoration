"""Finalize Phase 6: Compile Master CSV, Summary Table, and Visualizations."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
METRICS_DIR  = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR  = PROJECT_ROOT / "outputs"
TRAIN_BASE   = PROJECT_ROOT / "dataset" / "archive" / "Training"
DEGRADED_DIR = PROJECT_ROOT / "dataset" / "degraded"

print("=" * 70)
print("COMPILING PHASE 6 MASTER DATASETS & ARTIFACTS")
print("=" * 70)

# 1. Compile Master CSV
csv_files = [
    METRICS_DIR / "phase6_gaussian_metrics.csv",
    METRICS_DIR / "phase6_rician_metrics.csv",
    METRICS_DIR / "phase6_low_resolution_metrics.csv",
    METRICS_DIR / "phase6_motion_blur_metrics.csv"
]

dfs = []
for p in csv_files:
    print(f"Loading {p.name} ...")
    df_part = pd.read_csv(p)
    assert len(df_part) == 156800, f"Unexpected row count in {p.name}: {len(df_part)}"
    dfs.append(df_part)

df_master = pd.concat(dfs, ignore_index=True)
master_csv = METRICS_DIR / "phase6_restoration_metrics_master.csv"
df_master.to_csv(master_csv, index=False)
print(f"Master CSV saved: {master_csv.name} | Total rows: {len(df_master):,}")
assert len(df_master) == 627200, f"Expected 627,200 rows, got {len(df_master)}"

# 2. Compute Summary Statistics (mean ± std per deg, cat, sev, method)
print("\nComputing summary statistics (448 groups) ...")
df_summary = (
    df_master.groupby(['degradation', 'category', 'severity', 'method'])
    .agg(
        mean_mse=('mse', 'mean'),
        std_mse=('mse', 'std'),
        mean_psnr=('psnr', 'mean'),
        std_psnr=('psnr', 'std'),
        mean_ssim=('ssim', 'mean'),
        std_ssim=('ssim', 'std'),
        n_images=('mse', 'count')
    )
    .reset_index()
)
summary_csv = METRICS_DIR / "phase6_restoration_summary.csv"
df_summary.to_csv(summary_csv, index=False)
print(f"Summary CSV saved: {summary_csv.name} | Total rows: {len(df_summary)}")
assert len(df_summary) == 448, f"Expected 448 rows, got {len(df_summary)}"

# 3. Generate PSNR & SSIM vs Severity Plot
print("\nGenerating PSNR and SSIM comparison figure ...")
METHOD_LABELS = {
    'median_filter': 'Median Filter',
    'nlm': 'NLM',
    'wiener': 'Wiener Filter',
    'lanczos_sharpening': 'Lanczos+Sharp',
}
METHOD_COLORS = {
    'median_filter': '#1f77b4',
    'nlm': '#ff7f0e',
    'wiener': '#2ca02c',
    'lanczos_sharpening': '#d62728',
}
DEGRADATION_LABELS = {
    'gaussian': 'Gaussian Noise',
    'rician': 'Rician Noise',
    'low_resolution': 'Low Resolution',
    'motion_blur': 'Motion Blur',
}
SEVERITY_LEVELS = [10, 20, 30, 40, 50, 60, 70]
METHODS = ['median_filter', 'nlm', 'wiener', 'lanczos_sharpening']

fig, axes = plt.subplots(4, 2, figsize=(16, 22))
fig.suptitle(
    "Phase 6: Traditional Restoration Methods Comparison\n"
    "Mean PSNR (dB) and SSIM across All Categories & Severity Levels",
    fontsize=14, fontweight='bold', y=0.995
)

for row_idx, deg in enumerate(["gaussian", "rician", "low_resolution", "motion_blur"]):
    df_deg = df_summary[df_summary['degradation'] == deg]
    df_cat_avg = (
        df_deg.groupby(['severity', 'method'])[['mean_psnr', 'mean_ssim']]
        .mean()
        .reset_index()
    )

    for col_idx, metric in enumerate(['mean_psnr', 'mean_ssim']):
        ax = axes[row_idx, col_idx]
        for method in METHODS:
            sub = df_cat_avg[df_cat_avg['method'] == method].sort_values('severity')
            ax.plot(
                sub['severity'], sub[metric],
                marker='o', linewidth=2, markersize=6,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method]
            )
        ax.set_title(f"{DEGRADATION_LABELS[deg]} — {'PSNR (dB)' if metric == 'mean_psnr' else 'SSIM'}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Severity (%)")
        ax.set_ylabel("PSNR (dB)" if metric == 'mean_psnr' else "SSIM")
        ax.set_xticks(SEVERITY_LEVELS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right' if metric == 'mean_ssim' else 'lower left')

plt.tight_layout()
fig_path = FIGURES_DIR / "phase6_restoration_psnr_ssim_all.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Comparison figure saved: {fig_path.name}")

# 4. Generate Visual Sample Comparison Grid
print("\nGenerating sample visual restoration grid ...")
SAMPLE_CAT = "glioma"
SAMPLE_SEV = 50
clean_imgs = sorted([p for p in (TRAIN_BASE / SAMPLE_CAT).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
clean_sample_p = clean_imgs[0]

from src.preprocessing import normalize_image
from src.restoration import apply_all_methods

with Image.open(clean_sample_p) as img:
    clean_arr = normalize_image(img, target_size=(512, 512))

fig, axes = plt.subplots(4, 6, figsize=(22, 16))
fig.suptitle(
    f"Phase 6 Sample Visual Restorations\nCategory: {SAMPLE_CAT.title()} | Severity: {SAMPLE_SEV}%",
    fontsize=14, fontweight='bold'
)

DEG_CONFIG = {
    'gaussian':       (DEGRADED_DIR / 'gaussian',       f"{clean_sample_p.stem}_gaussian_{SAMPLE_SEV}.png"),
    'rician':         (DEGRADED_DIR / 'rician',         f"{clean_sample_p.stem}_rician_{SAMPLE_SEV}.png"),
    'low_resolution': (DEGRADED_DIR / 'low_resolution', f"{clean_sample_p.stem}_lowres_{SAMPLE_SEV}.png"),
    'motion_blur':    (DEGRADED_DIR / 'motion_blur',    f"{clean_sample_p.stem}_motion_{SAMPLE_SEV}.png"),
}

for row_idx, (deg_name, (deg_dir, deg_file)) in enumerate(DEG_CONFIG.items()):
    deg_p = deg_dir / SAMPLE_CAT / str(SAMPLE_SEV) / deg_file
    with Image.open(deg_p) as img:
        deg_arr = normalize_image(img, target_size=(512, 512))
    restored = apply_all_methods(deg_arr)

    imgs   = [clean_arr, deg_arr] + [restored[m] for m in METHODS]
    titles = ['Clean Original', f"{DEGRADATION_LABELS[deg_name]} ({SAMPLE_SEV}%)"] + [METHOD_LABELS[m] for m in METHODS]

    for col_idx, (arr, title) in enumerate(zip(imgs, titles)):
        ax = axes[row_idx, col_idx]
        ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
fig_samples_path = FIGURES_DIR / "phase6_sample_restorations.png"
fig.savefig(fig_samples_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Sample restorations figure saved: {fig_samples_path.name}")

print("\n" + "=" * 70)
print("PHASE 6 ALL DELIVERABLES GENERATED SUCCESSFULLY!")
print("=" * 70)
