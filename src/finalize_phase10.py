"""Finalize Phase 10: Combined Traditional MRI Restoration.
- Generates high-resolution performance comparison figures (PSNR & SSIM curves, sample restorations).
- Constructs and executes notebooks/10_combined_restoration.ipynb.
- Mirrors the executed notebook to OneDrive workspace.
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
ONEDRIVE_NOTEBOOKS = Path(r"C:\Users\gitan\OneDrive\Documents\Brain_MRI_Restoration\notebooks")
ARTIFACTS_DIR = Path(r"C:\Users\gitan\.gemini\antigravity\brain\57c45525-48a2-454d-9673-64c1e8e45b1e")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
SEVERITY_LEVELS = [10, 20, 30, 40, 50, 60, 70]
METHODS = ["median_filter", "nlm", "wiener", "lanczos_sharpening"]
METHOD_LABELS = {
    "median_filter": "Median Filter (3x3)",
    "nlm": "Non-Local Means (NLM)",
    "wiener": "Wiener Filter (5x5)",
    "lanczos_sharpening": "Lanczos + Sharpening"
}
METHOD_COLORS = {
    "median_filter": "#1f77b4",
    "nlm": "#2ca02c",
    "wiener": "#ff7f0e",
    "lanczos_sharpening": "#d62728"
}

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
TRAIN_SUMMARY_CSV = METRICS_DIR / "phase10_combined_training_summary.csv"
TEST_SUMMARY_CSV  = METRICS_DIR / "phase10_combined_testing_summary.csv"

TRAIN_CLEAN_DIR  = PROJECT_ROOT / "dataset" / "archive" / "Training"
TRAIN_DEG_DIR    = PROJECT_ROOT / "dataset" / "degraded" / "combined"
TRAIN_SAMPLE_DIR = PROJECT_ROOT / "outputs" / "samples" / "phase10_restored"


def generate_performance_curves():
    print("Generating Phase 10 performance comparison curves (PSNR & SSIM)...")
    df_train = pd.read_csv(TRAIN_SUMMARY_CSV)
    df_test  = pd.read_csv(TEST_SUMMARY_CSV)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.22, hspace=0.30)

    # 1. PSNR vs Severity (Training)
    ax1 = axes[0, 0]
    for method in METHODS:
        sub = df_train[df_train['restoration_method'] == method].groupby('severity')['mean_psnr'].mean()
        ax1.plot(sub.index, sub.values, marker='o', linewidth=2.2, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax1.set_title("Training Set: Mean PSNR vs Severity Level", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Severity Level (%)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("PSNR (dB) — Higher is Better", fontsize=11, fontweight='bold')
    ax1.set_xticks(SEVERITY_LEVELS)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10, loc='upper right')

    # 2. PSNR vs Severity (Testing)
    ax2 = axes[0, 1]
    for method in METHODS:
        sub = df_test[df_test['restoration_method'] == method].groupby('severity')['mean_psnr'].mean()
        ax2.plot(sub.index, sub.values, marker='s', linewidth=2.2, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax2.set_title("Testing Set: Mean PSNR vs Severity Level (Unseen Data)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Severity Level (%)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("PSNR (dB) — Higher is Better", fontsize=11, fontweight='bold')
    ax2.set_xticks(SEVERITY_LEVELS)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10, loc='upper right')

    # 3. SSIM vs Severity (Training)
    ax3 = axes[1, 0]
    for method in METHODS:
        sub = df_train[df_train['restoration_method'] == method].groupby('severity')['mean_ssim'].mean()
        ax3.plot(sub.index, sub.values, marker='o', linewidth=2.2, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax3.set_title("Training Set: Mean SSIM vs Severity Level", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Severity Level (%)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("SSIM (0.0 to 1.0) — Higher is Better", fontsize=11, fontweight='bold')
    ax3.set_xticks(SEVERITY_LEVELS)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(fontsize=10, loc='upper right')

    # 4. SSIM vs Severity (Testing)
    ax4 = axes[1, 1]
    for method in METHODS:
        sub = df_test[df_test['restoration_method'] == method].groupby('severity')['mean_ssim'].mean()
        ax4.plot(sub.index, sub.values, marker='s', linewidth=2.2, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax4.set_title("Testing Set: Mean SSIM vs Severity Level (Unseen Data)", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Severity Level (%)", fontsize=11, fontweight='bold')
    ax4.set_ylabel("SSIM (0.0 to 1.0) — Higher is Better", fontsize=11, fontweight='bold')
    ax4.set_xticks(SEVERITY_LEVELS)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(fontsize=10, loc='upper right')

    fig.suptitle(
        "Phase 10: Comparative Evaluation of Traditional Restoration Methods Under Combined MRI Degradation\n"
        "(Clean -> Low-Resolution -> Motion Blur -> Rician Noise across Training vs. Testing)",
        fontsize=15, fontweight='bold', y=0.98
    )

    out_fig = OUTPUTS_DIR / "phase10_psnr_ssim_comparison.png"
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved performance figure to: {out_fig}")

    if ARTIFACTS_DIR.exists():
        shutil.copy2(out_fig, ARTIFACTS_DIR / "phase10_psnr_ssim_comparison.png")


def generate_sample_restorations():
    print("Generating Phase 10 sample restoration grid...")
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 14))
    plt.subplots_adjust(wspace=0.04, hspace=0.15)

    col_headers = [
        "Clean Original",
        "Combined 40% Degraded",
        "Median Filter (3x3)",
        "Non-Local Means (NLM)",
        "Wiener Filter (5x5)",
        "Lanczos + Sharpening"
    ]

    for row_idx, cat in enumerate(CATEGORIES):
        clean_files = sorted(list((TRAIN_CLEAN_DIR / cat).glob("*.*")))
        clean_p = clean_files[0]
        stem = clean_p.stem

        # Clean
        with Image.open(clean_p) as img:
            clean_arr = np.array(img.convert('L').resize((512, 512)))
        axes[row_idx, 0].imshow(clean_arr, cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 0].set_ylabel(cat.capitalize(), fontsize=14, fontweight='bold')
        axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_headers[0], fontsize=12, fontweight='bold', pad=8)

        # Degraded 40%
        deg_p = TRAIN_DEG_DIR / cat / "40" / f"{stem}_combined_40.png"
        with Image.open(deg_p) as img:
            deg_arr = np.array(img)
        axes[row_idx, 1].imshow(deg_arr, cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 1].set_xticks([]); axes[row_idx, 1].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 1].set_title(col_headers[1], fontsize=12, fontweight='bold', pad=8)

        # 4 Restored Methods
        for col_idx, method in enumerate(METHODS, start=2):
            rest_p = TRAIN_SAMPLE_DIR / method / cat / "40" / f"{stem}_combined_40_restored_{method}.png"
            if rest_p.exists():
                with Image.open(rest_p) as img:
                    rest_arr = np.array(img)
            else:
                rest_arr = np.zeros((512, 512), dtype=np.uint8)
            axes[row_idx, col_idx].imshow(rest_arr, cmap='gray', vmin=0, vmax=255)
            axes[row_idx, col_idx].set_xticks([]); axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_headers[col_idx], fontsize=12, fontweight='bold', pad=8)

    fig.suptitle(
        "Phase 10: Visual Comparison of Restored Brain MRI Scans at 40% Combined Severity\n"
        "(Clean Reference vs. Combined Distortion vs. 4 Traditional Restoration Techniques)",
        fontsize=15, fontweight='bold', y=0.98
    )

    out_fig = OUTPUTS_DIR / "phase10_sample_restorations.png"
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample restorations figure to: {out_fig}")

    if ARTIFACTS_DIR.exists():
        shutil.copy2(out_fig, ARTIFACTS_DIR / "phase10_sample_restorations.png")


def build_and_execute_notebook():
    print("Building 10_combined_restoration.ipynb...")
    nb = new_notebook()

    # Cell 1: Markdown Overview
    c1 = new_markdown_cell(
        "# Phase 10: Comparative Analysis of Traditional Restoration Methods Under Combined MRI Degradation\n\n"
        "## Academic Project Context\n"
        "**Project Title:** *Comparative Analysis of Brain MRI Image Restoration Techniques Under Different Degradation Conditions*\n\n"
        "### Objective\n"
        "In this phase, we conduct rigorous comparative evaluation of four traditional image restoration filters on MRI scans subject to simultaneous multi-degradation:\n"
        "$$\\mathbf{I}_{\\text{clean}} \\xrightarrow{\\text{Low-Res}} \\mathbf{I}_{\\text{LR}} \\xrightarrow{\\text{Motion Blur}} \\mathbf{I}_{\\text{LR+Blur}} \\xrightarrow{\\text{Rician Noise}} \\mathbf{I}_{\\text{combined}}$$\n\n"
        "### Restoration Methods Evaluated:\n"
        "1. **Median Filter (3x3):** Non-linear spatial rank filter targeting impulse and non-Gaussian noise.\n"
        "2. **Non-Local Means (NLM):** Non-local patch redundancy filter exploiting anatomical structural symmetry.\n"
        "3. **Adaptive Wiener Filter (5x5):** Local minimum mean-squared error (MMSE) filter based on local spatial statistics.\n"
        "4. **Lanczos Sharpening:** High-frequency spatial unsharp masking to counteract downsampling and motion blur.\n\n"
        "### Evaluation Protocol & Scale:\n"
        "- **Training Combined Restoration:** 39,200 degraded scans $\\times$ 4 filters = **156,800 evaluation records**\n"
        "- **Testing Combined Restoration:** 11,200 degraded scans $\\times$ 4 filters = **44,800 evaluation records**\n"
        "- **Total Quantitative Records:** **201,600 evaluations** evaluated against corresponding clean original scans using MSE, PSNR, and SSIM.\n"
    )

    # Cell 2: Imports & Configuration
    c2 = new_code_cell(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from PIL import Image\n"
        "import matplotlib.pyplot as plt\n\n"
        "PROJECT_ROOT = Path(r'C:\\Users\\gitan\\BrainMRI_Project')\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "from src.restoration import RESTORATION_METHODS\n\n"
        "METRICS_DIR = PROJECT_ROOT / 'outputs' / 'metrics'\n"
        "TRAIN_METRICS_CSV = METRICS_DIR / 'phase10_combined_training_metrics.csv'\n"
        "TRAIN_SUMMARY_CSV = METRICS_DIR / 'phase10_combined_training_summary.csv'\n"
        "TEST_METRICS_CSV  = METRICS_DIR / 'phase10_combined_testing_metrics.csv'\n"
        "TEST_SUMMARY_CSV  = METRICS_DIR / 'phase10_combined_testing_summary.csv'\n\n"
        "CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary']\n"
        "SEVERITY_LEVELS = [10, 20, 30, 40, 50, 60, 70]\n"
        "METHODS = ['median_filter', 'nlm', 'wiener', 'lanczos_sharpening']\n\n"
        "print('=== Phase 10: Environment and Paths Initialized ===')\n"
        "print(f'Training Metrics CSV: {TRAIN_METRICS_CSV}')\n"
        "print(f'Testing Metrics CSV:  {TEST_METRICS_CSV}')\n"
    )

    # Cell 3: Verification of CSV Records
    c3 = new_code_cell(
        "df_train = pd.read_csv(TRAIN_METRICS_CSV)\n"
        "df_test  = pd.read_csv(TEST_METRICS_CSV)\n\n"
        "print(f'Training Combined Evaluations: {len(df_train):,} / 156,800 expected (Match: {len(df_train) == 156800})')\n"
        "print(f'Testing Combined Evaluations:  {len(df_test):,} / 44,800 expected (Match: {len(df_test) == 44800})')\n"
        "print(f'Total Combined Quantitative Records: {len(df_train) + len(df_test):,} / 201,600 expected')\n\n"
        "# Verify 0 missing values or errors\n"
        "train_nulls = df_train[['MSE', 'PSNR', 'SSIM']].isna().sum().sum()\n"
        "test_nulls  = df_test[['MSE', 'PSNR', 'SSIM']].isna().sum().sum()\n"
        "print(f'Training NaN/Null Count: {train_nulls}')\n"
        "print(f'Testing NaN/Null Count:  {test_nulls}')\n"
    )

    # Cell 4: Overall Performance Table
    c4 = new_code_cell(
        "df_train_sum = pd.read_csv(TRAIN_SUMMARY_CSV)\n"
        "df_test_sum  = pd.read_csv(TEST_SUMMARY_CSV)\n\n"
        "table_train = df_train_sum.groupby('restoration_method').agg({\n"
        "    'mean_psnr': 'mean',\n"
        "    'mean_ssim': 'mean',\n"
        "    'mean_mse':  'mean'\n"
        "}).reset_index()\n"
        "table_train.columns = ['Restoration Method', 'Train Mean PSNR (dB)', 'Train Mean SSIM', 'Train Mean MSE']\n\n"
        "table_test = df_test_sum.groupby('restoration_method').agg({\n"
        "    'mean_psnr': 'mean',\n"
        "    'mean_ssim': 'mean',\n"
        "    'mean_mse':  'mean'\n"
        "}).reset_index()\n"
        "table_test.columns = ['Restoration Method', 'Test Mean PSNR (dB)', 'Test Mean SSIM', 'Test Mean MSE']\n\n"
        "overall = pd.merge(table_train, table_test, on='Restoration Method')\n"
        "print('=== Phase 10: Overall Restoration Performance on Combined Multi-Degradation ===')\n"
        "print(overall.to_string(index=False))\n"
    )

    # Cell 5: PSNR / SSIM Curves Plot
    c5 = new_code_cell(
        "fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))\n"
        "plt.subplots_adjust(wspace=0.22, hspace=0.30)\n\n"
        "colors = {'median_filter': '#1f77b4', 'nlm': '#2ca02c', 'wiener': '#ff7f0e', 'lanczos_sharpening': '#d62728'}\n"
        "labels = {'median_filter': 'Median Filter (3x3)', 'nlm': 'Non-Local Means', 'wiener': 'Wiener Filter (5x5)', 'lanczos_sharpening': 'Lanczos + Sharpening'}\n\n"
        "# 1. PSNR Train\n"
        "for m in METHODS:\n"
        "    s = df_train_sum[df_train_sum['restoration_method'] == m].groupby('severity')['mean_psnr'].mean()\n"
        "    axes[0, 0].plot(s.index, s.values, marker='o', linewidth=2, label=labels[m], color=colors[m])\n"
        "axes[0, 0].set_title('Training: PSNR vs Severity', fontweight='bold')\n"
        "axes[0, 0].set_xlabel('Severity (%)'); axes[0, 0].set_ylabel('PSNR (dB)'); axes[0, 0].grid(True, linestyle='--'); axes[0, 0].legend()\n\n"
        "# 2. PSNR Test\n"
        "for m in METHODS:\n"
        "    s = df_test_sum[df_test_sum['restoration_method'] == m].groupby('severity')['mean_psnr'].mean()\n"
        "    axes[0, 1].plot(s.index, s.values, marker='s', linewidth=2, label=labels[m], color=colors[m])\n"
        "axes[0, 1].set_title('Testing (Unseen): PSNR vs Severity', fontweight='bold')\n"
        "axes[0, 1].set_xlabel('Severity (%)'); axes[0, 1].set_ylabel('PSNR (dB)'); axes[0, 1].grid(True, linestyle='--'); axes[0, 1].legend()\n\n"
        "# 3. SSIM Train\n"
        "for m in METHODS:\n"
        "    s = df_train_sum[df_train_sum['restoration_method'] == m].groupby('severity')['mean_ssim'].mean()\n"
        "    axes[1, 0].plot(s.index, s.values, marker='o', linewidth=2, label=labels[m], color=colors[m])\n"
        "axes[1, 0].set_title('Training: SSIM vs Severity', fontweight='bold')\n"
        "axes[1, 0].set_xlabel('Severity (%)'); axes[1, 0].set_ylabel('SSIM'); axes[1, 0].grid(True, linestyle='--'); axes[1, 0].legend()\n\n"
        "# 4. SSIM Test\n"
        "for m in METHODS:\n"
        "    s = df_test_sum[df_test_sum['restoration_method'] == m].groupby('severity')['mean_ssim'].mean()\n"
        "    axes[1, 1].plot(s.index, s.values, marker='s', linewidth=2, label=labels[m], color=colors[m])\n"
        "axes[1, 1].set_title('Testing (Unseen): SSIM vs Severity', fontweight='bold')\n"
        "axes[1, 1].set_xlabel('Severity (%)'); axes[1, 1].set_ylabel('SSIM'); axes[1, 1].grid(True, linestyle='--'); axes[1, 1].legend()\n\n"
        "fig.suptitle('Phase 10: Quantitative Restoration Performance under Combined Degradation', fontsize=15, fontweight='bold', y=0.98)\n"
        "plt.show()\n"
    )

    # Cell 6: Visual Inspection of Restored Images
    c6 = new_code_cell(
        "TRAIN_CLEAN_DIR  = PROJECT_ROOT / 'dataset' / 'archive' / 'Training'\n"
        "TRAIN_DEG_DIR    = PROJECT_ROOT / 'dataset' / 'degraded' / 'combined'\n"
        "TRAIN_SAMPLE_DIR = PROJECT_ROOT / 'outputs' / 'samples' / 'phase10_restored'\n\n"
        "fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 14))\n"
        "plt.subplots_adjust(wspace=0.04, hspace=0.15)\n"
        "col_headers = ['Clean Original', 'Combined 40%', 'Median (3x3)', 'NLM', 'Wiener (5x5)', 'Lanczos+Sharp']\n\n"
        "for row_idx, cat in enumerate(CATEGORIES):\n"
        "    clean_files = sorted(list((TRAIN_CLEAN_DIR / cat).glob('*.*')))\n"
        "    clean_p = clean_files[0]\n"
        "    stem = clean_p.stem\n\n"
        "    # Clean\n"
        "    with Image.open(clean_p) as img:\n"
        "        clean_arr = np.array(img.convert('L').resize((512, 512)))\n"
        "    axes[row_idx, 0].imshow(clean_arr, cmap='gray', vmin=0, vmax=255)\n"
        "    axes[row_idx, 0].set_ylabel(cat.capitalize(), fontsize=14, fontweight='bold')\n"
        "    axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])\n"
        "    if row_idx == 0: axes[row_idx, 0].set_title(col_headers[0], fontsize=12, fontweight='bold', pad=8)\n\n"
        "    # Degraded 40%\n"
        "    deg_p = TRAIN_DEG_DIR / cat / '40' / f'{stem}_combined_40.png'\n"
        "    with Image.open(deg_p) as img: deg_arr = np.array(img)\n"
        "    axes[row_idx, 1].imshow(deg_arr, cmap='gray', vmin=0, vmax=255)\n"
        "    axes[row_idx, 1].set_xticks([]); axes[row_idx, 1].set_yticks([])\n"
        "    if row_idx == 0: axes[row_idx, 1].set_title(col_headers[1], fontsize=12, fontweight='bold', pad=8)\n\n"
        "    # 4 Restored Methods\n"
        "    for col_idx, method in enumerate(METHODS, start=2):\n"
        "        rest_p = TRAIN_SAMPLE_DIR / method / cat / '40' / f'{stem}_combined_40_restored_{method}.png'\n"
        "        with Image.open(rest_p) as img: rest_arr = np.array(img)\n"
        "        axes[row_idx, col_idx].imshow(rest_arr, cmap='gray', vmin=0, vmax=255)\n"
        "        axes[row_idx, col_idx].set_xticks([]); axes[row_idx, col_idx].set_yticks([])\n"
        "        if row_idx == 0: axes[row_idx, col_idx].set_title(col_headers[col_idx], fontsize=12, fontweight='bold', pad=8)\n\n"
        "fig.suptitle('Phase 10: Visual Comparison of Restored Brain MRI Scans at 40% Combined Severity', fontsize=15, fontweight='bold', y=0.98)\n"
        "plt.show()\n"
    )

    nb.cells = [c1, c2, c3, c4, c5, c6]

    nb_path = PROJECT_ROOT / "notebooks" / "10_combined_restoration.ipynb"
    print(f"Writing notebook to {nb_path}...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("Executing notebook to populate all cell outputs...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook executed and saved successfully!")

    # Mirror to OneDrive
    if ONEDRIVE_NOTEBOOKS.exists():
        dest = ONEDRIVE_NOTEBOOKS / "10_combined_restoration.ipynb"
        shutil.copy2(nb_path, dest)
        print(f"Mirrored executed notebook to {dest}")


def main():
    generate_performance_curves()
    generate_sample_restorations()
    build_and_execute_notebook()
    print("\nPhase 10 Finalization Complete!")


if __name__ == "__main__":
    main()
