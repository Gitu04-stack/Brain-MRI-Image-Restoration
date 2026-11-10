"""Finalize Phase 9: Combined Degradation.
- Generates high-resolution comparative visualization figure (4 categories x 8 columns).
- Constructs and executes notebooks/09_combined_degradation.ipynb.
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

from src.degradation import (
    LOW_RES_SCALE_MAP,
    LOW_RES_DIM_MAP,
    MOTION_BLUR_KERNEL_MAP,
    RICIAN_SIGMA_MAP,
    SEVERITY_LEVELS
)

CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
TRAIN_CLEAN_DIR = PROJECT_ROOT / "dataset" / "archive" / "Training"
TEST_CLEAN_DIR  = PROJECT_ROOT / "dataset" / "archive" / "Testing"
TRAIN_DEG_DIR   = PROJECT_ROOT / "dataset" / "degraded" / "combined"
TEST_DEG_DIR    = PROJECT_ROOT / "dataset" / "degraded_test" / "combined"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"


def generate_comparison_figure():
    print("Generating Phase 9 combined degradation comparison figure...")
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(24, 13))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    col_titles = ["Clean (Original)"] + [f"Combined {s}%" for s in SEVERITY_LEVELS]

    for row_idx, cat in enumerate(CATEGORIES):
        clean_files = sorted(list((TRAIN_CLEAN_DIR / cat).glob("*.*")))
        rep_img_path = clean_files[0]
        stem = rep_img_path.stem

        # Clean original
        with Image.open(rep_img_path) as img:
            clean_arr = np.array(img.convert('L').resize((512, 512)))

        axes[row_idx, 0].imshow(clean_arr, cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 0].set_ylabel(cat.capitalize(), fontsize=14, fontweight='bold')
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=13, fontweight='bold', pad=10)

        # Severities
        for col_idx, sev in enumerate(SEVERITY_LEVELS, start=1):
            deg_path = TRAIN_DEG_DIR / cat / str(sev) / f"{stem}_combined_{sev}.png"
            with Image.open(deg_path) as img:
                deg_arr = np.array(img)

            axes[row_idx, col_idx].imshow(deg_arr, cmap='gray', vmin=0, vmax=255)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=13, fontweight='bold', pad=10)

    fig.suptitle(
        "Phase 9: Combined Multi-Degradation Progression (Clean -> Low-Res -> Motion Blur -> Rician Noise)\n"
        "Evaluated Across All 4 Brain MRI Categories (Glioma, Meningioma, No Tumor, Pituitary)",
        fontsize=16, fontweight='bold', y=0.98
    )

    out_fig = OUTPUTS_DIR / "phase9_combined_degradation_comparison.png"
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure to: {out_fig}")

    # Mirror to artifacts directory
    if ARTIFACTS_DIR.exists():
        art_fig = ARTIFACTS_DIR / "phase9_combined_degradation_comparison.png"
        shutil.copy2(out_fig, art_fig)
        print(f"Mirrored comparison figure to artifact directory: {art_fig}")


def build_and_execute_notebook():
    print("Building 09_combined_degradation.ipynb...")
    nb = new_notebook()

    # Cell 1: Markdown Overview
    c1 = new_markdown_cell(
        "# Phase 9: Controlled Combined Degradation Experiment (All 4 Categories)\n\n"
        "## Academic Project Context\n"
        "**Project Title:** *Comparative Analysis of Brain MRI Image Restoration Techniques Under Different Degradation Conditions*\n\n"
        "### Objective\n"
        "In this phase, we model realistic clinical multi-degradation where MRI scans simultaneously experience:\n"
        "1. **Low-Resolution Downsampling & Interpolation:** Simulating rapid low-field MRI acquisition or thick slice truncation.\n"
        "2. **Directional Linear Motion Blur:** Simulating involuntary patient motion (tremor, respiration, swallowing) at $\\theta = 45^\\circ$.\n"
        "3. **Non-Gaussian Rician Coil Noise:** Simulating intrinsic RF thermal coil noise via physical bivariate Gaussian complex quadrature summation.\n\n"
        "### Strict Pipeline Formulation & Order\n"
        "The degradation pipeline strictly follows the physical clinical sequence:\n\n"
        "$$\\mathbf{I}_{\\text{clean}} \\xrightarrow{\\text{Low-Res}} \\mathbf{I}_{\\text{LR}} \\xrightarrow{\\text{Motion Blur}} \\mathbf{I}_{\\text{LR+Blur}} \\xrightarrow{\\text{Rician Noise}} \\mathbf{I}_{\\text{combined}}$$\n\n"
        "- **Training Dataset:** 5,600 clean scans $\\times$ 7 severities = **39,200 degraded training scans** saved to `dataset/degraded/combined/`\n"
        "- **Testing Dataset:** 1,600 clean scans $\\times$ 7 severities = **11,200 degraded testing scans** saved to `dataset/degraded_test/combined/`\n"
        "- **Total Generated Scans:** **50,400 lossless 8-bit Grayscale PNG images** ($512 \\times 512$).\n"
        "- **Original Immutability:** Original clean images (`dataset/archive/`) remain 100% read-only and unmodified.\n"
    )

    # Cell 2: Imports & Configuration
    c2 = new_code_cell(
        "import os\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from PIL import Image\n"
        "import matplotlib.pyplot as plt\n\n"
        "PROJECT_ROOT = Path(r'C:\\Users\\gitan\\BrainMRI_Project')\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "from src.degradation import (\n"
        "    apply_combined_degradation,\n"
        "    LOW_RES_SCALE_MAP,\n"
        "    LOW_RES_DIM_MAP,\n"
        "    MOTION_BLUR_KERNEL_MAP,\n"
        "    RICIAN_SIGMA_MAP,\n"
        "    SEVERITY_LEVELS\n"
        ")\n\n"
        "CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary']\n"
        "TRAIN_CLEAN_DIR = PROJECT_ROOT / 'dataset' / 'archive' / 'Training'\n"
        "TEST_CLEAN_DIR  = PROJECT_ROOT / 'dataset' / 'archive' / 'Testing'\n"
        "TRAIN_DEG_DIR   = PROJECT_ROOT / 'dataset' / 'degraded' / 'combined'\n"
        "TEST_DEG_DIR    = PROJECT_ROOT / 'dataset' / 'degraded_test' / 'combined'\n\n"
        "print('=== Phase 9: Environment & Directory Configuration ===')\n"
        "print(f'Project Root: {PROJECT_ROOT}')\n"
        "print(f'Training Degraded Base: {TRAIN_DEG_DIR}')\n"
        "print(f'Testing Degraded Base:  {TEST_DEG_DIR}')\n"
    )

    # Cell 3: Mathematical Mapping Table
    c3 = new_code_cell(
        "mapping_data = []\n"
        "for sev in SEVERITY_LEVELS:\n"
        "    scale = LOW_RES_SCALE_MAP[sev]\n"
        "    dim = LOW_RES_DIM_MAP[sev]\n"
        "    k_len = MOTION_BLUR_KERNEL_MAP[sev]\n"
        "    sigma = RICIAN_SIGMA_MAP[sev]\n"
        "    mapping_data.append({\n"
        "        'Severity (%)': f'{sev}%',\n"
        "        'LR Scale Factor': f'{scale:.4f}',\n"
        "        'Downsampled Resolution': f'{dim}x{dim}',\n"
        "        'Motion Kernel (L)': f'{k_len} px (45°)',\n"
        "        'Rician Noise Sigma (σ)': f'{sigma:.2f}',\n"
        "        'Clinical Distortion Model': f'LR({dim}x{dim}) + Blur({k_len}px) + Rician(σ={sigma:.2f})'\n"
        "    })\n\n"
        "df_mapping = pd.DataFrame(mapping_data)\n"
        "print('=== Phase 9: Combined Multi-Degradation Severity Parameter Mapping ===')\n"
        "print(df_mapping.to_string(index=False))\n"
    )

    # Cell 4: Audit Verification Training
    c4 = new_code_cell(
        "train_audit = []\n"
        "all_train_matched = True\n"
        "for cat in CATEGORIES:\n"
        "    clean_count = len(list((TRAIN_CLEAN_DIR / cat).glob('*.*')))\n"
        "    for sev in SEVERITY_LEVELS:\n"
        "        sev_dir = TRAIN_DEG_DIR / cat / str(sev)\n"
        "        deg_count = len(list(sev_dir.glob('*.png')))\n"
        "        matched = (deg_count == clean_count)\n"
        "        if not matched:\n"
        "            all_train_matched = False\n"
        "        train_audit.append({\n"
        "            'Dataset': 'Training',\n"
        "            'Category': cat,\n"
        "            'Severity': f'{sev}%',\n"
        "            'Clean Count': clean_count,\n"
        "            'Degraded Count': deg_count,\n"
        "            'Match': matched\n"
        "        })\n\n"
        "df_train_audit = pd.DataFrame(train_audit)\n"
        "print('=== Training Combined Degraded Dataset Audit (39,200 Scans) ===')\n"
        "print(df_train_audit.to_string(index=False))\n"
        "print(f'\\nTotal Training Combined Degraded Images: {df_train_audit[\"Degraded Count\"].sum()} / 39,200 expected')\n"
        "print(f'All Training Counts Match: {all_train_matched}')\n"
    )

    # Cell 5: Audit Verification Testing
    c5 = new_code_cell(
        "test_audit = []\n"
        "all_test_matched = True\n"
        "for cat in CATEGORIES:\n"
        "    clean_count = len(list((TEST_CLEAN_DIR / cat).glob('*.*')))\n"
        "    for sev in SEVERITY_LEVELS:\n"
        "        sev_dir = TEST_DEG_DIR / cat / str(sev)\n"
        "        deg_count = len(list(sev_dir.glob('*.png')))\n"
        "        matched = (deg_count == clean_count)\n"
        "        if not matched:\n"
        "            all_test_matched = False\n"
        "        test_audit.append({\n"
        "            'Dataset': 'Testing',\n"
        "            'Category': cat,\n"
        "            'Severity': f'{sev}%',\n"
        "            'Clean Count': clean_count,\n"
        "            'Degraded Count': deg_count,\n"
        "            'Match': matched\n"
        "        })\n\n"
        "df_test_audit = pd.DataFrame(test_audit)\n"
        "print('=== Testing Combined Degraded Dataset Audit (11,200 Scans) ===')\n"
        "print(df_test_audit.to_string(index=False))\n"
        "print(f'\\nTotal Testing Combined Degraded Images: {df_test_audit[\"Degraded Count\"].sum()} / 11,200 expected')\n"
        "print(f'All Testing Counts Match: {all_test_matched}')\n"
    )

    # Cell 6: Image Integrity & Format Audit
    c6 = new_code_cell(
        "sample_records = []\n"
        "for cat in CATEGORIES:\n"
        "    first_clean = sorted(list((TRAIN_CLEAN_DIR / cat).glob('*.*')))[0]\n"
        "    stem = first_clean.stem\n"
        "    for sev in [10, 40, 70]:\n"
        "        p = TRAIN_DEG_DIR / cat / str(sev) / f'{stem}_combined_{sev}.png'\n"
        "        with Image.open(p) as img:\n"
        "            arr = np.array(img)\n"
        "            sample_records.append({\n"
        "                'Category': cat,\n"
        "                'Severity': f'{sev}%',\n"
        "                'Resolution': f'{img.size[0]}x{img.size[1]}',\n"
        "                'Mode': img.mode,\n"
        "                'Min Pixel': int(arr.min()),\n"
        "                'Max Pixel': int(arr.max()),\n"
        "                'Dtype': str(arr.dtype)\n"
        "            })\n\n"
        "print('=== Sample Format, Mode & Dynamic Range Audit ===')\n"
        "print(pd.DataFrame(sample_records).to_string(index=False))\n\n"
        "# Verify original datasets remain untouched\n"
        "train_orig = {cat: len(list((TRAIN_CLEAN_DIR / cat).glob('*.*'))) for cat in CATEGORIES}\n"
        "test_orig  = {cat: len(list((TEST_CLEAN_DIR / cat).glob('*.*'))) for cat in CATEGORIES}\n"
        "print(f'\\nOriginal Training Image Counts (Untouched): {train_orig}')\n"
        "print(f'Original Testing Image Counts (Untouched):  {test_orig}')\n"
    )

    # Cell 7: Full Visual Comparison Plot
    c7 = new_code_cell(
        "fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(24, 13))\n"
        "plt.subplots_adjust(wspace=0.05, hspace=0.15)\n\n"
        "col_titles = ['Clean (Original)'] + [f'Combined {s}%' for s in SEVERITY_LEVELS]\n\n"
        "for row_idx, cat in enumerate(CATEGORIES):\n"
        "    clean_files = sorted(list((TRAIN_CLEAN_DIR / cat).glob('*.*')))\n"
        "    rep_img_path = clean_files[0]\n"
        "    stem = rep_img_path.stem\n\n"
        "    # Clean original\n"
        "    with Image.open(rep_img_path) as img:\n"
        "        clean_arr = np.array(img.convert('L').resize((512, 512)))\n\n"
        "    axes[row_idx, 0].imshow(clean_arr, cmap='gray', vmin=0, vmax=255)\n"
        "    axes[row_idx, 0].set_ylabel(cat.capitalize(), fontsize=14, fontweight='bold')\n"
        "    axes[row_idx, 0].set_xticks([])\n"
        "    axes[row_idx, 0].set_yticks([])\n"
        "    if row_idx == 0:\n"
        "        axes[row_idx, 0].set_title(col_titles[0], fontsize=13, fontweight='bold', pad=10)\n\n"
        "    # Severities\n"
        "    for col_idx, sev in enumerate(SEVERITY_LEVELS, start=1):\n"
        "        deg_path = TRAIN_DEG_DIR / cat / str(sev) / f'{stem}_combined_{sev}.png'\n"
        "        with Image.open(deg_path) as img:\n"
        "            deg_arr = np.array(img)\n\n"
        "        axes[row_idx, col_idx].imshow(deg_arr, cmap='gray', vmin=0, vmax=255)\n"
        "        axes[row_idx, col_idx].set_xticks([])\n"
        "        axes[row_idx, col_idx].set_yticks([])\n"
        "        if row_idx == 0:\n"
        "            axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=13, fontweight='bold', pad=10)\n\n"
        "fig.suptitle(\n"
        "    'Phase 9: Combined Multi-Degradation Progression (Clean -> Low-Res -> Motion Blur -> Rician Noise)\\n'\n"
        "    'Evaluated Across All 4 Brain MRI Categories (Glioma, Meningioma, No Tumor, Pituitary)',\n"
        "    fontsize=16, fontweight='bold', y=0.98\n"
        ")\n"
        "plt.show()\n"
    )

    nb.cells = [c1, c2, c3, c4, c5, c6, c7]

    nb_path = PROJECT_ROOT / "notebooks" / "09_combined_degradation.ipynb"
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
        dest = ONEDRIVE_NOTEBOOKS / "09_combined_degradation.ipynb"
        shutil.copy2(nb_path, dest)
        print(f"Mirrored executed notebook to {dest}")


def main():
    generate_comparison_figure()
    build_and_execute_notebook()
    print("\nPhase 9 Finalization Complete!")


if __name__ == "__main__":
    main()
