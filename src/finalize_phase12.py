"""Finalize Phase 12: Deep Residual U-Net Training.
- Generates training & validation loss/PSNR/SSIM curves.
- Generates visual comparison of Clean vs. Degraded vs. Traditional vs. Deep ResUNet.
- Builds, executes, and mirrors notebooks/12_cnn_dnn_training.ipynb.
"""

import os
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

import torch

PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
ONEDRIVE_NOTEBOOKS = Path(r"C:\Users\gitan\OneDrive\Documents\Brain_MRI_Restoration\notebooks")
ARTIFACTS_DIR = Path(r"C:\Users\gitan\.gemini\antigravity\brain\57c45525-48a2-454d-9673-64c1e8e45b1e")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResUNet
from src.restoration import apply_median_filter, apply_wiener_filter

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR  = PROJECT_ROOT / "outputs" / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRAIN_LOG_CSV   = METRICS_DIR / "phase12_training_loss.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_restoration_model.pth"

CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
TRAIN_CLEAN_DIR = PROJECT_ROOT / "dataset" / "archive" / "Training"
TRAIN_DEG_DIR   = PROJECT_ROOT / "dataset" / "degraded" / "combined"


def generate_training_curves():
    print("Generating Phase 12 training curves...")
    df_log = pd.read_csv(TRAIN_LOG_CSV)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.22, hspace=0.30)

    # 1. Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(df_log['epoch'], df_log['train_loss'], marker='o', linewidth=2.2, label='Training Loss (Charbonnier)', color='#1f77b4')
    ax1.plot(df_log['epoch'], df_log['val_loss'], marker='s', linewidth=2.2, label='Validation Loss', color='#ff7f0e')
    ax1.set_title("Training & Validation Loss vs. Epoch", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Loss", fontsize=11, fontweight='bold')
    ax1.set_xticks(df_log['epoch'])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # 2. Validation PSNR vs Epoch
    ax2 = axes[0, 1]
    ax2.plot(df_log['epoch'], df_log['val_psnr'], marker='^', linewidth=2.5, label='ResUNet Validation PSNR', color='#2ca02c')
    ax2.axhline(10.42, color='#d62728', linestyle='--', linewidth=2.0, label='Traditional Baseline (Wiener: 10.42 dB)')
    ax2.set_title("Validation PSNR vs. Epoch (Surpassing Traditional Baseline)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax2.set_ylabel("PSNR (dB) — Higher is Better", fontsize=11, fontweight='bold')
    ax2.set_xticks(df_log['epoch'])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10, loc='lower right')

    # 3. Validation SSIM vs Epoch
    ax3 = axes[1, 0]
    ax3.plot(df_log['epoch'], df_log['val_ssim'], marker='d', linewidth=2.5, label='ResUNet Validation SSIM', color='#9467bd')
    ax3.axhline(0.1808, color='#d62728', linestyle='--', linewidth=2.0, label='Traditional Baseline (Wiener: 0.1808)')
    ax3.set_title("Validation SSIM vs. Epoch (Structural Fidelity Gains)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax3.set_ylabel("SSIM — Higher is Better", fontsize=11, fontweight='bold')
    ax3.set_xticks(df_log['epoch'])
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(fontsize=10, loc='lower right')

    # 4. Learning Rate Schedule
    ax4 = axes[1, 1]
    ax4.plot(df_log['epoch'], df_log['learning_rate'], marker='x', linewidth=2.0, label='Cosine Annealing Learning Rate', color='#8c564b')
    ax4.set_title("Cosine Annealing Learning Rate Schedule", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Epoch", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Learning Rate", fontsize=11, fontweight='bold')
    ax4.set_xticks(df_log['epoch'])
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(fontsize=10)

    fig.suptitle(
        "Phase 12: Deep Residual U-Net (ResUNet) Convergence & Restoration Progression\n"
        "(Combined Multi-Degradation: Low-Resolution + Linear Motion Blur + Complex Rician Noise)",
        fontsize=15, fontweight='bold', y=0.98
    )

    out_fig1 = OUTPUTS_DIR / "phase12_training_curves.png"
    plt.savefig(out_fig1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves figure: {out_fig1}")

    if ARTIFACTS_DIR.exists():
        shutil.copy2(out_fig1, ARTIFACTS_DIR / "phase12_training_curves.png")


def generate_visual_comparison():
    print("Generating visual comparison of Clean vs. Degraded vs. Traditional vs. ResUNet...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 15))
    plt.subplots_adjust(wspace=0.04, hspace=0.15)

    col_headers = [
        "Clean Reference",
        "Combined 40% Degraded",
        "Median Filter (3x3)",
        "Wiener Filter (5x5)",
        "Deep ResUNet (Ours)"
    ]

    for row_idx, cat in enumerate(CATEGORIES):
        clean_files = sorted(list((TRAIN_CLEAN_DIR / cat).glob("*.*")))
        clean_p = clean_files[0]
        stem = clean_p.stem
        deg_p = TRAIN_DEG_DIR / cat / "40" / f"{stem}_combined_40.png"

        with Image.open(clean_p) as img_c:
            clean_arr = np.array(img_c.convert('L').resize((512, 512)), dtype=np.float32) / 255.0
        with Image.open(deg_p) as img_d:
            deg_arr = np.array(img_d, dtype=np.float32) / 255.0

        # Traditional restorations
        median_arr = apply_median_filter(deg_arr)
        wiener_arr = apply_wiener_filter(deg_arr)

        # Deep restoration
        deg_tensor = torch.from_numpy(deg_arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred_tensor = model(deg_tensor)
        deep_arr = pred_tensor.squeeze().cpu().numpy()

        images = [clean_arr, deg_arr, median_arr, wiener_arr, deep_arr]

        for col_idx, arr in enumerate(images):
            axes[row_idx, col_idx].imshow(np.clip(arr * 255, 0, 255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
            axes[row_idx, col_idx].set_xticks([]); axes[row_idx, col_idx].set_yticks([])
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(cat.capitalize(), fontsize=14, fontweight='bold')
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_headers[col_idx], fontsize=13, fontweight='bold', pad=8)

    fig.suptitle(
        "Phase 12: Visual Comparison of Deep ResUNet Restoration vs. Traditional Baselines (40% Combined Severity)\n"
        "(Glioma, Meningioma, No Tumor, Pituitary)",
        fontsize=15, fontweight='bold', y=0.98
    )

    out_fig2 = OUTPUTS_DIR / "phase12_sample_deep_restorations.png"
    plt.savefig(out_fig2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visual comparison figure: {out_fig2}")

    if ARTIFACTS_DIR.exists():
        shutil.copy2(out_fig2, ARTIFACTS_DIR / "phase12_sample_deep_restorations.png")


def build_and_execute_notebook():
    print("Building 12_cnn_dnn_training.ipynb...")
    nb = new_notebook()

    # Cell 1: Markdown Overview
    c1 = new_markdown_cell(
        "# Phase 12: Deep Residual U-Net (ResUNet) Restoration Model Training\n\n"
        "## Academic Project Context\n"
        "**Project Title:** *Comparative Analysis of Brain MRI Image Restoration Techniques Under Different Degradation Conditions*\n\n"
        "### Objectives\n"
        "1. Train a deep convolutional image restoration model on paired **Combined MRI Multi-Degradation** scans (39,200 training scans).\n"
        "2. Address compound physical artifacts: **Low-Resolution Downsampling + Linear Motion Blur + Complex Rician Coil Noise**.\n"
        "3. Surpass the traditional baseline ceiling established in Phase 10 & 11 (Wiener: **10.42 dB** PSNR, **0.1808** SSIM).\n"
        "4. Employ GPU acceleration (NVIDIA RTX 2050, 4 GB VRAM) with mixed precision and Charbonnier smooth L1 loss.\n"
    )

    # Cell 2: Imports & Model Inspection
    c2 = new_code_cell(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import torch\n\n"
        "PROJECT_ROOT = Path(r'C:\\Users\\gitan\\BrainMRI_Project')\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "from src.models import ResUNet, count_parameters\n\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "model = ResUNet(in_channels=1, out_channels=1, base_channels=32).to(device)\n"
        "print('=== Phase 12: Architecture & Hardware Configuration ===')\n"
        "print(f'Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"})')\n"
        "print(f'Model Architecture: Residual U-Net (ResUNet)')\n"
        "print(f'Trainable Parameters: {count_parameters(model):,}')\n"
        "print(f'Global Residual Learning: Enabled (I_clean = clamp(I_deg + Residual, 0, 1))')\n"
    )

    # Cell 3: Training Log Analysis
    c3 = new_code_cell(
        "METRICS_DIR = PROJECT_ROOT / 'outputs' / 'metrics'\n"
        "TRAIN_LOG_CSV = METRICS_DIR / 'phase12_training_loss.csv'\n"
        "df_log = pd.read_csv(TRAIN_LOG_CSV)\n"
        "print('=== Phase 12: Training Epoch Metrics Log ===')\n"
        "print(df_log.to_string(index=False))\n\n"
        "best_epoch = df_log.loc[df_log['val_psnr'].idxmax()]\n"
        "print(f'\\nPeak Validation PSNR: {best_epoch[\"val_psnr\"]:.2f} dB at Epoch {int(best_epoch[\"epoch\"])}')\n"
        "print(f'Peak Validation SSIM: {best_epoch[\"val_ssim\"]:.4f}')\n"
        "print(f'Traditional Baseline (Wiener): 10.42 dB | ResUNet Improvement: +{best_epoch[\"val_psnr\"] - 10.42:.2f} dB')\n"
    )

    # Cell 4: Training & Convergence Curves Plot
    c4 = new_code_cell(
        "fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))\n"
        "plt.subplots_adjust(wspace=0.22, hspace=0.30)\n\n"
        "# 1. Loss\n"
        "axes[0, 0].plot(df_log['epoch'], df_log['train_loss'], marker='o', linewidth=2.2, label='Train Loss (Charbonnier)', color='#1f77b4')\n"
        "axes[0, 0].plot(df_log['epoch'], df_log['val_loss'], marker='s', linewidth=2.2, label='Val Loss', color='#ff7f0e')\n"
        "axes[0, 0].set_title('Training & Validation Loss vs. Epoch', fontweight='bold')\n"
        "axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].grid(True, linestyle='--'); axes[0, 0].legend()\n\n"
        "# 2. PSNR\n"
        "axes[0, 1].plot(df_log['epoch'], df_log['val_psnr'], marker='^', linewidth=2.5, label='ResUNet Val PSNR', color='#2ca02c')\n"
        "axes[0, 1].axhline(10.42, color='#d62728', linestyle='--', linewidth=2.0, label='Traditional Baseline (Wiener: 10.42 dB)')\n"
        "axes[0, 1].set_title('Validation PSNR vs. Epoch', fontweight='bold')\n"
        "axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('PSNR (dB)'); axes[0, 1].grid(True, linestyle='--'); axes[0, 1].legend(loc='lower right')\n\n"
        "# 3. SSIM\n"
        "axes[1, 0].plot(df_log['epoch'], df_log['val_ssim'], marker='d', linewidth=2.5, label='ResUNet Val SSIM', color='#9467bd')\n"
        "axes[1, 0].axhline(0.1808, color='#d62728', linestyle='--', linewidth=2.0, label='Traditional Baseline (Wiener: 0.1808)')\n"
        "axes[1, 0].set_title('Validation SSIM vs. Epoch', fontweight='bold')\n"
        "axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('SSIM'); axes[1, 0].grid(True, linestyle='--'); axes[1, 0].legend(loc='lower right')\n\n"
        "# 4. LR\n"
        "axes[1, 1].plot(df_log['epoch'], df_log['learning_rate'], marker='x', linewidth=2.0, label='Learning Rate', color='#8c564b')\n"
        "axes[1, 1].set_title('Cosine Annealing Learning Rate Schedule', fontweight='bold')\n"
        "axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('LR'); axes[1, 1].grid(True, linestyle='--'); axes[1, 1].legend()\n\n"
        "fig.suptitle('Phase 12: Deep Residual U-Net Convergence and Performance Progression', fontsize=15, fontweight='bold', y=0.98)\n"
        "plt.show()\n"
    )

    # Cell 5: Visual Restoration Comparison
    c5 = new_code_cell(
        "from PIL import Image\n"
        "from src.restoration import apply_median_filter, apply_wiener_filter\n\n"
        "MODELS_DIR = PROJECT_ROOT / 'outputs' / 'models'\n"
        "BEST_MODEL_PATH = MODELS_DIR / 'best_restoration_model.pth'\n"
        "checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)\n"
        "model.load_state_dict(checkpoint['model_state_dict'])\n"
        "model.eval()\n\n"
        "TRAIN_CLEAN_DIR = PROJECT_ROOT / 'dataset' / 'archive' / 'Training'\n"
        "TRAIN_DEG_DIR   = PROJECT_ROOT / 'dataset' / 'degraded' / 'combined'\n"
        "CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary']\n\n"
        "fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 15))\n"
        "plt.subplots_adjust(wspace=0.04, hspace=0.15)\n"
        "col_headers = ['Clean Reference', 'Combined 40% Degraded', 'Median Filter (3x3)', 'Wiener Filter (5x5)', 'Deep ResUNet (Ours)']\n\n"
        "for row_idx, cat in enumerate(CATEGORIES):\n"
        "    clean_p = sorted(list((TRAIN_CLEAN_DIR / cat).glob('*.*')))[0]\n"
        "    stem = clean_p.stem\n"
        "    deg_p = TRAIN_DEG_DIR / cat / '40' / f'{stem}_combined_40.png'\n\n"
        "    with Image.open(clean_p) as img_c:\n"
        "        clean_arr = np.array(img_c.convert('L').resize((512, 512)), dtype=np.float32) / 255.0\n"
        "    with Image.open(deg_p) as img_d:\n"
        "        deg_arr = np.array(img_d, dtype=np.float32) / 255.0\n\n"
        "    med_arr = apply_median_filter(deg_arr)\n"
        "    wien_arr = apply_wiener_filter(deg_arr)\n\n"
        "    deg_tensor = torch.from_numpy(deg_arr).unsqueeze(0).unsqueeze(0).to(device)\n"
        "    with torch.no_grad():\n"
        "        with torch.amp.autocast('cuda'):\n"
        "            pred_tensor = model(deg_tensor)\n"
        "    deep_arr = pred_tensor.squeeze().cpu().numpy()\n\n"
        "    images = [clean_arr, deg_arr, med_arr, wien_arr, deep_arr]\n"
        "    for col_idx, arr in enumerate(images):\n"
        "        axes[row_idx, col_idx].imshow(np.clip(arr * 255, 0, 255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)\n"
        "        axes[row_idx, col_idx].set_xticks([]); axes[row_idx, col_idx].set_yticks([])\n"
        "        if col_idx == 0: axes[row_idx, col_idx].set_ylabel(cat.capitalize(), fontsize=14, fontweight='bold')\n"
        "        if row_idx == 0: axes[row_idx, col_idx].set_title(col_headers[col_idx], fontsize=13, fontweight='bold', pad=8)\n\n"
        "fig.suptitle('Visual Comparison of Deep ResUNet Restoration vs. Traditional Baselines (40% Combined Severity)', fontsize=15, fontweight='bold', y=0.98)\n"
        "plt.show()\n"
    )

    nb.cells = [c1, c2, c3, c4, c5]

    nb_path = PROJECT_ROOT / "notebooks" / "12_cnn_dnn_training.ipynb"
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
        dest = ONEDRIVE_NOTEBOOKS / "12_cnn_dnn_training.ipynb"
        shutil.copy2(nb_path, dest)
        print(f"Mirrored executed notebook to {dest}")


def main():
    generate_training_curves()
    generate_visual_comparison()
    build_and_execute_notebook()
    print("\nPhase 12 Finalization Complete!")


if __name__ == "__main__":
    main()
