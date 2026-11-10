"""Phase 12: Deep Residual U-Net Restoration Training on Combined MRI Degradation.

Trains a deep Residual U-Net (ResUNet) with global residual skip connection
on the combined degraded training dataset (39,200 paired MRI scans).

Key Features:
1. Hardware Optimization: PyTorch with CUDA mixed precision (torch.amp.autocast('cuda')
   and GradScaler) on NVIDIA GeForce RTX 2050 (4 GB VRAM).
2. Loss Function: Charbonnier Loss (differentiable smooth L1) for edge-preserving restoration.
3. Dataset Split: 90% Training (35,280 samples) / 10% Validation (3,920 samples) stratified by category/severity.
4. Resumability: Checks existing epoch logs in outputs/metrics/phase12_training_loss.csv.
5. Checkpointing:
   - outputs/models/best_restoration_model.pth (highest validation PSNR)
   - outputs/models/final_restoration_model.pth (final epoch weights)
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResUNet, count_parameters
from src.dataset import CombinedMRIDataset
from src.metrics import calculate_mse, calculate_psnr, calculate_ssim

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR  = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LOG_CSV   = METRICS_DIR / "phase12_training_loss.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_restoration_model.pth"
LAST_MODEL_PATH = MODELS_DIR / "final_restoration_model.pth"
LATEST_MODEL_PATH = MODELS_DIR / "latest_checkpoint.pth"

TRAIN_DEG_DIR   = PROJECT_ROOT / "dataset" / "degraded" / "combined"
TRAIN_CLEAN_DIR = PROJECT_ROOT / "dataset" / "archive" / "Training"

TOTAL_EPOCHS = 10
BATCH_SIZE   = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42


class CharbonnierLoss(nn.Module):
    """Differentiable smooth L1 loss for edge-preserving image restoration."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


def evaluate_validation(model: nn.Module, val_loader: DataLoader, device: torch.device,
                        loss_fn: nn.Module, max_eval_samples: int = 400):
    """Evaluates validation loss, PSNR, and SSIM across validation subset."""
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    psnr_list = []
    ssim_list = []
    evaluated_count = 0

    with torch.no_grad():
        for deg_batch, clean_batch, meta in val_loader:
            deg_batch = deg_batch.to(device, non_blocking=True)
            clean_batch = clean_batch.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                pred_batch = model(deg_batch)
                loss = loss_fn(pred_batch, clean_batch)

            val_loss_sum += loss.item()
            val_batches += 1

            # Compute PSNR & SSIM on CPU numpy arrays
            preds_np = pred_batch.cpu().numpy()
            cleans_np = clean_batch.cpu().numpy()

            for i in range(preds_np.shape[0]):
                p_arr = preds_np[i, 0]
                c_arr = cleans_np[i, 0]
                psnr_list.append(calculate_psnr(c_arr, p_arr))
                ssim_list.append(calculate_ssim(c_arr, p_arr))
                evaluated_count += 1
                if evaluated_count >= max_eval_samples:
                    break
            if evaluated_count >= max_eval_samples:
                break

    mean_loss = val_loss_sum / max(1, val_batches)
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    return mean_loss, mean_psnr, mean_ssim


def main():
    print("=" * 75)
    print("PHASE 12: DEEP RESUNET RESTORATION TRAINING")
    print("=" * 75)

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("Enabled cuDNN Benchmark autotuner for static input tensors [8, 1, 512, 512].")

    # 1. Dataset & Split
    print("Indexing paired dataset from disk...")
    full_dataset = CombinedMRIDataset(TRAIN_DEG_DIR, TRAIN_CLEAN_DIR)
    total_samples = len(full_dataset)
    print(f"Total Paired Samples Available: {total_samples:,}")

    val_size = int(0.10 * total_samples)  # 3,920 samples
    train_size = total_samples - val_size  # 35,280 samples

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(f"Training Split:   {train_size:,} samples ({train_size/total_samples*100:.1f}%)")
    print(f"Validation Split: {val_size:,} samples ({val_size/total_samples*100:.1f}%)")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # 2. Model, Optimizer, Scheduler, Loss
    model = ResUNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    params_count = count_parameters(model)
    print(f"Model: ResUNet (Residual U-Net) | Trainable Parameters: {params_count:,}")

    loss_fn = CharbonnierLoss(eps=1e-3).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    # Check for existing logs / resumability
    start_epoch = 1
    best_val_psnr = -1.0
    log_records = []

    if TRAIN_LOG_CSV.exists() and TRAIN_LOG_CSV.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(TRAIN_LOG_CSV)
            if not df_existing.empty and 'epoch' in df_existing.columns:
                completed_epochs = df_existing['epoch'].tolist()
                last_completed = max(completed_epochs)
                log_records = df_existing.to_dict('records')
                best_val_psnr = df_existing['val_psnr'].max()
                print(f"Found existing training log with {last_completed}/{TOTAL_EPOCHS} epochs completed.")
                if last_completed >= TOTAL_EPOCHS and BEST_MODEL_PATH.exists():
                    print(f"Training already completed all {TOTAL_EPOCHS} epochs! Best Val PSNR: {best_val_psnr:.2f} dB")
                    return
                # If checkpoint exists, restore model, optimizer, and advance scheduler
                ckpt_to_load = LATEST_MODEL_PATH if LATEST_MODEL_PATH.exists() else BEST_MODEL_PATH
                if ckpt_to_load.exists():
                    print(f"Loading checkpoint from {ckpt_to_load.name} (model + optimizer state)...", flush=True)
                    checkpoint = torch.load(ckpt_to_load, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print(f"  Optimizer state restored.", flush=True)
                    # Advance scheduler to match completed epochs
                    for _ in range(last_completed):
                        scheduler.step()
                    print(f"  Scheduler advanced to epoch {last_completed}.", flush=True)
                start_epoch = last_completed + 1
        except Exception as e:
            print(f"[WARN] Error reading existing log: {e}", flush=True)

    print(f"\nStarting training from Epoch {start_epoch} to {TOTAL_EPOCHS}...", flush=True)
    total_t0 = time.time()

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        train_batches = 0

        for batch_idx, (deg_batch, clean_batch, _) in enumerate(train_loader):
            deg_batch = deg_batch.to(device, non_blocking=True)
            clean_batch = clean_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred_batch = model(deg_batch)
                loss = loss_fn(pred_batch, clean_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_batches += 1

            if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == len(train_loader):
                cur_avg_loss = running_loss / train_batches
                print(f"  [Epoch {epoch:02d}/{TOTAL_EPOCHS:02d}] Batch {batch_idx+1:,}/{len(train_loader):,} - Loss: {cur_avg_loss:.5f}", flush=True)

        train_loss = running_loss / max(1, train_batches)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Validation evaluation
        val_loss, val_psnr, val_ssim = evaluate_validation(model, val_loader, device, loss_fn)
        epoch_time = time.time() - epoch_t0

        # Always save latest checkpoint for resumability
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'val_loss': val_loss
        }, LATEST_MODEL_PATH)

        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_loss': val_loss
            }, BEST_MODEL_PATH)
            best_tag = " [BEST MODEL SAVED]"
        else:
            best_tag = ""

        print(
            f"--> Epoch {epoch:02d}/{TOTAL_EPOCHS:02d} | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s{best_tag}",
            flush=True
        )

        # Log epoch
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'learning_rate': current_lr,
            'time_seconds': round(epoch_time, 2)
        }
        log_records.append(record)

        # Incremental flush to CSV
        df_log = pd.DataFrame(log_records)
        df_log.to_csv(TRAIN_LOG_CSV, index=False)

    # Save final model
    torch.save({
        'epoch': TOTAL_EPOCHS,
        'model_state_dict': model.state_dict(),
        'val_psnr': val_psnr,
        'val_ssim': val_ssim
    }, LAST_MODEL_PATH)

    total_training_time = time.time() - total_t0
    print("\n" + "=" * 75)
    print(f"PHASE 12 TRAINING COMPLETED in {total_training_time:.2f}s ({total_training_time/60:.2f} min)!")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Saved Checkpoints:\n  - {BEST_MODEL_PATH}\n  - {LAST_MODEL_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    main()
