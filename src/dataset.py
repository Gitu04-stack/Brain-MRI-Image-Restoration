"""PyTorch Dataset Module for Brain MRI Restoration.

Pairs combined degraded MRI scans with their corresponding clean ground-truth scans.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project")
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]
SEVERITY_LEVELS = [10, 20, 30, 40, 50, 60, 70]
CLEAN_EXTS = {".jpg", ".jpeg", ".png"}


class CombinedMRIDataset(Dataset):
    """PyTorch Dataset for pairing degraded MRI slices with clean ground-truth reference scans."""
    def __init__(self,
                 degraded_base_dir: Union[str, Path],
                 clean_base_dir: Union[str, Path],
                 categories: Optional[List[str]] = None,
                 severities: Optional[List[int]] = None,
                 transform = None):
        super().__init__()
        self.deg_base = Path(degraded_base_dir)
        self.clean_base = Path(clean_base_dir)
        self.categories = categories or CATEGORIES
        self.severities = severities or SEVERITY_LEVELS
        self.transform = transform

        # Build clean index: category -> {stem: clean_path}
        self.clean_index: Dict[str, Dict[str, Path]] = {}
        for cat in self.categories:
            cat_dir = self.clean_base / cat
            self.clean_index[cat] = {
                p.stem: p for p in cat_dir.iterdir()
                if p.suffix.lower() in CLEAN_EXTS
            }

        # Build pair list: (deg_path, clean_path, category, severity)
        self.samples: List[Tuple[Path, Path, str, int]] = []
        for cat in self.categories:
            for sev in self.severities:
                deg_dir = self.deg_base / cat / str(sev)
                if not deg_dir.exists():
                    continue
                for deg_file in sorted(deg_dir.glob("*.png")):
                    # stem format: {original_stem}_combined_{sev}
                    stem_parts = deg_file.stem.split("_combined_")
                    clean_stem = stem_parts[0]
                    if clean_stem in self.clean_index[cat]:
                        clean_path = self.clean_index[cat][clean_stem]
                        self.samples.append((deg_file, clean_path, cat, sev))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        deg_p, clean_p, cat, sev = self.samples[idx]

        # Load degraded image (512x512 Grayscale PNG)
        with Image.open(deg_p) as img_deg:
            deg_arr = np.array(img_deg, dtype=np.float32) / 255.0

        # Load clean reference image (standardized to 512x512 Grayscale)
        with Image.open(clean_p) as img_clean:
            img_clean_gray = img_clean.convert('L')
            if img_clean_gray.size != (512, 512):
                img_clean_gray = img_clean_gray.resize((512, 512), resample=Image.Resampling.BILINEAR)
            clean_arr = np.array(img_clean_gray, dtype=np.float32) / 255.0

        # Convert to float32 tensor of shape (1, 512, 512)
        deg_tensor = torch.from_numpy(deg_arr).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_arr).unsqueeze(0)

        meta = {
            'filename': deg_p.name,
            'category': cat,
            'severity': sev
        }

        return deg_tensor, clean_tensor, meta
