"""Preprocessing module for Brain MRI restoration project.
Provides standardization, grayscale conversion, normalization, and lossless saving.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Loads an image from disk using PIL.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        PIL Image instance.
    """
    return Image.open(image_path)


def normalize_image(image: Union[Image.Image, np.ndarray], 
                    target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Standardizes MRI slice to single-channel Grayscale 'L', resizes to target_size,
    and normalizes to float32 in [0.0, 1.0].
    
    Args:
        image: PIL Image or numpy array.
        target_size: (width, height) desired spatial resolution.
        
    Returns:
        2D numpy array with shape (target_size[1], target_size[0]), dtype float32, values in [0.0, 1.0].
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        
    # Convert to single-channel Grayscale 'L'
    img_gray = image.convert('L')
    
    # Resize if not already matching target_size
    if img_gray.size != target_size:
        img_gray = img_gray.resize(target_size, resample=Image.Resampling.BILINEAR)
        
    # Cast to float32 and normalize to [0.0, 1.0]
    arr_normalized = np.array(img_gray, dtype=np.float32) / 255.0
    return arr_normalized


def save_image(array_norm: np.ndarray, 
               output_path: Union[str, Path], 
               format: str = "PNG", 
               compress_level: int = 1) -> None:
    """Converts a normalized float32 array in [0.0, 1.0] to uint8 [0, 255]
    and writes to disk as a lossless image.
    
    Args:
        array_norm: 2D numpy array in [0.0, 1.0].
        output_path: Destination path.
        format: Image format (default 'PNG').
        compress_level: PNG compression level (1 = fastest lossless).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    uint8_arr = np.round(np.clip(array_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    img_out = Image.fromarray(uint8_arr, mode='L')
    img_out.save(out_p, format=format, compress_level=compress_level)
