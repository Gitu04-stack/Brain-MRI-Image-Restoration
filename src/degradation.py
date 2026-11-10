"""Degradation module for Brain MRI restoration project.
Provides mathematical models for Gaussian noise, Rician noise, low resolution, motion blur, and combined distortions.
"""

from pathlib import Path
from typing import Dict, Union, Optional
import cv2
import numpy as np
from PIL import Image

from .preprocessing import normalize_image, save_image


SEVERITY_LEVELS = [10, 20, 30, 40, 50, 60, 70]

# Gaussian Noise Mapping: Severity percentage S% -> standard deviation sigma = S / 100
GAUSSIAN_SIGMA_MAP: Dict[int, float] = {
    10: 0.10,
    20: 0.20,
    30: 0.30,
    40: 0.40,
    50: 0.50,
    60: 0.60,
    70: 0.70
}


def add_gaussian_noise(image_array: np.ndarray, 
                       sigma: float, 
                       seed: Optional[int] = None) -> np.ndarray:
    """Applies Additive White Gaussian Noise (AWGN) to a normalized float32 image array in [0.0, 1.0].
    
    Model:
        I_noisy(x, y) = clip(I_clean(x, y) + eta(x, y), 0.0, 1.0)
        eta(x, y) ~ N(0, sigma^2)
        
    Args:
        image_array: 2D numpy array in [0.0, 1.0].
        sigma: Standard deviation of Gaussian distribution.
        seed: Optional random seed for reproducible noise generation.
        
    Returns:
        Noisy 2D numpy array clipped to [0.0, 1.0].
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random
    noise = rng.normal(loc=0.0, scale=sigma, size=image_array.shape).astype(np.float32)
    noisy_array = np.clip(image_array + noise, 0.0, 1.0)
    return noisy_array


def process_single_image_gaussian(clean_path_str: str, 
                                  category: str, 
                                  output_base_dir_str: str, 
                                  base_seed: int = 42,
                                  image_idx: int = 0) -> Dict[str, Union[str, int, bool]]:
    """Worker function for batch / multiprocessing Gaussian noise generation.
    Preprocesses one clean image and independently generates all 7 severity levels.
    
    Args:
        clean_path_str: Path to clean source image.
        category: MRI category (glioma, meningioma, notumor, pituitary).
        output_base_dir_str: Base output path (e.g. dataset/degraded/gaussian).
        base_seed: Base random seed.
        image_idx: Index of image to produce independent deterministic seed per slice.
        
    Returns:
        Status dictionary with results.
    """
    clean_p = Path(clean_path_str)
    out_base = Path(output_base_dir_str)
    stem = clean_p.stem
    
    try:
        with Image.open(clean_p) as img:
            clean_arr = normalize_image(img, target_size=(512, 512))
            
        # Independently generate each severity level from clean_arr
        for sev, sigma in GAUSSIAN_SIGMA_MAP.items():
            # Deterministic, unique seed per image and severity
            seed = (base_seed * 100000 + image_idx * 100 + sev) % (2**31 - 1)
            noisy_arr = add_gaussian_noise(clean_arr, sigma=sigma, seed=seed)
            
            out_file = out_base / category / str(sev) / f"{stem}_gaussian_{sev}.png"
            save_image(noisy_arr, out_file, format="PNG", compress_level=1)
            
        return {
            "filename": clean_p.name,
            "category": category,
            "success": True,
            "error": ""
        }
    except Exception as e:
        return {
            "filename": clean_p.name,
            "category": category,
            "success": False,
            "error": str(e)
        }


def run_worker_gaussian(item: tuple) -> Dict[str, Union[str, int, bool]]:
    """Multiprocessing wrapper for process_single_image_gaussian."""
    clean_p, cat, out_base, base_seed, idx = item
    return process_single_image_gaussian(
        clean_path_str=clean_p,
        category=cat,
        output_base_dir_str=out_base,
        base_seed=base_seed,
        image_idx=idx
    )


# Rician Noise Mapping: Severity percentage S% -> standard deviation sigma = S / 100
RICIAN_SIGMA_MAP: Dict[int, float] = {
    10: 0.10,
    20: 0.20,
    30: 0.30,
    40: 0.40,
    50: 0.50,
    60: 0.60,
    70: 0.70
}


def add_rician_noise(image_array: np.ndarray, 
                     sigma: float, 
                     seed: Optional[int] = None) -> np.ndarray:
    """Applies MRI Rician noise to a normalized float32 image array in [0.0, 1.0].
    
    Model:
        Real channel:      R = I_clean + eta_1,  eta_1 ~ N(0, sigma^2)
        Imaginary channel: I = 0 + eta_2,        eta_2 ~ N(0, sigma^2)
        Magnitude MRI:     M = sqrt(R^2 + I^2)
        
    Args:
        image_array: 2D numpy array in [0.0, 1.0].
        sigma: Standard deviation of orthogonal complex Gaussian noise channels.
        seed: Optional random seed for reproducible noise generation.
        
    Returns:
        Rician-degraded 2D numpy array clipped to [0.0, 1.0].
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random
    eta1 = rng.normal(loc=0.0, scale=sigma, size=image_array.shape).astype(np.float32)
    eta2 = rng.normal(loc=0.0, scale=sigma, size=image_array.shape).astype(np.float32)
    magnitude = np.sqrt((image_array + eta1) ** 2 + (eta2 ** 2))
    return np.clip(magnitude, 0.0, 1.0).astype(np.float32)


def process_single_image_rician(clean_path_str: str, 
                                category: str, 
                                output_base_dir_str: str, 
                                base_seed: int = 42,
                                image_idx: int = 0) -> Dict[str, Union[str, int, bool]]:
    """Worker function for batch / multiprocessing Rician noise generation.
    Preprocesses one clean image and independently generates all 7 severity levels.
    """
    clean_p = Path(clean_path_str)
    out_base = Path(output_base_dir_str)
    stem = clean_p.stem
    
    try:
        with Image.open(clean_p) as img:
            clean_arr = normalize_image(img, target_size=(512, 512))
            
        # Independently generate each severity level from clean_arr
        for sev, sigma in RICIAN_SIGMA_MAP.items():
            # Distinct deterministic seed for Rician generation
            seed = (base_seed * 200000 + image_idx * 100 + sev) % (2**31 - 1)
            noisy_arr = add_rician_noise(clean_arr, sigma=sigma, seed=seed)
            
            out_file = out_base / category / str(sev) / f"{stem}_rician_{sev}.png"
            save_image(noisy_arr, out_file, format="PNG", compress_level=1)
            
        return {
            "filename": clean_p.name,
            "category": category,
            "success": True,
            "error": ""
        }
    except Exception as e:
        return {
            "filename": clean_p.name,
            "category": category,
            "success": False,
            "error": str(e)
        }


def run_worker_rician(item: tuple) -> Dict[str, Union[str, int, bool]]:
    """Multiprocessing wrapper for process_single_image_rician."""
    clean_p, cat, out_base, base_seed, idx = item
    return process_single_image_rician(
        clean_path_str=clean_p,
        category=cat,
        output_base_dir_str=out_base,
        base_seed=base_seed,
        image_idx=idx
    )


# Low-Resolution Mapping: Severity percentage S% -> downsampled spatial retention (100 - S)%
# S=10% retains 90% dimension (461x461), S=50% retains 50% dimension (256x256), S=70% retains 30% dimension (154x154)
LOW_RES_SCALE_MAP: Dict[int, float] = {
    10: 0.90,
    20: 0.80,
    30: 0.70,
    40: 0.60,
    50: 0.50,
    60: 0.40,
    70: 0.30
}

LOW_RES_DIM_MAP: Dict[int, int] = {
    sev: int(round(512 * scale)) for sev, scale in LOW_RES_SCALE_MAP.items()
}


def apply_low_resolution(image_array: np.ndarray, 
                         scale_factor: float, 
                         target_size: tuple = (512, 512)) -> np.ndarray:
    """Downsamples image array to low-resolution intermediate size, then resizes back
    to target_size (512x512) so it can be directly compared against clean original.
    
    Model:
        1. Low-res grid: D_lr = round(D_target * scale_factor)
        2. Downsample:   I_lr = downsample(I_clean, (D_lr, D_lr)) using area averaging
        3. Upsample:     I_recon = upsample(I_lr, target_size) using bilinear interpolation
        
    Args:
        image_array: 2D numpy array in [0.0, 1.0].
        scale_factor: Linear scale retention (0.30 to 0.90).
        target_size: Common evaluation dimension (width, height).
        
    Returns:
        Degraded 2D numpy array with shape target_size, clipped to [0.0, 1.0].
    """
    orig_w, orig_h = target_size
    lr_w = int(round(orig_w * scale_factor))
    lr_h = int(round(orig_h * scale_factor))
    
    uint8_img = np.round(np.clip(image_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img, mode='L')
    
    # Downsample using area box resampling
    lr_img = pil_img.resize((lr_w, lr_h), resample=Image.Resampling.BOX)
    
    # Upsample back to common comparison dimension (512x512) using bilinear interpolation
    recon_img = lr_img.resize(target_size, resample=Image.Resampling.BILINEAR)
    
    return np.array(recon_img, dtype=np.float32) / 255.0


def process_single_image_lowres(clean_path_str: str, 
                                category: str, 
                                output_base_dir_str: str, 
                                base_seed: int = 42,
                                image_idx: int = 0) -> Dict[str, Union[str, int, bool]]:
    """Worker function for batch / multiprocessing Low Resolution degradation generation.
    Preprocesses one clean image and independently generates all 7 severity levels.
    """
    clean_p = Path(clean_path_str)
    out_base = Path(output_base_dir_str)
    stem = clean_p.stem
    
    try:
        with Image.open(clean_p) as img:
            clean_arr = normalize_image(img, target_size=(512, 512))
            
        # Independently generate each severity level from clean_arr
        for sev, scale in LOW_RES_SCALE_MAP.items():
            degraded_arr = apply_low_resolution(clean_arr, scale_factor=scale, target_size=(512, 512))
            
            out_file = out_base / category / str(sev) / f"{stem}_lowres_{sev}.png"
            save_image(degraded_arr, out_file, format="PNG", compress_level=1)
            
        return {
            "filename": clean_p.name,
            "category": category,
            "success": True,
            "error": ""
        }
    except Exception as e:
        return {
            "filename": clean_p.name,
            "category": category,
            "success": False,
            "error": str(e)
        }


def run_worker_lowres(item: tuple) -> Dict[str, Union[str, int, bool]]:
    """Multiprocessing wrapper for process_single_image_lowres."""
    clean_p, cat, out_base, base_seed, idx = item
    return process_single_image_lowres(
        clean_path_str=clean_p,
        category=cat,
        output_base_dir_str=out_base,
        base_seed=base_seed,
        image_idx=idx
    )


# Motion Blur Mapping: Severity percentage S% -> motion blur kernel length L = 4 * (S / 10) + 1
# S=10% -> L=5, S=20% -> L=9, S=30% -> L=13, S=40% -> L=17, S=50% -> L=21, S=60% -> L=25, S=70% -> L=29
MOTION_BLUR_KERNEL_MAP: Dict[int, int] = {
    10: 5,
    20: 9,
    30: 13,
    40: 17,
    50: 21,
    60: 25,
    70: 29
}


def get_motion_blur_kernel(kernel_length: int, angle_deg: float = 45.0) -> np.ndarray:
    """Generates a normalized 2D linear motion blur Point Spread Function (PSF).
    
    Args:
        kernel_length: Odd integer length in pixels.
        angle_deg: Motion trajectory orientation angle in degrees (default 45.0).
        
    Returns:
        Normalized 2D float32 kernel summing to 1.0.
    """
    kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
    center = kernel_length // 2
    cv2.line(kernel, (0, center), (kernel_length - 1, center), 1.0, thickness=1)
    
    if angle_deg != 0.0:
        rot_matrix = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
        kernel = cv2.warpAffine(kernel, rot_matrix, (kernel_length, kernel_length))
        
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def apply_motion_blur(image_array: np.ndarray, 
                      kernel_length: int, 
                      angle_deg: float = 45.0) -> np.ndarray:
    """Convolves image array with a linear motion blur PSF kernel.
    
    Model:
        I_blurred(x, y) = I_clean(x, y) * h(x, y)
        sum(h) = 1.0
        
    Args:
        image_array: 2D numpy array in [0.0, 1.0].
        kernel_length: PSF kernel length in pixels.
        angle_deg: Motion angle in degrees.
        
    Returns:
        Blurred 2D numpy array clipped to [0.0, 1.0].
    """
    kernel = get_motion_blur_kernel(kernel_length, angle_deg=angle_deg)
    blurred = cv2.filter2D(image_array, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def process_single_image_motion(clean_path_str: str, 
                                category: str, 
                                output_base_dir_str: str, 
                                base_seed: int = 42,
                                image_idx: int = 0) -> Dict[str, Union[str, int, bool]]:
    """Worker function for batch / multiprocessing Motion Blur degradation generation.
    Preprocesses one clean image and independently generates all 7 severity levels.
    """
    clean_p = Path(clean_path_str)
    out_base = Path(output_base_dir_str)
    stem = clean_p.stem
    
    try:
        with Image.open(clean_p) as img:
            clean_arr = normalize_image(img, target_size=(512, 512))
            
        # Independently generate each severity level from clean_arr
        for sev, k_len in MOTION_BLUR_KERNEL_MAP.items():
            blurred_arr = apply_motion_blur(clean_arr, kernel_length=k_len, angle_deg=45.0)
            
            out_file = out_base / category / str(sev) / f"{stem}_motion_{sev}.png"
            save_image(blurred_arr, out_file, format="PNG", compress_level=1)
            
        return {
            "filename": clean_p.name,
            "category": category,
            "success": True,
            "error": ""
        }
    except Exception as e:
        return {
            "filename": clean_p.name,
            "category": category,
            "success": False,
            "error": str(e)
        }


def run_worker_motion(item: tuple) -> Dict[str, Union[str, int, bool]]:
    """Multiprocessing wrapper for process_single_image_motion."""
    clean_p, cat, out_base, base_seed, idx = item
    return process_single_image_motion(
        clean_path_str=clean_p,
        category=cat,
        output_base_dir_str=out_base,
        base_seed=base_seed,
        image_idx=idx
    )


# Combined Degradation: Fixed pipeline: Clean -> Low-Res -> Motion Blur -> Rician Noise
def apply_combined_degradation(image_array: np.ndarray,
                               severity: int,
                               seed: Optional[int] = None) -> np.ndarray:
    """Applies realistic combined MRI distortion in fixed sequential order:
    1. Low-Resolution downsampling/upsampling
    2. Linear Motion Blur PSF convolution
    3. Non-Gaussian MRI Rician magnitude noise

    Args:
        image_array: Normalized 2D float32 array in [0.0, 1.0].
        severity: Integer severity percentage (10, 20, 30, 40, 50, 60, 70).
        seed: Random seed for reproducible Rician noise.

    Returns:
        Combined degraded float32 array in [0.0, 1.0].
    """
    scale_factor  = LOW_RES_SCALE_MAP[severity]
    kernel_length = MOTION_BLUR_KERNEL_MAP[severity]
    sigma         = RICIAN_SIGMA_MAP[severity]

    # Step 1: Low-Resolution down/upsampling
    step1_lr = apply_low_resolution(image_array, scale_factor=scale_factor, target_size=(512, 512))

    # Step 2: Linear Motion Blur
    step2_blur = apply_motion_blur(step1_lr, kernel_length=kernel_length, angle_deg=45.0)

    # Step 3: Complex Rician noise
    step3_rician = add_rician_noise(step2_blur, sigma=sigma, seed=seed)

    return step3_rician


def process_single_image_combined(clean_path_str: str,
                                  category: str,
                                  output_base_dir_str: str,
                                  base_seed: int = 42,
                                  image_idx: int = 0) -> Dict[str, Union[str, int, bool]]:
    """Worker function for batch / multiprocessing combined degradation generation.
    Preprocesses one clean image and independently generates all 7 severity levels.
    """
    clean_p  = Path(clean_path_str)
    out_base = Path(output_base_dir_str)
    stem     = clean_p.stem

    try:
        with Image.open(clean_p) as img:
            clean_arr = normalize_image(img, target_size=(512, 512))

        for sev in SEVERITY_LEVELS:
            # Deterministic, unique seed per image and severity for Rician noise
            seed = (base_seed * 300000 + image_idx * 100 + sev) % (2**31 - 1)
            deg_arr = apply_combined_degradation(clean_arr, severity=sev, seed=seed)

            out_file = out_base / category / str(sev) / f"{stem}_combined_{sev}.png"
            save_image(deg_arr, out_file, format="PNG", compress_level=1)

        return {
            "filename": clean_p.name,
            "category": category,
            "success":  True,
            "error":    ""
        }
    except Exception as e:
        return {
            "filename": clean_p.name,
            "category": category,
            "success":  False,
            "error":    str(e)
        }


def run_worker_combined(item: tuple) -> Dict[str, Union[str, int, bool]]:
    """Multiprocessing wrapper for process_single_image_combined."""
    clean_p, cat, out_base, base_seed, idx = item
    return process_single_image_combined(
        clean_path_str=clean_p,
        category=cat,
        output_base_dir_str=out_base,
        base_seed=base_seed,
        image_idx=idx
    )


