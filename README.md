# Comparative Analysis of Brain MRI Image Restoration Techniques Under Different Degradation Conditions

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20%7C%20CUDA%2012.8-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()

> **BCA Final Year Capstone Project**  
> **Repository:** [https://github.com/Gitu04-stack/Brain-MRI-Image-Restoration](https://github.com/Gitu04-stack/Brain-MRI-Image-Restoration)

---

## 📌 Abstract

Magnetic Resonance Imaging (MRI) is a primary non-invasive diagnostic modality in clinical neuro-oncology. However, MRI data acquisition is inherently vulnerable to physical and physiological artifacts, including complex Rician noise from thermal coil perturbations, Additive White Gaussian Noise (AWGN), linear motion artifacts from involuntary patient motion, and spatial truncation (low resolution) induced by scan-time constraints. 

This research project conducts a rigorous mathematical and empirical comparative analysis of image restoration paradigms across **7,200 clinical T1-weighted Brain MRI scans**. We systematically simulate single and combined degradation phenomena across **seven severity levels (10% to 70%)** across four pathological classes (**Glioma, Meningioma, Pituitary, and No Tumor**). We benchmark classical spatial and statistical filters (**Adaptive Wiener Filtering, Spatial Median Filtering, Non-Local Means (NLM), and Lanczos-inspired Sharpening**) against a deep **Residual U-Net (ResUNet)** architecture equipped with multi-scale skip connections, edge-preserving Charbonnier smooth L1 loss, and global residual learning.

---

## 🎯 Research Objectives

1. **Systematic Degradation Modeling**: Formulate mathematical representations of MRI physical distortions: AWGN, complex Rician magnitude distribution, anisotropic linear motion blur, and spatial low-resolution subsampling.
2. **Multi-Severity Benchmarking**: Synthesize and evaluate distortions across 7 standardized severity levels (10%, 20%, 30%, 40%, 50%, 60%, 70%) on 5,600 training scans and 1,600 unseen testing scans.
3. **Classical Filtering Evaluation**: Empirically evaluate 2D Median Filtering, Non-Local Means (NLM), Adaptive Wiener Filtering, and Lanczos edge sharpening under identical experimental controls.
4. **Deep Residual Learning**: Design, optimize, and train a 4-level Residual U-Net (1,926,433 parameters) on combined multi-degradations to surpass classical filtering limits.
5. **Quantitative & Visual Validation**: Quantitatively assess restoration fidelity using Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR in dB), and Structural Similarity Index (SSIM).
6. **Clinical Web Dashboard**: Deploy an interactive, responsive Flask-based web application providing real-time artifact simulation, filtering execution, side-by-side visualization, and empirical research analytics.

---

## 🔬 Scientific Methodology

### 1. Mathematical Degradation Models

```mermaid
flowchart LR
    Clean["Clean MRI<br/>512x512 Grayscale"] --> LowRes["1. Low-Resolution<br/>Matrix Subsampling"]
    LowRes --> Motion["2. Linear Motion Blur<br/>Point Spread Function (45°)"]
    Motion --> Rician["3. Complex Rician Noise<br/>Magnitude Transformation"]
    Rician --> Degraded["Combined Degraded Slice<br/>Severity: 10% to 70%"]
```

* **Additive White Gaussian Noise (AWGN)**:
  $$\tilde{I}(x, y) = \text{clip}\left(I(x, y) + \eta(x, y), 0.0, 1.0\right), \quad \eta \sim \mathcal{N}(0, \sigma^2), \quad \sigma = \frac{S}{100}$$
* **Complex Rician Noise**:
  $$\tilde{I}(x, y) = \sqrt{\left(I(x, y) + \eta_r(x, y)\right)^2 + \eta_i(x, y)^2}, \quad \eta_r, \eta_i \sim \mathcal{N}(0, \sigma^2)$$
* **Low-Resolution Degradation**:
  Spatially downsamples the $512 \times 512$ matrix to target dimensions $D \in [460 \times 460 \dots 154 \times 154]$ via bilinear interpolation, followed by dimension restoration to $512 \times 512$.
* **Linear Motion Blur**:
  Convolves image slices with an oriented line kernel of length $L \in [3 \dots 15]$ at $\theta = 45^\circ$:
  $$\tilde{I} = I * K_\theta$$
* **Multi-Phase Combined Distortion**:
  Applies the sequence: $\text{Clean} \rightarrow \text{Low-Resolution} \rightarrow \text{Motion Blur} \rightarrow \text{Rician Noise}$.

---

### 2. Restoration Algorithms

1. **Adaptive Wiener Filter ($5 \times 5$ window)**:
   Minimizes local Mean Squared Error by estimating local mean $\mu$ and variance $\sigma^2$ across spatial neighborhoods.
2. **Spatial Median Filter ($3 \times 3$ kernel)**:
   Non-linear spatial filter replacing central pixel intensities with median rank values, robust against extreme outliers.
3. **Non-Local Means (NLM)**:
   Exploits self-similarity in MRI anatomical structures by computing weighted patch averages across search windows ($h=10$).
4. **Lanczos Sharpening (Unsharp Masking)**:
   Recovers high-frequency boundary energy attenuated by downsampling and motion blur:
   $$I_{\text{sharp}} = \text{clip}\left(I + 1.5 \cdot (I - \text{GaussianBlur}(I, 1.0)), 0.0, 1.0\right)$$
5. **Deep Residual U-Net (ResUNet)**:
   A 4-level convolutional encoder-decoder with global residual skip learning:
   $$I_{\text{restored}} = \text{clip}\left(I_{\text{degraded}} + \mathcal{F}(I_{\text{degraded}}; \Theta), 0.0, 1.0\right)$$
   Trained with edge-preserving **Charbonnier Loss**:
   $$\mathcal{L}_{\text{char}}(y, \hat{y}) = \frac{1}{N}\sum_{i=1}^N \sqrt{(y_i - \hat{y}_i)^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$

---

### 3. Quantitative Evaluation Metrics

* **Mean Squared Error (MSE)**:
  $$\text{MSE} = \frac{1}{M \cdot N} \sum_{x=1}^M \sum_{y=1}^N \left[I_{\text{clean}}(x,y) - I_{\text{restored}}(x,y)\right]^2$$
* **Peak Signal-to-Noise Ratio (PSNR)**:
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right) \quad (\text{MAX} = 1.0)$$
* **Structural Similarity Index (SSIM)**:
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

---

## 📊 Key Experimental Findings

### Phase 11 Traditional Restoration Benchmark (44,800 Unseen Test Scans)

| Rank | Restoration Algorithm | Test PSNR (dB) ↑ | Std PSNR (±) | Test SSIM ↑ | Std SSIM (±) | Test MSE ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Adaptive Wiener Filter (5×5)** | **10.42 dB** | ±5.76 | **0.1808** | ±0.1367 | **0.1612** |
| **#2** | Spatial Median Filter (3×3) | 10.13 dB | ±6.15 | 0.1208 | ±0.1343 | 0.1874 |
| **#3** | Non-Local Means (NLM) | 8.82 dB | ±5.26 | 0.0528 | ±0.0767 | 0.2108 |
| **#4** | Lanczos Sharpening | 6.91 dB | ±3.46 | 0.0209 | ±0.0297 | 0.2557 |

### Phase 12 Deep ResUNet Performance vs. Traditional Baseline

| Metric | Traditional Wiener Baseline | Deep ResUNet (Epoch 6 Peak) | Performance Gain |
| :--- | :---: | :---: | :---: |
| **Validation PSNR** | 10.42 dB | **21.07 dB** | **+10.65 dB (Doubled Signal Quality)** |
| **Validation SSIM** | 0.1808 | **0.5001** | **+0.3193 (2.77× Structural Fidelity)** |
| **Training Loss** | N/A | **0.04828** | Continuous Downward Convergence |

---

## 🗂️ Project Repository Structure

```text
Brain-MRI-Image-Restoration/
├── .gitignore                     # Git tracking exclusions
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Verified Python dependencies
├── notebooks/                     # Executed research Jupyter Notebooks
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_gaussian_noise.ipynb
│   ├── 03_rician_noise.ipynb
│   ├── 04_low_resolution.ipynb
│   ├── 05_motion_blur.ipynb
│   ├── 06_traditional_restoration.ipynb
│   ├── 07_metrics_and_training_comparison.ipynb
│   ├── 08_testing_evaluation.ipynb
│   ├── 09_combined_degradation.ipynb
│   ├── 10_combined_restoration.ipynb
│   └── 11_combined_testing.ipynb
├── src/                           # Modular scientific source code
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch CombinedMRIDataset
│   ├── degradation.py             # Degradation mathematical models
│   ├── metrics.py                 # MSE, PSNR, SSIM evaluation
│   ├── models.py                  # Deep Residual U-Net architecture
│   ├── preprocessing.py           # Normalization & image standardization
│   ├── restoration.py             # Classical restoration filter implementations
│   ├── run_phase12_training.py    # Resumable PyTorch training engine (AMP + cuDNN)
│   └── ...                        # Phase automation & worker scripts
├── outputs/                       # Verified research outputs & metrics
│   ├── metrics/
│   │   ├── phase11_traditional_benchmark_summary.csv
│   │   ├── phase11_combined_category_breakdown.csv
│   │   ├── phase11_combined_severity_breakdown.csv
│   │   ├── phase12_training_loss.csv
│   │   └── ...                    # Statistical summaries
│   └── *.png                      # High-resolution benchmark figures
└── webapp/                        # Modern Flask Web Application
    ├── app.py                     # Flask backend & API routing
    ├── requirements.txt           # Webapp dependencies
    ├── README.md                  # Webapp setup guide
    ├── static/                    # CSS, JS, and uploads
    └── templates/                 # Bootstrap 5 responsive templates
        ├── base.html
        ├── index.html
        ├── upload.html
        ├── degrade.html
        ├── restore.html
        ├── compare.html
        ├── metrics.html
        ├── benchmarks.html
        └── deeplearning.html
```

---

## ⚙️ Installation & Setup (Windows)

### 1. Prerequisites
* **Operating System**: Windows 10 / 11
* **Environment Manager**: Anaconda or Miniconda
* **Compute (Optional for Traditional Filters; Recommended for CNN)**: NVIDIA GPU with CUDA support (tested on RTX 2050 4GB).

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Gitu04-stack/Brain-MRI-Image-Restoration.git
cd Brain-MRI-Image-Restoration

# Create and activate Conda environment
conda create -n brainmri python=3.10 -y
conda activate brainmri

# Install verified dependencies
pip install -r requirements.txt
```

### 3. Launching the Interactive Web Application
```powershell
cd webapp
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 📈 Current Project Status

* [x] **Phase 1**: Dataset verification and pathological stratification (7,200 scans).
* [x] **Phase 2**: Additive White Gaussian Noise (AWGN) modeling and evaluation.
* [x] **Phase 3**: Complex Rician Noise simulation and distribution testing.
* [x] **Phase 4**: Low-Resolution downsampling and interpolation analysis.
* [x] **Phase 5**: Point Spread Function (PSF) linear motion blur modeling.
* [x] **Phase 6**: Traditional restoration benchmarking on training splits (627,200 evaluations).
* [x] **Phase 7**: Training metric aggregation and statistical analysis.
* [x] **Phase 8**: Unseen testing split traditional evaluation (179,200 evaluations).
* [x] **Phase 9**: Combined multi-degradation dataset generation (50,400 paired scans).
* [x] **Phase 10**: Traditional restoration benchmarking on combined distortions (201,600 evaluations).
* [x] **Phase 11**: Final traditional benchmark synthesis (Wiener confirmed as #1 baseline).
* [x] **Phase 12 (In Active Progress)**: Deep Residual U-Net training (Epoch 6 of 10 completed with cuDNN autotuning; peak PSNR **21.07 dB**, peak SSIM **0.5001**; Epoch 7 actively training).
* [ ] **Phase 13**: Unseen testing evaluation of trained Deep ResUNet model against Phase 11 baseline.

---

## ⚠️ Limitations & Future Scope

* **Memory Constraints**: High-resolution $512 \times 512$ batch processing on consumer 4 GB VRAM GPUs requires mixed-precision (AMP) and batch size $\le 8$.
* **Motion Complexity**: Current motion modeling employs 1D linear angle kernels; non-rigid patient head rotation and non-linear trajectory modeling represent promising future directions.
* **Diffusion & Vision Transformers**: Extending beyond convolutional U-Nets to Denoising Diffusion Probabilistic Models (DDPM) or Medical Swin-Transformers represents an exciting avenue for generative detail synthesis.

---

## 📜 License & Academic Integrity

This project is submitted in partial fulfillment of the requirements for the **Bachelor of Computer Applications (BCA)** degree. All data, experimental evaluations, and code implementations adhere strictly to academic integrity standards.
