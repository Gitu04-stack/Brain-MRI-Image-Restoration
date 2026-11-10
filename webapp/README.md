# Brain MRI Image Restoration — Web Application

A modern, responsive Flask web application for clinical Brain MRI image restoration research.

Developed as part of the BCA Final Year Capstone Project:
**"Comparative Analysis of Brain MRI Image Restoration Techniques Under Different Degradation Conditions"**

---

## Key Features

1. **Academic Dashboard**: High-level overview of the 7,200 verified MRI dataset, multi-degradation models, and traditional vs. deep learning benchmarks.
2. **Standardized MRI Intake**: Upload local T1-weighted MRI slices or load authentic reference scans (Glioma, Meningioma, Pituitary, No Tumor) standardized to $512 \times 512$ grayscale float32 $[0.0, 1.0]$.
3. **Artifact Degradation Simulator**: Mathematically simulates:
   - Additive White Gaussian Noise (AWGN)
   - Complex Rician Noise (magnitude distribution)
   - Low-Resolution Subsampling (matrix truncation)
   - Linear Motion Blur (45° angle)
   - Multi-Phase Combined Distortion (Low-Res $\rightarrow$ Motion Blur $\rightarrow$ Rician Noise)
   Across 7 severity levels ($10\%$ through $70\%$).
4. **Spatial & Statistical Restoration Engine**:
   - Adaptive Wiener Filter ($5 \times 5$ window) — **Rank #1 Benchmark**
   - 2D Spatial Median Filter ($3 \times 3$ kernel)
   - Non-Local Means (NLM Denoising)
   - Lanczos-inspired Edge Sharpening (unsharp mask)
   - One-click multi-method concurrent execution
5. **Triplet & Side-by-Side Comparison**: Direct visual inspection comparing Clean Reference, Degraded Input, and Restored Outputs.
6. **Quantitative Metric Evaluation**: Automated calculation and detailed mathematical breakdown of Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR in dB), and Structural Similarity Index (SSIM).
7. **Empirical Research Benchmarks**: Interactive charts (Chart.js) and statistical summaries drawn directly from verified project experiment tables across 44,800 testing runs.
8. **Deep ResUNet Model Telemetry**: Displays the 4-level deep residual U-Net architecture (1,926,433 weights), Charbonnier loss formulation, and live Phase 12 training metrics.

---

## Directory Structure

```text
BrainMRI_Project/
├── webapp/
│   ├── app.py                # Main Flask application and verified routes
│   ├── requirements.txt      # Web application dependencies
│   ├── README.md             # Launch & documentation guide
│   ├── static/
│   │   ├── css/style.css     # Modern responsive styling (Navy/Slate/Indigo)
│   │   ├── js/main.js        # Drag-and-drop upload & interactive controls
│   │   └── uploads/          # Isolated session uploads and outputs
│   └── templates/
│       ├── base.html         # Responsive sidebar layout & topbar
│       ├── index.html        # Dashboard overview
│       ├── upload.html       # Image upload & reference inspector
│       ├── degrade.html      # Degradation simulation controls
│       ├── restore.html      # Restoration filter execution
│       ├── compare.html      # Side-by-side comparison
│       ├── metrics.html      # Performance metrics breakdown
│       ├── benchmarks.html   # Empirical benchmark charts
│       └── deeplearning.html # Deep learning architecture & telemetry
```

---

## How to Launch on Windows

### 1. Open PowerShell or Command Prompt
Activate the project Conda environment:
```powershell
conda activate brainmri
```

### 2. Navigate to the Webapp Directory
```powershell
cd C:\Users\gitan\BrainMRI_Project\webapp
```

### 3. Start the Flask Server
```powershell
python app.py
```

### 4. Open in Web Browser
Navigate to:
```text
http://127.0.0.1:5000
```
