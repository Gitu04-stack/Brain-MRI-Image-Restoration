"""Generates professional diagrams for the Brain MRI Restoration Final Project Report & Presentation:
1. System Architecture Diagram
2. Data Flow Diagram Level 0 (Context Level)
3. Data Flow Diagram Level 1
4. Data Flow Diagram Level 2 (Restoration Subsystem)
5. Entity-Relationship (ER) Diagram
6. Web Application Architecture Diagram
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = Path(r"C:\Users\gitan\BrainMRI_Project\outputs\diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_system_architecture():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(6, 6.6, "SYSTEM ARCHITECTURE: BRAIN MRI RESTORATION PLATFORM", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1A365D')

    # Color palette
    c_blue = '#2B6CB0'
    c_teal = '#319795'
    c_orange = '#DD6B20'
    c_purple = '#805AD5'
    c_dark = '#2D3748'
    c_bg = '#EDF2F7'

    # Module 1: Input Data
    ax.add_patch(patches.FancyBboxPatch((0.5, 3.8), 2.2, 2.2, boxstyle="round,pad=0.1", fc='#EBF8FF', ec=c_blue, lw=2))
    ax.text(1.6, 5.5, "1. INPUT & PREPROCESS", ha='center', va='center', fontweight='bold', fontsize=10, color=c_blue)
    ax.text(1.6, 4.7, "• 7,200 Clinical Scans\n• 4 Pathological Classes\n• Grayscale 512×512\n• Min-Max [0.0, 1.0]", 
            ha='center', va='center', fontsize=8, color=c_dark)

    # Module 2: Degradation Engine
    ax.add_patch(patches.FancyBboxPatch((3.2, 3.8), 2.4, 2.2, boxstyle="round,pad=0.1", fc='#FFFAF0', ec=c_orange, lw=2))
    ax.text(4.4, 5.5, "2. DEGRADATION ENGINE", ha='center', va='center', fontweight='bold', fontsize=10, color=c_orange)
    ax.text(4.4, 4.7, "• Gaussian Noise (AWGN)\n• Complex Rician Noise\n• Low-Res Subsampling\n• 45° Linear Motion Blur\n• 7 Severities (10%-70%)", 
            ha='center', va='center', fontsize=8, color=c_dark)

    # Module 3A: Classical Filters
    ax.add_patch(patches.FancyBboxPatch((6.1, 4.8), 2.5, 1.4, boxstyle="round,pad=0.1", fc='#F0FFF4', ec=c_teal, lw=2))
    ax.text(7.35, 5.8, "3A. CLASSICAL FILTERS", ha='center', va='center', fontweight='bold', fontsize=9, color=c_teal)
    ax.text(7.35, 5.25, "• Adaptive Wiener (5×5)\n• Median Filter (3×3)\n• Non-Local Means (NLM)\n• Lanczos Sharpening", 
            ha='center', va='center', fontsize=8, color=c_dark)

    # Module 3B: Deep Learning
    ax.add_patch(patches.FancyBboxPatch((6.1, 3.1), 2.5, 1.4, boxstyle="round,pad=0.1", fc='#FAF5FF', ec=c_purple, lw=2))
    ax.text(7.35, 4.1, "3B. DEEP LEARNING", ha='center', va='center', fontweight='bold', fontsize=9, color=c_purple)
    ax.text(7.35, 3.55, "• Deep ResUNet (4-Level)\n• 1.92M Parameters\n• Charbonnier Loss\n• Global Residual Skip", 
            ha='center', va='center', fontsize=8, color=c_dark)

    # Module 4: Quantitative Evaluation
    ax.add_patch(patches.FancyBboxPatch((9.1, 3.8), 2.4, 2.2, boxstyle="round,pad=0.1", fc='#EBF8FF', ec=c_blue, lw=2))
    ax.text(10.3, 5.5, "4. EVALUATION SUITE", ha='center', va='center', fontweight='bold', fontsize=10, color=c_blue)
    ax.text(10.3, 4.7, "• MSE Calculation\n• PSNR (dB) Peak Quality\n• SSIM Structural Metric\n• 11,200 Unseen Tests\n• Best Method Ranking", 
            ha='center', va='center', fontsize=8, color=c_dark)

    # Module 5: Clinical Web Application
    ax.add_patch(patches.FancyBboxPatch((1.5, 0.6), 9.0, 2.0, boxstyle="round,pad=0.1", fc='#F7FAFC', ec='#4A5568', lw=2))
    ax.text(6.0, 2.2, "5. CLINICAL RESEARCH WEB DASHBOARD (FLASK + JINJA2 + BOOTSTRAP 5)", 
            ha='center', va='center', fontweight='bold', fontsize=11, color='#2D3748')
    ax.text(6.0, 1.35, "• Interactive Slice Upload & Sample Presets  • Real-time Artifact Simulation (5 Types & 7 Severities)\n• 5-Way Concurrent Restoration Benchmark  • Triplet Visual Inspection & Metric Comparison\n• Empirical Recommendation Engine  • High-Resolution Restored Image Export", 
            ha='center', va='center', fontsize=8.5, color='#4A5568')

    # Arrows
    arrow_kw = dict(arrowstyle='->', lw=2, color='#4A5568')
    ax.annotate('', xy=(3.2, 4.9), xytext=(2.7, 4.9), arrowprops=arrow_kw)
    ax.annotate('', xy=(6.1, 5.5), xytext=(5.6, 5.1), arrowprops=arrow_kw)
    ax.annotate('', xy=(6.1, 3.8), xytext=(5.6, 4.7), arrowprops=arrow_kw)
    ax.annotate('', xy=(9.1, 5.1), xytext=(8.6, 5.5), arrowprops=arrow_kw)
    ax.annotate('', xy=(9.1, 4.7), xytext=(8.6, 3.8), arrowprops=arrow_kw)

    ax.annotate('', xy=(6.0, 2.6), xytext=(6.0, 3.1), arrowprops=arrow_kw)
    ax.annotate('', xy=(10.3, 2.6), xytext=(10.3, 3.8), arrowprops=arrow_kw)
    ax.annotate('', xy=(1.6, 2.6), xytext=(1.6, 3.8), arrowprops=arrow_kw)

    plt.tight_layout()
    out_p = OUTPUT_DIR / "system_architecture.png"
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"[OK] Saved {out_p.name}")


def create_dfd_diagrams():
    # DFD Level 0 (Context Diagram)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    ax.text(5, 4.6, "DATA FLOW DIAGRAM (DFD) - LEVEL 0 (CONTEXT DIAGRAM)", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#1A365D')

    # External Entity: Clinical Researcher / Radiologist
    ax.add_patch(patches.Rectangle((0.6, 1.8), 2.2, 1.6, fc='#E2E8F0', ec='#2D3748', lw=2))
    ax.text(1.7, 2.6, "Radiologist /\nClinical Researcher\n(External Entity)", ha='center', va='center', fontweight='bold', fontsize=9)

    # Process 0.0: Central System
    circle = patches.Circle((5.0, 2.6), 1.3, fc='#EBF8FF', ec='#2B6CB0', lw=2.5)
    ax.add_patch(circle)
    ax.text(5.0, 2.8, "0.0", ha='center', va='center', fontweight='bold', fontsize=12, color='#2B6CB0')
    ax.text(5.0, 2.3, "Brain MRI Image\nRestoration &\nAnalysis System", ha='center', va='center', fontweight='bold', fontsize=9)

    # Data Store: MRI Archive & Checkpoints
    ax.add_patch(patches.Rectangle((7.5, 1.8), 2.0, 1.6, fc='#FEFCBF', ec='#D69E2E', lw=2))
    ax.text(8.5, 2.6, "D1: Clean Scans\n& ResUNet Weights\n(Data Store)", ha='center', va='center', fontweight='bold', fontsize=9)

    # Flows
    arrow_kw = dict(arrowstyle='->', lw=1.8, color='#2D3748')
    # User -> System
    ax.annotate('', xy=(3.7, 2.9), xytext=(2.8, 2.9), arrowprops=arrow_kw)
    ax.text(3.25, 3.2, "Upload Slice /\nDegradation Config", ha='center', va='center', fontsize=7.5, color='#2B6CB0')

    # System -> User
    ax.annotate('', xy=(2.8, 2.3), xytext=(3.7, 2.3), arrowprops=arrow_kw)
    ax.text(3.25, 2.0, "Restored Slices,\nMSE/PSNR/SSIM, Rec.", ha='center', va='center', fontsize=7.5, color='#319795')

    # System <-> Data Store
    ax.annotate('', xy=(7.5, 2.8), xytext=(6.3, 2.8), arrowprops=arrow_kw)
    ax.text(6.9, 3.1, "Fetch Scans/Weights", ha='center', va='center', fontsize=7.5)
    ax.annotate('', xy=(6.3, 2.4), xytext=(7.5, 2.4), arrowprops=arrow_kw)
    ax.text(6.9, 2.1, "Metrics Logging", ha='center', va='center', fontsize=7.5)

    plt.tight_layout()
    out_p = OUTPUT_DIR / "dfd_level0.png"
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"[OK] Saved {out_p.name}")

    # DFD Level 1
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(5.5, 5.6, "DATA FLOW DIAGRAM (DFD) - LEVEL 1", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#1A365D')

    procs = [
        (1.2, 3.8, "1.0", "Image\nIngestion &\nNorm (512×512)"),
        (3.5, 3.8, "2.0", "Artifact\nDegradation\nModeling"),
        (5.8, 3.8, "3.0", "Multi-Filter &\nDeep ResUNet\nRestoration"),
        (8.1, 3.8, "4.0", "Quantitative\nMSE/PSNR/SSIM\nEvaluation"),
        (5.8, 1.4, "5.0", "Web Dashboard\nVisualization &\nExport")
    ]

    for x, y, num, title in procs:
        circ = patches.Circle((x, y), 0.75, fc='#EBF8FF', ec='#2B6CB0', lw=2)
        ax.add_patch(circ)
        ax.text(x, y + 0.25, num, ha='center', va='center', fontweight='bold', fontsize=10, color='#2B6CB0')
        ax.text(x, y - 0.2, title, ha='center', va='center', fontsize=7.5, fontweight='semibold')

    # Connectors
    ax.annotate('', xy=(2.75, 3.8), xytext=(1.95, 3.8), arrowprops=arrow_kw)
    ax.text(2.35, 4.15, "Clean\n[0.0, 1.0]", ha='center', va='center', fontsize=7)

    ax.annotate('', xy=(5.05, 3.8), xytext=(4.25, 3.8), arrowprops=arrow_kw)
    ax.text(4.65, 4.15, "Degraded\nMatrix", ha='center', va='center', fontsize=7)

    ax.annotate('', xy=(7.35, 3.8), xytext=(6.55, 3.8), arrowprops=arrow_kw)
    ax.text(6.95, 4.15, "Restored\nOutput", ha='center', va='center', fontsize=7)

    ax.annotate('', xy=(6.3, 2.15), xytext=(7.8, 3.2), arrowprops=arrow_kw)
    ax.text(7.3, 2.6, "Fidelity Scores", ha='center', va='center', fontsize=7)

    ax.annotate('', xy=(5.8, 2.15), xytext=(5.8, 3.05), arrowprops=arrow_kw)
    ax.text(6.3, 2.6, "Render\nImage", ha='center', va='center', fontsize=7)

    # Data stores
    ax.add_patch(patches.Rectangle((0.4, 1.2), 1.8, 1.0, fc='#FEFCBF', ec='#D69E2E', lw=1.5))
    ax.text(1.3, 1.7, "D1: T1 MRI Scans\n(7,200 Files)", ha='center', va='center', fontsize=8, fontweight='bold')
    ax.annotate('', xy=(1.2, 3.05), xytext=(1.3, 2.2), arrowprops=arrow_kw)

    ax.add_patch(patches.Rectangle((8.6, 1.2), 2.0, 1.0, fc='#FEFCBF', ec='#D69E2E', lw=1.5))
    ax.text(9.6, 1.7, "D2: Benchmark CSVs\n& Metric History", ha='center', va='center', fontsize=8, fontweight='bold')
    ax.annotate('', xy=(9.6, 2.2), xytext=(8.6, 3.2), arrowprops=arrow_kw)

    plt.tight_layout()
    out_p = OUTPUT_DIR / "dfd_level1.png"
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"[OK] Saved {out_p.name}")

    # DFD Level 2 (Restoration Subsystem 3.0)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(5.5, 5.6, "DATA FLOW DIAGRAM (DFD) - LEVEL 2: RESTORATION PROCESS (3.0)", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#1A365D')

    # Input node
    ax.add_patch(patches.Rectangle((0.4, 2.3), 1.6, 1.2, fc='#E2E8F0', ec='#4A5568', lw=1.5))
    ax.text(1.2, 2.9, "Degraded MRI\nMatrix\n(512×512)", ha='center', va='center', fontsize=8, fontweight='bold')

    # Sub-processes
    subs = [
        (4.0, 4.4, "3.1", "Adaptive Wiener Filter\nLocal Mean/Variance (5×5)"),
        (4.0, 3.2, "3.2", "Spatial Median Filter\nRank 3×3 Neighborhood"),
        (4.0, 2.0, "3.3", "Non-Local Means (NLM)\nPatch Self-Similarity"),
        (4.0, 0.8, "3.4", "Lanczos Sharpening\nHigh-Pass Unsharp Mask"),
        (7.5, 2.6, "3.5", "Deep Residual U-Net\nResidual Skip Connections\n(PyTorch / CUDA)")
    ]

    for x, y, num, desc in subs:
        c = '#FAF5FF' if num == '3.5' else '#EBF8FF'
        b = '#805AD5' if num == '3.5' else '#2B6CB0'
        circ = patches.Circle((x, y), 0.6, fc=c, ec=b, lw=2)
        ax.add_patch(circ)
        ax.text(x, y + 0.15, num, ha='center', va='center', fontweight='bold', fontsize=9, color=b)
        ax.text(x, y - 0.15, "Process", ha='center', va='center', fontsize=7)
        ax.text(x + 0.75, y, desc, ha='left', va='center', fontsize=7.5, color='#2D3748', fontweight='semibold')

    # Model Weights Store
    ax.add_patch(patches.Rectangle((7.0, 4.4), 2.2, 0.9, fc='#FEFCBF', ec='#D69E2E', lw=1.5))
    ax.text(8.1, 4.85, "D3: ResUNet Weights\n(best_restoration_model.pth)", ha='center', va='center', fontsize=7.5, fontweight='bold')
    ax.annotate('', xy=(7.5, 3.2), xytext=(7.8, 4.4), arrowprops=arrow_kw)

    # Connections from Input
    for y_pos in [4.4, 3.2, 2.0, 0.8]:
        ax.annotate('', xy=(3.4, y_pos), xytext=(2.0, 2.9), arrowprops=arrow_kw)
    ax.annotate('', xy=(6.9, 2.6), xytext=(2.0, 2.9), arrowprops=dict(arrowstyle='->', lw=2, color='#805AD5'))

    plt.tight_layout()
    out_p = OUTPUT_DIR / "dfd_level2.png"
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"[OK] Saved {out_p.name}")


def create_er_diagram():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(5, 5.6, "ENTITY-RELATIONSHIP (ER) DIAGRAM: MRI RESTORATION DATABASE", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#1A365D')

    # Entities
    entities = [
        (1.5, 3.8, "PATIENT_SCAN", ["# Scan_ID (PK)", "• Slice_Index", "• Pathological_Class", "• Dimension (512×512)", "• Split (Train/Test)"]),
        (5.0, 3.8, "DEGRADATION_INSTANCE", ["# Deg_ID (PK)", "# Scan_ID (FK)", "• Distortion_Type", "• Severity_Level (10-70%)", "• Parameter_Value"]),
        (8.5, 3.8, "RESTORATION_OUTPUT", ["# Restored_ID (PK)", "# Deg_ID (FK)", "• Algorithm_Name", "• Execution_Time_ms", "• Artifact_Path"]),
        (5.0, 1.2, "EVALUATION_METRIC", ["# Metric_ID (PK)", "# Restored_ID (FK)", "• MSE_Value", "• PSNR_dB", "• SSIM_Score", "• Timestamp"])
    ]

    for x, y, name, attrs in entities:
        ax.add_patch(patches.FancyBboxPatch((x - 1.2, y - 1.0), 2.4, 2.0, boxstyle="round,pad=0.05", fc='#EDF2F7', ec='#2B6CB0', lw=2))
        ax.add_patch(patches.Rectangle((x - 1.2, y + 0.6), 2.4, 0.4, fc='#2B6CB0', ec='#2B6CB0'))
        ax.text(x, y + 0.8, name, ha='center', va='center', fontweight='bold', fontsize=8.5, color='white')
        ax.text(x - 1.0, y - 0.15, "\n".join(attrs), ha='left', va='center', fontsize=7.5, color='#2D3748', family='monospace')

    # Relationships
    line_kw = dict(lw=2, color='#4A5568')
    # Scan -> Degradation
    ax.plot([2.7, 3.8], [3.8, 3.8], **line_kw)
    ax.text(3.25, 4.0, "1 : N\nUndergoes", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#DD6B20')

    # Degradation -> Restoration
    ax.plot([6.2, 7.3], [3.8, 3.8], **line_kw)
    ax.text(6.75, 4.0, "1 : N\nRestored By", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#319795')

    # Restoration -> Evaluation
    ax.plot([8.5, 8.5, 6.2], [2.8, 1.2, 1.2], **line_kw)
    ax.text(7.5, 1.4, "1 : 1 Evaluated As", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#805AD5')

    plt.tight_layout()
    out_p = OUTPUT_DIR / "er_diagram.png"
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"[OK] Saved {out_p.name}")


if __name__ == '__main__':
    create_system_architecture()
    create_dfd_diagrams()
    create_er_diagram()
    print("All architectural diagrams generated successfully.")
