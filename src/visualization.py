"""Visualization module for Brain MRI restoration project.
Provides comparative grid plotting and metric curve plotting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_comparison(rows_data: List[Dict[str, Any]], 
                    column_names: List[str], 
                    title: str, 
                    figsize: Optional[Tuple[int, int]] = None,
                    output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Plots a comparative grid showing slices across conditions/severities.
    
    Args:
        rows_data: List of dicts, each dict mapping column_name -> 2D numpy array or Image.
        column_names: List of column header names.
        title: Overall super-title.
        figsize: Figure size tuple.
        output_path: Optional path to save figure.
    """
    n_rows = len(rows_data)
    n_cols = len(column_names)
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3.5)
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)
        
    for r, row_dict in enumerate(rows_data):
        row_label = row_dict.get("_row_label", "")
        for c, col_name in enumerate(column_names):
            ax = axes[r, c]
            img = row_dict.get(col_name)
            if img is not None:
                if isinstance(img, Image.Image):
                    img = np.array(img) / 255.0
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"{row_label}\n{col_name}" if r == 0 and row_label else col_name, fontsize=9)
            ax.axis('off')
            
    plt.suptitle(title, fontsize=13, y=0.99)
    plt.tight_layout()
    
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches='tight')
        
    return fig
