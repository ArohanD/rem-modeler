"""Interface helpers for the GIS 584 project."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling


def preview_tif(raster_path: str | Path, band: int = 1) -> None:
    """
    Open and display a raster file with interactive pixel value viewer.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    band : int, optional
        Band number to display (1-indexed), default is 1
    """
    raster_path = Path(raster_path)
    
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    
    with rasterio.open(raster_path) as src:
        if band < 1 or band > src.count:
            raise ValueError(f"Band {band} out of range. File has {src.count} band(s).")
        
        # Downsample for fast display (max 2000x2000 pixels)
        max_dim = 2000
        height, width = src.height, src.width
        scale = max(height / max_dim, width / max_dim, 1.0)
        out_height = int(height / scale)
        out_width = int(width / scale)
        
        # Read downsampled data for display
        display_data = src.read(
            band,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear
        )
        
        # Read full resolution data for pixel lookups (stored in memory)
        full_data = src.read(band)
        
        nodata = src.nodata
        transform = src.transform
        
        # Mask nodata
        if nodata is not None:
            display_data = np.ma.masked_equal(display_data, nodata)
            full_data = np.ma.masked_equal(full_data, nodata)
        
        # Quick stats from display data
        vmin, vmax = np.nanpercentile(display_data.compressed() if np.ma.is_masked(display_data) else display_data, [2, 98])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(display_data, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Value')
    
    title = ax.set_title(f'{raster_path.name} (Band {band})\nHover for pixel values')
    
    def on_move(event):
        if event.inaxes != ax:
            return
        
        # Convert display coords to full resolution coords
        display_col = int(event.xdata + 0.5)
        display_row = int(event.ydata + 0.5)
        
        if 0 <= display_row < out_height and 0 <= display_col < out_width:
            # Map to full resolution
            full_col = int(display_col * scale)
            full_row = int(display_row * scale)
            
            if 0 <= full_row < height and 0 <= full_col < width:
                value = full_data[full_row, full_col]
                
                # Get geo coords
                geo_x, geo_y = transform * (full_col + 0.5, full_row + 0.5)
                
                val_str = "NoData" if np.ma.is_masked(value) else f"{value:.2f}"
                title.set_text(
                    f'{raster_path.name} (Band {band})\n'
                    f'Pixel: ({full_row}, {full_col}) | Value: {val_str} | Coords: ({geo_x:.1f}, {geo_y:.1f})'
                )
                fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.tight_layout()
    plt.show()


__all__ = ["preview_tif"]