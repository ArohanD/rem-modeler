"""Interface helpers for the GIS 584 project."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import rasterio
from rasterio.enums import Resampling


def preview_tif(raster_path: str | Path, band: int = 1, minmax: tuple[float | None, float | None] = (None, None)) -> tuple[float, float]:
    """
    Open and display a raster file with interactive pixel value viewer.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    band : int, optional
        Band number to display (1-indexed), default is 1
    minmax : tuple[float | None, float | None], optional
        Min and max values for color range. If (None, None), auto-calculated from data.
        
    Returns
    -------
    tuple[float, float]
        The final (min, max) values when the window is closed.
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
        
        # Calculate data range for sliders
        data_for_stats = display_data.compressed() if np.ma.is_masked(display_data) else display_data
        data_min = float(np.nanmin(data_for_stats))
        data_max = float(np.nanmax(data_for_stats))
        
        # Use provided minmax or auto-calculate
        if minmax[0] is not None and minmax[1] is not None:
            init_vmin, init_vmax = minmax
        else:
            init_vmin, init_vmax = np.nanpercentile(data_for_stats, [2, 98])
    
    # Create figure with space for input controls
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes([0.1, 0.25, 0.8, 0.7])
    
    im = ax.imshow(display_data, cmap='YlGnBu', vmin=init_vmin, vmax=init_vmax)
    cbar = plt.colorbar(im, ax=ax, label='Value')
    
    title = ax.set_title(f'{raster_path.name} (Band {band})\nHover for pixel values')
    
    # Create text input boxes and buttons
    ax_min = plt.axes([0.15, 0.12, 0.12, 0.04])
    ax_max = plt.axes([0.32, 0.12, 0.12, 0.04])
    ax_update = plt.axes([0.49, 0.12, 0.12, 0.04])
    ax_done = plt.axes([0.66, 0.12, 0.12, 0.04])
    
    textbox_min = TextBox(ax_min, 'Min:', initial=f'{init_vmin:.2f}')
    textbox_max = TextBox(ax_max, 'Max:', initial=f'{init_vmax:.2f}')
    button_update = Button(ax_update, 'Update')
    button_done = Button(ax_done, 'Done')
    
    # Store current values to return when window closes
    current_values = {'min': init_vmin, 'max': init_vmax}
    
    def update(event):
        try:
            vmin = float(textbox_min.text)
            vmax = float(textbox_max.text)
            if vmin < vmax:
                im.set_clim(vmin, vmax)
                current_values['min'] = vmin
                current_values['max'] = vmax
                fig.canvas.draw_idle()
        except ValueError:
            pass  # Ignore invalid input
    
    def done(event):
        plt.close(fig)
    
    button_update.on_clicked(update)
    button_done.on_clicked(done)
    
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
    plt.show()
    
    return current_values['min'], current_values['max']


__all__ = ["preview_tif"]