"""Interface helpers for the GIS 584 project."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, Slider
import rasterio
from rasterio.enums import Resampling


@dataclass
class ViewerState:
    """State container for interactive raster viewer."""
    raster_path: Path
    band: int
    display_data: np.ndarray
    full_data: np.ndarray
    transform: Any
    height: int
    width: int
    scale: float
    out_height: int
    out_width: int
    vmin: float
    vmax: float
    user_inputs: dict[str, Any]


def _load_raster_data(raster_path: Path, band: int, minmax: tuple[float | None, float | None] = (None, None)) -> ViewerState:
    """Load and prepare raster data for display."""
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
        
        # Read full resolution data for pixel lookups
        full_data = src.read(band)
        
        nodata = src.nodata
        transform = src.transform
        
        # Mask nodata
        if nodata is not None:
            display_data = np.ma.masked_equal(display_data, nodata)
            full_data = np.ma.masked_equal(full_data, nodata)
        
        # Calculate initial color range
        data_for_stats = display_data.compressed() if np.ma.is_masked(display_data) else display_data
        
        if minmax[0] is not None and minmax[1] is not None:
            vmin, vmax = minmax
        else:
            vmin, vmax = np.nanpercentile(data_for_stats, [2, 98])
        
        return ViewerState(
            raster_path=raster_path,
            band=band,
            display_data=display_data,
            full_data=full_data,
            transform=transform,
            height=height,
            width=width,
            scale=scale,
            out_height=out_height,
            out_width=out_width,
            vmin=float(vmin),
            vmax=float(vmax),
            user_inputs={}
        )


def interactive_raster_viewer(
    raster_path: str | Path,
    band: int = 1,
    minmax: tuple[float | None, float | None] = (None, None),
    widgets: list[Callable[[ViewerState, Any, Any], None]] | None = None
) -> ViewerState:
    """
    Create an interactive raster viewer with configurable widgets.
    
    This is the core viewer function that displays a raster and collects user input
    through customizable widgets. The viewer stays open until the user closes it.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    band : int, optional
        Band number to display (1-indexed), default is 1
    minmax : tuple[float | None, float | None], optional
        Initial min/max values for color range
    widgets : list of callables, optional
        List of widget builder functions. Each receives (state, fig, im) and should
        add widgets that update state.user_inputs
        
    Returns
    -------
    ViewerState
        Final state with user_inputs populated by widgets
        
    Examples
    --------
    >>> def minmax_widget(state, fig, im):
    ...     # Add min/max input widgets
    ...     pass
    >>> state = interactive_raster_viewer("raster.tif", widgets=[minmax_widget])
    >>> print(state.user_inputs['min'], state.user_inputs['max'])
    """
    raster_path = Path(raster_path)
    
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    
    # Load raster data
    state = _load_raster_data(raster_path, band, minmax)
    
    # Create figure
    fig = plt.figure(figsize=(10, 9))
    ax = plt.axes([0.1, 0.25, 0.8, 0.7])
    
    im = ax.imshow(state.display_data, cmap='YlGnBu', vmin=state.vmin, vmax=state.vmax)
    plt.colorbar(im, ax=ax, label='Value')
    
    title = ax.set_title(f'{state.raster_path.name} (Band {state.band})\nHover for pixel values')
    
    # Set up hover functionality
    def on_move(event):
        if event.inaxes != ax:
            return
        
        display_col = int(event.xdata + 0.5)
        display_row = int(event.ydata + 0.5)
        
        if 0 <= display_row < state.out_height and 0 <= display_col < state.out_width:
            full_col = int(display_col * state.scale)
            full_row = int(display_row * state.scale)
            
            if 0 <= full_row < state.height and 0 <= full_col < state.width:
                value = state.full_data[full_row, full_col]
                geo_x, geo_y = state.transform * (full_col + 0.5, full_row + 0.5)
                
                val_str = "NoData" if np.ma.is_masked(value) else f"{value:.2f}"
                title.set_text(
                    f'{state.raster_path.name} (Band {state.band})\n'
                    f'Pixel: ({full_row}, {full_col}) | Value: {val_str} | Coords: ({geo_x:.1f}, {geo_y:.1f})'
                )
                fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    
    # Add custom widgets if provided
    if widgets:
        for widget_builder in widgets:
            widget_builder(state, fig, im)
    
    plt.show()
    
    return state


# Widget Builders
# These functions add interactive controls to the viewer

def minmax_widget(state: ViewerState, fig: Any, im: Any) -> None:
    """
    Widget for collecting min/max color range values.
    
    Adds text boxes for min/max input, Update button to preview changes,
    and Done button to close and save values.
    
    Updates state.user_inputs with 'min' and 'max' keys.
    """
    # Store widgets to prevent garbage collection
    widgets = {}
    
    # Create text input boxes and buttons
    ax_min = plt.axes([0.15, 0.12, 0.12, 0.04])
    ax_max = plt.axes([0.32, 0.12, 0.12, 0.04])
    ax_update = plt.axes([0.49, 0.12, 0.12, 0.04])
    ax_done = plt.axes([0.66, 0.12, 0.12, 0.04])
    
    widgets['textbox_min'] = TextBox(ax_min, 'Min:', initial=f'{state.vmin:.2f}')
    widgets['textbox_max'] = TextBox(ax_max, 'Max:', initial=f'{state.vmax:.2f}')
    widgets['button_update'] = Button(ax_update, 'Update')
    widgets['button_done'] = Button(ax_done, 'Done')
    
    # Initialize user_inputs
    state.user_inputs['min'] = state.vmin
    state.user_inputs['max'] = state.vmax
    
    def update(event):
        try:
            vmin = float(widgets['textbox_min'].text)
            vmax = float(widgets['textbox_max'].text)
            if vmin < vmax:
                im.set_clim(vmin, vmax)
                state.user_inputs['min'] = vmin
                state.user_inputs['max'] = vmax
                fig.canvas.draw_idle()
        except ValueError:
            pass
    
    def done(event):
        plt.close(fig)
    
    widgets['button_update'].on_clicked(update)
    widgets['button_done'].on_clicked(done)


# Convenience Functions

def display_raster(
    raster_path: str | Path,
    band: int = 1,
    minmax: tuple[float | None, float | None] = (None, None)
) -> None:
    """
    Display a raster with interactive pixel value viewer (no input collection).
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    band : int, optional
        Band number to display (1-indexed), default is 1
    minmax : tuple[float | None, float | None], optional
        Min and max values for color range. If (None, None), auto-calculated.
    """
    interactive_raster_viewer(raster_path, band, minmax, widgets=None)


def interactive_min_max(raster_path: str | Path, band: int = 1, minmax: tuple[float | None, float | None] = (None, None)) -> tuple[float, float]:
    """
    Display a raster and collect min/max elevation values from user input.
    
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
        The final (min, max) values when user clicks Done button.
    """
    state = interactive_raster_viewer(raster_path, band, minmax, widgets=[minmax_widget])
    return state.user_inputs['min'], state.user_inputs['max']


__all__ = ["interactive_raster_viewer", "minmax_widget", "interactive_min_max", "display_raster", "ViewerState"]