"""Streamlit-compatible interface helpers for the GIS 584 project."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import rasterio
from rasterio.enums import Resampling
import plotly.graph_objects as go
import matplotlib.cm as cm
from PIL import Image
import io
import base64


def _load_raster_data(
    raster_path: Path, 
    band: int, 
    minmax: tuple[float | None, float | None] = (None, None),
    max_dim: int = 1000  # Reduced from 2000 for better performance
) -> tuple[np.ndarray, np.ndarray, Any, int, int, float]:
    """Load and prepare raster data for display."""
    with rasterio.open(raster_path) as src:
        if band < 1 or band > src.count:
            raise ValueError(f"Band {band} out of range. File has {src.count} band(s).")
        
        # Downsample for fast display
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
        
        # Note: Full resolution data loading removed for performance
        # We use display_data for hover values instead
        full_data = display_data  # Use display data as placeholder
        
        nodata = src.nodata
        transform = src.transform
        
        # Mask nodata
        if nodata is not None:
            display_data = np.ma.masked_equal(display_data, nodata)
            full_data = np.ma.masked_equal(full_data, nodata)
        
        # Calculate initial color range
        data_for_stats = display_data.compressed() if np.ma.is_masked(display_data) else display_data
        
        if minmax is not None and minmax[0] is not None and minmax[1] is not None:
            vmin, vmax = minmax
        else:
            vmin, vmax = np.nanpercentile(data_for_stats, [2, 98])
        
        return display_data, full_data, transform, height, width, float(vmin), float(vmax), scale


def _compute_hillshade(
    elevation: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0
) -> np.ndarray:
    """
    Compute hillshade from elevation data.
    
    Parameters
    ----------
    elevation : np.ndarray
        Elevation data
    azimuth : float
        Light source azimuth angle (degrees, 0-360)
    altitude : float
        Light source altitude angle (degrees, 0-90)
    z_factor : float
        Vertical exaggeration factor
        
    Returns
    -------
    np.ndarray
        Hillshade values (0-255)
    """
    # Convert to radians
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)
    
    # Calculate gradients
    x, y = np.gradient(elevation * z_factor)
    
    # Calculate slope and aspect
    slope = np.pi/2.0 - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    
    # Calculate hillshade
    shaded = np.sin(altitude_rad) * np.sin(slope) + \
             np.cos(altitude_rad) * np.cos(slope) * \
             np.cos(azimuth_rad - aspect)
    
    # Scale to 0-255
    shaded = (shaded + 1) / 2 * 255
    
    return shaded.astype(np.uint8)


def _matplotlib_to_plotly_colorscale(cmap_name: str, n_colors: int = 256) -> list[list[float | str]]:
    """Convert matplotlib colormap to Plotly colorscale."""
    cmap = cm.get_cmap(cmap_name)
    colors = []
    for i in range(n_colors):
        rgba = cmap(i / (n_colors - 1))
        colors.append([i / (n_colors - 1), f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'])
    return colors


def display_raster_streamlit(
    raster_path: str | Path,
    band: int = 1,
    minmax: tuple[float | None, float | None] = (None, None),
    cmap: str = 'YlGnBu',
    title: str | None = None,
    show_colorbar: bool = True
) -> tuple[np.ndarray, float, float]:
    """
    Display a raster in Streamlit with interactive Plotly visualization.
    
    Features:
    - Zoom and pan
    - Hover to see pixel values and coordinates
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    band : int, optional
        Band number to display (1-indexed), default is 1
    minmax : tuple[float | None, float | None], optional
        Min and max values for color range. If (None, None), auto-calculated.
    cmap : str, optional
        Colormap name, default is 'YlGnBu'
    title : str, optional
        Plot title
    show_colorbar : bool, optional
        Whether to show colorbar, default is True
        
    Returns
    -------
    tuple[np.ndarray, float, float]
        (display_data, vmin, vmax) - the displayed data and color range
    """
    raster_path = Path(raster_path)
    
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    
    # Load raster data
    display_data, full_data, transform, height, width, vmin, vmax, scale = _load_raster_data(raster_path, band, minmax)
    
    # Handle masked arrays
    if np.ma.is_masked(display_data):
        # Convert masked array to regular array with NaN
        display_data = np.ma.filled(display_data, np.nan)
    
    # Get colormap
    colorscale = _matplotlib_to_plotly_colorscale(cmap)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Flip the y-axis for proper display
    display_data_flipped = np.flipud(display_data)
    
    # Use Plotly's built-in hover with custom hovertemplate for better performance
    # This is much faster than pre-computing hover text for every pixel
    fig.add_trace(go.Heatmap(
        z=display_data_flipped,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(title='Value') if show_colorbar else None,
        hovertemplate=(
            'X: %{x:.0f}<br>'
            'Y: %{y:.0f}<br>'
            'Value: %{z:.2f}<extra></extra>'
        )
    ))
    
    # Set title and labels
    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(title=f'{raster_path.name} (Band {band})')
    
    fig.update_xaxes(title_text='X (pixels)')
    fig.update_yaxes(title_text='Y (pixels)')
    
    # Set aspect ratio and layout
    fig.update_layout(
        width=1000,
        height=800,
        autosize=True,
        dragmode='pan',
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return display_data, vmin, vmax


def display_raster_with_hillshade_streamlit(
    raster_path: str | Path,
    band: int = 1,
    minmax: tuple[float | None, float | None] = (None, None),
    alpha: float = 0.3,
    exaggeration: float = 1.0,
    altitude: float = 45.0,
    azimuth: float = 315.0,
    cmap: str = 'YlGnBu',
    title: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Display a raster with hillshade overlay in Streamlit with interactive Plotly visualization.
    
    Features:
    - Zoom and pan
    - Hover to see pixel values and coordinates
    - Hillshade overlay with adjustable transparency
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    band : int, optional
        Band number to display (1-indexed), default is 1
    minmax : tuple[float | None, float | None], optional
        Min and max values for color range
    alpha : float, optional
        Hillshade transparency (0-1), default is 0.3
    exaggeration : float, optional
        Vertical exaggeration factor, default is 1.0
    altitude : float, optional
        Light source altitude angle (degrees, 0-90), default is 45.0
    azimuth : float, optional
        Light source azimuth angle (degrees, 0-360), default is 315.0
    cmap : str, optional
        Colormap name, default is 'YlGnBu'
    title : str, optional
        Plot title
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (display_data, hillshade) - the displayed elevation data and hillshade
    """
    raster_path = Path(raster_path)
    
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    
    # Load raster data
    display_data, full_data, transform, height, width, vmin, vmax, scale = _load_raster_data(raster_path, band, minmax)
    
    # Handle masked arrays
    if np.ma.is_masked(display_data):
        display_data = np.ma.filled(display_data, np.nan)
    
    # Compute hillshade (matches original matplotlib implementation)
    hillshade = _compute_hillshade(
        display_data,
        altitude=altitude,
        azimuth=azimuth,
        z_factor=exaggeration
    )
    
    # Get colormap for colorbar
    colorscale = _matplotlib_to_plotly_colorscale(cmap)
    
    # Apply hillshade effect to elevation data for display
    # Normalize hillshade to 0-1 (grayscale)
    hillshade_norm = hillshade / 255.0
    
    # Apply hillshade effect: darken elevation values based on hillshade
    # Formula: display_value = elevation * (1 - alpha * (1 - hillshade_factor))
    # This darkens shadowed areas while preserving elevation colors
    hillshade_factor = 1 - alpha * (1 - hillshade_norm)
    display_data_shaded = display_data * hillshade_factor
    
    # Flip for display
    display_data_shaded_flipped = np.flipud(display_data_shaded)
    display_data_flipped = np.flipud(display_data)
    
    # Create Plotly figure (same approach as display_raster_streamlit)
    fig = go.Figure()
    
    # Display the shaded elevation data with the colormap
    fig.add_trace(go.Heatmap(
        z=display_data_shaded_flipped,
        colorscale=colorscale,
        zmin=vmin * (1 - alpha),  # Adjust min to account for darkening
        zmax=vmax,
        colorbar=dict(title='Elevation'),
        hovertemplate=(
            'X: %{x:.0f}<br>'
            'Y: %{y:.0f}<br>'
            'Elevation: %{z:.2f}<extra></extra>'
        ),
    ))
    
    # Set title and labels
    if title:
        fig.update_layout(title=title)
    else:
        fig.update_layout(title=f'{raster_path.name} (Band {band}) with Hillshade')
    
    fig.update_xaxes(title_text='X (pixels)')
    fig.update_yaxes(title_text='Y (pixels)')
    
    # Set aspect ratio and layout
    fig.update_layout(
        width=1000,
        height=800,
        autosize=True,
        dragmode='pan',
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return display_data, hillshade


def get_raster_stats_dict(raster_path: str | Path) -> dict[str, Any]:
    """
    Get raster statistics as a dictionary (useful for Streamlit display).
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
        
    Returns
    -------
    dict
        Dictionary containing raster statistics
    """
    raster_path = Path(raster_path)
    
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    
    stats_dict = {
        'file': str(raster_path),
        'bands': {}
    }
    
    with rasterio.open(raster_path) as src:
        stats_dict['width'] = src.width
        stats_dict['height'] = src.height
        stats_dict['count'] = src.count
        stats_dict['crs'] = str(src.crs) if src.crs else None
        stats_dict['transform'] = list(src.transform)
        
        for band_num in range(1, src.count + 1):
            band = src.read(band_num)
            nodata = src.nodata
            
            if nodata is not None:
                band = np.ma.masked_equal(band, nodata)
            
            data = band.compressed() if np.ma.is_masked(band) else band.flatten()
            data = data[~np.isnan(data)]
            
            if len(data) > 0:
                stats_dict['bands'][band_num] = {
                    'min': float(np.nanmin(data)),
                    'max': float(np.nanmax(data)),
                    'mean': float(np.nanmean(data)),
                    'std': float(np.nanstd(data)),
                    'nodata': float(nodata) if nodata is not None else None
                }
    
    return stats_dict


__all__ = [
    "display_raster_streamlit",
    "display_raster_with_hillshade_streamlit",
    "get_raster_stats_dict",
    "_compute_hillshade"
]

