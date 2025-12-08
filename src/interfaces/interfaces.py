"""Interface helpers for the GIS 584 project."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
from dataclasses import dataclass
import json
import urllib.request
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, Slider
from matplotlib.patches import Polygon as MplPolygon
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from affine import Affine
from scipy import ndimage
from skimage.morphology import skeletonize
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, transform as shapely_transform
from pyproj import Transformer


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
    
    # Add custom widgets if provided - store returned widgets to prevent garbage collection
    _widget_refs = []
    if widgets:
        for widget_builder in widgets:
            widget_dict = widget_builder(state, fig, im)
            if widget_dict:
                _widget_refs.append(widget_dict)
    
    plt.show()
    
    return state


# Helper Functions

def _skeleton_to_linestring(skeleton: np.ndarray, transform=None) -> LineString | None:
    """
    Convert a binary skeleton to a Shapely LineString.
    
    Parameters
    ----------
    skeleton : np.ndarray
        Binary array where True indicates skeleton pixels
    transform : affine.Affine, optional
        Rasterio transform for converting pixel coords to geographic coords
        
    Returns
    -------
    LineString or None
        LineString following the skeleton path
    """
    # Find skeleton pixels
    y_coords, x_coords = np.where(skeleton)
    
    if len(x_coords) < 2:
        return None
    
    # Order points to form a continuous line (simple approach: use a path)
    # Start from one end and follow connected pixels
    points = list(zip(y_coords, x_coords))
    
    if len(points) < 2:
        return None
    
    # Build connectivity graph and trace path
    visited = set()
    path = []
    
    # Start from first point
    current = points[0]
    path.append(current)
    visited.add(current)
    
    # Follow connected neighbors
    while len(visited) < len(points):
        y, x = current
        found_next = False
        
        # Check 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                if neighbor in points and neighbor not in visited:
                    path.append(neighbor)
                    visited.add(neighbor)
                    current = neighbor
                    found_next = True
                    break
            if found_next:
                break
        
        if not found_next:
            # Dead end or disconnected component - find nearest unvisited
            remaining = [p for p in points if p not in visited]
            if not remaining:
                break
            # Find closest unvisited point
            distances = [abs(p[0] - current[0]) + abs(p[1] - current[1]) for p in remaining]
            current = remaining[np.argmin(distances)]
            path.append(current)
            visited.add(current)
    
    # Convert to coordinates (swap back to x,y order)
    coords = [(x, y) for y, x in path]
    
    # Apply transform if provided
    if transform is not None:
        coords = [transform * (x + 0.5, y + 0.5) for x, y in coords]
    
    if len(coords) < 2:
        return None
    
    return LineString(coords)


def _extract_centerline(
    elevation: np.ndarray,
    min_elev: float,
    max_elev: float,
    smooth_sigma: float = 2.0
) -> np.ndarray:
    """
    Extract river centerline using morphological skeletonization.
    
    Parameters
    ----------
    elevation : np.ndarray
        Elevation data
    min_elev : float
        Minimum elevation defining river extent
    max_elev : float
        Maximum elevation defining river extent
    smooth_sigma : float
        Smoothing parameter for morphological operations
        
    Returns
    -------
    np.ndarray
        Binary array where True indicates centerline pixels
    """
    # Create binary mask of river (elevations within range)
    river_mask = (elevation >= min_elev) & (elevation <= max_elev)
    
    # Handle masked arrays
    if np.ma.is_masked(elevation):
        river_mask = river_mask & ~elevation.mask
    
    # Fill small holes in the river mask
    river_mask = ndimage.binary_fill_holes(river_mask)
    
    # Smooth the mask with morphological operations
    smoothing_size = max(3, int(smooth_sigma))
    river_mask = ndimage.binary_closing(river_mask, structure=np.ones((smoothing_size, smoothing_size)))
    river_mask = ndimage.binary_opening(river_mask, structure=np.ones((smoothing_size, smoothing_size)))
    
    # Skeletonize directly
    skeleton = skeletonize(river_mask)
    
    # Remove very small disconnected components (< 10 pixels)
    labeled, num_features = ndimage.label(skeleton)
    if num_features > 1:
        # Keep components with at least 10 pixels
        for label in range(1, num_features + 1):
            component_size = np.sum(labeled == label)
            if component_size < 10:
                skeleton[labeled == label] = False
    
    return skeleton


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


# Widget Builders
# These functions add interactive controls to the viewer

def minmax_widget(state: ViewerState, fig: Any, im: Any) -> dict:
    """
    Widget for collecting min/max color range values.
    
    Adds text boxes for min/max input, Update button to preview changes,
    and Done button to close and save values.
    
    Updates state.user_inputs with 'min' and 'max' keys.
    
    Returns
    -------
    dict
        Dictionary of widgets to keep them from being garbage collected
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
    
    return widgets


def hillshade_widget(state: ViewerState, fig: Any, im: Any) -> dict:
    """
    Widget for adding and customizing a hillshade overlay.
    
    Adds text boxes for transparency (alpha), exaggeration (z-factor), and altitude angle,
    Update button to preview changes, and Done button to close and save values.
    
    Updates state.user_inputs with 'hillshade_alpha', 'hillshade_exaggeration', and 'hillshade_altitude' keys.
    
    Returns
    -------
    dict
        Dictionary of widgets to keep them from being garbage collected
    """
    # Store widgets to prevent garbage collection
    widgets = {}
    
    # Initial values
    initial_alpha = 0.3
    initial_exaggeration = 1.0
    initial_altitude = 45.0  # degrees
    
    # Create text input boxes and buttons (3 rows)
    # Row 1: Alpha, Exaggeration
    ax_alpha = plt.axes([0.15, 0.16, 0.12, 0.04])
    ax_exag = plt.axes([0.32, 0.16, 0.12, 0.04])
    
    # Row 2: Altitude
    ax_altitude = plt.axes([0.15, 0.10, 0.12, 0.04])
    
    # Row 3: Buttons
    ax_update = plt.axes([0.49, 0.10, 0.12, 0.04])
    ax_done = plt.axes([0.66, 0.10, 0.12, 0.04])
    
    widgets['textbox_alpha'] = TextBox(ax_alpha, 'Alpha:', initial=f'{initial_alpha:.2f}')
    widgets['textbox_exag'] = TextBox(ax_exag, 'Exag:', initial=f'{initial_exaggeration:.2f}')
    widgets['textbox_altitude'] = TextBox(ax_altitude, 'Altitude°:', initial=f'{initial_altitude:.1f}')
    widgets['button_update'] = Button(ax_update, 'Update')
    widgets['button_done'] = Button(ax_done, 'Done')
    
    # Initialize user_inputs
    state.user_inputs['hillshade_alpha'] = initial_alpha
    state.user_inputs['hillshade_exaggeration'] = initial_exaggeration
    state.user_inputs['hillshade_altitude'] = initial_altitude
    
    # Get the axes from the image
    ax = im.axes
    
    # Compute initial hillshade
    hillshade = _compute_hillshade(
        state.display_data, 
        altitude=initial_altitude,
        z_factor=initial_exaggeration
    )
    
    # Add hillshade overlay
    widgets['hillshade_im'] = ax.imshow(
        hillshade,
        cmap='gray',
        alpha=initial_alpha,
        vmin=0,
        vmax=255,
        zorder=2  # Render on top of the raster
    )
    
    # Ensure raster is below
    im.set_zorder(1)
    
    def update(event):
        try:
            alpha = float(widgets['textbox_alpha'].text)
            exaggeration = float(widgets['textbox_exag'].text)
            altitude = float(widgets['textbox_altitude'].text)
            
            # Validate inputs
            if 0 <= alpha <= 1 and exaggeration > 0 and 0 <= altitude <= 90:
                # Recompute hillshade with new parameters
                hillshade = _compute_hillshade(
                    state.display_data, 
                    altitude=altitude,
                    z_factor=exaggeration
                )
                
                # Update hillshade image
                widgets['hillshade_im'].set_data(hillshade)
                widgets['hillshade_im'].set_alpha(alpha)
                
                # Store values
                state.user_inputs['hillshade_alpha'] = alpha
                state.user_inputs['hillshade_exaggeration'] = exaggeration
                state.user_inputs['hillshade_altitude'] = altitude
                
                fig.canvas.draw_idle()
        except ValueError:
            pass
    
    def done(event):
        plt.close(fig)
    
    widgets['button_update'].on_clicked(update)
    widgets['button_done'].on_clicked(done)
    
    return widgets


def manual_centerline_widget(state: ViewerState, fig: Any, im: Any) -> dict:
    """
    Widget for drawing a mask over the river, then extracting centerline within it.
    
    Paint over the river area like a brush, then extract centerline automatically
    within the painted region.
    
    Updates state.user_inputs with 'centerline_geom' (Shapely LineString).
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from matplotlib.patches import Polygon as MplPolygon
    
    # Store widgets and data
    widgets = {}
    drawn_strokes = []  # List of stroke paths (each stroke is a list of points)
    current_stroke = []
    drawing_active = {'enabled': False, 'is_drawing': False}
    brush_width = {'value': 20}  # Width of the brush in pixels (mutable dict)
    
    # Create controls - 2 rows
    # Row 1: Brush size input
    ax_brush_label = plt.axes([0.12, 0.17, 0.08, 0.04])
    ax_brush_size = plt.axes([0.21, 0.17, 0.10, 0.04])
    
    # Row 2: Action buttons
    ax_start = plt.axes([0.12, 0.10, 0.15, 0.04])
    ax_clear = plt.axes([0.32, 0.10, 0.15, 0.04])
    ax_extract = plt.axes([0.52, 0.10, 0.15, 0.04])
    ax_done = plt.axes([0.72, 0.10, 0.15, 0.04])
    
    # Add label manually
    ax_brush_label.text(0.5, 0.5, 'Brush:', ha='center', va='center', fontsize=10)
    ax_brush_label.set_xlim(0, 1)
    ax_brush_label.set_ylim(0, 1)
    ax_brush_label.axis('off')
    
    widgets['textbox_brush'] = TextBox(ax_brush_size, '', initial=f'{brush_width["value"]}')
    widgets['button_start'] = Button(ax_start, 'Start Brush')
    widgets['button_clear'] = Button(ax_clear, 'Clear')
    widgets['button_extract'] = Button(ax_extract, 'Extract')
    widgets['button_done'] = Button(ax_done, 'Done')
    
    # Update brush width callback
    def update_brush_size(text):
        try:
            val = int(float(text))
            if val > 0:
                brush_width['value'] = val
                print(f"Brush size updated to: {val}")
        except:
            pass
    
    widgets['textbox_brush'].on_submit(update_brush_size)
    
    # Get the axes from the image
    ax = im.axes
    
    # Apply hillshade if parameters are provided
    if 'hillshade_alpha' in state.user_inputs and 'hillshade_exaggeration' in state.user_inputs:
        print("Applying hillshade overlay...")
        hillshade = _compute_hillshade(
            state.display_data,
            altitude=state.user_inputs.get('hillshade_altitude', 45.0),
            z_factor=state.user_inputs['hillshade_exaggeration']
        )
        
        widgets['hillshade_im'] = ax.imshow(
            hillshade,
            cmap='gray',
            alpha=state.user_inputs['hillshade_alpha'],
            vmin=0,
            vmax=255,
            zorder=2
        )
        im.set_zorder(1)
    
    # Overlays
    widgets['mask_patches'] = []
    widgets['centerline_line'] = None
    
    print("\n" + "="*60)
    print("BRUSH & EXTRACT CENTERLINE")
    print("="*60)
    print("1. (Optional) Adjust brush size (default: 20 pixels)")
    print("2. Click 'Start Brush' button")
    print("3. Click and drag to paint over the river area")
    print("4. Paint multiple strokes to cover the whole river")
    print("5. Click 'Extract' to compute centerline in painted area")
    print("6. Review result - you can paint more and re-extract")
    print("7. Click 'Done' to save (or 'Clear' to start over)")
    print("="*60 + "\n")
    
    def update_mask_display():
        """Update the mask display."""
        # Remove old patches
        for patch in widgets['mask_patches']:
            patch.remove()
        widgets['mask_patches'].clear()
        
        # Draw all strokes as semi-transparent polygons
        for stroke in drawn_strokes:
            if len(stroke) < 2:
                continue
            
            # Create a buffered polygon around the stroke
            stroke_line = LineString([(p[1], p[0]) for p in stroke])
            buffered = stroke_line.buffer(brush_width['value'] / 2)
            
            if buffered.geom_type == 'Polygon':
                coords = np.array(buffered.exterior.coords)
                patch = MplPolygon(coords, facecolor='yellow', edgecolor='orange',
                                  alpha=0.3, linewidth=2, zorder=3)
                ax.add_patch(patch)
                widgets['mask_patches'].append(patch)
        
        fig.canvas.draw_idle()
    
    def start_drawing(event):
        """Enable drawing mode."""
        if not drawing_active['enabled']:
            drawing_active['enabled'] = True
            widgets['button_start'].label.set_text('Brushing...')
            widgets['button_start'].color = '0.85'
            fig.canvas.draw_idle()
            print("Brush mode enabled. Click and drag to paint over the river.")
    
    def on_press(event):
        """Start drawing stroke when mouse is pressed."""
        if not drawing_active['enabled']:
            return
        if event.inaxes != ax or event.button != 1:
            return
        
        drawing_active['is_drawing'] = True
        current_stroke.clear()
        current_stroke.append((event.ydata, event.xdata))
    
    def on_motion(event):
        """Capture points while dragging."""
        if not drawing_active['is_drawing']:
            return
        if event.inaxes != ax:
            return
        
        # Add point
        current_stroke.append((event.ydata, event.xdata))
        
        # Show live preview
        if len(current_stroke) >= 2:
            stroke_line = LineString([(p[1], p[0]) for p in current_stroke])
            buffered = stroke_line.buffer(brush_width['value'] / 2)
            
            # Remove previous temp patch if exists
            if 'temp_patch' in widgets and widgets['temp_patch'] is not None:
                widgets['temp_patch'].remove()
            
            if buffered.geom_type == 'Polygon':
                coords = np.array(buffered.exterior.coords)
                widgets['temp_patch'] = MplPolygon(coords, facecolor='yellow', edgecolor='orange',
                                                   alpha=0.3, linewidth=2, zorder=3)
                ax.add_patch(widgets['temp_patch'])
                fig.canvas.draw_idle()
    
    def on_release(event):
        """Finish stroke when mouse is released."""
        if drawing_active['is_drawing']:
            drawing_active['is_drawing'] = False
            if len(current_stroke) >= 2:
                drawn_strokes.append(list(current_stroke))
                print(f"Stroke {len(drawn_strokes)} added ({len(current_stroke)} points)")
                update_mask_display()
    
    def clear_all(event):
        """Clear all drawn strokes."""
        drawn_strokes.clear()
        current_stroke.clear()
        print("Cleared all brush strokes")
        update_mask_display()
        
        # Remove centerline if exists
        if widgets['centerline_line'] is not None:
            widgets['centerline_line'].remove()
            widgets['centerline_line'] = None
            fig.canvas.draw_idle()
    
    def extract_centerline(event):
        """Extract centerline within the painted mask."""
        if not drawn_strokes:
            print("ERROR: Paint over the river first! Click 'Start Brush' and drag.")
            return
        
        print("\nExtracting centerline within painted area...")
        
        # Create mask from all strokes
        mask = np.zeros(state.display_data.shape, dtype=bool)
        
        for stroke in drawn_strokes:
            if len(stroke) < 2:
                continue
            
            stroke_line = LineString([(p[1], p[0]) for p in stroke])
            buffered = stroke_line.buffer(brush_width['value'] / 2)
            
            # Rasterize the polygon into the mask
            if buffered.geom_type == 'Polygon':
                from matplotlib.path import Path as MplPath
                coords = np.array(buffered.exterior.coords)
                path = MplPath(coords)
                
                # Create grid of all points
                y_grid, x_grid = np.mgrid[0:state.out_height, 0:state.out_width]
                points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
                
                # Check which points are inside
                inside = path.contains_points(points)
                inside_2d = inside.reshape(state.out_height, state.out_width)
                mask |= inside_2d
        
        # Extract centerline within mask
        masked_elevation = state.display_data.copy()
        masked_elevation[~mask] = np.nan
        
        # Use the existing centerline extraction but only on masked region
        centerline_binary = _extract_centerline(masked_elevation, state.vmin, state.vmax, smooth_sigma=3.0)
        
        # Convert to LineString in DISPLAY pixel coordinates (no transform)
        centerline_display = _skeleton_to_linestring(centerline_binary, transform=None)
        
        if centerline_display is None or len(centerline_display.coords) < 2:
            print("ERROR: Could not extract centerline. Try painting a wider area.")
            return
        
        # Also create a geographic version for export (scaled up and transformed)
        # Create scaled transform for display resolution
        scaled_transform = state.transform * Affine.scale(state.scale)
        centerline_geom = _skeleton_to_linestring(centerline_binary, transform=scaled_transform)
        
        # Store BOTH in state
        state.user_inputs['centerline_geom'] = centerline_geom  # Geographic coords
        state.user_inputs['centerline_display'] = centerline_display  # Display coords
        
        # Display centerline using display coordinates
        if widgets['centerline_line'] is not None:
            widgets['centerline_line'].remove()
            widgets['centerline_line'] = None
        
        # Extract display coordinates (already in pixel space)
        display_coords = list(centerline_display.coords)
        
        print(f"  Display coord range: x=[{min(c[0] for c in display_coords):.1f}, {max(c[0] for c in display_coords):.1f}], "
              f"y=[{min(c[1] for c in display_coords):.1f}, {max(c[1] for c in display_coords):.1f}]")
        print(f"  Image extent: x=[0, {state.out_width}], y=[0, {state.out_height}]")
        
        if len(display_coords) >= 2:
            coords_array = np.array(display_coords)
            widgets['centerline_line'], = ax.plot(
                coords_array[:, 0], coords_array[:, 1],
                'r-', linewidth=4, zorder=10, alpha=1.0
            )
        
        print(f"✓ Centerline extracted: {len(centerline_geom.coords)} points")
        print(f"  Line length: {centerline_geom.length:.2f} map units")
        print(f"  Centerline plotted on axes")
        fig.canvas.draw_idle()
    
    def done(event):
        """Finish and save."""
        if 'centerline_geom' not in state.user_inputs:
            print("ERROR: Extract centerline first! Click 'Extract' button.")
            return
        
        print("\nCenterline saved!")
        plt.close(fig)
    
    # Connect events
    widgets['press_cid'] = fig.canvas.mpl_connect('button_press_event', on_press)
    widgets['motion_cid'] = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    widgets['release_cid'] = fig.canvas.mpl_connect('button_release_event', on_release)
    
    widgets['button_start'].on_clicked(start_drawing)
    widgets['button_clear'].on_clicked(clear_all)
    widgets['button_extract'].on_clicked(extract_centerline)
    widgets['button_done'].on_clicked(done)
    
    return widgets


def centerline_widget(state: ViewerState, fig: Any, im: Any) -> dict:
    """
    Widget for computing and displaying river centerline.
    
    Automatically computes centerline from the elevation data and color range.
    Provides controls to adjust smoothing and regenerate.
    
    If hillshade parameters exist in state.user_inputs, applies the hillshade overlay.
    
    Updates state.user_inputs with 'centerline' (the binary centerline array) and 'centerline_smooth'.
    
    Returns
    -------
    dict
        Dictionary of widgets to keep them from being garbage collected
    """
    # Store widgets to prevent garbage collection
    widgets = {}
    
    # Initial values
    initial_smooth = 2.0
    
    # Create text input boxes and buttons
    ax_smooth = plt.axes([0.15, 0.12, 0.12, 0.04])
    ax_update = plt.axes([0.32, 0.12, 0.12, 0.04])
    ax_done = plt.axes([0.49, 0.12, 0.12, 0.04])
    
    widgets['textbox_smooth'] = TextBox(ax_smooth, 'Smooth:', initial=f'{initial_smooth:.1f}')
    widgets['button_update'] = Button(ax_update, 'Regenerate')
    widgets['button_done'] = Button(ax_done, 'Done')
    
    # Get the axes from the image
    ax = im.axes
    
    # Apply hillshade if parameters are provided
    if 'hillshade_alpha' in state.user_inputs and 'hillshade_exaggeration' in state.user_inputs:
        print("Applying hillshade overlay...")
        hillshade = _compute_hillshade(
            state.display_data,
            altitude=state.user_inputs.get('hillshade_altitude', 45.0),
            z_factor=state.user_inputs['hillshade_exaggeration']
        )
        
        widgets['hillshade_im'] = ax.imshow(
            hillshade,
            cmap='gray',
            alpha=state.user_inputs['hillshade_alpha'],
            vmin=0,
            vmax=255,
            zorder=2
        )
        im.set_zorder(1)
    
    # Compute initial centerline
    print("Computing river centerline...")
    centerline = _extract_centerline(
        state.display_data,
        state.vmin,
        state.vmax,
        smooth_sigma=initial_smooth
    )
    
    # Convert to LineString
    centerline_geom = _skeleton_to_linestring(centerline, state.transform)
    
    # Initialize user_inputs
    state.user_inputs['centerline'] = centerline
    state.user_inputs['centerline_geom'] = centerline_geom
    state.user_inputs['centerline_smooth'] = initial_smooth
    
    # Thicken the centerline for better visibility
    centerline_display = ndimage.binary_dilation(centerline, iterations=3)
    
    # Create a colored overlay for the centerline (e.g., red)
    centerline_rgba = np.zeros((*centerline.shape, 4))
    centerline_rgba[centerline_display, :] = [1, 0, 0, 1]  # Red with full opacity
    
    # Add centerline overlay
    widgets['centerline_im'] = ax.imshow(
        centerline_rgba,
        zorder=3  # Render on top of everything
    )
    
    pixel_count = np.sum(centerline)
    line_info = f" ({len(centerline_geom.coords)} coords)" if centerline_geom else " (no line)"
    print(f"Centerline extracted: {pixel_count} pixels{line_info}")
    
    def update(event):
        try:
            smooth = float(widgets['textbox_smooth'].text)
            
            if smooth >= 0:
                print(f"Regenerating centerline with smooth={smooth}...")
                # Recompute centerline
                centerline = _extract_centerline(
                    state.display_data,
                    state.vmin,
                    state.vmax,
                    smooth_sigma=smooth
                )
                
                # Convert to LineString
                centerline_geom = _skeleton_to_linestring(centerline, state.transform)
                
                # Thicken for display
                centerline_display = ndimage.binary_dilation(centerline, iterations=3)
                
                # Update overlay
                centerline_rgba = np.zeros((*centerline.shape, 4))
                centerline_rgba[centerline_display, :] = [1, 0, 0, 1]
                
                widgets['centerline_im'].set_data(centerline_rgba)
                
                # Store values
                state.user_inputs['centerline'] = centerline
                state.user_inputs['centerline_geom'] = centerline_geom
                state.user_inputs['centerline_smooth'] = smooth
                
                pixel_count = np.sum(centerline)
                line_info = f" ({len(centerline_geom.coords)} coords)" if centerline_geom else " (no line)"
                print(f"Centerline updated: {pixel_count} pixels{line_info}")
                fig.canvas.draw_idle()
        except ValueError:
            pass
    
    def done(event):
        plt.close(fig)
    
    widgets['button_update'].on_clicked(update)
    widgets['button_done'].on_clicked(done)
    
    return widgets


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


def interactive_hillshade(
    raster_path: str | Path, 
    band: int = 1, 
    minmax: tuple[float | None, float | None] = (None, None)
) -> tuple[float, float, float]:
    """
    Display a raster with hillshade overlay and collect transparency/exaggeration/altitude values.
    
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
    tuple[float, float, float]
        (alpha, exaggeration, altitude) - hillshade transparency, vertical exaggeration, and altitude angle (degrees)
    """
    state = interactive_raster_viewer(raster_path, band, minmax, widgets=[hillshade_widget])
    return (
        state.user_inputs['hillshade_alpha'], 
        state.user_inputs['hillshade_exaggeration'],
        state.user_inputs['hillshade_altitude']
    )

def _query_osm_waterways(bbox_wgs84: tuple[float, float, float, float]) -> list[dict]:
    """
    Query OpenStreetMap Overpass API for waterways within a bounding box.
    
    Parameters
    ----------
    bbox_wgs84 : tuple
        Bounding box in WGS84 (min_lon, min_lat, max_lon, max_lat)
        
    Returns
    -------
    list[dict]
        List of OSM way elements with their nodes and coordinates
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    # Overpass uses (south, west, north, east) format
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    
    # Query for waterways (rivers, streams, canals)
    query = f"""
    [out:json][timeout:60];
    (
      way["waterway"="river"]({bbox_str});
      way["waterway"="stream"]({bbox_str});
      way["waterway"="canal"]({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """
    
    url = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    
    print(f"Querying OSM Overpass API for waterways in bbox: {bbox_wgs84}")
    
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("User-Agent", "GIS584-RiverCenterline/1.0")
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error querying Overpass API: {e}")
        return []
    
    return result.get("elements", [])


def _build_linestrings_from_osm(elements: list[dict]) -> list[LineString]:
    """
    Build Shapely LineStrings from OSM Overpass API elements.
    
    Parameters
    ----------
    elements : list[dict]
        OSM elements from Overpass API response
        
    Returns
    -------
    list[LineString]
        List of Shapely LineStrings representing waterways
    """
    # Separate nodes and ways
    nodes = {}
    ways = []
    
    for elem in elements:
        if elem["type"] == "node":
            nodes[elem["id"]] = (elem["lon"], elem["lat"])
        elif elem["type"] == "way":
            ways.append(elem)
    
    linestrings = []
    for way in ways:
        coords = []
        for node_id in way.get("nodes", []):
            if node_id in nodes:
                coords.append(nodes[node_id])
        
        if len(coords) >= 2:
            linestrings.append(LineString(coords))
    
    return linestrings


def _densify_line(line: LineString, max_segment_length: float) -> LineString:
    """
    Densify a LineString by adding points so no segment exceeds max_segment_length.
    
    This ensures the line has enough points to follow curves when snapping.
    
    Parameters
    ----------
    line : LineString
        Input line geometry
    max_segment_length : float
        Maximum distance between consecutive points (in map units)
        
    Returns
    -------
    LineString
        Densified line with additional interpolated points
    """
    coords = list(line.coords)
    densified_coords = [coords[0]]
    
    for i in range(1, len(coords)):
        start = np.array(coords[i-1])
        end = np.array(coords[i])
        segment_length = np.linalg.norm(end - start)
        
        if segment_length > max_segment_length:
            # Add intermediate points
            num_segments = int(np.ceil(segment_length / max_segment_length))
            for j in range(1, num_segments):
                t = j / num_segments
                interp_point = start + t * (end - start)
                densified_coords.append(tuple(interp_point))
        
        densified_coords.append(coords[i])
    
    return LineString(densified_coords)


def _snap_centerline_to_channel(
    centerline: LineString,
    raster_path: Path,
    search_radius: float = 100.0,
    river_elev_range: tuple[float, float] | None = None,
    point_spacing: float = 50.0
) -> LineString:
    """
    Snap a centerline to the actual river channel by finding local elevation minima.
    
    First densifies the line to ensure enough points to follow curves, then snaps
    each point to the local elevation minimum within the search radius.
    
    Parameters
    ----------
    centerline : LineString
        The input centerline geometry in the raster's CRS
    raster_path : Path
        Path to the elevation raster
    search_radius : float, optional
        Search radius in map units (meters) to look for channel bottom. Default 100m.
    river_elev_range : tuple[float, float], optional
        (min_elev, max_elev) expected elevation range of the river. If provided,
        only considers points within this range as valid channel locations.
    point_spacing : float, optional
        Maximum distance between points along the line (meters). Smaller values
        allow the line to follow tighter curves. Default 50m.
        
    Returns
    -------
    LineString
        Snapped centerline geometry
    """
    # Densify the line first to ensure we have enough points to follow curves
    original_point_count = len(centerline.coords)
    centerline = _densify_line(centerline, point_spacing)
    print(f"Densified line: {original_point_count} -> {len(centerline.coords)} points (spacing: {point_spacing}m)")
    
    with rasterio.open(raster_path) as src:
        transform = src.transform
        data = src.read(1)
        nodata = src.nodata
        
        # Mask nodata values
        if nodata is not None:
            data = np.ma.masked_equal(data, nodata)
    
    # Get pixel size for search window calculation
    pixel_size = abs(transform[0])  # Assumes square pixels
    search_pixels = int(search_radius / pixel_size)
    
    snapped_coords = []
    original_coords = list(centerline.coords)
    
    print(f"Snapping {len(original_coords)} centerline points to channel (search radius: {search_radius}m)...")
    
    for i, (x, y) in enumerate(original_coords):
        # Convert geographic coords to pixel coords
        col, row = ~transform * (x, y)
        col, row = int(col), int(row)
        
        # Define search window
        row_min = max(0, row - search_pixels)
        row_max = min(data.shape[0], row + search_pixels + 1)
        col_min = max(0, col - search_pixels)
        col_max = min(data.shape[1], col + search_pixels + 1)
        
        # Extract search window
        window = data[row_min:row_max, col_min:col_max]
        
        if window.size == 0 or (np.ma.is_masked(window) and window.mask.all()):
            # No valid data in window, keep original point
            snapped_coords.append((x, y))
            continue
        
        # If river elevation range is provided, mask out values outside range
        if river_elev_range is not None:
            min_elev, max_elev = river_elev_range
            valid_mask = (window >= min_elev) & (window <= max_elev)
            if not valid_mask.any():
                # No points in river elevation range, keep original
                snapped_coords.append((x, y))
                continue
            # Set non-river pixels to a high value so they won't be selected as minimum
            window = np.where(valid_mask, window, np.inf)
        
        # Find the minimum elevation location in the window
        if np.ma.is_masked(window):
            window_filled = window.filled(np.inf)
        else:
            window_filled = window
            
        local_min_idx = np.unravel_index(np.argmin(window_filled), window_filled.shape)
        
        # Convert back to geographic coordinates
        snap_row = row_min + local_min_idx[0]
        snap_col = col_min + local_min_idx[1]
        snap_x, snap_y = transform * (snap_col + 0.5, snap_row + 0.5)
        
        snapped_coords.append((snap_x, snap_y))
    
    # Smooth the snapped centerline to remove zigzag artifacts
    snapped_line = LineString(snapped_coords)
    
    # Apply a simple moving average smoothing
    if len(snapped_coords) > 5:
        smoothed_coords = []
        coords_array = np.array(snapped_coords)
        
        # Keep first 2 and last 2 points as-is
        smoothed_coords.extend(snapped_coords[:2])
        
        # Apply 5-point moving average to middle points
        for i in range(2, len(coords_array) - 2):
            avg_x = np.mean(coords_array[i-2:i+3, 0])
            avg_y = np.mean(coords_array[i-2:i+3, 1])
            smoothed_coords.append((avg_x, avg_y))
        
        smoothed_coords.extend(snapped_coords[-2:])
        snapped_line = LineString(smoothed_coords)
    
    print(f"Centerline snapped successfully")
    return snapped_line


def derive_centerline(
    raster_path: str | Path, 
    minmax: tuple[float | None, float | None] = (None, None),
    hillshade_params: tuple[float, float, float] | None = None,
    show_overlay: bool = True,
    snap_to_channel: bool = True,
    snap_radius: float = 100.0
) -> LineString | None:
    """
    Derive the river centerline by querying OpenStreetMap Overpass API.
    
    This function extracts the bounding box from the raster, queries OSM for
    waterways within that area, and returns a merged centerline geometry.
    Optionally snaps the centerline to the actual channel bottom using elevation data.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    minmax : tuple[float | None, float | None], optional
        Min and max values for color range, also used to define river elevation range
        for snapping. If (None, None), auto-calculated from data.
    hillshade_params : tuple[float, float, float], optional
        (alpha, exaggeration, altitude) for hillshade overlay. If None, no hillshade applied.
    show_overlay : bool, optional
        If True, display an interactive plot with the centerline overlaid on the raster.
    snap_to_channel : bool, optional
        If True (default), snap the OSM centerline to the actual channel bottom
        using the elevation data. This corrects for OSM georeferencing errors.
    snap_radius : float, optional
        Search radius in meters for snapping to channel. Default 100m.
        
    Returns
    -------
    LineString or None
        Shapely LineString representing the river centerline in the raster's CRS,
        or None if no waterways found.
    """
    raster_path = Path(raster_path)
    
    # Get raster bounds and CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds  # (left, bottom, right, top) = (minx, miny, maxx, maxy)
        
    print(f"Raster CRS: {raster_crs}")
    print(f"Raster bounds: {raster_bounds}")
    
    # Transform bounds to WGS84 for Overpass API query
    bounds_wgs84 = transform_bounds(raster_crs, "EPSG:4326", *raster_bounds)
    print(f"Bounds in WGS84: {bounds_wgs84}")
    
    # Query OSM for waterways
    elements = _query_osm_waterways(bounds_wgs84)
    
    if not elements:
        print("No waterway elements found in the bounding box.")
        return None
    
    print(f"Retrieved {len(elements)} OSM elements")
    
    # Build LineStrings from OSM data
    linestrings = _build_linestrings_from_osm(elements)
    
    if not linestrings:
        print("No valid waterway geometries found.")
        return None
    
    print(f"Built {len(linestrings)} waterway LineStrings")
    
    # Merge all linestrings into a single geometry
    merged = linemerge(linestrings)
    
    # If result is MultiLineString, take the longest segment or keep as-is
    if isinstance(merged, MultiLineString):
        # Find the longest linestring
        longest = max(merged.geoms, key=lambda g: g.length)
        centerline_wgs84 = longest
        print(f"Merged result is MultiLineString with {len(merged.geoms)} parts, using longest segment")
    else:
        centerline_wgs84 = merged
    
    # Transform centerline from WGS84 back to raster CRS
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    
    def transform_coords(x, y):
        return transformer.transform(x, y)
    
    centerline_raster_crs = shapely_transform(transform_coords, centerline_wgs84)
    
    print(f"Centerline has {len(centerline_raster_crs.coords)} points")
    
    # Snap centerline to actual channel if requested
    if snap_to_channel:
        # Use minmax as the river elevation range if provided
        river_elev_range = minmax if (minmax[0] is not None and minmax[1] is not None) else None
        centerline_raster_crs = _snap_centerline_to_channel(
            centerline_raster_crs,
            raster_path,
            search_radius=snap_radius,
            river_elev_range=river_elev_range
        )
    
    # Display overlay if requested
    if show_overlay:
        display_centerline_overlay(raster_path, centerline_raster_crs, minmax, hillshade_params=hillshade_params)
    
    return centerline_raster_crs


def display_centerline_overlay(
    raster_path: str | Path,
    centerline: LineString,
    minmax: tuple[float | None, float | None] = (None, None),
    band: int = 1,
    hillshade_params: tuple[float, float, float] | None = None
) -> None:
    """
    Display the raster with the river centerline overlaid, optionally with hillshade.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    centerline : LineString
        Shapely LineString in the same CRS as the raster
    minmax : tuple[float | None, float | None], optional
        Min and max values for color range. If (None, None), auto-calculated.
    band : int, optional
        Band number to display (1-indexed), default is 1
    hillshade_params : tuple[float, float, float], optional
        (alpha, exaggeration, altitude) for hillshade overlay. If None, no hillshade.
    """
    raster_path = Path(raster_path)
    
    with rasterio.open(raster_path) as src:
        # Read data with optional downsampling for display
        max_dim = 2000
        height, width = src.height, src.width
        scale = max(height / max_dim, width / max_dim, 1.0)
        out_height = int(height / scale)
        out_width = int(width / scale)
        
        data = src.read(
            band,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear
        )
        
        # Get extent for proper coordinate display
        transform = src.transform
        left, bottom, right, top = src.bounds
        extent = [left, right, bottom, top]
        
        # Mask nodata
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_equal(data, nodata)
        
        # Calculate color range
        if minmax[0] is not None and minmax[1] is not None:
            vmin, vmax = minmax
        else:
            data_for_stats = data.compressed() if np.ma.is_masked(data) else data
            vmin, vmax = np.nanpercentile(data_for_stats, [2, 98])
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display raster (base layer)
    im = ax.imshow(
        data, 
        cmap='YlGnBu',  # Use same colormap as other viewers
        vmin=vmin, 
        vmax=vmax,
        extent=extent,
        origin='upper',
        zorder=1
    )
    plt.colorbar(im, ax=ax, label='Elevation', shrink=0.8)
    
    # Apply hillshade overlay if parameters provided
    if hillshade_params is not None:
        alpha, exaggeration, altitude = hillshade_params
        print(f"Applying hillshade: alpha={alpha}, exaggeration={exaggeration}, altitude={altitude}°")
        
        hillshade = _compute_hillshade(
            data,
            altitude=altitude,
            z_factor=exaggeration
        )
        
        ax.imshow(
            hillshade,
            cmap='gray',
            alpha=alpha,
            vmin=0,
            vmax=255,
            extent=extent,
            origin='upper',
            zorder=2
        )
    
    # Overlay the centerline (on top of everything)
    if centerline is not None and not centerline.is_empty:
        x, y = centerline.xy
        # Draw with a glow effect for visibility
        ax.plot(x, y, 'w-', linewidth=4, alpha=0.6, zorder=3)  # White outline
        ax.plot(x, y, color='#FF4500', linewidth=2.5, label='River Centerline (OSM)', alpha=0.95, zorder=4)
        
        # Mark start and end points
        ax.plot(x[0], y[0], 'go', markersize=12, markeredgecolor='white', 
                markeredgewidth=2, label='Start', zorder=5)
        ax.plot(x[-1], y[-1], 'ro', markersize=12, markeredgecolor='white', 
                markeredgewidth=2, label='End', zorder=5)
    
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    
    title = f'{raster_path.name}\nRiver Centerline from OpenStreetMap'
    if hillshade_params:
        title += f'\n(Hillshade: α={alpha:.1f}, z={exaggeration:.1f}x, alt={altitude:.0f}°)'
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def interactive_osm_centerline(
    raster_path: str | Path,
    osm_centerline: LineString,
    minmax: tuple[float | None, float | None] = (None, None),
    hillshade_params: tuple[float, float, float] | None = None,
    initial_snap_radius: float = 100.0
) -> LineString:
    """
    Interactive interface to adjust OSM centerline snapping parameters.
    
    Displays the raster with the centerline and provides controls to adjust:
    - Snap radius (how far to search for channel bottom)
    - Point spacing (how densely to sample points for snapping)
    - Show/hide centerline to inspect terrain
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    osm_centerline : LineString
        The raw OSM centerline (already in raster CRS)
    minmax : tuple[float | None, float | None], optional
        Min and max values for color range display.
    hillshade_params : tuple[float, float, float], optional
        (alpha, exaggeration, altitude) for hillshade overlay.
    initial_snap_radius : float, optional
        Initial snap radius in meters. Default 100.
        
    Returns
    -------
    LineString
        The snapped centerline
    """
    raster_path = Path(raster_path)
    
    # Load raster data
    with rasterio.open(raster_path) as src:
        max_dim = 2000
        height, width = src.height, src.width
        scale = max(height / max_dim, width / max_dim, 1.0)
        out_height = int(height / scale)
        out_width = int(width / scale)
        
        data = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear
        )
        
        left, bottom, right, top = src.bounds
        extent = [left, right, bottom, top]
        
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_equal(data, nodata)
        
        if minmax[0] is not None and minmax[1] is not None:
            vmin, vmax = minmax
        else:
            data_for_stats = data.compressed() if np.ma.is_masked(data) else data
            vmin, vmax = np.nanpercentile(data_for_stats, [2, 98])
    
    # State for the interactive viewer
    state = {
        'snap_radius': initial_snap_radius,
        'point_spacing': 20.0,
        'original_centerline': osm_centerline,
        'current_centerline': osm_centerline,
        'centerline_line': None,
        'centerline_glow': None,
        'start_marker': None,
        'end_marker': None,
        'radius_patches': [],
        'line_visible': True,
    }
    
    # Create figure with space for controls
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes([0.08, 0.28, 0.84, 0.65])
    
    # Display raster
    im = ax.imshow(
        data, 
        cmap='YlGnBu',
        vmin=vmin, 
        vmax=vmax,
        extent=extent,
        origin='upper',
        zorder=1
    )
    plt.colorbar(im, ax=ax, label='Elevation', shrink=0.8)
    
    # Apply hillshade if provided
    if hillshade_params is not None:
        alpha, exaggeration, altitude = hillshade_params
        hillshade = _compute_hillshade(data, altitude=altitude, z_factor=exaggeration)
        ax.imshow(
            hillshade,
            cmap='gray',
            alpha=alpha,
            vmin=0,
            vmax=255,
            extent=extent,
            origin='upper',
            zorder=2
        )
    
    # Initial centerline plot
    def plot_centerline(line):
        # Remove old plots
        if state['centerline_glow']:
            state['centerline_glow'].remove()
        if state['centerline_line']:
            state['centerline_line'].remove()
        if state['start_marker']:
            state['start_marker'].remove()
        if state['end_marker']:
            state['end_marker'].remove()
        
        if line is not None and not line.is_empty:
            x, y = line.xy
            state['centerline_glow'], = ax.plot(x, y, 'w-', linewidth=4, alpha=0.6, zorder=3)
            state['centerline_line'], = ax.plot(x, y, color='#FF4500', linewidth=2.5, zorder=4)
            state['start_marker'], = ax.plot(x[0], y[0], 'go', markersize=10, 
                                              markeredgecolor='white', markeredgewidth=2, zorder=5)
            state['end_marker'], = ax.plot(x[-1], y[-1], 'ro', markersize=10,
                                            markeredgecolor='white', markeredgewidth=2, zorder=5)
        
        fig.canvas.draw_idle()

    def clear_radius_overlay():
        for patch in state['radius_patches']:
            patch.remove()
        state['radius_patches'].clear()

    def update_radius_overlay(radius):
        clear_radius_overlay()
        line = state['current_centerline']
        if line is None or line.is_empty or radius <= 0:
            fig.canvas.draw_idle()
            return
        buffer_geom = line.buffer(radius, resolution=24)
        polygons = [buffer_geom] if buffer_geom.geom_type == 'Polygon' else list(buffer_geom.geoms)
        for poly in polygons:
            coords = np.asarray(poly.exterior.coords)
            patch = MplPolygon(
                coords,
                facecolor='#ff00ff',
                edgecolor='#8b008b',
                alpha=0.35,
                linewidth=1.8,
                linestyle='--',
                zorder=2.4
            )
            patch.set_visible(state['line_visible'])
            ax.add_patch(patch)
            state['radius_patches'].append(patch)
        fig.canvas.draw_idle()
    
    # Plot initial centerline
    plot_centerline(osm_centerline)
    update_radius_overlay(state['snap_radius'])
    
    title = ax.set_title(f'{raster_path.name}\nOSM Centerline - Adjust snapping parameters below')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    
    # Create control widgets - simplified interface
    # Stacked sliders
    ax_spacing = plt.axes([0.22, 0.16, 0.65, 0.025])  # Top slider
    ax_radius = plt.axes([0.22, 0.12, 0.65, 0.025])   # Bottom slider
    slider_spacing = Slider(ax_spacing, 'Point Spacing (m)', 5, 100, valinit=20, valstep=5)
    slider_radius = Slider(ax_radius, 'Snap Radius (m)', 10, 300, valinit=initial_snap_radius, valstep=10)
    state['point_spacing'] = 20.0  # Default to tighter spacing for better curve following
    slider_radius.on_changed(lambda val: update_radius_overlay(val))
    
    # Action buttons row
    ax_recompute = plt.axes([0.15, 0.05, 0.15, 0.045])
    ax_show_hide = plt.axes([0.35, 0.05, 0.15, 0.045])
    ax_reset = plt.axes([0.55, 0.05, 0.12, 0.045])
    ax_done = plt.axes([0.72, 0.05, 0.12, 0.045])
    
    btn_recompute = Button(ax_recompute, 'Recompute')
    btn_show_hide = Button(ax_show_hide, 'Hide Line')
    btn_reset = Button(ax_reset, 'Reset')
    btn_done = Button(ax_done, 'Done')
    
    # Status text
    ax_status = plt.axes([0.15, 0.01, 0.70, 0.04])
    ax_status.axis('off')
    status_text = ax_status.text(0, 0.5, 'Ready. Adjust sliders and click "Recompute".',
                                  fontsize=10, verticalalignment='center')
    
    def update_status(msg):
        status_text.set_text(msg)
        fig.canvas.draw_idle()
    
    def toggle_visibility(event):
        state['line_visible'] = not state['line_visible']
        btn_show_hide.label.set_text('Show Line' if not state['line_visible'] else 'Hide Line')
        
        # Toggle visibility of all centerline elements
        if state['centerline_glow']:
            state['centerline_glow'].set_visible(state['line_visible'])
        if state['centerline_line']:
            state['centerline_line'].set_visible(state['line_visible'])
        if state['start_marker']:
            state['start_marker'].set_visible(state['line_visible'])
        if state['end_marker']:
            state['end_marker'].set_visible(state['line_visible'])
        for patch in state['radius_patches']:
            patch.set_visible(state['line_visible'])
        
        fig.canvas.draw_idle()
    
    def recompute(event):
        state['snap_radius'] = slider_radius.val
        state['point_spacing'] = slider_spacing.val
        
        update_status(f'Snapping to lowest elevation: spacing={state["point_spacing"]:.0f}m, radius={state["snap_radius"]:.0f}m...')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
        try:
            snapped = _snap_centerline_to_channel(
                state['original_centerline'],
                raster_path,
                search_radius=state['snap_radius'],
                river_elev_range=None,  # Always snap to lowest elevation
                point_spacing=state['point_spacing']
            )
            state['current_centerline'] = snapped
            plot_centerline(snapped)
            update_radius_overlay(slider_radius.val)
            
            # Ensure visibility matches state
            if not state['line_visible']:
                if state['centerline_glow']:
                    state['centerline_glow'].set_visible(False)
                if state['centerline_line']:
                    state['centerline_line'].set_visible(False)
                if state['start_marker']:
                    state['start_marker'].set_visible(False)
                if state['end_marker']:
                    state['end_marker'].set_visible(False)
                for patch in state['radius_patches']:
                    patch.set_visible(False)
            
            update_status(f'✓ Done! {len(snapped.coords)} points | Spacing: {state["point_spacing"]:.0f}m | Radius: {state["snap_radius"]:.0f}m')
        except Exception as e:
            update_status(f'✗ Error: {str(e)}')
    
    def reset(event):
        slider_radius.set_val(initial_snap_radius)
        slider_spacing.set_val(20)
        state['point_spacing'] = 20.0
        state['line_visible'] = True
        btn_show_hide.label.set_text('Hide Line')
        state['current_centerline'] = state['original_centerline']
        plot_centerline(state['original_centerline'])
        update_radius_overlay(slider_radius.val)
        update_status('Reset. Click "Recompute" to snap to channel.')
    
    def done(event):
        plt.close(fig)
    
    btn_recompute.on_clicked(recompute)
    btn_show_hide.on_clicked(toggle_visibility)
    btn_reset.on_clicked(reset)
    btn_done.on_clicked(done)
    
    # Instructions
    print("\n" + "="*60)
    print("CENTERLINE SNAPPING")
    print("="*60)
    print("1. Adjust 'Point Spacing' - lower = follows curves better (10-25m)")
    print("2. Adjust 'Snap Radius' - search distance for channel (50-150m)")
    print("3. Click 'Recompute' to snap centerline to lowest elevations")
    print("4. Click 'Hide/Show Line' to see the terrain underneath")
    print("5. Click 'Done' when satisfied")
    print("="*60 + "\n")
    
    plt.show()
    
    return state['current_centerline'], state['snap_radius'], state['point_spacing']


def derive_centerline_interactive(
    raster_path: str | Path,
    minmax: tuple[float | None, float | None] = (None, None),
    hillshade_params: tuple[float, float, float] | None = None
) -> LineString | None:
    """
    Derive centerline from OSM with interactive snapping adjustment.
    
    This is a convenience function that:
    1. Queries OSM for the river centerline
    2. Opens an interactive viewer to adjust snapping parameters
    3. Returns the final adjusted centerline
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    minmax : tuple[float | None, float | None], optional
        Min and max values for color range and initial elevation range.
    hillshade_params : tuple[float, float, float], optional
        (alpha, exaggeration, altitude) for hillshade overlay.
        
    Returns
    -------
    LineString or None
        The adjusted centerline, or None if no waterways found.
    """
    raster_path = Path(raster_path)
    
    # Get raster bounds and CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        
    print(f"Raster CRS: {raster_crs}")
    print(f"Raster bounds: {raster_bounds}")
    
    # Transform bounds to WGS84 for Overpass API query
    bounds_wgs84 = transform_bounds(raster_crs, "EPSG:4326", *raster_bounds)
    print(f"Bounds in WGS84: {bounds_wgs84}")
    
    # Query OSM for waterways
    elements = _query_osm_waterways(bounds_wgs84)
    
    if not elements:
        print("No waterway elements found in the bounding box.")
        return None
    
    print(f"Retrieved {len(elements)} OSM elements")
    
    # Build LineStrings from OSM data
    linestrings = _build_linestrings_from_osm(elements)
    
    if not linestrings:
        print("No valid waterway geometries found.")
        return None
    
    print(f"Built {len(linestrings)} waterway LineStrings")
    
    # Merge all linestrings
    merged = linemerge(linestrings)
    
    if isinstance(merged, MultiLineString):
        longest = max(merged.geoms, key=lambda g: g.length)
        centerline_wgs84 = longest
        print(f"Merged result is MultiLineString with {len(merged.geoms)} parts, using longest segment")
    else:
        centerline_wgs84 = merged
    
    # Transform to raster CRS
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    
    def transform_coords(x, y):
        return transformer.transform(x, y)
    
    osm_centerline = shapely_transform(transform_coords, centerline_wgs84)
    
    print(f"OSM Centerline has {len(osm_centerline.coords)} points")
    
    # Open interactive snapping interface
    return interactive_osm_centerline(
        raster_path,
        osm_centerline,
        minmax=minmax,
        hillshade_params=hillshade_params
    )


__all__ = [
    "interactive_raster_viewer",
    "minmax_widget",
    "hillshade_widget",
    "centerline_widget",
    "manual_centerline_widget",
    "interactive_min_max",
    "interactive_hillshade",
    "derive_centerline",
    "derive_centerline_interactive",
    "interactive_osm_centerline",
    "display_centerline_overlay",
    "display_raster",
    "ViewerState"
]