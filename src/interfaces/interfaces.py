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
from affine import Affine
from scipy import ndimage
from skimage.morphology import skeletonize
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge


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


def interactive_centerline(
    raster_path: str | Path,
    band: int = 1,
    minmax: tuple[float | None, float | None] = (None, None),
    hillshade_params: tuple[float, float, float] | None = None,
    manual: bool = False
) -> LineString:
    """
    Display a raster and compute/digitize river centerline.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file
    band : int, optional
        Band number to display (1-indexed), default is 1
    minmax : tuple[float | None, float | None], optional
        Min and max values defining river extent. If (None, None), auto-calculated.
    hillshade_params : tuple[float, float, float], optional
        (alpha, exaggeration, altitude) from previous hillshade step
    manual : bool, optional
        If True, use manual digitization. If False, use automatic extraction.
        
    Returns
    -------
    LineString
        Shapely LineString representing the river centerline in geographic coordinates
    """
    # Create initial state with hillshade params if provided
    state = _load_raster_data(Path(raster_path), band, minmax)
    
    if hillshade_params:
        state.user_inputs['hillshade_alpha'] = hillshade_params[0]
        state.user_inputs['hillshade_exaggeration'] = hillshade_params[1]
        state.user_inputs['hillshade_altitude'] = hillshade_params[2]
    
    # Now use the viewer with pre-populated state
    # We'll recreate the viewer but with the state already initialized
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
    
    # Add appropriate centerline widget
    if manual:
        widgets = manual_centerline_widget(state, fig, im)
    else:
        widgets = centerline_widget(state, fig, im)
    
    plt.show()
    
    return state.user_inputs.get('centerline_geom')


__all__ = [
    "interactive_raster_viewer",
    "minmax_widget",
    "hillshade_widget",
    "centerline_widget",
    "manual_centerline_widget",
    "interactive_min_max",
    "interactive_hillshade",
    "interactive_centerline",
    "display_raster",
    "ViewerState"
]