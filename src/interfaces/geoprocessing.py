"""Geoprocessing utilities for the GIS 584 project.

This module contains pure geoprocessing functions without UI dependencies.
"""

from __future__ import annotations

from pathlib import Path
import json
import urllib.request
import urllib.parse
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from shapely.geometry import LineString
import rasterio


def skeleton_to_linestring(skeleton: np.ndarray, transform=None) -> LineString | None:
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


def extract_centerline(
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


def compute_hillshade(
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


def query_osm_waterways(bbox_wgs84: tuple[float, float, float, float]) -> list[dict]:
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


def build_linestrings_from_osm(elements: list[dict]) -> list[LineString]:
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


def densify_line(line: LineString, max_segment_length: float) -> LineString:
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


def snap_centerline_to_channel(
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
    centerline = densify_line(centerline, point_spacing)
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


__all__ = [
    "skeleton_to_linestring",
    "extract_centerline",
    "compute_hillshade",
    "query_osm_waterways",
    "build_linestrings_from_osm",
    "densify_line",
    "snap_centerline_to_channel",
]

