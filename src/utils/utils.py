"""Utility helpers for raster processing tasks."""

from __future__ import annotations

from pathlib import Path
from osgeo import gdal
from rasterio import rasterio
from shapely.geometry import LineString, Point
import numpy as np


def merge_tifs(directory: str | Path, output_path: str | Path | None = None) -> Path:
    """Merge all ``.tif`` rasters in ``directory`` into a single GeoTIFF."""

    input_dir = Path(directory)

    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a directory")

    tif_files = sorted(input_dir.glob("*.tif"))

    if not tif_files:
        raise ValueError(f"No .tif files found in {input_dir}")

    out_path = Path(output_path or input_dir / "merged.tif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing output file if it exists
    if out_path.exists():
        try:
            out_path.unlink()
            print(f"Removed existing file: {out_path}")
        except PermissionError:
            raise RuntimeError(
                f"Cannot delete existing file {out_path}. "
                "It may be open in another program (QGIS, viewer, etc.). "
                "Please close any programs using this file and try again."
            )

    print(f"Merging {len(tif_files)} rasters to: {out_path}")
    print("Processing...")

    # Use GDAL's BuildVRT + Translate approach (faster than gdal_merge for large files)
    gdal.UseExceptions()
    
    # Step 1: Build a VRT (Virtual Dataset) - very fast, just metadata
    vrt_path = out_path.with_suffix('.vrt')
    print(f"Building VRT: {vrt_path}")
    vrt_options = gdal.BuildVRTOptions(resolution='highest', addAlpha=False)
    vrt_ds = gdal.BuildVRT(str(vrt_path), [str(f) for f in tif_files], options=vrt_options)
    
    if vrt_ds is None:
        raise RuntimeError("Failed to build VRT")
    
    vrt_ds = None  # Close VRT
    print(f"VRT created successfully")
    
    # Step 2: Translate VRT to GeoTIFF - this is where the actual merging happens
    print(f"Converting VRT to GeoTIFF: {out_path}")
    translate_options = gdal.TranslateOptions(
        format='GTiff',
        creationOptions=['TILED=YES', 'BIGTIFF=IF_SAFER']
    )
    
    ds = gdal.Translate(str(out_path), str(vrt_path), options=translate_options)
    
    if ds is None:
        raise RuntimeError(f"Failed to create merged raster at {out_path}")
    
    ds.FlushCache()
    ds = None  # Close dataset
    
    # Clean up VRT
    vrt_path.unlink()
    
    if not out_path.exists():
        raise RuntimeError(f"Failed to create merged raster at {out_path}")
    
    print(f"\n✓ Successfully merged to {out_path}")
    print(f"Output file size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    
    return out_path


def print_raster_stats(raster_path: str | Path) -> None:
    """
    Print the statistics of a raster file.

    Most importantly, we want to know the distribution of values in the raster
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise ValueError(f"Raster file {raster_path} does not exist")
    
    gdal.UseExceptions()
    ds = gdal.Open(str(raster_path))
    if ds is None:
        raise ValueError(f"Failed to open raster file {raster_path}")

    print(f"\n{'='*60}")
    print(f"Raster file: {raster_path}")
    print(f"{'='*60}")
    print(f"Dimensions: {ds.RasterXSize} x {ds.RasterYSize} pixels")
    print(f"Number of bands: {ds.RasterCount}")
    print(f"Projection: {ds.GetProjection()[:80]}...")  # Truncate long projection string
    print(f"Geotransform: {ds.GetGeoTransform()}")
    
    # Get statistics for each band (usually just 1 for elevation data)
    for band_num in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_num)
        
        print(f"\n--- Band {band_num} Statistics ---")
        
        # Get or compute statistics
        stats = band.GetStatistics(True, True)  # (approx_ok=True, force=True)
        print(f"  Min value:    {stats[0]:.2f}")
        print(f"  Max value:    {stats[1]:.2f}")
        print(f"  Mean value:   {stats[2]:.2f}")
        print(f"  Std Dev:      {stats[3]:.2f}")
        
        # Get histogram
        print(f"\n  Computing histogram...")
        hist = band.GetHistogram(min=stats[0], max=stats[1], buckets=10, approx_ok=False)
        
        # Print histogram in a readable format
        print(f"\n  Value Distribution (10 bins):")
        bin_width = (stats[1] - stats[0]) / 10
        for i, count in enumerate(hist):
            bin_min = stats[0] + (i * bin_width)
            bin_max = bin_min + bin_width
            bar = '█' * int(count / max(hist) * 50)  # Scale to 50 chars max
            print(f"    {bin_min:8.2f} - {bin_max:8.2f}: {int(count):10d} {bar}")
        
        # Check for NoData value
        nodata = band.GetNoDataValue()
        if nodata is not None:
            print(f"\n  NoData value: {nodata}")
        
        band = None  # Close band
    
    ds = None  # Close dataset
    print(f"\n{'='*60}\n")

def create_points_from_centerline(centerline: LineString, point_spacing: float) -> LineString:
    """
    Create evenly spaced points along a centerline and return as a new LineString.
    
    Parameters
    ----------
    centerline : LineString
        The input centerline geometry
    point_spacing : float
        Distance between consecutive points (in map units)
        
    Returns
    -------
    LineString
        A new LineString with points sampled at regular intervals
    """
    total_length = centerline.length
    
    # Generate distances along the line
    distances = []
    distance = 0.0
    while distance <= total_length:
        distances.append(distance)
        distance += point_spacing
    
    # Always include the endpoint
    if distances[-1] < total_length:
        distances.append(total_length)
    
    # Interpolate points at each distance
    coords = []
    for d in distances:
        point = centerline.interpolate(d)
        coords.append((point.x, point.y))
    
    return LineString(coords)


def sample_elevations_along_line(
    centerline: LineString, 
    raster_path: str | Path, 
    band: int = 1
) -> list[tuple[float, float, float]]:
    """
    Sample elevation values from a raster at each point along a centerline.
    
    Returns list of (x, y, elevation) tuples for valid sample points.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(band)
        transform = src.transform
        nodata = src.nodata

    samples = []
    for x, y in centerline.coords:
        # Convert geographic coords to pixel coords
        col, row = ~transform * (x, y)
        col, row = int(col), int(row)
        
        # Check if pixel is within raster bounds
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            value = data[row, col]
            
            # Skip nodata values
            if nodata is None or value != nodata:
                samples.append((x, y, float(value)))
    
    return samples

def idw_interpolate(points: LineString, raster_path: str | Path, band: int) -> np.ndarray:
    """
    Interpolate the values of a raster at the points using IDW.
    Parameters
    ----------
    points: LineString
        The points to interpolate the values at
    raster_path: str | Path
        The path to the raster file
    band: int
        The band to interpolate the values from
    """

    




__all__ = ["merge_tifs", "print_raster_stats", "create_points_from_centerline"]
