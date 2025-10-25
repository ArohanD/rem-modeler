"""Utility helpers for raster processing tasks."""

from __future__ import annotations

from pathlib import Path
from osgeo import gdal


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


__all__ = ["merge_tifs"]