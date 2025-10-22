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

    gdal.UseExceptions()
    dataset = gdal.Warp(
        destNameOrDestDS=str(out_path),
        srcDSOrSrcDSTab=[str(path) for path in tif_files],
    )

    if dataset is not None:
        dataset = None

    return out_path


__all__ = ["merge_tifs"]

