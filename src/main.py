"""Entry point for the GIS 584 project package."""

from pathlib import Path

from osgeo import gdal

from .utils import merge_tifs


def main() -> None:
    """Run the main application routine."""
    print("Hello from GIS 584 project!")

    gdal.UseExceptions()
    print(f"GDAL version: {gdal.VersionInfo('RELEASE_NAME')}")

    tutorial_dir = Path("data/tutorial_2")
    output_path = Path("outputs/tutorial_merged.tif")

    print(f"Merging rasters from {tutorial_dir} -> {output_path}")
    merged = merge_tifs(tutorial_dir, output_path)
    print(f"Merged raster written to {merged}")


    


if __name__ == "__main__":
    main()



