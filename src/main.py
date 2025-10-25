"""Entry point for the GIS 584 project package."""

from pathlib import Path

from osgeo import gdal

from .utils import merge_tifs, print_raster_stats
from .interfaces import preview_tif


def main() -> None:
    """Run the main application routine."""
    print("Hello from GIS 584 project!")

    gdal.UseExceptions()
    print(f"GDAL version: {gdal.VersionInfo('RELEASE_NAME')}")

    tutorial_dir = Path("data/tutorial_2")
    output_path = Path("outputs/tutorial_merged.tif")

    # Uncomment to merge rasters
    # print(f"Merging rasters from {tutorial_dir} -> {output_path}")
    # merged = merge_tifs(tutorial_dir, output_path)
    # print(f"Merged raster written to {merged}")

    merged = Path("outputs/tutorial_merged.tif")

    # Print Merged Raster Stats
    # Want to know the distribution of values in the merged raster
    print_raster_stats(merged)

    # Open an interactive panel of the raster where the user can get the value of
    # a pixel under the cursor for a specified band
    band = 1
    river_min_elevation, river_max_elevation = preview_tif(merged, band)
    
    print(f"Selected elevation range: {river_min_elevation:.2f} to {river_max_elevation:.2f}")


    


if __name__ == "__main__":
    main()







