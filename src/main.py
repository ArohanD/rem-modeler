"""Entry point for the GIS 584 project package."""

from pathlib import Path

from osgeo import gdal

from .utils import merge_tifs, print_raster_stats


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

    # Allow the user to input a min and max elevation for the river
    river_min_elevation = float(input("Enter the minimum elevation for the river: "))
    river_max_elevation = float(input("Enter the maximum elevation for the river: "))
    print(f"Proceeding with a minimum elevation of {river_min_elevation} and a maximum elevation of {river_max_elevation}")
    


if __name__ == "__main__":
    main()







