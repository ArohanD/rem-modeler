"""Entry point for the GIS 584 project package."""

from pathlib import Path

from osgeo import gdal

from src.utils.utils import create_points_from_centerline, interpolate, sample_elevations_along_line

from .utils import merge_tifs, print_raster_stats
from .interfaces import interactive_min_max, interactive_hillshade, interactive_osm_centerline, derive_centerline, derive_centerline_interactive


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
    # print_raster_stats(merged)  # Commented out due to Windows encoding issue with Unicode chars

    # Open an interactive panel of the raster where the user can get the value of
    # a pixel under the cursor for a specified band
    band = 1
    # river_min_elevation, river_max_elevation = interactive_min_max(merged, band)
    
    # print(f"Selected elevation range: {river_min_elevation:.2f} to {river_max_elevation:.2f}")

    # Allow the user to customize a hillshade on top of the current raster
    # alpha, exaggeration, altitude = interactive_hillshade(
    #     merged, 
    #     band, 
    #     minmax=(river_min_elevation, river_max_elevation)
    # )
    
    # print(f"Hillshade settings: alpha={alpha:.2f}, exaggeration={exaggeration:.2f}, altitude={altitude:.1f}°")

    # Derive centerline from OpenStreetMap with INTERACTIVE snapping adjustment
    centerline, snap_radius, point_spacing = derive_centerline_interactive(
        merged,
        minmax=(1300, 1320),
        hillshade_params=(0.5, 5, 45)  # alpha, exaggeration, altitude
    )

    # Create a point shapefile from the centerline, with the point spacing as the distance between points
    densified_centerline_points = create_points_from_centerline(centerline, point_spacing)

    # sample the elevations along the centerline
    elevations_along_centerline = sample_elevations_along_line(densified_centerline_points, merged, band)

    # interpolate the elevations along the centerline using IDW
    idw_interpolated_elevations = interpolate(elevations_along_centerline, merged, band)

    import pdb; pdb.set_trace()


    


if __name__ == "__main__":
    main()







