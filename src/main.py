"""Entry point for the GIS 584 project package."""

from pathlib import Path

from osgeo import gdal
import numpy as np
import rasterio
import matplotlib.pyplot as plt

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
    river_min_elevation, river_max_elevation = interactive_min_max(merged, band)
    
    print(f"Selected elevation range: {river_min_elevation:.2f} to {river_max_elevation:.2f}")

    # Allow the user to customize a hillshade on top of the current raster
    alpha, exaggeration, altitude = interactive_hillshade(
        merged, 
        band, 
        minmax=(river_min_elevation, river_max_elevation)
    )
    
    print(f"Hillshade settings: alpha={alpha:.2f}, exaggeration={exaggeration:.2f}, altitude={altitude:.1f}°")

    # Derive centerline from OpenStreetMap with INTERACTIVE snapping adjustment
    centerline, snap_radius, point_spacing = derive_centerline_interactive(
        merged,
        minmax=(river_min_elevation, river_max_elevation), 
        hillshade_params=(alpha, exaggeration, altitude)   # alpha, exaggeration, altitude
        # minmax=(1300, 1320),
        # hillshade_params=(0.5, 5, 45)  
    )

    # Create a point shapefile from the centerline, with the point spacing as the distance between points
    densified_centerline_points = create_points_from_centerline(centerline, point_spacing)

    # sample the elevations along the centerline
    elevations_along_centerline = sample_elevations_along_line(densified_centerline_points, merged, band)

    # interpolate the elevations along the centerline using IDW
    water_surface = interpolate(elevations_along_centerline, merged, band)

    # Create REM (Relative Elevation Model)
    print("\n" + "="*60)
    print("CREATING RELATIVE ELEVATION MODEL (REM)")
    print("="*60)
    
    # Read the original DEM
    with rasterio.open(merged) as src:
        dem = src.read(band).astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()
    
    # Calculate REM = DEM - water_surface
    rem = dem - water_surface
    
    # Handle nodata values
    if nodata is not None:
        mask = dem == nodata
        rem[mask] = nodata
        print(f"REM elevation range: {np.nanmin(rem[~mask]):.2f} to {np.nanmax(rem[~mask]):.2f}m")
    else:
        mask = np.zeros_like(rem, dtype=bool)
        print(f"REM elevation range: {rem.min():.2f} to {rem.max():.2f}m")
    
    # Display the REM (downsample for speed)
    print("\nDisplaying REM...")
    
    # Downsample for fast display (every 10th pixel)
    downsample_factor = 10
    rem_display = rem[::downsample_factor, ::downsample_factor].copy()
    
    # Mask nodata for display
    if nodata is not None:
        mask_display = mask[::downsample_factor, ::downsample_factor]
        rem_display[mask_display] = np.nan
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Display with color range 0-10 meters (typical for floodplain features). 5 looks prettier sometimes tho
    im = ax.imshow(rem_display, cmap='YlGnBu', vmin=0, vmax=5, interpolation='bilinear')
    ax.set_title('Relative Elevation Model (REM)\nHeight Above River Surface', fontsize=14, pad=15)
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height above river (m)', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Save the REM to file
    output_path = Path("outputs/rem_script.tif")
    output_path.parent.mkdir(exist_ok=True)
    
    profile.update(dtype=rasterio.float32, compress='lzw')
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(rem, 1)
    
    print(f"\n✓ REM saved to: {output_path}")
    print("="*60)


    


if __name__ == "__main__":
    main()







