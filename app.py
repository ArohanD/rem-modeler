"""Streamlit dashboard for GIS 584 raster processing workflow."""

import streamlit as st
from pathlib import Path
from osgeo import gdal

from src.utils import merge_tifs, print_raster_stats
from src.interfaces.streamlit_interfaces import (
    display_raster_streamlit,
    display_raster_with_hillshade_streamlit,
    get_raster_stats_dict
)

# Page configuration
st.set_page_config(
    page_title="GIS 584 Raster Processing",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'merged_raster_path' not in st.session_state:
    st.session_state.merged_raster_path = None
if 'raster_stats' not in st.session_state:
    st.session_state.raster_stats = None
if 'min_elevation' not in st.session_state:
    st.session_state.min_elevation = None
if 'max_elevation' not in st.session_state:
    st.session_state.max_elevation = None
if 'hillshade_alpha' not in st.session_state:
    st.session_state.hillshade_alpha = 0.3
if 'hillshade_exaggeration' not in st.session_state:
    st.session_state.hillshade_exaggeration = 1.0
if 'hillshade_altitude' not in st.session_state:
    st.session_state.hillshade_altitude = 45.0

# Title
st.title("🗺️ GIS 584 Raster Processing Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Workflow Steps")
step = st.sidebar.radio(
    "Select Step:",
    ["1. Merge Rasters", "2. View Statistics", "3. Set Elevation Range", "4. Add Hillshade"],
    index=0
)

# Initialize GDAL
gdal.UseExceptions()

# Step 1: Merge Rasters
if step == "1. Merge Rasters":
    st.header("Step 1: Merge Raster Files")
    st.markdown("Select a directory containing TIFF files to merge into a single raster.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Directory input
        tutorial_dir = st.text_input(
            "Raster Directory Path:",
            value="data/tutorial_2",
            help="Path to directory containing .tif files to merge"
        )
        
        output_path = st.text_input(
            "Output Path:",
            value="outputs/tutorial_merged.tif",
            help="Path where the merged raster will be saved"
        )
    
    with col2:
        st.markdown("### GDAL Info")
        try:
            gdal_version = gdal.VersionInfo('RELEASE_NAME')
            st.info(f"GDAL Version: {gdal_version}")
        except:
            st.warning("Could not retrieve GDAL version")
    
    # Check if output already exists
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        st.warning(f"⚠️ Output file already exists: {output_path}")
        if st.button("🗑️ Delete Existing File"):
            try:
                output_path_obj.unlink()
                st.success("File deleted successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting file: {e}")
    
    # Merge button
    if st.button("🔄 Merge Rasters", type="primary"):
        tutorial_dir_obj = Path(tutorial_dir)
        
        if not tutorial_dir_obj.exists():
            st.error(f"❌ Directory not found: {tutorial_dir}")
        else:
            # Count TIFF files
            tif_files = list(tutorial_dir_obj.glob("*.tif"))
            if not tif_files:
                st.error(f"❌ No .tif files found in {tutorial_dir}")
            else:
                st.info(f"Found {len(tif_files)} TIFF file(s) to merge")
                
                with st.spinner("Merging rasters... This may take a while..."):
                    try:
                        merged_path = merge_tifs(tutorial_dir_obj, output_path_obj)
                        st.session_state.merged_raster_path = str(merged_path)
                        st.success(f"✅ Successfully merged rasters to: {merged_path}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error merging rasters: {e}")
    
    # Display merged raster if it exists
    if st.session_state.merged_raster_path and Path(st.session_state.merged_raster_path).exists():
        st.markdown("---")
        st.subheader("📊 Merged Raster Preview")
        try:
            display_raster_streamlit(
                st.session_state.merged_raster_path,
                band=1,
                title="Merged Raster"
            )
        except Exception as e:
            st.error(f"Error displaying raster: {e}")
    
    # Also check default path
    default_merged = Path("outputs/tutorial_merged.tif")
    if default_merged.exists() and st.session_state.merged_raster_path is None:
        st.session_state.merged_raster_path = str(default_merged)
        st.info(f"📁 Found existing merged raster: {default_merged}")

# Step 2: View Statistics
elif step == "2. View Statistics":
    st.header("Step 2: View Raster Statistics")
    
    # Check for merged raster
    if st.session_state.merged_raster_path is None:
        default_merged = Path("outputs/tutorial_merged.tif")
        if default_merged.exists():
            st.session_state.merged_raster_path = str(default_merged)
        else:
            st.warning("⚠️ No merged raster found. Please complete Step 1 first.")
            st.stop()
    
    raster_path = Path(st.session_state.merged_raster_path)
    
    if not raster_path.exists():
        st.error(f"❌ Raster file not found: {raster_path}")
        st.stop()
    
    # Get statistics
    if st.session_state.raster_stats is None:
        with st.spinner("Computing statistics..."):
            st.session_state.raster_stats = get_raster_stats_dict(raster_path)
    
    stats = st.session_state.raster_stats
    
    # Display statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Raster Information")
        st.metric("Width", f"{stats['width']:,} pixels")
        st.metric("Height", f"{stats['height']:,} pixels")
        st.metric("Bands", stats['count'])
        if stats['crs']:
            st.text(f"CRS: {stats['crs']}")
    
    with col2:
        st.subheader("📊 Band Statistics")
        for band_num, band_stats in stats['bands'].items():
            with st.expander(f"Band {band_num}", expanded=(band_num == 1)):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Min", f"{band_stats['min']:.2f}")
                    st.metric("Max", f"{band_stats['max']:.2f}")
                with col_b:
                    st.metric("Mean", f"{band_stats['mean']:.2f}")
                    st.metric("Std Dev", f"{band_stats['std']:.2f}")
                if band_stats['nodata'] is not None:
                    st.info(f"NoData value: {band_stats['nodata']}")
    
    # Display raster
    st.markdown("---")
    st.subheader("🗺️ Raster Visualization")
    display_raster_streamlit(
        raster_path,
        band=1,
        title="Merged Raster"
    )
    
    # Print detailed stats to console/log
    if st.button("📝 Print Detailed Statistics"):
        with st.spinner("Computing detailed statistics..."):
            print_raster_stats(raster_path)
        st.success("Statistics printed to console/log")

# Step 3: Set Elevation Range
elif step == "3. Set Elevation Range":
    st.header("Step 3: Set Elevation Range")
    st.markdown("Adjust the min/max elevation values to focus on specific elevation ranges.")
    
    # Check for merged raster
    if st.session_state.merged_raster_path is None:
        default_merged = Path("outputs/tutorial_merged.tif")
        if default_merged.exists():
            st.session_state.merged_raster_path = str(default_merged)
        else:
            st.warning("⚠️ No merged raster found. Please complete Step 1 first.")
            st.stop()
    
    raster_path = Path(st.session_state.merged_raster_path)
    
    if not raster_path.exists():
        st.error(f"❌ Raster file not found: {raster_path}")
        st.stop()
    
    # Get statistics for default values
    if st.session_state.raster_stats is None:
        st.session_state.raster_stats = get_raster_stats_dict(raster_path)
    
    stats = st.session_state.raster_stats
    band_stats = stats['bands'].get(1, {})
    
    # Default values
    default_min = st.session_state.min_elevation if st.session_state.min_elevation is not None else band_stats.get('min', 0)
    default_max = st.session_state.max_elevation if st.session_state.max_elevation is not None else band_stats.get('max', 1000)
    
    # UI Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Elevation Range Controls")
        
        min_elevation = st.number_input(
            "Min Elevation:",
            value=float(default_min),
            step=1.0,
            format="%.2f",
            help="Minimum elevation value for color scaling"
        )
        
        max_elevation = st.number_input(
            "Max Elevation:",
            value=float(default_max),
            step=1.0,
            format="%.2f",
            help="Maximum elevation value for color scaling"
        )
        
        if min_elevation >= max_elevation:
            st.error("⚠️ Min elevation must be less than max elevation")
        else:
            st.session_state.min_elevation = min_elevation
            st.session_state.max_elevation = max_elevation
        
        st.info(f"**Range:** {min_elevation:.2f} to {max_elevation:.2f}")
        st.info(f"**Span:** {max_elevation - min_elevation:.2f}")
    
    with col2:
        st.subheader("📊 Current Statistics")
        st.metric("Global Min", f"{band_stats.get('min', 0):.2f}")
        st.metric("Global Max", f"{band_stats.get('max', 1000):.2f}")
        st.metric("Global Mean", f"{band_stats.get('mean', 0):.2f}")
    
    # Display raster with custom range
    st.markdown("---")
    st.subheader("🗺️ Raster with Custom Elevation Range")
    
    if min_elevation < max_elevation:
        try:
            display_raster_streamlit(
                raster_path,
                band=1,
                minmax=(min_elevation, max_elevation),
                title=f"Elevation Range: {min_elevation:.2f} - {max_elevation:.2f}"
            )
        except Exception as e:
            st.error(f"Error displaying raster: {e}")

# Step 4: Add Hillshade
elif step == "4. Add Hillshade":
    st.header("Step 4: Add Hillshade Overlay")
    st.markdown("Customize the hillshade overlay to enhance terrain visualization.")
    
    # Check for merged raster
    if st.session_state.merged_raster_path is None:
        default_merged = Path("outputs/tutorial_merged.tif")
        if default_merged.exists():
            st.session_state.merged_raster_path = str(default_merged)
        else:
            st.warning("⚠️ No merged raster found. Please complete Step 1 first.")
            st.stop()
    
    raster_path = Path(st.session_state.merged_raster_path)
    
    if not raster_path.exists():
        st.error(f"❌ Raster file not found: {raster_path}")
        st.stop()
    
    # Get elevation range (use from Step 3 or auto-calculate)
    minmax = None
    if st.session_state.min_elevation is not None and st.session_state.max_elevation is not None:
        minmax = (st.session_state.min_elevation, st.session_state.max_elevation)
    
    # UI Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Hillshade Controls")
        
        alpha = st.slider(
            "Transparency (Alpha):",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.hillshade_alpha,
            step=0.05,
            help="Hillshade overlay transparency (0 = transparent, 1 = opaque)"
        )
        
        exaggeration = st.slider(
            "Vertical Exaggeration:",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.hillshade_exaggeration,
            step=0.1,
            help="Vertical exaggeration factor for terrain relief"
        )
        
        altitude = st.slider(
            "Light Altitude (degrees):",
            min_value=0.0,
            max_value=90.0,
            value=st.session_state.hillshade_altitude,
            step=1.0,
            help="Altitude angle of light source (0 = horizon, 90 = overhead)"
        )
        
        azimuth = st.slider(
            "Light Azimuth (degrees):",
            min_value=0.0,
            max_value=360.0,
            value=315.0,
            step=15.0,
            help="Direction of light source (0 = North, 90 = East, 180 = South, 270 = West)"
        )
        
        # Update session state
        st.session_state.hillshade_alpha = alpha
        st.session_state.hillshade_exaggeration = exaggeration
        st.session_state.hillshade_altitude = altitude
        
        st.info(f"**Settings:**\n- Alpha: {alpha:.2f}\n- Exaggeration: {exaggeration:.2f}x\n- Altitude: {altitude:.1f}°\n- Azimuth: {azimuth:.1f}°")
    
    with col2:
        st.subheader("💡 Tips")
        st.markdown("""
        - **Alpha (0.2-0.4)**: Lower values show more of the base raster
        - **Exaggeration (1.0-3.0)**: Higher values emphasize terrain relief
        - **Altitude (30-60°)**: Higher angles create more uniform lighting
        - **Azimuth (315°)**: Northwest lighting is common for terrain visualization
        """)
    
    # Display raster with hillshade
    st.markdown("---")
    st.subheader("🗺️ Raster with Hillshade Overlay")
    
    try:
        display_raster_with_hillshade_streamlit(
            raster_path,
            band=1,
            minmax=minmax,
            alpha=alpha,
            exaggeration=exaggeration,
            altitude=altitude,
            azimuth=azimuth,
            title=f"Hillshade: α={alpha:.2f}, exag={exaggeration:.2f}x, alt={altitude:.1f}°, az={azimuth:.1f}°"
        )
    except Exception as e:
        st.error(f"Error displaying raster with hillshade: {e}")
    
    # Summary
    st.markdown("---")
    st.subheader("📋 Summary")
    col_sum1, col_sum2 = st.columns(2)
    
    with col_sum1:
        st.markdown("**Elevation Range:**")
        if st.session_state.min_elevation is not None and st.session_state.max_elevation is not None:
            st.write(f"- Min: {st.session_state.min_elevation:.2f}")
            st.write(f"- Max: {st.session_state.max_elevation:.2f}")
        else:
            st.write("- Using auto-calculated range")
    
    with col_sum2:
        st.markdown("**Hillshade Settings:**")
        st.write(f"- Alpha: {alpha:.2f}")
        st.write(f"- Exaggeration: {exaggeration:.2f}x")
        st.write(f"- Altitude: {altitude:.1f}°")

# Footer
st.markdown("---")
st.markdown("### GIS 584 Project - Raster Processing Dashboard")

