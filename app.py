"""Streamlit dashboard for GIS 584 raster processing workflow."""

import streamlit as st
from pathlib import Path
from osgeo import gdal
import traceback

from src.utils import print_raster_stats
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
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'raster_path' not in st.session_state:
    st.session_state.raster_path = None
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
if 'hillshade_azimuth' not in st.session_state:
    st.session_state.hillshade_azimuth = 315.0
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# Initialize GDAL
gdal.UseExceptions()

# Title
st.title("🗺️ GIS 584 Raster Processing Dashboard")
st.markdown("---")

# Try to find raster file
default_raster = Path("outputs/tutorial_merged.tif")
if st.session_state.raster_path is None and default_raster.exists():
    st.session_state.raster_path = str(default_raster)

# Main layout: Map on left, controls on right
map_col, control_col = st.columns([2, 1])

with map_col:
    st.subheader("🗺️ Map View")
    
    # Display map based on current step
    if st.session_state.raster_path and Path(st.session_state.raster_path).exists():
        raster_path = Path(st.session_state.raster_path)
        
        # Determine which visualization to show
        if st.session_state.current_step >= 3:
            # Show hillshade
            minmax = None
            if st.session_state.min_elevation is not None and st.session_state.max_elevation is not None:
                minmax = (st.session_state.min_elevation, st.session_state.max_elevation)
            
            try:
                display_raster_with_hillshade_streamlit(
                    raster_path,
                    band=1,
                    minmax=minmax,
                    alpha=st.session_state.hillshade_alpha,
                    exaggeration=st.session_state.hillshade_exaggeration,
                    altitude=st.session_state.hillshade_altitude,
                    azimuth=st.session_state.hillshade_azimuth,
                    title=None  # No title in map view
                )
            except Exception as e:
                st.error(f"Error displaying raster with hillshade: {e}")
                with st.expander("🔍 Full Error Details (Click to expand)"):
                    st.code(traceback.format_exc(), language="python")
        elif st.session_state.current_step >= 2:
            # Show with custom elevation range
            minmax = None
            if st.session_state.min_elevation is not None and st.session_state.max_elevation is not None:
                minmax = (st.session_state.min_elevation, st.session_state.max_elevation)
            
            try:
                display_raster_streamlit(
                    raster_path,
                    band=1,
                    minmax=minmax,
                    title=None
                )
            except Exception as e:
                st.error(f"Error displaying raster: {e}")
                with st.expander("🔍 Full Error Details (Click to expand)"):
                    st.code(traceback.format_exc(), language="python")
        else:
            # Show basic raster
            try:
                display_raster_streamlit(
                    raster_path,
                    band=1,
                    title=None
                )
            except Exception as e:
                st.error(f"Error displaying raster: {e}")
                with st.expander("🔍 Full Error Details (Click to expand)"):
                    st.code(traceback.format_exc(), language="python")
    else:
        st.info("👈 Load a raster file in Step 1 to view the map")

with control_col:
    st.subheader("📋 Workflow Steps")
    
    # Step 1: Load Raster and View Statistics
    with st.expander("Step 1: Load Raster & View Statistics", expanded=(st.session_state.current_step == 1)):
        raster_path_input = st.text_input(
            "Raster File Path:",
            value=st.session_state.raster_path or "outputs/tutorial_merged.tif",
            help="Path to the raster file to load"
        )
        
        if st.button("📂 Load Raster", type="primary"):
            raster_path_obj = Path(raster_path_input)
            if raster_path_obj.exists():
                st.session_state.raster_path = str(raster_path_obj)
                st.session_state.raster_stats = None  # Reset stats
                st.session_state.current_step = 1
                st.success("✅ Raster loaded successfully!")
                st.rerun()
            else:
                st.error(f"❌ File not found: {raster_path_input}")
        
        # Display statistics if raster is loaded
        if st.session_state.raster_path and Path(st.session_state.raster_path).exists():
            raster_path = Path(st.session_state.raster_path)
            
            if st.session_state.raster_stats is None:
                with st.spinner("Computing statistics..."):
                    st.session_state.raster_stats = get_raster_stats_dict(raster_path)
            
            stats = st.session_state.raster_stats
            
            st.markdown("### 📊 Statistics")
            st.metric("Width", f"{stats['width']:,} px")
            st.metric("Height", f"{stats['height']:,} px")
            st.metric("Bands", stats['count'])
            
            if 1 in stats['bands']:
                band_stats = stats['bands'][1]
                st.metric("Min Elevation", f"{band_stats['min']:.2f}")
                st.metric("Max Elevation", f"{band_stats['max']:.2f}")
                st.metric("Mean Elevation", f"{band_stats['mean']:.2f}")
            
            if st.button("➡️ Continue to Step 2"):
                st.session_state.current_step = 2
                st.rerun()
    
    # Step 2: Set Elevation Range
    with st.expander("Step 2: Set Elevation Range", expanded=(st.session_state.current_step == 2)):
        if st.session_state.raster_path is None or not Path(st.session_state.raster_path).exists():
            st.warning("⚠️ Please complete Step 1 first.")
        else:
            raster_path = Path(st.session_state.raster_path)
            
            # Get statistics for default values
            if st.session_state.raster_stats is None:
                st.session_state.raster_stats = get_raster_stats_dict(raster_path)
            
            stats = st.session_state.raster_stats
            band_stats = stats['bands'].get(1, {})
            
            # Default values
            default_min = st.session_state.min_elevation if st.session_state.min_elevation is not None else band_stats.get('min', 0)
            default_max = st.session_state.max_elevation if st.session_state.max_elevation is not None else band_stats.get('max', 1000)
            
            st.markdown("Adjust the elevation range to focus on specific areas.")
            
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
                st.error("⚠️ Min must be less than max")
            else:
                st.session_state.min_elevation = min_elevation
                st.session_state.max_elevation = max_elevation
                st.success(f"✅ Range: {min_elevation:.2f} to {max_elevation:.2f}")
            
            st.markdown("### 📊 Global Statistics")
            st.text(f"Min: {band_stats.get('min', 0):.2f}")
            st.text(f"Max: {band_stats.get('max', 1000):.2f}")
            st.text(f"Mean: {band_stats.get('mean', 0):.2f}")
            
            if st.button("➡️ Continue to Step 3"):
                st.session_state.current_step = 3
                st.rerun()
    
    # Step 3: Add Hillshade
    with st.expander("Step 3: Add Hillshade Overlay", expanded=(st.session_state.current_step == 3)):
        if st.session_state.raster_path is None or not Path(st.session_state.raster_path).exists():
            st.warning("⚠️ Please complete Step 1 first.")
        else:
            st.markdown("Customize the hillshade overlay to enhance terrain visualization.")
            
            alpha = st.slider(
                "Transparency (Alpha):",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.hillshade_alpha,
                step=0.05,
                help="Hillshade overlay transparency"
            )
            
            exaggeration = st.slider(
                "Vertical Exaggeration:",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.hillshade_exaggeration,
                step=0.1,
                help="Vertical exaggeration factor"
            )
            
            altitude = st.slider(
                "Light Altitude (degrees):",
                min_value=0.0,
                max_value=90.0,
                value=st.session_state.hillshade_altitude,
                step=1.0,
                help="Altitude angle of light source"
            )
            
            azimuth = st.slider(
                "Light Azimuth (degrees):",
                min_value=0.0,
                max_value=360.0,
                value=st.session_state.hillshade_azimuth,
                step=15.0,
                help="Direction of light source"
            )
            
            # Update session state
            st.session_state.hillshade_alpha = alpha
            st.session_state.hillshade_exaggeration = exaggeration
            st.session_state.hillshade_altitude = altitude
            st.session_state.hillshade_azimuth = azimuth
            
            st.markdown("### 💡 Tips")
            st.markdown("""
            - **Alpha (0.2-0.4)**: Lower values show more base raster
            - **Exaggeration (1.0-3.0)**: Higher = more relief
            - **Altitude (30-60°)**: Higher = more uniform lighting
            - **Azimuth (315°)**: Northwest lighting is standard
            """)
            
            st.markdown("### 📋 Current Settings")
            st.text(f"Alpha: {alpha:.2f}")
            st.text(f"Exaggeration: {exaggeration:.2f}x")
            st.text(f"Altitude: {altitude:.1f}°")
            st.text(f"Azimuth: {azimuth:.1f}°")
    
    # Summary section
    st.markdown("---")
    st.markdown("### 📋 Summary")
    
    if st.session_state.raster_path:
        st.success(f"✅ Raster: {Path(st.session_state.raster_path).name}")
    
    if st.session_state.min_elevation is not None:
        st.info(f"Elevation Range: {st.session_state.min_elevation:.2f} - {st.session_state.max_elevation:.2f}")
    
    if st.session_state.current_step >= 3:
        st.info(f"Hillshade: α={st.session_state.hillshade_alpha:.2f}, exag={st.session_state.hillshade_exaggeration:.2f}x")

# Footer
st.markdown("---")
st.markdown("### GIS 584 Project - Raster Processing Dashboard")
