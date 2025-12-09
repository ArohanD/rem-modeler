# REM Generator: Interactive Relative Elevation Model Creation

A Python-based tool for generating **Relative Elevation Models (REMs)** from Digital Elevation Models (DEMs) using river centerline interpolation. REMs highlight fluvial landforms and floodplain features by normalizing terrain elevation relative to the water surface.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![GDAL](https://img.shields.io/badge/GDAL-3.11+-green.svg)

## Overview

Relative Elevation Models (REMs) are a powerful visualization technique for analyzing river systems and floodplain geomorphology. Unlike traditional DEMs that show absolute elevation, REMs display **height above the local water surface**, making subtle floodplain features like levees, oxbow lakes, and terrace scarps dramatically visible.

This tool provides an interactive workflow for:
1. **Merging DEM tiles** (e.g., USGS 1-meter DEMs)
2. **Identifying river elevation ranges** via interactive visualization
3. **Extracting river centerlines** from OpenStreetMap or manual delineation
4. **Snapping centerlines** to actual channel bottoms using elevation data
5. **Interpolating water surface** elevation using Radial Basis Functions (RBF)
6. **Generating the REM** by subtracting water surface from the DEM

## Features

- **Interactive Raster Viewer**: Hover over pixels to see elevation values in real-time
- **Hillshade Overlay**: Configurable transparency, vertical exaggeration, and sun angle
- **Dual Centerline Methods**:
  - *Automatic*: Query OpenStreetMap Overpass API for waterway geometries
  - *Manual*: Brush-paint over the river channel and extract centerline via skeletonization
- **Adaptive Centerline Snapping**: Correct OSM georeferencing errors by snapping to local elevation minima
- **RBF Interpolation**: Thin-plate spline interpolation for smooth water surface modeling
- **Progress Tracking**: Real-time progress bars for long-running operations

## Installation

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** - A fast Python package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd code

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

That's it! `uv sync` automatically handles GDAL and all other dependencies.

## Usage

### Quick Start

```bash
# Run the main interactive workflow
python -m src.main <input_dem.tif> <output_rem.tif>
```

### Step-by-Step Workflow

#### 1. Merge DEM Tiles (Optional)

If you have multiple DEM tiles (e.g., USGS 1m tiles), merge them first:

```python
from src.utils import merge_tifs

merged = merge_tifs("data/tutorial/", "outputs/merged_dem.tif")
```

#### 2. Run the Interactive Workflow

```bash
python -m src.main outputs/merged_dem.tif my_rem.tif
```

The workflow will guide you through:

1. **Select Elevation Range**: An interactive viewer opens where you can hover over pixels to identify the river's elevation range. Enter min/max values and click "Update" to preview, then "Done" to proceed.

2. **Configure Hillshade**: Adjust transparency (alpha), vertical exaggeration, and sun altitude angle to enhance terrain visualization. Click "Done" when satisfied.

3. **Derive Centerline**: The tool queries OpenStreetMap for waterways in your study area and displays them overlaid on the DEM.

4. **Adjust Snapping Parameters**: Use the interactive interface to:
   - **Point Spacing**: Controls how densely points are sampled (10-25m for tight curves)
   - **Snap Radius**: Search distance for finding channel bottom (50-150m typical)
   - Click "Recompute" to snap the centerline to the lowest elevations
   - Click "Hide Line" to inspect the terrain underneath

5. **Generate REM**: The tool interpolates water surface elevation and subtracts it from the DEM to create the final REM.

## Data Sources

This tool is designed to work with high-resolution DEMs. Recommended sources:

- **USGS 3DEP 1-meter DEMs**: [The National Map](https://apps.nationalmap.gov/downloader/)
- **USGS 3DEP 1/3 arc-second (~10m)**: For larger study areas
- **State LiDAR Programs**: Many states provide free LiDAR-derived DEMs

Place your DEM tiles in the `data/` directory, organized by study area:

```
data/
├── tutorial/           # Tutorial data (Carson River, NV)
│   ├── USGS_one_meter_x27y435_NV_Reno_Carson_QL1_2017.tif
│   └── ...
├── snake_river/        # Your study area
│   └── snake_river.tif
└── outputs/            # Generated outputs
```

## Project Structure

```
code/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main entry point
│   ├── interfaces/
│   │   ├── interfaces.py       # Interactive viewers and widgets
│   │   └── geoprocessing.py    # Core geoprocessing functions
│   └── utils/
│       └── utils.py            # Utility functions (merge, interpolate)
├── data/                       # Input DEM data
├── outputs/                    # Generated REMs
├── pyproject.toml              # Project configuration
└── README.md
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `gdal` | Raster I/O, merging, reprojection |
| `rasterio` | High-level raster operations |
| `numpy` | Array operations |
| `scipy` | RBF interpolation, morphological operations |
| `scikit-image` | Skeletonization for centerline extraction |
| `shapely` | Geometry operations |
| `pyproj` | Coordinate transformations |
| `matplotlib` | Interactive visualization |
| `matplotlib-scalebar` | Map scale bars |
| `tqdm` | Progress bars |

## Algorithm Details

### Centerline Snapping

OpenStreetMap waterway geometries often have positional errors of 10-100+ meters. The snapping algorithm:

1. **Densifies** the centerline to ensure adequate point density for curve following
2. **Searches** within a configurable radius around each point
3. **Snaps** to the local elevation minimum (channel bottom)
4. **Smooths** the result with a 5-point moving average to remove zigzag artifacts

### Water Surface Interpolation

Uses **Radial Basis Function (RBF)** interpolation with a thin-plate spline kernel:

1. Samples elevation at points along the snapped centerline
2. Interpolates to a coarse grid (1/10 resolution) for performance
3. Upsamples back to full resolution using bilinear interpolation

This produces a smooth, physically plausible water surface that slopes continuously downstream.

## Tips for Best Results

1. **Choose an appropriate study area**: REMs work best for relatively straight river reaches (5-20 km). For longer reaches or rivers with significant elevation change, consider processing in segments.

2. **Set accurate elevation bounds**: The min/max elevation range should tightly bracket the river surface. Too wide a range includes floodplain; too narrow misses the channel.

3. **Adjust snap radius for your river**: 
   - Small streams: 30-50m
   - Medium rivers: 50-150m  
   - Large rivers: 100-300m

4. **Use tight point spacing for meandering rivers**: Set point spacing to 10-25m for rivers with tight bends.

## Troubleshooting

### "No waterway elements found"
- Your study area may not have mapped waterways in OpenStreetMap
- Use the manual brush-and-extract method instead: `manual_centerline_widget`

### GDAL installation errors
- Try using conda: `conda install -c conda-forge gdal`
- On Windows, pre-built wheels are available from [geospatial-wheels](https://nathanjmcdougall.github.io/geospatial-wheels-index/)

### Memory errors with large rasters
- Process in tiles or reduce the DEM resolution
- The interpolation uses a 1/10 coarse grid by default to manage memory

## Acknowledgments

- USGS 3DEP program for providing high-resolution elevation data
- OpenStreetMap contributors for waterway geometries
- The open-source geospatial Python ecosystem (GDAL, Rasterio, Shapely, etc.)
