# Legacy Files for Reuse

Files extracted from the old brake analysis project (`14.toyota-hackathon`) that we'll port to the new traction efficiency analysis.

## Python Code to Port

**`python/geometry.py`** - GPS and coordinate utilities:
- `resample_by_distance()` - Resample GPS to uniform spacing
- `smooth_periodic()` - Savitzky-Golay smoothing with periodic wrapping
- `compute_normals()` - Compute unit normals along polyline
- `rotate_coordinates()` - 2D rotation (for visualization only)

**`python/track_outline.py`** - Track centerline generation:
- `compute_centerline()` - Auto-generate centerline from GPS telemetry
- `save_centerline()` - Save centerline to CSV
- `load_centerline()` - Load centerline from CSV

**`python/data_processing.py`** - Data processing utilities:
- `convert_gps_to_meters()` - GPS (lat/lon) → UTM (x/y meters)
- `project_points_onto_centerline()` - Map GPS points → track distance

## Manual Assets

**`assets/corner_labels.json`** - Manually placed corner labels (C1-C17) for Barber Motorsports Park with UTM coordinates

**`assets/pit_lane.json`** - Pre-extracted pit lane GPS path (from vehicle 13, lap 2) - use as reference or copy

**`assets/Barber-Motorsports-Park.png`** - Track reference image for orientation

## Usage

These files are for reference when building `hackathon/src/geometry.py` and `hackathon/src/analysis.py`. Port the functions, don't just copy - we're simplifying and adapting for traction analysis.
