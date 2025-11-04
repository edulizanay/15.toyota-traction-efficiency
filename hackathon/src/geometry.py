# ABOUTME: GPS and track geometry utilities for traction efficiency analysis
# ABOUTME: Functions for GPS conversion, centerline generation, and spatial projection

import numpy as np
import pandas as pd
import math
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyproj import Transformer
from pathlib import Path


def convert_gps_to_meters(df):
    """
    Convert GPS lon/lat to local Cartesian coordinates in meters using UTM projection.

    Args:
        df: DataFrame with VBOX_Long_Minutes and VBOX_Lat_Min columns

    Returns:
        DataFrame with added x_meters and y_meters columns
    """
    # Use UTM zone 16N for Alabama (Barber Motorsports Park)
    # EPSG:32616 is UTM zone 16N
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)

    # Convert lon, lat to x, y in meters
    x_meters, y_meters = transformer.transform(
        df["VBOX_Long_Minutes"].values, df["VBOX_Lat_Min"].values
    )

    df["x_meters"] = x_meters
    df["y_meters"] = y_meters

    return df


def resample_by_distance(x, y, step_m=2.0, spike_threshold_m=10.0):
    """
    Resample GPS track to uniform distance spacing, removing spikes.

    Args:
        x: x coordinates in meters
        y: y coordinates in meters
        step_m: Target spacing between points in meters
        spike_threshold_m: Maximum allowed segment length (drop spikes exceeding this)

    Returns:
        tuple: (x_resampled, y_resampled, distances)
    """
    # Compute segment distances
    dx = np.diff(x)
    dy = np.diff(y)
    segment_dist = np.sqrt(dx**2 + dy**2)

    # Remove spikes (segments longer than threshold)
    spike_mask = segment_dist > spike_threshold_m
    if np.any(spike_mask):
        spike_indices = np.where(spike_mask)[0]
        print(f"  Removing {len(spike_indices)} spikes (>{spike_threshold_m}m)")

        # Keep only non-spike segments
        keep_mask = np.ones(len(x), dtype=bool)
        keep_mask[spike_indices + 1] = False  # Remove the point after the spike
        x = x[keep_mask]
        y = y[keep_mask]

        # Recalculate distances
        dx = np.diff(x)
        dy = np.diff(y)
        segment_dist = np.sqrt(dx**2 + dy**2)

    # Remove duplicates (zero distance)
    duplicate_mask = segment_dist < 1e-6
    if np.any(duplicate_mask):
        dup_indices = np.where(duplicate_mask)[0]
        print(f"  Removing {len(dup_indices)} duplicate points")

        keep_mask = np.ones(len(x), dtype=bool)
        keep_mask[dup_indices + 1] = False
        x = x[keep_mask]
        y = y[keep_mask]

        # Recalculate distances
        dx = np.diff(x)
        dy = np.diff(y)
        segment_dist = np.sqrt(dx**2 + dy**2)

    # Compute cumulative distance
    cumulative_dist = np.concatenate([[0], np.cumsum(segment_dist)])
    total_distance = cumulative_dist[-1]

    print(f"  Total track length: {total_distance:.1f}m")
    print(f"  Original points: {len(x)}")

    # Create uniform distance stations
    num_stations = int(np.ceil(total_distance / step_m)) + 1
    uniform_dist = np.linspace(0, total_distance, num_stations)

    # Interpolate x and y at uniform distances
    interp_x = interp1d(
        cumulative_dist, x, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    interp_y = interp1d(
        cumulative_dist, y, kind="linear", bounds_error=False, fill_value="extrapolate"
    )

    x_resampled = interp_x(uniform_dist)
    y_resampled = interp_y(uniform_dist)

    print(f"  Resampled to {len(x_resampled)} points (step={step_m}m)")

    return x_resampled, y_resampled, uniform_dist


def smooth_periodic(x, y, window_length=31, polyorder=3, wrap_count=25):
    """
    Apply Savitzky-Golay smoothing with periodic wrapping to avoid endpoint kink.

    Args:
        x: x coordinates
        y: y coordinates
        window_length: Smoothing window size (must be odd)
        polyorder: Polynomial order for Savitzky-Golay
        wrap_count: Number of points to wrap from each end

    Returns:
        tuple: (x_smooth, y_smooth)
    """
    if window_length % 2 == 0:
        window_length += 1  # Must be odd

    if window_length >= len(x):
        print(
            f"  Warning: Window ({window_length}) >= points ({len(x)}), reducing to {len(x) // 3 | 1}"
        )
        window_length = (len(x) // 3) | 1  # Make odd
        if window_length < 5:
            print("  Warning: Too few points for smoothing, skipping")
            return x, y

    # Wrap points for periodic smoothing
    x_wrapped = np.concatenate([x[-wrap_count:], x, x[:wrap_count]])
    y_wrapped = np.concatenate([y[-wrap_count:], y, y[:wrap_count]])

    # Apply Savitzky-Golay filter
    x_smooth_wrapped = savgol_filter(
        x_wrapped, window_length, polyorder, mode="nearest"
    )
    y_smooth_wrapped = savgol_filter(
        y_wrapped, window_length, polyorder, mode="nearest"
    )

    # Extract the middle (unwrap)
    x_smooth = x_smooth_wrapped[wrap_count:-wrap_count]
    y_smooth = y_smooth_wrapped[wrap_count:-wrap_count]

    print(f"  Applied Savitzky-Golay (window={window_length}, poly={polyorder})")

    return x_smooth, y_smooth


def compute_centerline(
    telemetry_df,
    vehicle_number=None,
    lap_number=None,
    resample_step_m=2.0,
    spike_threshold_m=10.0,
    savgol_window=31,
    savgol_poly=3,
    wrap_count=25,
):
    """
    Compute track centerline from GPS coordinates with distance-based smoothing.

    Args:
        telemetry_df: DataFrame with GPS coordinates (x_meters, y_meters)
        vehicle_number: Specific vehicle to use (default: first vehicle in data)
        lap_number: Specific lap to use (default: lap with most GPS data)
        resample_step_m: Target spacing for resampling in meters (default: 2.0)
        spike_threshold_m: Maximum segment length before considering it a spike (default: 10.0)
        savgol_window: Savitzky-Golay window size (default: 31)
        savgol_poly: Savitzky-Golay polynomial order (default: 3)
        wrap_count: Points to wrap for periodic smoothing (default: 25)

    Returns:
        tuple: (smoothed_x, smoothed_y) - smoothed centerline coordinates (not closed)
    """
    # If no vehicle specified, use first vehicle in data
    if vehicle_number is None:
        vehicle_number = telemetry_df["vehicle_number"].iloc[0]
        print(f"Using vehicle #{vehicle_number}")

    # Filter to specific vehicle
    vehicle_data = telemetry_df[telemetry_df["vehicle_number"] == vehicle_number].copy()

    # If no lap specified, find lap with most complete GPS data
    if lap_number is None:
        lap_counts = vehicle_data.groupby("lap").size()
        lap_number = lap_counts.idxmax()
        print(f"Using lap #{lap_number} (most complete GPS data)")

    # Get lap data
    lap_data = vehicle_data[vehicle_data["lap"] == lap_number].copy()

    # Sort by timestamp to ensure correct order
    lap_data = lap_data.sort_values("timestamp")

    print(f"Track outline points: {len(lap_data)}")

    # Extract coordinates
    x = lap_data["x_meters"].values
    y = lap_data["y_meters"].values

    # Phase 1: Resample by distance
    print("\nPhase 1: Distance-based resampling")
    x_resampled, y_resampled, _ = resample_by_distance(
        x, y, step_m=resample_step_m, spike_threshold_m=spike_threshold_m
    )

    # Phase 2: Smooth with periodic wrapping
    print("\nPhase 2: Periodic smoothing")
    x_smooth, y_smooth = smooth_periodic(
        x_resampled,
        y_resampled,
        window_length=savgol_window,
        polyorder=savgol_poly,
        wrap_count=wrap_count,
    )

    print(f"\n✓ Final smoothed track: {len(x_smooth)} points")

    return x_smooth, y_smooth


def project_points_onto_centerline(points_x, points_y, centerline_x, centerline_y):
    """
    Project points onto track centerline and return track distances.

    Args:
        points_x: Array of x coordinates to project
        points_y: Array of y coordinates to project
        centerline_x: Array of centerline x coordinates
        centerline_y: Array of centerline y coordinates

    Returns:
        Array of track distances (meters from start)
    """
    # Calculate cumulative distance along centerline
    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    cumulative_distance = np.concatenate([[0], np.cumsum(segment_lengths)])

    # For each point, find nearest centerline point
    track_distances = []

    for px, py in zip(points_x, points_y):
        distances = np.sqrt((centerline_x - px) ** 2 + (centerline_y - py) ** 2)
        nearest_idx = np.argmin(distances)
        track_distances.append(cumulative_distance[nearest_idx])

    return np.array(track_distances)


def rotate_coordinates(x, y, angle_degrees):
    """
    Rotate 2D coordinates counterclockwise by angle_degrees.

    NOTE: Use this ONLY for visualization in D3.js, NOT for data storage.

    Args:
        x: x-coordinates (array-like)
        y: y-coordinates (array-like)
        angle_degrees: rotation angle in degrees (positive = counterclockwise)

    Returns:
        x_rot, y_rot: rotated coordinates
    """
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    x = np.asarray(x)
    y = np.asarray(y)

    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a

    return x_rot, y_rot


def compute_normals(x, y):
    """
    Compute unit normal vectors along a polyline.

    Args:
        x: x coordinates (array-like)
        y: y coordinates (array-like)

    Returns:
        array of shape (N, 2) containing normal vectors [nx, ny]
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate tangent vectors using gradient
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Normalize tangent vectors
    mag = np.hypot(dx, dy)
    mag[mag == 0] = 1.0  # Avoid division by zero
    tx = dx / mag
    ty = dy / mag

    # Rotate tangent 90° CCW to get normal (perpendicular)
    nx = -ty
    ny = tx

    return np.column_stack([nx, ny])


def compute_track_boundaries(centerline_x, centerline_y, track_width_m=12.0):
    """
    Compute inner and outer track boundaries from centerline.

    Args:
        centerline_x: Centerline x coordinates
        centerline_y: Centerline y coordinates
        track_width_m: Total track width in meters (default: 12m)

    Returns:
        tuple: (inner_x, inner_y, outer_x, outer_y)
    """
    centerline_x = np.asarray(centerline_x)
    centerline_y = np.asarray(centerline_y)

    # Compute normal vectors
    normals = compute_normals(centerline_x, centerline_y)

    # Half-width offset
    half_width = track_width_m / 2.0

    # Offset centerline by ±half_width along normals
    inner_x = centerline_x - normals[:, 0] * half_width
    inner_y = centerline_y - normals[:, 1] * half_width

    outer_x = centerline_x + normals[:, 0] * half_width
    outer_y = centerline_y + normals[:, 1] * half_width

    return inner_x, inner_y, outer_x, outer_y


def save_centerline(x_smooth, y_smooth, output_path):
    """
    Save smoothed track centerline to CSV.

    Args:
        x_smooth: Smoothed x coordinates
        y_smooth: Smoothed y coordinates
        output_path: Path to save CSV
    """
    df_track = pd.DataFrame({"x_meters": x_smooth, "y_meters": y_smooth})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_track.to_csv(output_path, index=False)

    print(f"✓ Saved track centerline to: {output_path}")


def load_centerline(centerline_path):
    """
    Load track centerline from CSV.

    Args:
        centerline_path: Path to centerline CSV

    Returns:
        tuple: (x, y) coordinates from centerline
    """
    centerline_path = Path(centerline_path)
    if not centerline_path.exists():
        raise FileNotFoundError(f"Centerline not found: {centerline_path}")

    df_center = pd.read_csv(centerline_path)
    x_c = df_center["x_meters"].to_numpy()
    y_c = df_center["y_meters"].to_numpy()

    print(f"  Loaded centerline: {centerline_path}")
    return x_c, y_c


def save_track_boundaries(inner_x, inner_y, outer_x, outer_y, output_path):
    """
    Save track boundaries to JSON.

    Args:
        inner_x: Inner boundary x coordinates
        inner_y: Inner boundary y coordinates
        outer_x: Outer boundary x coordinates
        outer_y: Outer boundary y coordinates
        output_path: Path to save JSON
    """
    import json

    boundaries = {
        "inner": [{"x": float(x), "y": float(y)} for x, y in zip(inner_x, inner_y)],
        "outer": [{"x": float(x), "y": float(y)} for x, y in zip(outer_x, outer_y)],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(boundaries, f)

    print(f"✓ Saved track boundaries to: {output_path}")
