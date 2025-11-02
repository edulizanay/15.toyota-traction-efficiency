# ABOUTME: Pure geometric and numeric functions for GPS processing and coordinate transformations
# ABOUTME: No I/O or visualization dependencies - only numpy/scipy computations

import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


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


def compute_normals(x, y):
    """
    Compute unit normals along a polyline defined by (x, y).

    Args:
        x: x coordinates (array-like)
        y: y coordinates (array-like)

    Returns:
        array of shape (N, 2) for normals
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    mag = np.hypot(dx, dy)
    mag[mag == 0] = 1.0
    tx = dx / mag
    ty = dy / mag
    # Rotate tangent 90° CCW → normal
    nx = -ty
    ny = tx
    return np.column_stack([nx, ny])


def rotate_coordinates(x, y, angle_degrees):
    """
    Rotate 2D coordinates counterclockwise by angle_degrees.

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
