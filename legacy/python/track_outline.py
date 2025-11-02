# ABOUTME: Track centerline computation and base track figure generation
# ABOUTME: Handles GPS smoothing, centerline persistence, and Plotly track rendering

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from .geometry import resample_by_distance, smooth_periodic, compute_normals


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


def make_base_track_figure(
    telemetry_df,
    centerline_path=None,
    vehicle_number=None,
    lap_number=None,
    resample_step_m=2.0,
    spike_threshold_m=10.0,
    savgol_window=31,
    savgol_poly=3,
    wrap_count=25,
    track_width_m=18.0,
):
    """
    Build base track figure with centerline and track surface.

    If centerline_path exists, loads from file for consistency.
    Otherwise computes from telemetry data.

    Args:
        telemetry_df: DataFrame with GPS coordinates (x_meters, y_meters)
        centerline_path: Path to saved centerline CSV (optional, loads if exists)
        vehicle_number: Specific vehicle to use (default: first in data)
        lap_number: Specific lap to use (default: lap with most GPS data)
        resample_step_m: Target spacing for resampling in meters (default: 2.0)
        spike_threshold_m: Maximum segment length before considering it a spike (default: 10.0)
        savgol_window: Savitzky-Golay window size (default: 31)
        savgol_poly: Savitzky-Golay polynomial order (default: 3)
        wrap_count: Points to wrap for periodic smoothing (default: 25)
        track_width_m: Total width of track surface in meters (default: 18.0)

    Returns:
        tuple: (smoothed_x, smoothed_y, fig) - centerline coordinates and Plotly figure
    """
    # Load or compute centerline
    if centerline_path and Path(centerline_path).exists():
        x_smooth, y_smooth = load_centerline(centerline_path)
    else:
        x_smooth, y_smooth = compute_centerline(
            telemetry_df,
            vehicle_number=vehicle_number,
            lap_number=lap_number,
            resample_step_m=resample_step_m,
            spike_threshold_m=spike_threshold_m,
            savgol_window=savgol_window,
            savgol_poly=savgol_poly,
            wrap_count=wrap_count,
        )

    # Close the loop for visualization (add first point at end)
    x_closed = np.append(x_smooth, x_smooth[0])
    y_closed = np.append(y_smooth, y_smooth[0])

    # Create Plotly figure with dark theme
    fig = go.Figure()

    # Build track surface as fixed-width donut from centerline normals
    half_width = track_width_m / 2.0
    x_c = x_smooth
    y_c = y_smooth
    n_hat = compute_normals(x_c, y_c)

    # Left/outer and right/inner edges
    x_left = x_c + half_width * n_hat[:, 0]
    y_left = y_c + half_width * n_hat[:, 1]
    x_right = x_c - half_width * n_hat[:, 0]
    y_right = y_c - half_width * n_hat[:, 1]

    # Build ring path: left forward, right reversed, and close
    x_ring = np.concatenate([x_left, x_right[::-1], [x_left[0]]])
    y_ring = np.concatenate([y_left, y_right[::-1], [y_left[0]]])

    fig.add_trace(
        go.Scatter(
            x=x_ring,
            y=y_ring,
            mode="lines",
            fill="toself",
            line=dict(color="rgba(100,100,100,0.35)", width=1),
            fillcolor="rgba(255,255,255,0.07)",
            name="Track surface",
            hoverinfo="skip",
        )
    )

    # Thin cyan centerline (crisp reference line)
    fig.add_trace(
        go.Scatter(
            x=x_closed,
            y=y_closed,
            mode="lines",
            line=dict(color="#5cf", width=2),
            name="Centerline",
            hovertemplate="x: %{x:.1f}m<br>y: %{y:.1f}m<extra></extra>",
        )
    )

    # Update layout with dark theme
    fig.update_layout(
        title="Barber Motorsports Park - Track Outline",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="#ffffff", size=12),
        xaxis=dict(
            gridcolor="#333333",
            showgrid=True,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(gridcolor="#333333", showgrid=True, zeroline=False, visible=False),
        hovermode="closest",
        showlegend=True,
    )

    return x_smooth, y_smooth, fig
