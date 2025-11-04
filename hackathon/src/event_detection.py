# ABOUTME: Event detection for wheelspin, understeer, and oversteer
# ABOUTME: Uses rolling window analysis on telemetry signals

import numpy as np
from scipy.interpolate import interp1d


def detect_events(df, window_size=10):
    """
    Detect over-limit events using rolling window analysis.

    Args:
        df: DataFrame with telemetry (must have accx_can, accy_can, aps, Steering_Angle)
        window_size: Rolling window size in samples (default 10 = 0.5s at 20Hz)

    Returns:
        df with new boolean columns: wheelspin_event, understeer_event, oversteer_event
    """

    # Sort by timestamp to ensure proper ordering
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute rolling trends for each signal
    aps_trend = (
        df["aps"]
        .rolling(window_size)
        .apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    )
    accx_trend = (
        df["accx_can"]
        .rolling(window_size)
        .apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    )

    # For steering and accy, use absolute values (turn direction doesn't matter)
    steer_trend = (
        df["Steering_Angle"]
        .abs()
        .rolling(window_size)
        .apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    )
    accy_trend = (
        df["accy_can"]
        .abs()
        .rolling(window_size)
        .apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    )

    # Detect wheelspin: throttle increasing but forward acceleration decreasing
    df["wheelspin_event"] = (
        (aps_trend > 5)  # Throttle increasing by >5%
        & (accx_trend < -0.1)  # Forward acceleration dropping by >0.1g
        & (df["accx_can"] > 0)  # Must be in acceleration zone (not braking)
    )

    # Detect understeer: steering increasing but lateral G not responding
    accy_responsiveness = accy_trend / (steer_trend + 0.001)  # Avoid division by zero
    df["understeer_event"] = (
        (steer_trend > 10)  # Steering increasing by >10°
        & (accy_responsiveness < 0.01)  # Lateral G not responding (<0.01g per degree)
        & (df["accy_can"].abs() > 0.3)  # Must be in cornering zone
    )

    # Detect oversteer: sudden lateral G spike with forward acceleration drop
    accy_spike = df["accy_can"].abs().diff() > 0.2
    accx_drop = df["accx_can"].diff() < -0.15
    df["oversteer_event"] = (
        accy_spike
        & accx_drop
        & (df["accy_can"].abs() > 0.5)  # Must be in cornering zone
    )

    # Fill NaN values (from rolling windows) with False
    df["wheelspin_event"] = df["wheelspin_event"].fillna(False)
    df["understeer_event"] = df["understeer_event"].fillna(False)
    df["oversteer_event"] = df["oversteer_event"].fillna(False)

    return df


def _interpolate_envelope_radius(envelope_points, theta):
    """
    Interpolate the envelope boundary radius at a given angle.

    Args:
        envelope_points: List of dicts with 'accx', 'accy', 'total_g' keys
        theta: Angle in radians

    Returns:
        Boundary radius at the given angle
    """
    # Convert envelope points to polar coordinates
    angles = []
    radii = []
    for pt in envelope_points:
        angle = np.arctan2(pt["accy"], pt["accx"])
        radius = pt["total_g"]
        angles.append(angle)
        radii.append(radius)

    # Sort by angle for proper interpolation
    sorted_indices = np.argsort(angles)
    angles = np.array(angles)[sorted_indices]
    radii = np.array(radii)[sorted_indices]

    # Handle wrapping: duplicate first/last points for circular interpolation
    # Envelope spans [0, π/2] but we need to handle all quadrants
    angles_wrapped = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi])
    radii_wrapped = np.concatenate([radii, radii, radii])

    # Interpolate
    interpolator = interp1d(
        angles_wrapped,
        radii_wrapped,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    return float(interpolator(theta))


def classify_zone(zone_samples, envelope_data, zone_id):
    """
    Classify a single zone as Conservative, Aggressive, or Optimal.

    Args:
        zone_samples: DataFrame with samples for this (driver, lap, zone)
        envelope_data: Dict with envelope for this driver (from friction_envelopes.json)
        zone_id: Zone ID for envelope lookup

    Returns:
        Dict with classification results
    """
    # Make a copy to avoid SettingWithCopyWarning
    zone_samples = zone_samples.copy()

    # Calculate total G for this zone
    zone_samples["total_g"] = np.sqrt(
        zone_samples["accx_can"] ** 2 + zone_samples["accy_can"] ** 2
    )
    avg_total_g = zone_samples["total_g"].mean()

    # Calculate angle-aware utilization if envelope is available
    utilization = 0
    if envelope_data and "envelope_points" in envelope_data:
        envelope_points = envelope_data["envelope_points"]
        if envelope_points and len(envelope_points) > 2:
            # Compute per-sample utilization with angle awareness
            utilizations = []
            for _, row in zone_samples.iterrows():
                # Compute angle for this sample
                theta = np.arctan2(row["accy_can"], row["accx_can"])

                # Get envelope boundary at this angle
                boundary_radius = _interpolate_envelope_radius(envelope_points, theta)

                # Compute utilization for this sample
                if boundary_radius > 0:
                    sample_util = row["total_g"] / boundary_radius
                    utilizations.append(sample_util)

            # Aggregate using median (robust to outliers)
            if utilizations:
                utilization = np.median(utilizations)

    # Check if any events occurred
    wheelspin_occurred = zone_samples["wheelspin_event"].any()
    understeer_occurred = zone_samples["understeer_event"].any()
    oversteer_occurred = zone_samples["oversteer_event"].any()

    # Classify
    if wheelspin_occurred or understeer_occurred or oversteer_occurred:
        classification = "Aggressive"
    elif utilization < 0.95:
        classification = "Conservative"
    else:
        classification = "Optimal"

    return {
        "classification": classification,
        "avg_total_g": avg_total_g,
        "avg_utilization": utilization,
        "wheelspin": wheelspin_occurred,
        "understeer": understeer_occurred,
        "oversteer": oversteer_occurred,
        "sample_count": len(zone_samples),
    }
