# ABOUTME: Event detection for wheelspin, understeer, and oversteer
# ABOUTME: Uses rolling window analysis on telemetry signals

import numpy as np
from scipy.interpolate import interp1d

# Event detection thresholds
WHEELSPIN_APS_TREND_THRESHOLD = 7  # % throttle increase over window
WHEELSPIN_ACCX_TREND_THRESHOLD = -0.12  # g forward accel drop over window
WHEELSPIN_ACCX_MIN = 0  # g minimum forward accel

UNDERSTEER_STEER_TREND_THRESHOLD = 15  # degrees steering increase over window
UNDERSTEER_STEER_ABS_MIN = 10  # degrees minimum absolute steering angle
UNDERSTEER_RESPONSIVENESS_THRESHOLD = 0.005  # g/deg lateral response per steering
UNDERSTEER_ACCY_ABS_MIN = 0.3  # g minimum lateral accel

OVERSTEER_ACCY_SPIKE_THRESHOLD = 0.25  # g lateral accel spike
OVERSTEER_ACCX_DROP_THRESHOLD = -0.2  # g forward accel drop
OVERSTEER_ACCY_ABS_MIN = 0.5  # g minimum lateral accel

# Zone-level macro-event thresholds
MACRO_EVENT_COVERAGE_THRESHOLD = (
    0.10  # fraction of samples with events (balanced tweak from 0.11)
)
MACRO_EVENT_RUN_LENGTH_THRESHOLD = (
    6  # samples in longest contiguous event run (balanced tweak)
)
OPTIMAL_UTILIZATION_THRESHOLD = (
    0.81  # utilization threshold for optimal classification (lowered from 0.90)
)
AGGRESSIVE_HIGH_UTIL_THRESHOLD = (
    0.84  # require high utilization for aggressive classification
)


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
        (aps_trend > WHEELSPIN_APS_TREND_THRESHOLD)
        & (accx_trend < WHEELSPIN_ACCX_TREND_THRESHOLD)
        & (df["accx_can"] > WHEELSPIN_ACCX_MIN)
    )

    # Detect understeer: steering increasing but lateral G not responding
    accy_responsiveness = accy_trend / (steer_trend + 0.001)  # Avoid division by zero
    df["understeer_event"] = (
        (steer_trend > UNDERSTEER_STEER_TREND_THRESHOLD)
        & (df["Steering_Angle"].abs() >= UNDERSTEER_STEER_ABS_MIN)
        & (accy_responsiveness < UNDERSTEER_RESPONSIVENESS_THRESHOLD)
        & (df["accy_can"].abs() > UNDERSTEER_ACCY_ABS_MIN)
    )

    # Detect oversteer: sudden lateral G spike with forward acceleration drop
    accy_spike = df["accy_can"].abs().diff() > OVERSTEER_ACCY_SPIKE_THRESHOLD
    accx_drop = df["accx_can"].diff() < OVERSTEER_ACCX_DROP_THRESHOLD
    df["oversteer_event"] = (
        accy_spike & accx_drop & (df["accy_can"].abs() > OVERSTEER_ACCY_ABS_MIN)
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

    # Compute event union and coverage
    any_event = (
        zone_samples["wheelspin_event"]
        | zone_samples["understeer_event"]
        | zone_samples["oversteer_event"]
    )
    event_coverage = any_event.sum() / len(zone_samples) if len(zone_samples) > 0 else 0

    # Calculate maximum contiguous event run length
    max_event_run_len = 0
    current_run = 0
    for has_event in any_event:
        if has_event:
            current_run += 1
            max_event_run_len = max(max_event_run_len, current_run)
        else:
            current_run = 0

    # Determine if this is a macro-event (sustained aggressive behavior)
    # Changed to AND logic: both conditions must be met for sustained issue
    # Also requires high utilization to be "aggressive" (near-limit mistakes)
    is_macro_event = (
        event_coverage >= MACRO_EVENT_COVERAGE_THRESHOLD
        and max_event_run_len >= MACRO_EVENT_RUN_LENGTH_THRESHOLD
        and utilization >= AGGRESSIVE_HIGH_UTIL_THRESHOLD
    )

    # Classify zone
    if is_macro_event:
        classification = "Aggressive"
    elif utilization >= OPTIMAL_UTILIZATION_THRESHOLD:
        classification = "Optimal"
    else:
        classification = "Conservative"

    return {
        "classification": classification,
        "avg_total_g": avg_total_g,
        "avg_utilization": utilization,
        "wheelspin": wheelspin_occurred,
        "understeer": understeer_occurred,
        "oversteer": oversteer_occurred,
        "sample_count": len(zone_samples),
        "event_coverage": event_coverage,
        "max_event_run_len": max_event_run_len,
    }
