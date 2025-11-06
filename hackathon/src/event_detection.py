# ABOUTME: Event detection for wheelspin, understeer, and oversteer
# ABOUTME: Uses rolling window analysis on telemetry signals

import numpy as np
from scipy.interpolate import interp1d

# Event detection thresholds
WHEELSPIN_APS_TREND_THRESHOLD = 7  # % throttle increase over window
WHEELSPIN_ACCX_TREND_THRESHOLD = -0.12  # g forward accel drop over window
WHEELSPIN_ACCX_MIN = 0  # g minimum forward accel

UNDERSTEER_STEER_TREND_THRESHOLD = 5  # degrees minimum steering increase to evaluate
UNDERSTEER_STEER_ABS_MIN = 10  # degrees minimum absolute steering angle
UNDERSTEER_RESPONSIVENESS_THRESHOLD = 0.0025  # g/deg lateral response per steering
UNDERSTEER_ACCY_ABS_MIN = 0.3  # g minimum lateral accel
UNDERSTEER_LAG_SAMPLES = 6  # samples to shift accy forward (120ms at 20Hz = 6 samples)
UNDERSTEER_SUSTAINED_SAMPLES = (
    6  # samples required for sustained understeer (0.30s at 20Hz)
)

OVERSTEER_ACCY_SPIKE_THRESHOLD = 0.25  # g lateral accel spike
OVERSTEER_ACCX_DROP_THRESHOLD = -0.2  # g forward accel drop
OVERSTEER_ACCY_ABS_MIN = 0.5  # g minimum lateral accel

# Zone-level macro-event thresholds
MACRO_EVENT_COVERAGE_THRESHOLD = (
    0.10  # fraction of samples with events (balanced tweak from 0.11)
)
# Event-specific run-length thresholds (samples at 20Hz)
MACRO_EVENT_RUN_LENGTH_THRESHOLD_UNDERSTEER = 6  # sustained event (0.30s)
MACRO_EVENT_RUN_LENGTH_THRESHOLD_WHEELSPIN = 3  # spike event (0.15s)
MACRO_EVENT_RUN_LENGTH_THRESHOLD_OVERSTEER = 3  # spike event (0.15s)

OPTIMAL_UTILIZATION_THRESHOLD = (
    0.81  # utilization threshold for optimal classification (lowered from 0.90)
)
# Event-specific utilization thresholds for aggressive classification
AGGRESSIVE_UNDERSTEER_UTIL_THRESHOLD = 0.84  # understeer is a limit mistake
AGGRESSIVE_WHEELSPIN_UTIL_THRESHOLD = 0.70  # wheelspin happens below peak utilization
AGGRESSIVE_OVERSTEER_UTIL_THRESHOLD = 0.73  # oversteer happens below peak utilization


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

    # For steering, use absolute values (turn direction doesn't matter)
    steer_abs = df["Steering_Angle"].abs()

    # Time-align lateral G forward by 120ms (6 samples) to compensate for tire response lag
    accy_abs_aligned = df["accy_can"].abs().shift(-UNDERSTEER_LAG_SAMPLES)

    # Compute trends over rolling window
    steer_trend = steer_abs.rolling(window_size).apply(
        lambda x: x.iloc[-1] - x.iloc[0], raw=False
    )
    accy_trend = accy_abs_aligned.rolling(window_size).apply(
        lambda x: x.iloc[-1] - x.iloc[0], raw=False
    )

    # Detect wheelspin: throttle increasing but forward acceleration decreasing
    df["wheelspin_event"] = (
        (aps_trend > WHEELSPIN_APS_TREND_THRESHOLD)
        & (accx_trend < WHEELSPIN_ACCX_TREND_THRESHOLD)
        & (df["accx_can"] > WHEELSPIN_ACCX_MIN)
    )

    # Detect understeer: steering increasing but lateral G not responding
    # Only evaluate when steering increases by at least threshold
    # Responsiveness in g/deg
    accy_responsiveness = accy_trend / (steer_trend + 0.001)  # Avoid division by zero
    understeer_candidate = (
        (steer_trend >= UNDERSTEER_STEER_TREND_THRESHOLD)  # steering increasing ≥5°
        & (steer_abs >= UNDERSTEER_STEER_ABS_MIN)  # minimum absolute steering
        & (accy_responsiveness < UNDERSTEER_RESPONSIVENESS_THRESHOLD)  # poor response
        & (accy_abs_aligned > UNDERSTEER_ACCY_ABS_MIN)  # minimum lateral G
    )

    # Require sustained understeer: ≥6 samples (0.30s) within a rolling window
    # Count True values in rolling window
    sustained_count = understeer_candidate.rolling(UNDERSTEER_SUSTAINED_SAMPLES).sum()
    df["understeer_event"] = (sustained_count >= UNDERSTEER_SUSTAINED_SAMPLES).fillna(
        False
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

    # Compute per-event coverage
    num_samples = len(zone_samples)
    wheelspin_coverage = (
        zone_samples["wheelspin_event"].sum() / num_samples if num_samples > 0 else 0
    )
    understeer_coverage = (
        zone_samples["understeer_event"].sum() / num_samples if num_samples > 0 else 0
    )
    oversteer_coverage = (
        zone_samples["oversteer_event"].sum() / num_samples if num_samples > 0 else 0
    )

    # Compute per-event maximum contiguous run lengths
    def get_max_run(series):
        max_run, current = 0, 0
        for val in series:
            current = current + 1 if val else 0
            max_run = max(max_run, current)
        return max_run

    wheelspin_max_run = get_max_run(zone_samples["wheelspin_event"])
    understeer_max_run = get_max_run(zone_samples["understeer_event"])
    oversteer_max_run = get_max_run(zone_samples["oversteer_event"])

    # Check if each event qualifies as macro-event independently
    # Spike events use hybrid logic: broad presence at lower util OR sustained spike at higher util
    is_macro_wheelspin = (
        wheelspin_coverage >= MACRO_EVENT_COVERAGE_THRESHOLD and utilization >= 0.68
    ) or (
        wheelspin_max_run >= MACRO_EVENT_RUN_LENGTH_THRESHOLD_WHEELSPIN
        and utilization >= 0.73
    )
    is_macro_understeer = (
        understeer_coverage >= MACRO_EVENT_COVERAGE_THRESHOLD
        and understeer_max_run >= MACRO_EVENT_RUN_LENGTH_THRESHOLD_UNDERSTEER
        and utilization >= AGGRESSIVE_UNDERSTEER_UTIL_THRESHOLD
    )
    is_macro_oversteer = (
        oversteer_coverage >= MACRO_EVENT_COVERAGE_THRESHOLD and utilization >= 0.73
    ) or (oversteer_max_run >= 2 and utilization >= 0.75)

    # Zone is aggressive if ANY event qualifies as macro
    is_macro_event = is_macro_wheelspin or is_macro_understeer or is_macro_oversteer

    # Gate zone-level event flags: spike events use lower coverage threshold (20% vs 30%)
    wheelspin_occurred = wheelspin_coverage >= 0.20 or is_macro_wheelspin
    understeer_occurred = understeer_coverage >= 0.30 or is_macro_understeer
    oversteer_occurred = oversteer_coverage >= 0.20 or is_macro_oversteer

    # Identify primary event (longest max run, tie-break by coverage)
    primary_event = "none"
    if is_macro_event:
        runs = [
            (wheelspin_max_run, wheelspin_coverage, "wheelspin"),
            (understeer_max_run, understeer_coverage, "understeer"),
            (oversteer_max_run, oversteer_coverage, "oversteer"),
        ]
        primary_event = max(runs, key=lambda x: (x[0], x[1]))[2]

    # Classify zone
    if is_macro_event:
        classification = "Aggressive"
    elif utilization >= OPTIMAL_UTILIZATION_THRESHOLD:
        classification = "Optimal"
    else:
        classification = "Conservative"

    # Compute union metrics for backward compatibility
    any_event = (
        zone_samples["wheelspin_event"]
        | zone_samples["understeer_event"]
        | zone_samples["oversteer_event"]
    )
    event_coverage = any_event.sum() / num_samples if num_samples > 0 else 0
    max_event_run_len = get_max_run(any_event)

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
        # Per-event metrics
        "wheelspin_coverage": wheelspin_coverage,
        "understeer_coverage": understeer_coverage,
        "oversteer_coverage": oversteer_coverage,
        "wheelspin_max_run": wheelspin_max_run,
        "understeer_max_run": understeer_max_run,
        "oversteer_max_run": oversteer_max_run,
        "primary_event": primary_event,
    }
