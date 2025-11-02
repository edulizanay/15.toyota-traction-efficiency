# ABOUTME: Data processing pipeline for brake point analysis
# ABOUTME: Loads telemetry, detects brake events, assigns zones, and computes dispersion metrics

import pandas as pd
import numpy as np
import json
from pyproj import Transformer


# ============================================================================
# Data Loading Functions (from src/data_loaders.py)
# ============================================================================

def load_and_pivot_telemetry(telemetry_path, chunk_size=500000):
    """
    Load telemetry data in chunks, filter to needed parameters, and convert GPS to meters.

    Args:
        telemetry_path: Path to telemetry CSV file
        chunk_size: Number of rows to process at a time

    Returns:
        DataFrame with columns: vehicle_number, lap, timestamp, pbrake_f, pbrake_r,
                                VBOX_Long_Minutes, VBOX_Lat_Min, speed, x_meters, y_meters
    """
    # Parameters we need from telemetry
    needed_params = [
        "pbrake_f",
        "pbrake_r",
        "VBOX_Long_Minutes",
        "VBOX_Lat_Min",
        "speed",
    ]

    print(f"Loading telemetry from {telemetry_path}")
    print(f"Processing in chunks of {chunk_size:,} rows...")

    # Store pivoted chunks
    pivoted_chunks = []
    chunk_count = 0

    # Read in chunks
    for chunk in pd.read_csv(telemetry_path, chunksize=chunk_size):
        chunk_count += 1

        # Filter to only needed telemetry parameters
        chunk_filtered = chunk[chunk["telemetry_name"].isin(needed_params)].copy()

        if len(chunk_filtered) == 0:
            continue

        # Pivot: one row per (vehicle_number, lap, timestamp) with columns for each parameter
        chunk_pivoted = chunk_filtered.pivot_table(
            index=["vehicle_number", "lap", "timestamp"],
            columns="telemetry_name",
            values="telemetry_value",
            aggfunc="first",  # Take first value if duplicates
        ).reset_index()

        pivoted_chunks.append(chunk_pivoted)

        if chunk_count % 10 == 0:
            print(
                f"  Processed {chunk_count} chunks ({chunk_count * chunk_size:,} rows)..."
            )

    print(f"Total chunks processed: {chunk_count}")
    print("Concatenating chunks...")

    # Combine all chunks
    df = pd.concat(pivoted_chunks, ignore_index=True)

    # Drop rows with missing GPS data (can't use without coordinates)
    print(f"Rows before GPS filter: {len(df):,}")
    df = df.dropna(subset=["VBOX_Long_Minutes", "VBOX_Lat_Min"])
    print(f"Rows after GPS filter: {len(df):,}")

    # Convert GPS coordinates to meters using UTM projection
    print("Converting GPS coordinates to meters using UTM projection...")
    df = convert_gps_to_meters(df)

    print(
        f"Final dataset: {len(df):,} rows, {len(df['vehicle_number'].unique())} vehicles"
    )

    return df


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


def compute_brake_threshold_p5(df, percentile=5):
    """
    Calculate brake pressure threshold to detect brake onset.
    Uses P5 (5th percentile) of positive pressures to discard lowest 5% as noise.

    Args:
        df: Telemetry DataFrame with pbrake_f and pbrake_r columns
        percentile: Percentile threshold (default 5 to discard lowest 5%)

    Returns:
        Threshold value in bar
    """
    # Combine front and rear brake pressures
    all_brake_pressures = pd.concat([df["pbrake_f"].dropna(), df["pbrake_r"].dropna()])

    # Filter to positive pressures only (zeros are not braking)
    positive_pressures = all_brake_pressures[all_brake_pressures > 0]

    # Calculate P5 of positive pressures (discard lowest 5% as noise)
    threshold = np.percentile(positive_pressures, percentile)

    print(
        f"P{percentile} brake pressure threshold (positive pressures only): {threshold:.2f} bar"
    )
    print(f"Total brake pressure samples: {len(all_brake_pressures):,}")
    print(
        f"Positive brake pressure samples: {len(positive_pressures):,} ({100 * len(positive_pressures) / len(all_brake_pressures):.1f}%)"
    )
    print(
        f"Zero/negative samples excluded: {len(all_brake_pressures) - len(positive_pressures):,}"
    )

    return threshold


def load_usac_results(results_path):
    """
    Load USAC timing results.

    Args:
        results_path: Path to USAC results CSV file

    Returns:
        DataFrame with race results including driver numbers and fastest lap times
    """
    print(f"Loading USAC results from {results_path}")

    # USAC files use semicolon delimiter
    df = pd.read_csv(results_path, sep=";")

    # Convert FL_TIME (fastest lap time) to seconds for easier comparison
    df["FL_TIME_seconds"] = pd.to_timedelta(
        "00:" + df["FL_TIME"].astype(str)
    ).dt.total_seconds()

    print(f"Loaded {len(df)} drivers")
    print(
        f"Fastest lap: {df.loc[df['FL_TIME_seconds'].idxmin(), 'FL_TIME']} by car #{df.loc[df['FL_TIME_seconds'].idxmin(), 'NUMBER']}"
    )

    return df


# ============================================================================
# Brake Detection Functions (from src/brake_detection.py)
# ============================================================================

def detect_brake_onsets(df, threshold):
    """
    Detect brake onset events using rising-edge detection.

    Args:
        df: Telemetry DataFrame with pbrake_f, pbrake_r, x_meters, y_meters, timestamp, lap, vehicle_number
        threshold: Brake pressure threshold in bar (from P5 calculation)

    Returns:
        DataFrame with brake onset events: vehicle_number, lap, timestamp, x_meters, y_meters,
                                          brake_pressure, brake_type (front/rear)
    """
    print(f"Detecting brake events with threshold: {threshold:.2f} bar")

    # Sort by vehicle, lap, and timestamp to ensure correct order
    df = df.sort_values(["vehicle_number", "lap", "timestamp"]).copy()

    # Combine front and rear brake pressures (use max)
    df["brake_pressure"] = df[["pbrake_f", "pbrake_r"]].max(axis=1)

    # Determine which brake led (front or rear)
    df["brake_type"] = np.where(df["pbrake_f"] >= df["pbrake_r"], "front", "rear")

    # Mark samples where braking (pressure >= threshold)
    df["is_braking"] = df["brake_pressure"] >= threshold

    # Detect rising edges (transition from not braking to braking)
    # Group by vehicle and lap to handle edge detection per stint
    brake_events = []

    for (vehicle, lap), group in df.groupby(["vehicle_number", "lap"]):
        # Shift is_braking to detect transitions
        prev_braking = group["is_braking"].shift(1, fill_value=False)
        rising_edge = group["is_braking"] & (~prev_braking)

        # Extract brake onset events
        onsets = group[rising_edge].copy()

        if len(onsets) > 0:
            brake_events.append(
                onsets[
                    [
                        "vehicle_number",
                        "lap",
                        "timestamp",
                        "x_meters",
                        "y_meters",
                        "VBOX_Long_Minutes",
                        "VBOX_Lat_Min",
                        "brake_pressure",
                        "brake_type",
                        "pbrake_f",
                        "pbrake_r",
                    ]
                ]
            )

    if len(brake_events) == 0:
        print("❌ WARNING: No brake events detected!")
        return pd.DataFrame()

    # Combine all events
    df_events = pd.concat(brake_events, ignore_index=True)

    print(f"✓ Detected {len(df_events):,} brake onset events")
    print(
        f"  Events per vehicle (avg): {len(df_events) / df['vehicle_number'].nunique():.1f}"
    )
    print(
        f"  Front brake led: {(df_events['brake_type'] == 'front').sum():,} ({100 * (df_events['brake_type'] == 'front').sum() / len(df_events):.1f}%)"
    )
    print(
        f"  Rear brake led: {(df_events['brake_type'] == 'rear').sum():,} ({100 * (df_events['brake_type'] == 'rear').sum() / len(df_events):.1f}%)"
    )

    return df_events


# ============================================================================
# Racing Lap Filter (centralized helper)
# ============================================================================

def filter_racing_laps(brake_events_df, telemetry_df, min_lap_distance=3500, max_lap_distance=4000):
    """
    Filter brake events to only include racing laps (3500-4000m lap distance heuristic).

    Args:
        brake_events_df: DataFrame with brake events
        telemetry_df: Full telemetry DataFrame
        min_lap_distance: Minimum lap distance in meters (default: 3500)
        max_lap_distance: Maximum lap distance in meters (default: 4000)

    Returns:
        Filtered brake events DataFrame
    """
    print(f"Filtering to racing laps ({min_lap_distance}-{max_lap_distance}m)...")

    # Calculate lap distances from telemetry
    lap_distances = []
    for (vehicle, lap), group in telemetry_df.groupby(["vehicle_number", "lap"]):
        dx = np.diff(group["x_meters"].values)
        dy = np.diff(group["y_meters"].values)
        lap_dist = np.sum(np.sqrt(dx**2 + dy**2))
        lap_distances.append({
            "vehicle_number": vehicle,
            "lap": lap,
            "lap_distance": lap_dist
        })

    lap_dist_df = pd.DataFrame(lap_distances)

    # Filter to racing laps
    racing_laps = lap_dist_df[
        (lap_dist_df["lap_distance"] >= min_lap_distance) &
        (lap_dist_df["lap_distance"] <= max_lap_distance)
    ].copy()

    print(f"  Total laps: {len(lap_dist_df)}")
    print(f"  Racing laps: {len(racing_laps)}")

    # Merge to filter brake events
    df_filtered = brake_events_df.merge(
        racing_laps[["vehicle_number", "lap"]],
        on=["vehicle_number", "lap"],
        how="inner"
    )

    print(f"  Brake events before filter: {len(brake_events_df):,}")
    print(f"  Brake events after filter: {len(df_filtered):,}")

    return df_filtered


# ============================================================================
# Corner Detection Functions (from src/corner_detection.py)
# ============================================================================

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


def assign_brake_events_to_zones(brake_events_df, centerline_x, centerline_y, zones_json_path):
    """
    Assign brake events to zones based on track distance.

    Args:
        brake_events_df: DataFrame with x_meters, y_meters columns
        centerline_x: Array of centerline x coordinates
        centerline_y: Array of centerline y coordinates
        zones_json_path: Path to corner definitions JSON

    Returns:
        DataFrame with added track_distance and zone_id columns
    """
    # Load zone definitions
    with open(zones_json_path, "r") as f:
        zones = json.load(f)

    # Project brake points to track distance
    track_distances = project_points_onto_centerline(
        brake_events_df["x_meters"].values,
        brake_events_df["y_meters"].values,
        centerline_x,
        centerline_y,
    )

    brake_events_df = brake_events_df.copy()
    brake_events_df["track_distance"] = track_distances

    # Assign to zones
    def assign_zone(distance):
        for zone in zones:
            if zone["start_distance_m"] <= distance <= zone["end_distance_m"]:
                return zone["zone_id"]
        return None

    brake_events_df["zone_id"] = brake_events_df["track_distance"].apply(assign_zone)

    return brake_events_df


def compute_zone_bounds(brake_events_df, padding_m=20.0):
    """
    Calculate spatial boundaries (bounding box) for each brake zone.

    Args:
        brake_events_df: DataFrame with brake events (must have zone_id, x_meters, y_meters)
        padding_m: Padding in meters to add around each zone (default: 20m)

    Returns:
        dict: {zone_id: {x_min, x_max, y_min, y_max, center_x, center_y}}
    """
    bounds = {}
    grouped = brake_events_df[brake_events_df["zone_id"].notna()].groupby("zone_id")

    for zid, dfz in grouped:
        xmin, xmax = dfz["x_meters"].min(), dfz["x_meters"].max()
        ymin, ymax = dfz["y_meters"].min(), dfz["y_meters"].max()

        # Add padding
        xmin -= padding_m
        xmax += padding_m
        ymin -= padding_m
        ymax += padding_m

        bounds[int(zid)] = {
            "x_min": float(xmin),
            "x_max": float(xmax),
            "y_min": float(ymin),
            "y_max": float(ymax),
            "center_x": float((xmin + xmax) / 2.0),
            "center_y": float((ymin + ymax) / 2.0),
        }

    return bounds


# ============================================================================
# Consistency Analysis Functions (from src/consistency_analysis.py)
# ============================================================================

def compute_zone_dispersion(brake_events_df):
    """
    Calculate brake point dispersion (std dev) per driver per zone.

    Returns:
        DataFrame with columns: vehicle_number, zone_id, dispersion_meters, brake_count
    """
    # Filter to only brake events within zones
    in_zone = brake_events_df[brake_events_df["zone_id"].notna()].copy()

    results = []

    for (vehicle, zone), group in in_zone.groupby(["vehicle_number", "zone_id"]):
        if len(group) < 2:
            # Need at least 2 points to calculate std dev
            continue

        # Calculate std dev in x and y
        std_x = group["x_meters"].std()
        std_y = group["y_meters"].std()

        # Euclidean std dev (dispersion in meters)
        dispersion = np.sqrt(std_x**2 + std_y**2)

        results.append({
            "vehicle_number": vehicle,
            "zone_id": zone,
            "dispersion_meters": dispersion,
            "brake_count": len(group),
            "std_x": std_x,
            "std_y": std_y,
        })

    return pd.DataFrame(results)


def compute_zone_centroids(brake_events_df):
    """
    Calculate centroid (mean brake point position) per driver per zone.

    Args:
        brake_events_df: DataFrame with brake events (must have zone_id, x_meters, y_meters, vehicle_number)

    Returns:
        DataFrame with columns: vehicle_number, zone_id, centroid_x, centroid_y, brake_count
    """
    # Filter to only brake events within zones
    in_zone = brake_events_df[brake_events_df["zone_id"].notna()].copy()

    results = []

    for (vehicle, zone), group in in_zone.groupby(["vehicle_number", "zone_id"]):
        # Calculate mean position (centroid)
        centroid_x = group["x_meters"].mean()
        centroid_y = group["y_meters"].mean()

        results.append({
            "vehicle_number": vehicle,
            "zone_id": zone,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "brake_count": len(group),
        })

    return pd.DataFrame(results)


def summarize_driver_consistency(dispersion_by_zone_df):
    """
    Calculate average dispersion across all zones per driver.

    Returns:
        DataFrame with columns: vehicle_number, avg_dispersion_meters, zone_count, total_brake_count
    """
    summary = dispersion_by_zone_df.groupby("vehicle_number").agg({
        "dispersion_meters": "mean",
        "zone_id": "count",
        "brake_count": "sum",
    }).reset_index()

    summary.columns = ["vehicle_number", "avg_dispersion_meters", "zone_count", "total_brake_count"]

    return summary


def merge_usac_lap_times(driver_summary_df, usac_results_path):
    """
    Add lap time data from USAC results.

    Args:
        driver_summary_df: DataFrame with vehicle_number column
        usac_results_path: Path to USAC results CSV

    Returns:
        DataFrame with added fastest_lap_time column
    """
    # Load USAC results (semicolon-separated)
    usac = pd.read_csv(usac_results_path, sep=";")

    # Extract relevant columns: NUMBER (car number), FL_TIME (fastest lap time)
    usac_clean = usac[["NUMBER", "FL_TIME"]].copy()
    usac_clean.columns = ["vehicle_number", "fastest_lap_time"]

    # Convert lap time to seconds
    def lap_time_to_seconds(time_str):
        if pd.isna(time_str) or time_str == "":
            return np.nan
        try:
            parts = time_str.split(":")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except:
            return np.nan

    usac_clean["fastest_lap_seconds"] = usac_clean["fastest_lap_time"].apply(lap_time_to_seconds)

    # Merge with driver summary
    merged = driver_summary_df.merge(
        usac_clean[["vehicle_number", "fastest_lap_time", "fastest_lap_seconds"]],
        on="vehicle_number",
        how="left"
    )

    return merged
