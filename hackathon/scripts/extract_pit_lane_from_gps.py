# ABOUTME: Extract pit lane geometry from GPS telemetry data
# ABOUTME: Uses pit stop lap sequences to build complete pit lane path

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import json

from src.geometry import compute_normals


def extract_pit_lap_gps(telemetry_file, car_laps):
    """
    Extract GPS coordinates from specific car/lap combinations.

    Args:
        telemetry_file: Path to telemetry CSV file
        car_laps: List of (car_number, lap_number) tuples

    Returns:
        DataFrame with GPS coordinates sorted by timestamp
    """
    print("\n=== Extracting GPS from Pit Lane Laps ===")
    print(f"  Reading from: {telemetry_file}")
    print(f"  Target laps: {car_laps}")

    # Read telemetry data in chunks (file is 1.5GB)
    gps_data = []
    chunk_size = 100000

    for chunk_num, chunk in enumerate(
        pd.read_csv(telemetry_file, chunksize=chunk_size)
    ):
        if chunk_num % 10 == 0:
            print(f"  Processing chunk {chunk_num}...")

        # Filter for GPS telemetry only
        gps_chunk = chunk[
            chunk["telemetry_name"].isin(["VBOX_Long_Minutes", "VBOX_Lat_Min"])
        ]

        # Filter for our target car/lap combinations
        for car_num, lap_num in car_laps:
            lap_data = gps_chunk[
                (gps_chunk["vehicle_number"] == car_num) & (gps_chunk["lap"] == lap_num)
            ]

            if len(lap_data) > 0:
                gps_data.append(lap_data)

    # Combine all GPS data
    if not gps_data:
        raise ValueError("No GPS data found for specified laps!")

    all_gps = pd.concat(gps_data, ignore_index=True)

    print(f"  Found {len(all_gps)} GPS telemetry points")

    # Pivot to get lat/lon in columns
    gps_pivot = all_gps.pivot_table(
        index=["vehicle_number", "lap", "timestamp"],
        columns="telemetry_name",
        values="telemetry_value",
    ).reset_index()

    # Sort by timestamp to get chronological order
    gps_pivot = gps_pivot.sort_values("timestamp")

    print(f"  Result: {len(gps_pivot)} GPS coordinate pairs")
    print(
        f"  Lat range: {gps_pivot['VBOX_Lat_Min'].min():.6f} to {gps_pivot['VBOX_Lat_Min'].max():.6f}"
    )
    print(
        f"  Lon range: {gps_pivot['VBOX_Long_Minutes'].min():.6f} to {gps_pivot['VBOX_Long_Minutes'].max():.6f}"
    )

    return gps_pivot


def gps_to_utm(lat, lon, zone=16, hemisphere="N"):
    """
    Convert GPS coordinates to UTM.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        zone: UTM zone (Barber is Zone 16)
        hemisphere: 'N' or 'S'

    Returns:
        (easting, northing) in meters
    """
    from pyproj import Transformer

    # Create transformer from WGS84 (GPS) to UTM
    transformer = Transformer.from_crs(
        "EPSG:4326",  # WGS84
        f"EPSG:326{zone:02d}" if hemisphere == "N" else f"EPSG:327{zone:02d}",  # UTM
        always_xy=True,
    )

    easting, northing = transformer.transform(lon, lat)
    return easting, northing


def apply_rotation(x, y, angle_degrees):
    """
    Apply rotation to coordinates.

    Args:
        x, y: Coordinate arrays
        angle_degrees: Rotation angle (negative = clockwise)

    Returns:
        rotated x, y arrays
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Find center point
    cx = np.mean(x)
    cy = np.mean(y)

    # Translate to origin, rotate, translate back
    x_centered = x - cx
    y_centered = y - cy

    x_rot = x_centered * cos_a - y_centered * sin_a + cx
    y_rot = x_centered * sin_a + y_centered * cos_a + cy

    return x_rot, y_rot


def convert_gps_to_track_coordinates(gps_df, rotation_angle=-42):
    """
    Convert GPS data to track coordinate system (UTM + rotation).

    Args:
        gps_df: DataFrame with VBOX_Lat_Min and VBOX_Long_Minutes
        rotation_angle: Rotation to apply (default -42° like track centerline)

    Returns:
        Arrays of x_meters, y_meters in track coordinate system
    """
    print("\n=== Converting GPS to Track Coordinates ===")

    # Convert to UTM
    print("  Converting GPS to UTM Zone 16N...")
    easting = []
    northing = []

    for idx, row in gps_df.iterrows():
        e, n = gps_to_utm(row["VBOX_Lat_Min"], row["VBOX_Long_Minutes"])
        easting.append(e)
        northing.append(n)

    easting = np.array(easting)
    northing = np.array(northing)

    print("  UTM coordinates:")
    print(f"    Easting:  {easting.min():.1f} to {easting.max():.1f}m")
    print(f"    Northing: {northing.min():.1f} to {northing.max():.1f}m")

    # Apply rotation
    print(f"  Applying {rotation_angle}° rotation...")
    x_rot, y_rot = apply_rotation(easting, northing, rotation_angle)

    print("  Rotated coordinates:")
    print(f"    X: {x_rot.min():.1f} to {x_rot.max():.1f}m")
    print(f"    Y: {y_rot.min():.1f} to {y_rot.max():.1f}m")

    return x_rot, y_rot


def find_touching_segments(pit_x, pit_y, track_x, track_y, touch_threshold=15.0):
    """
    Find where pit lane path touches the track centerline.

    Args:
        pit_x, pit_y: Pit lane coordinates
        track_x, track_y: Track centerline coordinates
        touch_threshold: Distance threshold for "touching" (meters)

    Returns:
        Boolean array: True where pit lane touches track
    """
    print("\n=== Finding Touching vs Non-Touching Segments ===")
    print(f"  Touch threshold: {touch_threshold}m")

    # Build KD-tree for track
    track_points = np.column_stack([track_x, track_y])
    tree = KDTree(track_points)

    # Find distance from each pit point to nearest track point
    pit_points = np.column_stack([pit_x, pit_y])
    distances, _ = tree.query(pit_points)

    # Mark as touching if within threshold
    is_touching = distances < touch_threshold

    num_touching = np.sum(is_touching)
    num_separate = np.sum(~is_touching)

    print(
        f"  Touching points: {num_touching} ({num_touching / len(is_touching) * 100:.1f}%)"
    )
    print(
        f"  Separate points: {num_separate} ({num_separate / len(is_touching) * 100:.1f}%)"
    )
    print(f"  Distance range: {distances.min():.1f}m to {distances.max():.1f}m")

    return is_touching, distances


def extract_largest_non_touching_segment(pit_x, pit_y, is_touching):
    """
    Extract the largest continuous segment where pit lane is separate from track.

    Args:
        pit_x, pit_y: Pit lane coordinates
        is_touching: Boolean array marking touching points

    Returns:
        x, y arrays for the largest non-touching segment
    """
    print("\n=== Extracting Non-Touching Segment ===")

    # Find continuous non-touching segments
    segments = []
    current_segment = []

    for i in range(len(is_touching)):
        if not is_touching[i]:
            # Part of non-touching segment
            current_segment.append(i)
        else:
            # End of segment
            if len(current_segment) > 0:
                segments.append(current_segment)
                current_segment = []

    # Don't forget last segment
    if len(current_segment) > 0:
        segments.append(current_segment)

    print(f"  Found {len(segments)} non-touching segments:")
    for i, seg in enumerate(segments):
        print(f"    Segment {i + 1}: {len(seg)} points")

    if not segments:
        raise ValueError("No non-touching segments found!")

    # Find largest segment
    largest_seg = max(segments, key=len)
    largest_idx = segments.index(largest_seg)

    print(f"  Using largest segment #{largest_idx + 1} with {len(largest_seg)} points")

    # Extract coordinates
    seg_x = pit_x[largest_seg]
    seg_y = pit_y[largest_seg]

    # Calculate length
    dx = np.diff(seg_x)
    dy = np.diff(seg_y)
    length = np.sum(np.sqrt(dx**2 + dy**2))

    print(f"  Segment length: {length:.1f}m")

    return seg_x, seg_y


def add_width_boundaries(centerline_x, centerline_y, half_width_m=6.0):
    """Create inner and outer boundaries for pit lane."""
    print("\n=== Adding Width Boundaries ===")

    normals = compute_normals(centerline_x, centerline_y)

    inner_x = centerline_x - normals[:, 0] * half_width_m
    inner_y = centerline_y - normals[:, 1] * half_width_m

    outer_x = centerline_x + normals[:, 0] * half_width_m
    outer_y = centerline_y + normals[:, 1] * half_width_m

    print(f"  Created boundaries with {half_width_m * 2}m total width")

    return {
        "inner": {"x": inner_x, "y": inner_y},
        "outer": {"x": outer_x, "y": outer_y},
    }


def save_pit_lane_data(centerline_x, centerline_y, boundaries, output_path):
    """Save pit lane data to JSON."""

    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    total_length = np.sum(np.sqrt(dx**2 + dy**2))

    pit_lane_data = {
        "description": "Pit lane extracted from GPS telemetry data",
        "total_distance_m": float(total_length),
        "sample_interval_m": float(total_length / len(centerline_x)),
        "centerline": [
            {"x_meters": float(x), "y_meters": float(y)}
            for x, y in zip(centerline_x, centerline_y)
        ],
        "boundaries": {
            "inner": [
                {"x_meters": float(x), "y_meters": float(y)}
                for x, y in zip(boundaries["inner"]["x"], boundaries["inner"]["y"])
            ],
            "outer": [
                {"x_meters": float(x), "y_meters": float(y)}
                for x, y in zip(boundaries["outer"]["x"], boundaries["outer"]["y"])
            ],
        },
        "anchors": {},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(pit_lane_data, f, indent=2)

    print(f"\n✓ Saved pit lane data to: {output_path}")
    print(f"  Total length: {total_length:.1f}m")
    print(f"  Centerline points: {len(centerline_x)}")
    print(f"  Boundary points: {len(boundaries['inner']['x'])} per side")


def main():
    # Set working directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    print("=== Pit Lane Extraction from GPS Telemetry ===\n")

    # Paths
    telemetry_file = "data/input/R1_barber_telemetry_data.csv"
    centerline_path = "data/processed/track_centerline.csv"
    output_path = "data/processed/pit_lane.json"

    # Pit stop lap sequences
    # Car 46: Lap 8 (enter pit) + Lap 9 (exit pit)
    # Car 51: Lap 23 (enter pit) + Lap 24 (exit pit)
    pit_laps = [
        (46, 8),
        (46, 9),
        (51, 23),
        (51, 24),
    ]

    # Step 1: Extract GPS data from pit laps
    gps_df = extract_pit_lap_gps(telemetry_file, pit_laps)

    # Step 2: Convert to track coordinates (UTM only, no rotation)
    # Note: Rotation is applied at visualization time in track.js
    pit_x, pit_y = convert_gps_to_track_coordinates(gps_df, rotation_angle=0)

    # Step 3: Load track centerline
    print("\n=== Loading Track Centerline ===")
    track_df = pd.read_csv(centerline_path)
    track_x = track_df["x_meters"].values
    track_y = track_df["y_meters"].values
    print(f"  Loaded {len(track_x)} track centerline points")

    # Step 4: Find touching vs non-touching segments
    is_touching, distances = find_touching_segments(
        pit_x, pit_y, track_x, track_y, touch_threshold=15.0
    )

    # Step 5: Extract largest non-touching segment
    final_x, final_y = extract_largest_non_touching_segment(pit_x, pit_y, is_touching)

    # Step 6: Add width boundaries
    boundaries = add_width_boundaries(final_x, final_y, half_width_m=6.0)

    # Step 7: Save output
    save_pit_lane_data(final_x, final_y, boundaries, output_path)

    print("\n✓ Pit lane extraction complete!")
    print("\nNext step: Open hackathon/track.html to visualize the pit lane.")


if __name__ == "__main__":
    main()
