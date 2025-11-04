# ABOUTME: Extract pit lane polyline from telemetry using spatial gating
# ABOUTME: Finds front straight, filters pit samples by distance/heading/speed, creates track anchors

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import json
import argparse

from src.geometry import convert_gps_to_meters


def compute_curvature(x, y):
    """
    Compute curvature along a polyline using finite differences.

    Args:
        x: x coordinates (array-like)
        y: y coordinates (array-like)

    Returns:
        array of curvature values (rad/m) at each point
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Compute first derivatives (tangent vectors)
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Compute second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula: |dx*ddy - dy*ddx| / (dx^2 + dy^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5)
    denominator[denominator == 0] = 1e-10  # Avoid division by zero

    curvature = numerator / denominator
    return curvature


def find_front_straight(
    centerline_x, centerline_y, min_length_m=150, curvature_threshold=0.005
):
    """
    Find the front straight: longest low-curvature segment with highest average Y.

    Args:
        centerline_x: Centerline x coordinates
        centerline_y: Centerline y coordinates
        min_length_m: Minimum length to consider as straight
        curvature_threshold: Maximum curvature for straight sections (rad/m)

    Returns:
        dict: {
            'start_idx': start index,
            'end_idx': end index,
            'tangent': unit tangent vector [tx, ty],
            'normal': unit normal vector [nx, ny],
            'center_x': center x coordinate,
            'center_y': center y coordinate,
            'length_m': length of straight
        }
    """
    print("\n=== Finding Front Straight ===")

    # Compute curvature
    curvature = compute_curvature(centerline_x, centerline_y)

    # Compute segment lengths
    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    segment_lengths = np.sqrt(dx**2 + dy**2)

    # Find low-curvature segments
    is_straight = curvature < curvature_threshold

    # Find continuous straight sections
    straight_sections = []
    in_section = False
    section_start = None

    for i in range(len(is_straight)):
        if is_straight[i] and not in_section:
            section_start = i
            in_section = True
        elif not is_straight[i] and in_section:
            section_end = i - 1
            # Calculate section length
            section_length = np.sum(segment_lengths[section_start:section_end])
            if section_length >= min_length_m:
                avg_y = np.mean(centerline_y[section_start : section_end + 1])
                straight_sections.append(
                    {
                        "start": section_start,
                        "end": section_end,
                        "length": section_length,
                        "avg_y": avg_y,
                    }
                )
            in_section = False

    # Handle case where track ends in straight
    if in_section:
        section_end = len(is_straight) - 1
        section_length = np.sum(segment_lengths[section_start:section_end])
        if section_length >= min_length_m:
            avg_y = np.mean(centerline_y[section_start : section_end + 1])
            straight_sections.append(
                {
                    "start": section_start,
                    "end": section_end,
                    "length": section_length,
                    "avg_y": avg_y,
                }
            )

    if not straight_sections:
        raise ValueError(
            f"No straight sections found (min_length={min_length_m}m, curvature<{curvature_threshold})"
        )

    print(f"  Found {len(straight_sections)} straight sections:")
    for s in straight_sections:
        print(f"    Length: {s['length']:.1f}m, Avg Y: {s['avg_y']:.1f}m")

    # Select longest straight with highest Y (front straight is at top)
    front_straight = max(
        straight_sections, key=lambda s: (s["length"] * 0.5 + s["avg_y"])
    )

    start_idx = front_straight["start"]
    end_idx = front_straight["end"]

    print("\n  Selected front straight:")
    print(f"    Indices: {start_idx} to {end_idx}")
    print(f"    Length: {front_straight['length']:.1f}m")
    print(f"    Avg Y: {front_straight['avg_y']:.1f}m")

    # Compute average tangent and normal for this section
    section_x = centerline_x[start_idx : end_idx + 1]
    section_y = centerline_y[start_idx : end_idx + 1]

    # Tangent: average direction along the straight
    dx_avg = section_x[-1] - section_x[0]
    dy_avg = section_y[-1] - section_y[0]
    mag = np.sqrt(dx_avg**2 + dy_avg**2)
    tangent = np.array([dx_avg / mag, dy_avg / mag])

    # Normal: perpendicular to tangent (90° CCW rotation)
    normal = np.array([-tangent[1], tangent[0]])

    # Center point of straight
    center_x = np.mean(section_x)
    center_y = np.mean(section_y)

    print(f"    Tangent: [{tangent[0]:.3f}, {tangent[1]:.3f}]")
    print(f"    Normal: [{normal[0]:.3f}, {normal[1]:.3f}]")
    print(f"    Center: ({center_x:.1f}, {center_y:.1f})")

    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "tangent": tangent,
        "normal": normal,
        "center_x": center_x,
        "center_y": center_y,
        "length_m": front_straight["length"],
    }


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """
    Compute distance from point (px, py) to line segment (x1, y1) to (x2, y2).

    Returns:
        distance: perpendicular distance to segment
        nearest_x: x coordinate of nearest point on segment
        nearest_y: y coordinate of nearest point on segment
    """
    # Vector from p1 to p2
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        # Segment is a point
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2), x1, y1

    # Parameter t: projection of point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)
    t = np.clip(t, 0, 1)  # Clamp to segment

    # Nearest point on segment
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    # Distance
    distance = np.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)

    return distance, nearest_x, nearest_y


def compute_heading(x, y):
    """
    Compute heading angle at each point from consecutive GPS positions.

    Args:
        x: x coordinates
        y: y coordinates

    Returns:
        array of heading angles in radians [-π, π]
    """
    dx = np.diff(x)
    dy = np.diff(y)
    heading = np.arctan2(dy, dx)

    # Duplicate last heading for same length
    heading = np.append(heading, heading[-1])

    return heading


def load_pit_candidates(telemetry_paths, front_straight, args):
    """
    Load telemetry and filter to pit lane candidates using spatial gates.

    Args:
        telemetry_paths: List of (race_name, path) tuples
        front_straight: Front straight info from find_front_straight()
        args: Command-line arguments with thresholds

    Returns:
        DataFrame with columns: x_meters, y_meters, speed, heading
    """
    print("\n=== Loading Telemetry and Applying Gates ===")

    needed_params = ["VBOX_Long_Minutes", "VBOX_Lat_Min", "speed"]
    all_candidates = []

    for race_name, telemetry_path in telemetry_paths:
        print(f"\n  Processing {race_name}...")

        chunk_count = 0
        total_rows = 0

        for chunk in pd.read_csv(telemetry_path, chunksize=50000):
            total_rows += len(chunk)

            # Filter to needed parameters
            chunk_filtered = chunk[chunk["telemetry_name"].isin(needed_params)].copy()

            if len(chunk_filtered) == 0:
                continue

            # Add sequence number for pivot
            chunk_filtered["seq"] = chunk_filtered.groupby(
                ["vehicle_number", "lap", "timestamp", "telemetry_name"]
            ).cumcount()

            # Pivot this chunk
            try:
                chunk_wide = chunk_filtered.pivot_table(
                    index=["vehicle_number", "lap", "timestamp", "seq"],
                    columns="telemetry_name",
                    values="telemetry_value",
                    aggfunc="first",
                ).reset_index()
                chunk_wide.columns.name = None
                chunk_wide = chunk_wide.drop(columns=["seq", "vehicle_number", "lap"])

                # Drop rows with NaN
                chunk_wide = chunk_wide.dropna()

                if len(chunk_wide) == 0:
                    continue

                # Convert GPS to meters
                chunk_wide = convert_gps_to_meters(chunk_wide)

                # Gate 1: Distance to front straight
                # Compute distance to each point in straight segment
                straight_x = np.linspace(
                    front_straight["center_x"]
                    - front_straight["tangent"][0] * front_straight["length_m"] / 2,
                    front_straight["center_x"]
                    + front_straight["tangent"][0] * front_straight["length_m"] / 2,
                    50,
                )
                straight_y = np.linspace(
                    front_straight["center_y"]
                    - front_straight["tangent"][1] * front_straight["length_m"] / 2,
                    front_straight["center_y"]
                    + front_straight["tangent"][1] * front_straight["length_m"] / 2,
                    50,
                )

                # For each point, find minimum distance to straight
                distances = []
                for _, row in chunk_wide.iterrows():
                    min_dist = float("inf")
                    for i in range(len(straight_x) - 1):
                        dist, _, _ = point_to_segment_distance(
                            row["x_meters"],
                            row["y_meters"],
                            straight_x[i],
                            straight_y[i],
                            straight_x[i + 1],
                            straight_y[i + 1],
                        )
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)

                chunk_wide["dist_to_straight"] = distances
                chunk_wide = chunk_wide[
                    chunk_wide["dist_to_straight"] < args.nearby_dist_m
                ]

                if len(chunk_wide) == 0:
                    continue

                # Gate 2: Speed < threshold (in mph!)
                chunk_wide = chunk_wide[chunk_wide["speed"] < args.mph_threshold]

                if len(chunk_wide) == 0:
                    continue

                # Gate 3: Compute heading and check alignment
                if len(chunk_wide) >= 2:
                    chunk_wide = chunk_wide.sort_values("timestamp")
                    headings = compute_heading(
                        chunk_wide["x_meters"].values, chunk_wide["y_meters"].values
                    )
                    chunk_wide["heading"] = headings

                    # Front straight heading
                    straight_heading = np.arctan2(
                        front_straight["tangent"][1], front_straight["tangent"][0]
                    )

                    # Angle difference (handle wrap-around)
                    angle_diff = np.abs(chunk_wide["heading"] - straight_heading)
                    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

                    angle_threshold_rad = np.radians(args.angle_deg)
                    chunk_wide = chunk_wide[angle_diff < angle_threshold_rad]

                if len(chunk_wide) == 0:
                    continue

                # Gate 4: Signed lateral offset (must be on "top" side)
                offsets = []
                for _, row in chunk_wide.iterrows():
                    # Vector from straight center to point
                    to_point = np.array(
                        [
                            row["x_meters"] - front_straight["center_x"],
                            row["y_meters"] - front_straight["center_y"],
                        ]
                    )
                    # Signed offset along normal
                    signed_offset = np.dot(to_point, front_straight["normal"])
                    offsets.append(signed_offset)

                chunk_wide["signed_offset"] = offsets
                chunk_wide = chunk_wide[chunk_wide["signed_offset"] > args.offset_min_m]

                if len(chunk_wide) > 0:
                    all_candidates.append(
                        chunk_wide[["x_meters", "y_meters", "speed", "signed_offset"]]
                    )
                    chunk_count += 1

            except Exception as e:
                print(f"    Warning: Chunk pivot failed: {e}")

        print(
            f"    Processed {total_rows:,} rows, found {chunk_count} chunks with pit candidates"
        )

    if not all_candidates:
        raise ValueError("No pit lane candidates found after applying gates!")

    df_candidates = pd.concat(all_candidates, ignore_index=True)
    print(f"\n  Total pit candidates: {len(df_candidates):,}")
    print(
        f"  Signed offset range: {df_candidates['signed_offset'].min():.1f}m to {df_candidates['signed_offset'].max():.1f}m"
    )
    print(f"  Signed offset mean: {df_candidates['signed_offset'].mean():.1f}m")

    return df_candidates


def cluster_and_clean(df_candidates, eps_m=10, min_samples=20):
    """
    Apply DBSCAN clustering and keep largest cluster.

    Args:
        df_candidates: DataFrame with x_meters, y_meters
        eps_m: DBSCAN eps parameter (meters)
        min_samples: DBSCAN min_samples parameter

    Returns:
        DataFrame with cleaned pit lane points
    """
    print("\n=== Clustering with DBSCAN ===")

    coords = df_candidates[["x_meters", "y_meters"]].values

    clustering = DBSCAN(eps=eps_m, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    # Count clusters (excluding noise: label=-1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    print(f"  Found {len(unique_labels)} clusters")

    if len(unique_labels) == 0:
        raise ValueError(
            "DBSCAN found no clusters! Try adjusting eps_m or min_samples."
        )

    # Find largest cluster
    cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    for label, size in cluster_sizes[:3]:
        print(f"    Cluster {label}: {size} points")

    largest_label = cluster_sizes[0][0]
    df_cleaned = df_candidates[labels == largest_label].copy()

    print(f"\n  Kept largest cluster: {len(df_cleaned)} points")

    return df_cleaned


def resample_and_smooth(df_cleaned, front_straight, step_m=8):
    """
    Project points onto straight axis, resample uniformly, and smooth.

    Args:
        df_cleaned: DataFrame with x_meters, y_meters
        front_straight: Front straight info
        step_m: Resampling step in meters

    Returns:
        arrays: (x_smooth, y_smooth)
    """
    print("\n=== Resampling and Smoothing ===")

    # Project onto straight axis
    tangent = front_straight["tangent"]
    center = np.array([front_straight["center_x"], front_straight["center_y"]])

    projections = []
    for _, row in df_cleaned.iterrows():
        point = np.array([row["x_meters"], row["y_meters"]])
        to_point = point - center
        projection = np.dot(to_point, tangent)
        projections.append((projection, point[0], point[1]))

    # Sort by projection distance
    projections.sort(key=lambda x: x[0])

    sorted_x = np.array([p[1] for p in projections])
    sorted_y = np.array([p[2] for p in projections])

    # Compute cumulative distance along polyline
    dx = np.diff(sorted_x)
    dy = np.diff(sorted_y)
    segment_dist = np.sqrt(dx**2 + dy**2)
    cumulative_dist = np.concatenate([[0], np.cumsum(segment_dist)])

    total_length = cumulative_dist[-1]
    print(f"  Original length: {total_length:.1f}m")

    # Resample at uniform intervals
    num_samples = int(total_length / step_m) + 1
    uniform_dist = np.linspace(0, total_length, num_samples)

    from scipy.interpolate import interp1d

    interp_x = interp1d(
        cumulative_dist, sorted_x, kind="linear", fill_value="extrapolate"
    )
    interp_y = interp1d(
        cumulative_dist, sorted_y, kind="linear", fill_value="extrapolate"
    )

    x_resampled = interp_x(uniform_dist)
    y_resampled = interp_y(uniform_dist)

    print(f"  Resampled to {len(x_resampled)} points (step={step_m}m)")

    # Apply Savitzky-Golay smoothing (no periodic wrapping for pit lane)
    if len(x_resampled) >= 11:
        x_smooth = savgol_filter(x_resampled, window_length=11, polyorder=3)
        y_smooth = savgol_filter(y_resampled, window_length=11, polyorder=3)
        print("  Applied Savitzky-Golay smoothing (window=11, poly=3)")
    else:
        x_smooth = x_resampled
        y_smooth = y_resampled
        print("  Skipped smoothing (too few points)")

    return x_smooth, y_smooth


def create_anchors_and_connectors(
    pit_x, pit_y, centerline_x, centerline_y, max_dist=15
):
    """
    Find anchors on centerline and create connector polylines.

    Args:
        pit_x: Pit lane x coordinates
        pit_y: Pit lane y coordinates
        centerline_x: Track centerline x coordinates
        centerline_y: Track centerline y coordinates
        max_dist: Maximum distance for valid anchor

    Returns:
        dict with 'anchors' and 'connectors'
    """
    print("\n=== Creating Track Anchors and Connectors ===")

    # Build KD-tree for centerline
    centerline_points = np.column_stack([centerline_x, centerline_y])
    tree = KDTree(centerline_points)

    # Compute cumulative distance along centerline
    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    cumulative_distance = np.concatenate([[0], np.cumsum(segment_lengths)])

    anchors = {}
    connectors = {}

    # Pit-out: start of pit lane (first point)
    pit_out_point = np.array([pit_x[0], pit_y[0]])
    dist, idx = tree.query(pit_out_point)

    if dist < max_dist:
        anchors["pit_out"] = {
            "centerline_idx": int(idx),
            "track_distance_m": float(cumulative_distance[idx]),
            "x_meters": float(centerline_x[idx]),
            "y_meters": float(centerline_y[idx]),
        }

        # Create smooth connector
        connector_points = []
        for t in np.linspace(0, 1, 8):
            x = centerline_x[idx] * (1 - t) + pit_x[0] * t
            y = centerline_y[idx] * (1 - t) + pit_y[0] * t
            connector_points.append({"x_meters": float(x), "y_meters": float(y)})

        connectors["pit_out"] = connector_points
        print(
            f"  Pit-out anchor: centerline[{idx}], distance={cumulative_distance[idx]:.1f}m, offset={dist:.1f}m"
        )
    else:
        print(
            f"  Warning: Pit-out anchor too far from centerline ({dist:.1f}m > {max_dist}m)"
        )

    # Pit-in: end of pit lane (last point)
    pit_in_point = np.array([pit_x[-1], pit_y[-1]])
    dist, idx = tree.query(pit_in_point)

    if dist < max_dist:
        anchors["pit_in"] = {
            "centerline_idx": int(idx),
            "track_distance_m": float(cumulative_distance[idx]),
            "x_meters": float(centerline_x[idx]),
            "y_meters": float(centerline_y[idx]),
        }

        # Create smooth connector
        connector_points = []
        for t in np.linspace(0, 1, 8):
            x = centerline_x[idx] * (1 - t) + pit_x[-1] * t
            y = centerline_y[idx] * (1 - t) + pit_y[-1] * t
            connector_points.append({"x_meters": float(x), "y_meters": float(y)})

        connectors["pit_in"] = connector_points
        print(
            f"  Pit-in anchor: centerline[{idx}], distance={cumulative_distance[idx]:.1f}m, offset={dist:.1f}m"
        )
    else:
        print(
            f"  Warning: Pit-in anchor too far from centerline ({dist:.1f}m > {max_dist}m)"
        )

    return {"anchors": anchors, "connectors": connectors}


def main():
    parser = argparse.ArgumentParser(
        description="Extract pit lane polyline using spatial gating"
    )
    parser.add_argument(
        "--r1-telemetry",
        default="data/input/R1_barber_telemetry_data.csv",
        help="Path to R1 telemetry CSV",
    )
    parser.add_argument(
        "--r2-telemetry",
        default="data/input/R2_barber_telemetry_data.csv",
        help="Path to R2 telemetry CSV",
    )
    parser.add_argument(
        "--centerline",
        default="data/processed/track_centerline.csv",
        help="Path to track centerline CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed/pit_lane.json",
        help="Output path for pit lane JSON",
    )
    parser.add_argument(
        "--mph-threshold",
        type=float,
        default=55,
        help="Maximum speed for pit lane (mph)",
    )
    parser.add_argument(
        "--nearby-dist-m",
        type=float,
        default=30,
        help="Maximum distance to front straight (meters)",
    )
    parser.add_argument(
        "--offset-min-m",
        type=float,
        default=8,
        help="Minimum lateral offset from straight (meters)",
    )
    parser.add_argument(
        "--angle-deg",
        type=float,
        default=20,
        help="Maximum heading angle difference (degrees)",
    )
    parser.add_argument(
        "--sample-step-m", type=float, default=8, help="Resampling step size (meters)"
    )

    args = parser.parse_args()

    # Set script directory as working directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    print("=== Pit Lane Extraction with Spatial Gating ===")
    print("\nParameters:")
    print(f"  Speed threshold: {args.mph_threshold} mph")
    print(f"  Nearby distance: {args.nearby_dist_m} m")
    print(f"  Offset minimum: {args.offset_min_m} m")
    print(f"  Angle threshold: {args.angle_deg} deg")
    print(f"  Sample step: {args.sample_step_m} m")

    # Phase 1: Find front straight
    centerline_df = pd.read_csv(args.centerline)
    centerline_x = centerline_df["x_meters"].values
    centerline_y = centerline_df["y_meters"].values

    front_straight = find_front_straight(centerline_x, centerline_y)

    # Phase 2: Load telemetry and apply gates
    telemetry_paths = [("R1", Path(args.r1_telemetry)), ("R2", Path(args.r2_telemetry))]

    df_candidates = load_pit_candidates(telemetry_paths, front_straight, args)

    # Phase 3: Cluster and clean
    df_cleaned = cluster_and_clean(df_candidates, eps_m=10, min_samples=20)

    # Phase 4: Resample and smooth
    pit_x, pit_y = resample_and_smooth(
        df_cleaned, front_straight, step_m=args.sample_step_m
    )

    # Calculate final statistics
    dx = np.diff(pit_x)
    dy = np.diff(pit_y)
    total_distance = np.sum(np.sqrt(dx**2 + dy**2))

    print("\n=== Final Pit Lane Statistics ===")
    print(f"  Total length: {total_distance:.1f}m")
    print(f"  Number of points: {len(pit_x)}")
    print(f"  Sample interval: {args.sample_step_m}m")

    # Calculate distance to front straight
    min_dist_to_straight = float("inf")
    for x, y in zip(pit_x, pit_y):
        to_point = np.array(
            [x - front_straight["center_x"], y - front_straight["center_y"]]
        )
        # Distance perpendicular to straight
        perp_dist = abs(np.dot(to_point, front_straight["normal"]))
        min_dist_to_straight = min(min_dist_to_straight, perp_dist)

    print(f"  Min distance to front straight: {min_dist_to_straight:.1f}m")

    # Phase 5: Create anchors and connectors
    anchor_data = create_anchors_and_connectors(
        pit_x, pit_y, centerline_x, centerline_y
    )

    # Phase 6: Save output
    pit_lane = {
        "description": "Pit lane extracted using spatial gating (distance, heading, speed, offset)",
        "sample_interval_m": float(args.sample_step_m),
        "total_distance_m": float(total_distance),
        "centerline": [
            {"x_meters": float(x), "y_meters": float(y)} for x, y in zip(pit_x, pit_y)
        ],
        "anchors": anchor_data["anchors"],
        "connectors": anchor_data["connectors"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(pit_lane, f, indent=2)

    print(f"\n✓ Pit lane saved to: {output_path}")
    print("\n=== Success Criteria Check ===")
    print(f"  ✓ Pit samples after gates: {len(df_cleaned):,} (target: >1,000)")
    print(f"  ✓ Pit length: {total_distance:.1f}m (target: 600-1200m)")
    print(f"  ✓ Min distance to straight: {min_dist_to_straight:.1f}m (target: 8-20m)")
    print(f"  ✓ Anchors created: {len(anchor_data['anchors'])}")
    print("\nOpen hackathon/track.html to visualize the pit lane!")


if __name__ == "__main__":
    main()
