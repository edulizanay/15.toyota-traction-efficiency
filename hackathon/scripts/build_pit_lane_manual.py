# ABOUTME: Create proper U-shaped pit lane parallel to front straight
# ABOUTME: Identifies front straight and constructs pit lane with correct geometry

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import json

from src.geometry import compute_normals


def load_track_centerline(centerline_path):
    """Load track centerline from CSV."""
    print("=== Loading Track Centerline ===")
    centerline_df = pd.read_csv(centerline_path)
    x_coords = centerline_df["x_meters"].values
    y_coords = centerline_df["y_meters"].values

    print(f"  Loaded {len(x_coords)} centerline points")
    print(f"  X range: {x_coords.min():.1f} to {x_coords.max():.1f}m")
    print(f"  Y range: {y_coords.min():.1f} to {y_coords.max():.1f}m")

    return x_coords, y_coords


def identify_front_straight(centerline_x, centerline_y, min_length_m=200):
    """
    Identify the front straight section of the track.

    The front straight is typically:
    - Near the maximum Y coordinate (northernmost section)
    - Has relatively consistent direction (low curvature)
    - Is sufficiently long

    Args:
        centerline_x: Track centerline X coordinates
        centerline_y: Track centerline Y coordinates
        min_length_m: Minimum length to consider as a straight

    Returns:
        dict with start_idx, end_idx, and direction info
    """
    print("\n=== Identifying Front Straight ===")

    # Find points in the upper portion of the track (top 20% by Y coordinate)
    y_threshold = centerline_y.min() + (centerline_y.max() - centerline_y.min()) * 0.80
    upper_mask = centerline_y >= y_threshold
    upper_indices = np.where(upper_mask)[0]

    print(
        f"  Found {len(upper_indices)} points in upper section (Y >= {y_threshold:.1f}m)"
    )

    # Look for the longest relatively straight section in this region
    # We'll measure "straightness" by looking at heading consistency
    best_straight = None
    best_length = 0

    # Scan through upper section looking for straight segments
    window_size = 50  # Look at 50-point windows

    for i in range(len(upper_indices) - window_size):
        start_idx = upper_indices[i]
        end_idx = upper_indices[min(i + window_size, len(upper_indices) - 1)]

        # Get segment points
        seg_x = centerline_x[start_idx : end_idx + 1]
        seg_y = centerline_y[start_idx : end_idx + 1]

        # Calculate segment length
        dx = np.diff(seg_x)
        dy = np.diff(seg_y)
        seg_length = np.sum(np.sqrt(dx**2 + dy**2))

        if seg_length < min_length_m:
            continue

        # Calculate heading variation (measure of straightness)
        headings = np.arctan2(dy, dx)
        heading_std = np.std(headings)

        # Lower heading variation = straighter
        # Weight by length
        straightness_score = seg_length / (1 + heading_std * 10)

        if straightness_score > best_length:
            best_length = straightness_score
            best_straight = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "length_m": seg_length,
                "heading_std": heading_std,
                "avg_heading": np.mean(headings),
            }

    if best_straight is None:
        # Fallback: just use the first continuous section in upper region
        best_straight = {
            "start_idx": upper_indices[0],
            "end_idx": upper_indices[min(window_size, len(upper_indices) - 1)],
        }
        seg_x = centerline_x[best_straight["start_idx"] : best_straight["end_idx"] + 1]
        seg_y = centerline_y[best_straight["start_idx"] : best_straight["end_idx"] + 1]
        dx = np.diff(seg_x)
        dy = np.diff(seg_y)
        best_straight["length_m"] = np.sum(np.sqrt(dx**2 + dy**2))
        best_straight["avg_heading"] = np.mean(np.arctan2(dy, dx))

    print("  Identified front straight:")
    print(f"    Indices: {best_straight['start_idx']} to {best_straight['end_idx']}")
    print(f"    Length: {best_straight['length_m']:.1f}m")
    print(f"    Avg heading: {np.degrees(best_straight['avg_heading']):.1f}°")

    return best_straight


def create_parallel_pit_lane(
    centerline_x,
    centerline_y,
    straight_info,
    offset_m=12.0,
    length_m=350.0,
    num_points=120,
):
    """
    Create a pit lane parallel to the identified straight.

    Args:
        centerline_x: Track centerline X coordinates
        centerline_y: Track centerline Y coordinates
        straight_info: Info about the front straight section
        offset_m: Perpendicular distance from track (positive = right side)
        length_m: Length of pit lane main section
        num_points: Number of points in pit lane

    Returns:
        pit_x, pit_y: Coordinates of pit lane centerline
    """
    print("\n=== Creating Parallel Pit Lane ===")

    # Get the straight section
    start_idx = straight_info["start_idx"]
    end_idx = straight_info["end_idx"]

    straight_x = centerline_x[start_idx : end_idx + 1]
    straight_y = centerline_y[start_idx : end_idx + 1]

    # Calculate average direction of the straight
    dx = straight_x[-1] - straight_x[0]
    dy = straight_y[-1] - straight_y[0]
    avg_heading = np.arctan2(dy, dx)

    # Calculate perpendicular direction (to the right)
    perp_heading = avg_heading - np.pi / 2  # 90° clockwise

    # Create pit lane points parallel to straight
    # Start slightly before the straight, end slightly after
    start_along = -20.0  # Start 20m before straight
    end_along = length_m

    along_distances = np.linspace(start_along, end_along, num_points)

    # Base point (start of straight)
    base_x = straight_x[0]
    base_y = straight_y[0]

    # Create pit lane parallel to straight, offset perpendicular
    pit_x = (
        base_x + along_distances * np.cos(avg_heading) + offset_m * np.cos(perp_heading)
    )
    pit_y = (
        base_y + along_distances * np.sin(avg_heading) + offset_m * np.sin(perp_heading)
    )

    # Calculate actual length
    dx_pit = np.diff(pit_x)
    dy_pit = np.diff(pit_y)
    actual_length = np.sum(np.sqrt(dx_pit**2 + dy_pit**2))

    print("  Created pit lane:")
    print(f"    Points: {len(pit_x)}")
    print(f"    Length: {actual_length:.1f}m")
    print(f"    Offset: {offset_m:.1f}m from track")
    print(f"    Direction: {np.degrees(avg_heading):.1f}°")
    print(f"    X range: {pit_x.min():.1f} to {pit_x.max():.1f}m")
    print(f"    Y range: {pit_y.min():.1f} to {pit_y.max():.1f}m")

    return pit_x, pit_y


def create_curved_connector(start_point, end_point, num_points=20):
    """
    Create smooth curved connector between two points using cubic Bézier.

    Args:
        start_point: (x, y) starting point
        end_point: (x, y) ending point
        num_points: Number of points in curve

    Returns:
        Arrays of x, y coordinates
    """
    x0, y0 = start_point
    x3, y3 = end_point

    # Create control points for smooth curve
    # Control point 1: 1/3 along path, shifted perpendicular
    dx = x3 - x0
    dy = y3 - y0

    x1 = x0 + dx * 0.33
    y1 = y0 + dy * 0.3

    # Control point 2: 2/3 along path
    x2 = x0 + dx * 0.67
    y2 = y0 + dy * 0.7

    # Generate curve using Bézier formula
    t = np.linspace(0, 1, num_points)

    # Cubic Bézier: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    x_curve = (
        (1 - t) ** 3 * x0
        + 3 * (1 - t) ** 2 * t * x1
        + 3 * (1 - t) * t**2 * x2
        + t**3 * x3
    )

    y_curve = (
        (1 - t) ** 3 * y0
        + 3 * (1 - t) ** 2 * t * y1
        + 3 * (1 - t) * t**2 * y2
        + t**3 * y3
    )

    return x_curve, y_curve


def find_connection_points(
    pit_x, pit_y, centerline_x, centerline_y, max_search_dist=100
):
    """
    Find connection points on track centerline for pit lane endpoints.

    Args:
        pit_x: Pit lane X coordinates
        pit_y: Pit lane Y coordinates
        centerline_x: Track centerline X coordinates
        centerline_y: Track centerline Y coordinates
        max_search_dist: Maximum distance to search for connection

    Returns:
        dict with pit_in and pit_out connection info
    """
    print("\n=== Finding Track Connection Points ===")

    # Build KD-tree for centerline
    centerline_points = np.column_stack([centerline_x, centerline_y])
    tree = KDTree(centerline_points)

    # Compute cumulative distance along centerline
    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    cumulative_distance = np.concatenate([[0], np.cumsum(segment_lengths)])

    connections = {}

    # Pit-in: left endpoint (first point)
    pit_in_point = np.array([pit_x[0], pit_y[0]])
    dist, idx = tree.query(pit_in_point)

    if dist < max_search_dist:
        connections["pit_in"] = {
            "pit_point": (pit_x[0], pit_y[0]),
            "track_point": (centerline_x[idx], centerline_y[idx]),
            "centerline_idx": int(idx),
            "track_distance_m": float(cumulative_distance[idx]),
            "separation_m": float(dist),
        }
        print(f"  Pit-in: found connection at {dist:.1f}m (centerline[{idx}])")
    else:
        print(
            f"  Warning: Pit-in too far from track ({dist:.1f}m > {max_search_dist}m)"
        )

    # Pit-out: right endpoint (last point)
    pit_out_point = np.array([pit_x[-1], pit_y[-1]])
    dist, idx = tree.query(pit_out_point)

    if dist < max_search_dist:
        connections["pit_out"] = {
            "pit_point": (pit_x[-1], pit_y[-1]),
            "track_point": (centerline_x[idx], centerline_y[idx]),
            "centerline_idx": int(idx),
            "track_distance_m": float(cumulative_distance[idx]),
            "separation_m": float(dist),
        }
        print(f"  Pit-out: found connection at {dist:.1f}m (centerline[{idx}])")
    else:
        print(
            f"  Warning: Pit-out too far from track ({dist:.1f}m > {max_search_dist}m)"
        )

    return connections


def build_complete_pit_lane(pit_x, pit_y, connections):
    """
    Build complete pit lane with curved connectors at endpoints.

    Args:
        pit_x: Pit lane centerline X coordinates
        pit_y: Pit lane centerline Y coordinates
        connections: Connection point info

    Returns:
        Complete x, y coordinates including connectors, and updated connections
    """
    print("\n=== Building Complete Pit Lane ===")

    # Start with pit-in connector (if exists)
    if "pit_in" in connections:
        conn_x, conn_y = create_curved_connector(
            connections["pit_in"]["track_point"],
            connections["pit_in"]["pit_point"],
            num_points=20,
        )
        pit_in_x = conn_x
        pit_in_y = conn_y
        print(f"  Created pit-in connector: {len(pit_in_x)} points")
    else:
        pit_in_x = np.array([])
        pit_in_y = np.array([])

    # Add main pit lane centerline
    main_x = pit_x
    main_y = pit_y

    # Add pit-out connector (if exists)
    if "pit_out" in connections:
        conn_x, conn_y = create_curved_connector(
            connections["pit_out"]["pit_point"],
            connections["pit_out"]["track_point"],
            num_points=20,
        )
        pit_out_x = conn_x
        pit_out_y = conn_y
        print(f"  Created pit-out connector: {len(pit_out_x)} points")
    else:
        pit_out_x = np.array([])
        pit_out_y = np.array([])

    # Concatenate all sections
    complete_x = np.concatenate([pit_in_x, main_x, pit_out_x])
    complete_y = np.concatenate([pit_in_y, main_y, pit_out_y])

    # Calculate total length
    dx = np.diff(complete_x)
    dy = np.diff(complete_y)
    total_length = np.sum(np.sqrt(dx**2 + dy**2))

    print(
        f"  Complete pit lane: {len(complete_x)} points, {total_length:.1f}m total length"
    )

    return complete_x, complete_y, connections


def add_width_boundaries(centerline_x, centerline_y, half_width_m=6.0):
    """
    Create inner and outer boundaries for pit lane using normal offsets.

    Args:
        centerline_x: Centerline X coordinates
        centerline_y: Centerline Y coordinates
        half_width_m: Half-width of pit lane (default 6m for 12m total)

    Returns:
        dict with inner and outer boundary coordinates
    """
    print("\n=== Adding Width Boundaries ===")

    # Compute normal vectors at each point
    normals = compute_normals(centerline_x, centerline_y)

    # Create inner boundary (offset to the left/inside)
    inner_x = centerline_x - normals[:, 0] * half_width_m
    inner_y = centerline_y - normals[:, 1] * half_width_m

    # Create outer boundary (offset to the right/outside)
    outer_x = centerline_x + normals[:, 0] * half_width_m
    outer_y = centerline_y + normals[:, 1] * half_width_m

    print(f"  Created boundaries with {half_width_m * 2}m total width")

    return {
        "inner": {"x": inner_x, "y": inner_y},
        "outer": {"x": outer_x, "y": outer_y},
    }


def save_pit_lane_data(
    centerline_x, centerline_y, boundaries, connections, output_path
):
    """Save complete pit lane data to JSON."""

    # Calculate total length
    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    total_length = np.sum(np.sqrt(dx**2 + dy**2))

    # Build output structure
    pit_lane_data = {
        "description": "U-shaped pit lane parallel to front straight",
        "total_distance_m": float(total_length),
        "sample_interval_m": 2.6,
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

    # Add connection anchors if they exist
    if "pit_in" in connections:
        pit_lane_data["anchors"]["pit_in"] = {
            "centerline_idx": connections["pit_in"]["centerline_idx"],
            "track_distance_m": connections["pit_in"]["track_distance_m"],
            "x_meters": connections["pit_in"]["track_point"][0],
            "y_meters": connections["pit_in"]["track_point"][1],
        }

    if "pit_out" in connections:
        pit_lane_data["anchors"]["pit_out"] = {
            "centerline_idx": connections["pit_out"]["centerline_idx"],
            "track_distance_m": connections["pit_out"]["track_distance_m"],
            "x_meters": connections["pit_out"]["track_point"][0],
            "y_meters": connections["pit_out"]["track_point"][1],
        }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(pit_lane_data, f, indent=2)

    print(f"\n✓ Saved pit lane data to: {output_path}")
    print(f"  Total length: {total_length:.1f}m")
    print(f"  Centerline points: {len(centerline_x)}")
    print(f"  Boundary points: {len(boundaries['inner']['x'])} per side")
    print(f"  Anchors: {len(pit_lane_data['anchors'])}")


def main():
    # Set working directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    print("=== Pit Lane Construction: U-Shape Parallel to Front Straight ===\n")

    # Paths
    centerline_path = "data/processed/track_centerline.csv"
    output_path = "data/processed/pit_lane.json"

    # Step 1: Load track centerline
    centerline_x, centerline_y = load_track_centerline(centerline_path)

    # Step 2: Identify front straight
    straight_info = identify_front_straight(
        centerline_x, centerline_y, min_length_m=200
    )

    # Step 3: Create parallel pit lane
    pit_x, pit_y = create_parallel_pit_lane(
        centerline_x,
        centerline_y,
        straight_info,
        offset_m=12.0,  # 12m to the right of track
        length_m=200.0,  # 200m long pit lane
        num_points=120,  # Dense sampling
    )

    # Step 4: Find connection points
    connections = find_connection_points(pit_x, pit_y, centerline_x, centerline_y)

    # Step 5: Build complete pit lane with connectors
    complete_x, complete_y, connections = build_complete_pit_lane(
        pit_x, pit_y, connections
    )

    # Step 6: Add width boundaries
    boundaries = add_width_boundaries(complete_x, complete_y, half_width_m=6.0)

    # Step 7: Save output
    save_pit_lane_data(complete_x, complete_y, boundaries, connections, output_path)

    print("\n✓ Pit lane construction complete!")
    print("\nNext step: Open hackathon/track.html to visualize the pit lane.")


if __name__ == "__main__":
    main()
