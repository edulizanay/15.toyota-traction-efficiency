# ABOUTME: Extend pit lane beginning to connect with track and apply aggressive smoothing
# ABOUTME: Fixes "squabbly" edges by smoothing boundaries and extending to proper connection point

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import json
import csv
from scipy.ndimage import gaussian_filter1d

from src.geometry import compute_normals


def load_pit_lane(input_path):
    """Load existing pit lane data."""
    print("=== Loading Existing Pit Lane ===")
    print(f"  Reading from: {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    x_coords = np.array([p["x_meters"] for p in data["centerline"]])
    y_coords = np.array([p["y_meters"] for p in data["centerline"]])

    print(f"  Loaded {len(x_coords)} centerline points")
    print(f"  Length: {data['total_distance_m']:.1f}m")

    return x_coords, y_coords, data


def load_track_centerline(centerline_path):
    """Load track centerline from CSV."""
    print("\n=== Loading Track Centerline ===")

    track_points = []
    with open(centerline_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_points.append(
                {"x": float(row["x_meters"]), "y": float(row["y_meters"])}
            )

    print(f"  Loaded {len(track_points)} track centerline points")
    return track_points


def extend_pit_lane_beginning(pit_x, pit_y, track_points, num_extension_points=8):
    """
    Extend pit lane beginning to connect smoothly with track.

    Args:
        pit_x: Existing pit lane X coordinates
        pit_y: Existing pit lane Y coordinates
        track_points: Track centerline points
        num_extension_points: Number of points to add at beginning

    Returns:
        Extended x, y coordinates
    """
    print("\n=== Extending Pit Lane Beginning ===")

    # Find closest track point to pit beginning
    pit_begin = np.array([pit_x[0], pit_y[0]])
    min_dist = float("inf")
    closest_idx = -1

    for i, tp in enumerate(track_points):
        dx = tp["x"] - pit_begin[0]
        dy = tp["y"] - pit_begin[1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    track_connect = track_points[closest_idx]
    print(f"  Closest track point: index={closest_idx}, dist={min_dist:.2f}m")
    print(f"    Track: ({track_connect['x']:.2f}, {track_connect['y']:.2f})")
    print(f"    Pit:   ({pit_begin[0]:.2f}, {pit_begin[1]:.2f})")

    # Calculate track tangent direction at connection point
    # Use points before and after to get direction
    window = 5
    track_before = track_points[max(0, closest_idx - window)]
    track_after = track_points[min(len(track_points) - 1, closest_idx + window)]
    track_direction = np.array(
        [track_after["x"] - track_before["x"], track_after["y"] - track_before["y"]]
    )
    track_direction = track_direction / np.linalg.norm(track_direction)

    # Calculate pit lane tangent direction at beginning
    # Use first few points to get initial direction
    pit_direction = np.array([pit_x[3] - pit_x[0], pit_y[3] - pit_y[0]])
    pit_direction = pit_direction / np.linalg.norm(pit_direction)

    print(
        f"  Track direction: angle={np.degrees(np.arctan2(track_direction[1], track_direction[0])):.1f}°"
    )
    print(
        f"  Pit direction: angle={np.degrees(np.arctan2(pit_direction[1], pit_direction[0])):.1f}°"
    )

    # Create smooth extension using cubic Bezier
    # We want to go from track_connect to pit_begin smoothly
    t = np.linspace(0, 1, num_extension_points)

    # Bezier curve control points
    p0 = np.array([track_connect["x"], track_connect["y"]])
    p3 = pit_begin

    # Control points that respect tangent directions
    # p1: Start from track, follow track direction
    control_dist = min_dist * 0.4
    p1 = p0 + track_direction * control_dist

    # p2: Arrive at pit, following pit direction backwards
    p2 = p3 - pit_direction * control_dist

    # Cubic Bezier: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    extension_x = (
        (1 - t) ** 3 * p0[0]
        + 3 * (1 - t) ** 2 * t * p1[0]
        + 3 * (1 - t) * t**2 * p2[0]
        + t**3 * p3[0]
    )

    extension_y = (
        (1 - t) ** 3 * p0[1]
        + 3 * (1 - t) ** 2 * t * p1[1]
        + 3 * (1 - t) * t**2 * p2[1]
        + t**3 * p3[1]
    )

    # Prepend extension to pit lane (excluding last point which overlaps with pit_x[0])
    extended_x = np.concatenate([extension_x[:-1], pit_x])
    extended_y = np.concatenate([extension_y[:-1], pit_y])

    print(f"  Added {num_extension_points - 1} extension points")
    print(f"  New total: {len(extended_x)} points")

    return extended_x, extended_y


def smooth_boundaries(inner_x, inner_y, outer_x, outer_y, sigma=3.0):
    """
    Apply Gaussian smoothing to boundary coordinates for smoother edges.

    Args:
        inner_x, inner_y: Inner boundary coordinates
        outer_x, outer_y: Outer boundary coordinates
        sigma: Gaussian kernel width (higher = smoother)

    Returns:
        Smoothed boundary coordinates
    """
    print("\n=== Smoothing Boundaries ===")
    print(f"  Applying Gaussian smoothing with sigma={sigma}")

    # Apply Gaussian smoothing to each coordinate array
    # mode='nearest' to avoid edge effects
    inner_x_smooth = gaussian_filter1d(inner_x, sigma=sigma, mode="nearest")
    inner_y_smooth = gaussian_filter1d(inner_y, sigma=sigma, mode="nearest")
    outer_x_smooth = gaussian_filter1d(outer_x, sigma=sigma, mode="nearest")
    outer_y_smooth = gaussian_filter1d(outer_y, sigma=sigma, mode="nearest")

    print("  Smoothing complete")

    return inner_x_smooth, inner_y_smooth, outer_x_smooth, outer_y_smooth


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


def save_complete_pit_lane(centerline_x, centerline_y, boundaries, output_path):
    """Save complete pit lane with boundaries to JSON."""

    dx = np.diff(centerline_x)
    dy = np.diff(centerline_y)
    total_length = np.sum(np.sqrt(dx**2 + dy**2))

    pit_lane_data = {
        "description": "Manually selected pit lane with extended beginning and smoothed boundaries",
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

    print(f"\n✓ Saved complete pit lane to: {output_path}")
    print(f"  Total length: {total_length:.1f}m")
    print(f"  Centerline points: {len(centerline_x)}")
    print(f"  Boundary points: {len(boundaries['inner']['x'])} per side")


def main():
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    print("=== Extend and Smooth Pit Lane ===\n")

    input_path = "data/processed/pit_lane.json"
    centerline_path = "data/processed/track_centerline.csv"
    output_path = "data/processed/pit_lane.json"

    # Step 1: Load existing pit lane
    pit_x, pit_y, original_data = load_pit_lane(input_path)

    # Step 2: Load track centerline
    track_points = load_track_centerline(centerline_path)

    # Step 3: Extend pit lane beginning
    extended_x, extended_y = extend_pit_lane_beginning(
        pit_x, pit_y, track_points, num_extension_points=8
    )

    # Step 4: Add boundaries
    boundaries = add_width_boundaries(extended_x, extended_y, half_width_m=6.0)

    # Step 5: Smooth boundaries
    (
        boundaries["inner"]["x"],
        boundaries["inner"]["y"],
        boundaries["outer"]["x"],
        boundaries["outer"]["y"],
    ) = smooth_boundaries(
        boundaries["inner"]["x"],
        boundaries["inner"]["y"],
        boundaries["outer"]["x"],
        boundaries["outer"]["y"],
        sigma=3.0,  # Aggressive smoothing
    )

    # Step 6: Save complete pit lane
    save_complete_pit_lane(extended_x, extended_y, boundaries, output_path)

    print("\n✓ Complete!")
    print("\nNext: Open hackathon/track.html to see the improved pit lane!")


if __name__ == "__main__":
    main()
