# ABOUTME: Add width boundaries to manually selected pit lane centerline
# ABOUTME: Converts pit_lane_manual.json to complete pit_lane.json format

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import json

from src.geometry import compute_normals


def load_manual_pit_lane(input_path):
    """Load manually selected pit lane centerline."""
    print("\n=== Loading Manual Pit Lane ===")
    print(f"  Reading from: {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    x_coords = np.array([p["x_meters"] for p in data["centerline"]])
    y_coords = np.array([p["y_meters"] for p in data["centerline"]])

    print(f"  Loaded {len(x_coords)} centerline points")

    # Remove duplicate consecutive points
    print("\n=== Removing Duplicate Points ===")
    unique_mask = np.ones(len(x_coords), dtype=bool)
    for i in range(1, len(x_coords)):
        if x_coords[i] == x_coords[i - 1] and y_coords[i] == y_coords[i - 1]:
            unique_mask[i] = False

    x_coords = x_coords[unique_mask]
    y_coords = y_coords[unique_mask]

    duplicates_removed = np.sum(~unique_mask)
    print(f"  Removed {duplicates_removed} duplicate points")
    print(f"  Remaining: {len(x_coords)} unique points")

    # Recalculate length
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    total_length = np.sum(np.sqrt(dx**2 + dy**2))
    print(f"  Updated length: {total_length:.1f}m")

    return x_coords, y_coords, data


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
        "description": "Manually selected pit lane from vehicle #46 GPS trace",
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

    print("=== Add Boundaries to Manual Pit Lane ===\n")

    # Use the file from Downloads
    input_path = "/Users/eduardolizana/Downloads/pit_lane_manual.json"
    output_path = "data/processed/pit_lane.json"

    # Step 1: Load manual pit lane
    centerline_x, centerline_y, original_data = load_manual_pit_lane(input_path)

    # Step 2: Add boundaries
    boundaries = add_width_boundaries(centerline_x, centerline_y, half_width_m=6.0)

    # Step 3: Save complete pit lane
    save_complete_pit_lane(centerline_x, centerline_y, boundaries, output_path)

    print("\n✓ Complete!")
    print("\nNext: Open hackathon/track.html to see the pit lane!")


if __name__ == "__main__":
    main()
