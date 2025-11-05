#!/usr/bin/env python3
# ABOUTME: Preprocesses RacetrackMarks.svg into clean JSON geometry
# ABOUTME: Converts complex SVG paths to simple coordinate arrays for runtime use

import re
import json
from pathlib import Path


def parse_svg_path(path_d):
    """Parse SVG path data into absolute coordinate commands"""
    commands = []
    current_x, current_y = 0, 0

    # Split path into commands
    tokens = re.findall(r"[MmLlCcZz]|[-+]?[0-9]*\.?[0-9]+", path_d)

    i = 0
    while i < len(tokens):
        cmd = tokens[i]

        if cmd in ["M", "m"]:  # MoveTo
            x = float(tokens[i + 1])
            y = float(tokens[i + 2])
            if cmd == "m":  # relative
                x += current_x
                y += current_y
            commands.append(("M", x, y))
            current_x, current_y = x, y
            i += 3

        elif cmd in ["L", "l"]:  # LineTo
            x = float(tokens[i + 1])
            y = float(tokens[i + 2])
            if cmd == "l":  # relative
                x += current_x
                y += current_y
            commands.append(("L", x, y))
            current_x, current_y = x, y
            i += 3

        elif cmd in ["C", "c"]:  # CurveTo (cubic bezier)
            # We'll approximate curves with their endpoints for simplicity
            x1 = float(tokens[i + 1])
            y1 = float(tokens[i + 2])
            x2 = float(tokens[i + 3])
            y2 = float(tokens[i + 4])
            x = float(tokens[i + 5])
            y = float(tokens[i + 6])
            if cmd == "c":  # relative
                x1 += current_x
                y1 += current_y
                x2 += current_x
                y2 += current_y
                x += current_x
                y += current_y
            commands.append(("C", x1, y1, x2, y2, x, y))
            current_x, current_y = x, y
            i += 7

        elif cmd in ["Z", "z"]:  # Close path
            commands.append(("Z",))
            i += 1
        else:
            i += 1

    return commands


def commands_to_points(commands, num_samples=50):
    """Convert path commands to point array, sampling curves"""
    points = []

    for cmd in commands:
        if cmd[0] == "M":
            points.append([cmd[1], cmd[2]])
        elif cmd[0] == "L":
            points.append([cmd[1], cmd[2]])
        elif cmd[0] == "C":
            # Sample cubic bezier curve
            x0, y0 = points[-1] if points else (cmd[1], cmd[2])
            x1, y1, x2, y2, x3, y3 = cmd[1:]
            for t in [i / num_samples for i in range(1, num_samples + 1)]:
                # Cubic bezier formula
                x = (
                    (1 - t) ** 3 * x0
                    + 3 * (1 - t) ** 2 * t * x1
                    + 3 * (1 - t) * t**2 * x2
                    + t**3 * x3
                )
                y = (
                    (1 - t) ** 3 * y0
                    + 3 * (1 - t) ** 2 * t * y1
                    + 3 * (1 - t) * t**2 * y2
                    + t**3 * y3
                )
                points.append([x, y])

    return points


def apply_svg_transform(points):
    """Apply the SVG's embedded transform: translate(0,239) scale(0.1,-0.1)

    The transform is applied as: translate first, then scale
    But in the SVG it's written as: translate(0,239) scale(0.1,-0.1)
    Matrix multiplication order: scale(point) then translate

    Actually, SVG transforms are: transform(x,y) = translate(scale(x,y))
    So: new_x = old_x * 0.1 + 0
        new_y = old_y * (-0.1) + 239
    """
    transformed = []
    for x, y in points:
        # Apply transform as per SVG: translate(scale(point))
        x_final = x * 0.1 + 0
        y_final = y * (-0.1) + 239
        transformed.append([x_final, y_final])
    return transformed


def rotate_180(points):
    """Rotate points 180 degrees around their center"""
    # Find center
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2

    # Rotate 180° around center
    rotated = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        # 180° rotation: negate both components
        x_rot = cx - dx
        y_rot = cy - dy
        rotated.append([x_rot, y_rot])

    return rotated


def normalize_to_meters(points, target_width_m=5.0):
    """Normalize points to target size in meters, centered at origin"""
    # Get bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y

    # Scale to target width
    scale = target_width_m / width if width > 0 else 1.0

    # Center at origin
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    normalized = []
    for x, y in points:
        x_norm = (x - cx) * scale
        y_norm = (y - cy) * scale
        normalized.append([round(x_norm, 4), round(y_norm, 4)])

    return normalized, width * scale, height * scale


def main():
    # Input and output paths
    script_dir = Path(__file__).parent
    svg_path = script_dir.parent / "data" / "assets" / "RacetrackMarks.svg"
    output_path = script_dir.parent / "data" / "assets" / "racetrack_mark_geometry.json"

    print(f"Reading SVG from: {svg_path}")

    # Read SVG content
    with open(svg_path, "r") as f:
        svg_content = f.read()

    # Extract path data (looking for the 3 stripe paths)
    path_pattern = r'<path d="([^"]+)"'
    path_matches = re.findall(path_pattern, svg_content)

    print(f"Found {len(path_matches)} paths in SVG")

    # Process each stripe
    stripes = []
    for i, path_d in enumerate(path_matches):
        print(f"\nProcessing stripe {i + 1}...")

        # Parse path commands
        commands = parse_svg_path(path_d)
        print(f"  - Parsed {len(commands)} commands")

        # Convert to points with more samples for smooth curves
        points = commands_to_points(commands, num_samples=10)
        print(f"  - Generated {len(points)} points")

        # Apply SVG transform to get to viewBox coordinates (300×239)
        points = apply_svg_transform(points)

        # Rotate 180° to correct the direction
        points = rotate_180(points)

        # Normalize to meters (target width will be the target size)
        points, width, height = normalize_to_meters(points, target_width_m=4.0)
        print(f"  - Normalized to {width:.2f}m × {height:.2f}m")

        stripes.append({"points": points})

    # Calculate overall bounding box
    all_points = [p for stripe in stripes for p in stripe["points"]]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    geometry = {
        "stripes": stripes,
        "boundingBox": {
            "width": round(max(xs) - min(xs), 2),
            "height": round(max(ys) - min(ys), 2),
        },
        "metadata": {
            "description": "Preprocessed racetrack mark geometry (3 diagonal stripes)",
            "units": "meters",
            "preprocessed_from": "RacetrackMarks.svg",
            "orientation": "Rotated 180° from original, ready to align with track direction",
        },
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geometry, f, indent=2)

    print(f"\n✓ Geometry written to: {output_path}")
    print(
        f"  Bounding box: {geometry['boundingBox']['width']}m × {geometry['boundingBox']['height']}m"
    )
    print(f"  Total stripes: {len(stripes)}")


if __name__ == "__main__":
    main()
