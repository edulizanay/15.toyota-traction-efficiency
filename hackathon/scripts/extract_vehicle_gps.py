# ABOUTME: Extract GPS data for specific vehicle to identify pit stops visually
# ABOUTME: Creates JSON output for web visualization

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json
from pyproj import Transformer


def extract_vehicle_gps(telemetry_file, vehicle_number):
    """Extract all GPS data for a specific vehicle."""
    print(f"\n=== Extracting GPS for Vehicle #{vehicle_number} ===")
    print(f"  Reading from: {telemetry_file}")

    gps_data = []
    chunk_size = 100000

    for chunk_num, chunk in enumerate(
        pd.read_csv(telemetry_file, chunksize=chunk_size)
    ):
        if chunk_num % 10 == 0:
            print(f"  Processing chunk {chunk_num}...")

        # Filter for this vehicle and GPS telemetry only
        gps_chunk = chunk[
            (chunk["vehicle_number"] == vehicle_number)
            & (chunk["telemetry_name"].isin(["VBOX_Long_Minutes", "VBOX_Lat_Min"]))
        ]

        if len(gps_chunk) > 0:
            gps_data.append(gps_chunk)

    if not gps_data:
        raise ValueError(f"No GPS data found for vehicle #{vehicle_number}")

    all_gps = pd.concat(gps_data, ignore_index=True)
    print(f"  Found {len(all_gps)} GPS telemetry points")

    # Pivot to get lat/lon in columns
    gps_pivot = all_gps.pivot_table(
        index=["vehicle_number", "lap", "timestamp"],
        columns="telemetry_name",
        values="telemetry_value",
    ).reset_index()

    # Sort by lap and timestamp
    gps_pivot = gps_pivot.sort_values(["lap", "timestamp"])

    print(f"  Result: {len(gps_pivot)} GPS coordinate pairs")
    print(f"  Laps: {gps_pivot['lap'].min()} to {gps_pivot['lap'].max()}")

    return gps_pivot


def convert_to_utm(gps_df):
    """Convert GPS to UTM coordinates."""
    print("\n=== Converting to UTM ===")

    transformer = Transformer.from_crs(
        "EPSG:4326",  # WGS84
        "EPSG:32616",  # UTM Zone 16N
        always_xy=True,
    )

    easting = []
    northing = []

    for idx, row in gps_df.iterrows():
        e, n = transformer.transform(row["VBOX_Long_Minutes"], row["VBOX_Lat_Min"])
        easting.append(e)
        northing.append(n)

    gps_df["x_meters"] = easting
    gps_df["y_meters"] = northing

    print(f"  X range: {min(easting):.1f} to {max(easting):.1f}m")
    print(f"  Y range: {min(northing):.1f} to {max(northing):.1f}m")

    return gps_df


def extract_all_coordinates(gps_df):
    """Extract all GPS coordinates (no splitting)."""
    print("\n=== Extracting All Coordinates ===")

    coordinates = [
        {"x": float(row["x_meters"]), "y": float(row["y_meters"])}
        for _, row in gps_df.iterrows()
    ]

    print(f"  Total coordinates: {len(coordinates)}")

    return coordinates


def save_visualization_data(coordinates, vehicle_number, output_path):
    """Save data for web visualization."""

    output = {
        "vehicle_number": vehicle_number,
        "total_points": len(coordinates),
        "coordinates": coordinates,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved visualization data to: {output_path}")
    print(f"  Vehicle: #{vehicle_number}")
    print(f"  Total points: {len(coordinates)}")


def main():
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    print("=== Vehicle GPS Extraction ===\n")

    telemetry_file = "data/input/R1_barber_telemetry_data.csv"
    vehicle_number = 46  # Car that made a pit stop
    output_path = "data/processed/vehicle_46_gps.json"

    # Step 1: Extract GPS data
    gps_df = extract_vehicle_gps(telemetry_file, vehicle_number)

    # Step 2: Convert to UTM
    gps_df = convert_to_utm(gps_df)

    # Step 3: Extract all coordinates
    coordinates = extract_all_coordinates(gps_df)

    # Step 4: Save for visualization
    save_visualization_data(coordinates, vehicle_number, output_path)

    print("\n✓ Extraction complete!")
    print("\nNext: Open vehicle_13_viewer.html to visualize")


if __name__ == "__main__":
    main()
