# ABOUTME: Generate track centerline and boundaries from telemetry GPS data  
# ABOUTME: Combines R1 + R2 telemetry for better statistical confidence

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from geometry import (
    convert_gps_to_meters,
    compute_centerline,
    save_centerline,
    compute_track_boundaries,
    save_track_boundaries,
)

print("=" * 80)
print("TRACK CENTERLINE GENERATION")
print("=" * 80)
print()

# Paths
INPUT_DIR = Path(__file__).parent.parent / "data" / "input"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

print("Step 1: Loading telemetry data (this may take a minute)...")
print("-" * 80)

# Load R1 telemetry (GPS data only)
print("Loading R1 telemetry...")
r1_chunks = []
for chunk in pd.read_csv(
    INPUT_DIR / "R1_barber_telemetry_data.csv",
    chunksize=500000,
):
    # Filter to GPS parameters only
    gps_chunk = chunk[
        chunk["telemetry_name"].isin(["VBOX_Long_Minutes", "VBOX_Lat_Min"])
    ].copy()

    if len(gps_chunk) > 0:
        # Pivot to get one row per (vehicle, lap, timestamp)
        pivoted = gps_chunk.pivot_table(
            index=["vehicle_number", "lap", "timestamp"],
            columns="telemetry_name",
            values="telemetry_value",
            aggfunc="first",
        ).reset_index()

        r1_chunks.append(pivoted)

r1_gps = pd.concat(r1_chunks, ignore_index=True)
print(f"  R1: {len(r1_gps):,} GPS samples")

# Load R2 telemetry (GPS data only)
print("Loading R2 telemetry...")
r2_chunks = []
for chunk in pd.read_csv(
    INPUT_DIR / "R2_barber_telemetry_data.csv",
    chunksize=500000,
):
    # Filter to GPS parameters only
    gps_chunk = chunk[
        chunk["telemetry_name"].isin(["VBOX_Long_Minutes", "VBOX_Lat_Min"])
    ].copy()

    if len(gps_chunk) > 0:
        # Pivot to get one row per (vehicle, lap, timestamp)
        pivoted = gps_chunk.pivot_table(
            index=["vehicle_number", "lap", "timestamp"],
            columns="telemetry_name",
            values="telemetry_value",
            aggfunc="first",
        ).reset_index()

        r2_chunks.append(pivoted)

r2_gps = pd.concat(r2_chunks, ignore_index=True)
print(f"  R2: {len(r2_gps):,} GPS samples")

# Use R1 only for centerline (using both R1+R2 would double-trace the track)
# Note: We'll use combined R1+R2 later for envelope construction
print()

print("Step 2: Converting GPS to UTM meters...")
print("-" * 80)
r1_gps = convert_gps_to_meters(r1_gps)
print(f"✓ Converted {len(r1_gps):,} points to UTM coordinates")
print()

print("Step 3: Generating track centerline...")
print("-" * 80)
print("Using R1 only (avoiding double-trace from R1+R2)")
centerline_x, centerline_y = compute_centerline(r1_gps)
print()

print("Step 4: Saving centerline...")
print("-" * 80)
save_centerline(centerline_x, centerline_y, OUTPUT_DIR / "track_centerline.csv")
print()

print("Step 5: Computing track boundaries...")
print("-" * 80)
print("Using 12m track width")
inner_x, inner_y, outer_x, outer_y = compute_track_boundaries(
    centerline_x, centerline_y, track_width_m=12.0
)
print(f"✓ Computed inner/outer boundaries ({len(inner_x)} points each)")
print()

print("Step 6: Saving track boundaries...")
print("-" * 80)
save_track_boundaries(
    inner_x, inner_y, outer_x, outer_y, OUTPUT_DIR / "track_boundaries.json"
)
print()

print("=" * 80)
print("COMPLETE!")
print("=" * 80)
print()
print(f"Track centerline saved to: {OUTPUT_DIR / 'track_centerline.csv'}")
print(f"  Points: {len(centerline_x)}")
print(f"  X range: {centerline_x.min():.1f}m to {centerline_x.max():.1f}m")
print(f"  Y range: {centerline_y.min():.1f}m to {centerline_y.max():.1f}m")
print()
print(f"Track boundaries saved to: {OUTPUT_DIR / 'track_boundaries.json'}")
print(f"  Track width: 12.0m")
print()
print("Next: Open http://localhost:8000/dashboard.html to visualize the track!")
print()
