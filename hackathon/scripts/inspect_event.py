# ABOUTME: Quick script to inspect specific events from telemetry
# ABOUTME: Shows raw signal values that triggered event detection

import sys
import json
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.geometry import (
    convert_gps_to_meters,
    load_centerline,
    project_points_onto_centerline,
)
from src.event_detection import detect_events

# Event to inspect (from lap_classifications.csv)
RACE = "R1"
VEHICLE = 0
LAP = 2
ZONE_ID = 8

# Load data
print(f"Inspecting: {RACE}, Vehicle {VEHICLE}, Lap {LAP}, Zone {ZONE_ID}")

# Load turn zones
with open("data/processed/turn_zones.json") as f:
    turn_zones = json.load(f)

zone = [z for z in turn_zones if z["zone_id"] == ZONE_ID][0]
print(f"\nZone: {zone['name']}")
print(f"Distance: {zone['start_distance_m']:.0f}m - {zone['end_distance_m']:.0f}m")

# Load centerline
centerline_x, centerline_y = load_centerline("data/processed/track_centerline.csv")

# Load telemetry for this specific lap
telemetry_path = f"data/input/{RACE}_barber_telemetry_data.csv"
needed_params = [
    "VBOX_Long_Minutes",
    "VBOX_Lat_Min",
    "accx_can",
    "accy_can",
    "aps",
    "Steering_Angle",
]

print(f"\nLoading telemetry for vehicle {VEHICLE}, lap {LAP}...")

# Read in chunks and filter
chunks = []
for chunk in pd.read_csv(telemetry_path, chunksize=50000):
    filtered = chunk[
        (chunk["vehicle_number"] == VEHICLE)
        & (chunk["lap"] == LAP)
        & (chunk["telemetry_name"].isin(needed_params))
    ]
    if len(filtered) > 0:
        chunks.append(filtered)

if not chunks:
    print("No data found!")
    sys.exit(1)

# Combine and pivot
df = pd.concat(chunks, ignore_index=True)
df_wide = df.pivot_table(
    index=["vehicle_number", "lap", "timestamp"],
    columns="telemetry_name",
    values="telemetry_value",
    aggfunc="first",
).reset_index()
df_wide.columns.name = None
df_wide = df_wide.dropna()

print(f"Loaded {len(df_wide)} samples for this lap")

# Convert GPS and project
df_wide = convert_gps_to_meters(df_wide)
track_distances = project_points_onto_centerline(
    df_wide["x_meters"].values,
    df_wide["y_meters"].values,
    centerline_x,
    centerline_y,
)
df_wide["track_distance_m"] = track_distances

# Detect events
df_wide = detect_events(df_wide, window_size=10)

# Filter to zone
zone_data = df_wide[
    (df_wide["track_distance_m"] >= zone["start_distance_m"])
    & (df_wide["track_distance_m"] <= zone["end_distance_m"])
].copy()

print(f"\n{len(zone_data)} samples in this zone")

# Show wheelspin events
wheelspin_samples = zone_data[zone_data["wheelspin_event"]]
print(f"\nWheelspin events detected: {len(wheelspin_samples)}")

if len(wheelspin_samples) > 0:
    print("\nShowing first wheelspin event:")
    print("=" * 80)
    event = wheelspin_samples.iloc[0]
    print(f"Timestamp: {event['timestamp']}")
    print(f"aps (throttle): {event['aps']:.1f}%")
    print(f"accx_can (forward G): {event['accx_can']:.3f}g")
    print(f"accy_can (lateral G): {event['accy_can']:.3f}g")
    print(f"Steering_Angle: {event['Steering_Angle']:.1f}°")

    # Show context (5 samples before and after)
    idx = zone_data[zone_data["wheelspin_event"]].index[0]
    context_start = max(0, zone_data.index.get_loc(idx) - 5)
    context_end = min(len(zone_data), zone_data.index.get_loc(idx) + 6)
    context = zone_data.iloc[context_start:context_end]

    print("\nContext (5 samples before and after):")
    print("=" * 80)
    print("Time | Throttle | AccX | AccY | Wheelspin?")
    print("-" * 80)
    for _, row in context.iterrows():
        ws = "***WHEELSPIN***" if row["wheelspin_event"] else ""
        print(
            f"{row['timestamp']} | {row['aps']:5.1f}% | {row['accx_can']:+.3f}g | {row['accy_can']:+.3f}g | {ws}"
        )
else:
    print("\nNo wheelspin events in this zone (threshold not met)")
