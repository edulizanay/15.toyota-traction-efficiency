# ABOUTME: Diagnostic script to trace why specific corners are flagged as aggressive
# ABOUTME: Loads telemetry and walks through event detection logic step-by-step

import pandas as pd
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.event_detection import (
    WHEELSPIN_APS_TREND_THRESHOLD,
    WHEELSPIN_ACCX_TREND_THRESHOLD,
    WHEELSPIN_ACCX_MIN,
    UNDERSTEER_STEER_TREND_THRESHOLD,
    UNDERSTEER_STEER_ABS_MIN,
    UNDERSTEER_RESPONSIVENESS_THRESHOLD,
    UNDERSTEER_ACCY_ABS_MIN,
    OVERSTEER_ACCY_SPIKE_THRESHOLD,
    OVERSTEER_ACCX_DROP_THRESHOLD,
    OVERSTEER_ACCY_ABS_MIN,
    MACRO_EVENT_COVERAGE_THRESHOLD,
    MACRO_EVENT_RUN_LENGTH_THRESHOLD,
)

# Load classifications
script_dir = Path(__file__).parent.parent
classifications_path = script_dir / "data/processed/lap_classifications.csv"
classifications = pd.read_csv(classifications_path)

print("\n=== Classification Distribution ===")
print(classifications["classification"].value_counts())
print(f"\nTotal zones: {len(classifications)}")

# Find aggressive corners
aggressive = classifications[classifications["classification"] == "Aggressive"]
print(f"\n=== Aggressive Zones: {len(aggressive)} ===")

# Pick a specific example - let's find one with understeer
understeer_examples = aggressive[aggressive["understeer"] == True]
if len(understeer_examples) > 0:
    example = understeer_examples.iloc[0]
    print("\nSelected Example (Understeer):")
    print(f"  Race: {example['race']}")
    print(f"  Vehicle: {example['vehicle_number']}")
    print(f"  Lap: {example['lap']}")
    print(f"  Zone: {example['zone_name']} (Zone {example['zone_id']})")
    print(f"  Classification: {example['classification']}")
    print(f"  Wheelspin: {example['wheelspin']}")
    print(f"  Understeer: {example['understeer']}")
    print(f"  Oversteer: {example['oversteer']}")
    print(f"  Event Coverage: {float(example['event_coverage']) * 100:.1f}%")
    print(f"  Max Event Run: {int(example['max_event_run_len'])} samples")
    print(f"  Avg Utilization: {float(example['avg_utilization']) * 100:.1f}%")
    print(f"  Sample Count: {int(example['sample_count'])}")

    # Now we need to load the actual telemetry for this corner
    # This requires loading the full telemetry data
    print("\n=== Loading Full Telemetry ===")
    print("(This may take a moment...)")

    # Determine which telemetry file to load
    race_name = example['race']
    telemetry_path = script_dir / f"data/input/{race_name}_barber_telemetry_data.csv"

    if not telemetry_path.exists():
        print(f"ERROR: Telemetry file not found: {telemetry_path}")
        sys.exit(1)

    # Load only the data we need (filter by vehicle and lap first)
    needed_params = ["accx_can", "accy_can", "aps", "Steering_Angle", "VBOX_Long_Minutes", "VBOX_Lat_Min"]

    print(f"Loading telemetry for Vehicle {example['vehicle_number']}, Lap {example['lap']}...")

    # Read in chunks and filter
    chunks = []
    for chunk in pd.read_csv(telemetry_path, chunksize=50000):
        # Filter to our vehicle and lap
        filtered = chunk[
            (chunk["vehicle_number"] == example['vehicle_number']) &
            (chunk["lap"] == example['lap']) &
            (chunk["telemetry_name"].isin(needed_params))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if len(chunks) == 0:
        print("ERROR: No telemetry found for this corner")
        sys.exit(1)

    telemetry = pd.concat(chunks, ignore_index=True)

    # Pivot to wide format
    telemetry["seq"] = telemetry.groupby(["vehicle_number", "lap", "timestamp", "telemetry_name"]).cumcount()
    telemetry_wide = telemetry.pivot_table(
        index=["vehicle_number", "lap", "timestamp", "seq"],
        columns="telemetry_name",
        values="telemetry_value",
        aggfunc="first",
    ).reset_index()
    telemetry_wide.columns.name = None
    telemetry_wide = telemetry_wide.drop(columns=["seq"])
    telemetry_wide = telemetry_wide.dropna()

    print(f"Loaded {len(telemetry_wide)} telemetry samples")

    # Convert GPS to meters
    from src.geometry import convert_gps_to_meters, load_centerline, project_points_onto_centerline

    telemetry_wide = convert_gps_to_meters(telemetry_wide)

    # Load turn zones
    turn_zones_path = script_dir / "data/processed/turn_zones.json"
    with open(turn_zones_path) as f:
        turn_zones = json.load(f)

    # Find the zone definition
    zone_def = next(z for z in turn_zones if z["zone_id"] == example['zone_id'])

    # Project to track distance
    centerline_path = script_dir / "data/processed/track_centerline.csv"
    centerline_x, centerline_y = load_centerline(centerline_path)
    track_distances = project_points_onto_centerline(
        telemetry_wide["x_meters"].values,
        telemetry_wide["y_meters"].values,
        centerline_x,
        centerline_y,
    )
    telemetry_wide["track_distance"] = track_distances

    # Filter to zone
    zone_data = telemetry_wide[
        (telemetry_wide["track_distance"] >= zone_def["start_distance_m"]) &
        (telemetry_wide["track_distance"] <= zone_def["end_distance_m"])
    ].copy()

    print(f"\n=== Zone Telemetry: {len(zone_data)} samples ===")

    # Sort by timestamp
    zone_data = zone_data.sort_values("timestamp").reset_index(drop=True)

    # Now let's manually run the event detection logic
    print("\n=== Running Event Detection ===")

    window_size = 10

    # Compute rolling trends
    aps_trend = zone_data["aps"].rolling(window_size).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    accx_trend = zone_data["accx_can"].rolling(window_size).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    steer_trend = zone_data["Steering_Angle"].abs().rolling(window_size).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    accy_trend = zone_data["accy_can"].abs().rolling(window_size).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)

    # Detect wheelspin
    zone_data["wheelspin_event"] = (
        (aps_trend > WHEELSPIN_APS_TREND_THRESHOLD) &
        (accx_trend < WHEELSPIN_ACCX_TREND_THRESHOLD) &
        (zone_data["accx_can"] > WHEELSPIN_ACCX_MIN)
    )

    # Detect understeer
    accy_responsiveness = accy_trend / (steer_trend + 0.001)
    zone_data["understeer_event"] = (
        (steer_trend > UNDERSTEER_STEER_TREND_THRESHOLD) &
        (zone_data["Steering_Angle"].abs() >= UNDERSTEER_STEER_ABS_MIN) &
        (accy_responsiveness < UNDERSTEER_RESPONSIVENESS_THRESHOLD) &
        (zone_data["accy_can"].abs() > UNDERSTEER_ACCY_ABS_MIN)
    )

    # Detect oversteer
    accy_spike = zone_data["accy_can"].abs().diff() > OVERSTEER_ACCY_SPIKE_THRESHOLD
    accx_drop = zone_data["accx_can"].diff() < OVERSTEER_ACCX_DROP_THRESHOLD
    zone_data["oversteer_event"] = (
        accy_spike & accx_drop & (zone_data["accy_can"].abs() > OVERSTEER_ACCY_ABS_MIN)
    )

    # Fill NaN
    zone_data["wheelspin_event"] = zone_data["wheelspin_event"].fillna(False)
    zone_data["understeer_event"] = zone_data["understeer_event"].fillna(False)
    zone_data["oversteer_event"] = zone_data["oversteer_event"].fillna(False)

    # Count events
    wheelspin_count = zone_data["wheelspin_event"].sum()
    understeer_count = zone_data["understeer_event"].sum()
    oversteer_count = zone_data["oversteer_event"].sum()

    print("\nEvent Counts:")
    print(f"  Wheelspin: {wheelspin_count}")
    print(f"  Understeer: {understeer_count}")
    print(f"  Oversteer: {oversteer_count}")

    # Compute event coverage
    any_event = zone_data["wheelspin_event"] | zone_data["understeer_event"] | zone_data["oversteer_event"]
    event_coverage = any_event.sum() / len(zone_data)

    # Calculate max run length
    max_event_run_len = 0
    current_run = 0
    for has_event in any_event:
        if has_event:
            current_run += 1
            max_event_run_len = max(max_event_run_len, current_run)
        else:
            current_run = 0

    print("\nMacro-Event Metrics:")
    print(f"  Event Coverage: {event_coverage * 100:.1f}%")
    print(f"  Max Event Run: {max_event_run_len} samples")
    print(f"  Coverage Threshold: {MACRO_EVENT_COVERAGE_THRESHOLD * 100:.1f}%")
    print(f"  Run Length Threshold: {MACRO_EVENT_RUN_LENGTH_THRESHOLD} samples")

    # Determine if macro-event
    is_macro_event = (
        event_coverage >= MACRO_EVENT_COVERAGE_THRESHOLD
        or max_event_run_len >= MACRO_EVENT_RUN_LENGTH_THRESHOLD
    )

    print(f"\nIs Macro-Event? {is_macro_event}")
    if event_coverage >= MACRO_EVENT_COVERAGE_THRESHOLD:
        print(f"  ✓ Event coverage ({event_coverage * 100:.1f}%) >= threshold ({MACRO_EVENT_COVERAGE_THRESHOLD * 100:.1f}%)")
    if max_event_run_len >= MACRO_EVENT_RUN_LENGTH_THRESHOLD:
        print(f"  ✓ Max event run ({max_event_run_len}) >= threshold ({MACRO_EVENT_RUN_LENGTH_THRESHOLD})")

    # Show a few samples where understeer was detected
    if understeer_count > 0:
        print("\n=== Understeer Event Samples ===")
        understeer_samples = zone_data[zone_data["understeer_event"] == True].head(3)

        for idx, row in understeer_samples.iterrows():
            print(f"\nSample {idx}:")
            print(f"  Steering Angle: {row['Steering_Angle']:.1f}°")
            print(f"  Lateral G: {row['accy_can']:.3f}g")
            print(f"  Longitudinal G: {row['accx_can']:.3f}g")
            print(f"  Throttle: {row['aps']:.1f}%")

            # Calculate the trends for this sample
            if idx >= window_size:
                prev_samples = zone_data.iloc[idx-window_size:idx+1]
                steer_change = prev_samples["Steering_Angle"].abs().iloc[-1] - prev_samples["Steering_Angle"].abs().iloc[0]
                accy_change = prev_samples["accy_can"].abs().iloc[-1] - prev_samples["accy_can"].abs().iloc[0]
                responsiveness = accy_change / (steer_change + 0.001)

                print(f"  Steering Trend (over 10 samples): {steer_change:.1f}°")
                print(f"  Lateral G Trend: {accy_change:.3f}g")
                print(f"  Responsiveness: {responsiveness:.4f} g/deg")
                print(f"  Threshold: {UNDERSTEER_RESPONSIVENESS_THRESHOLD:.4f} g/deg")

    print("\n=== Analysis Complete ===")

else:
    print("\nNo understeer examples found in aggressive classifications")
