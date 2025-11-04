# ABOUTME: Traction efficiency analysis - turn detection, friction envelopes, and lap classification
# ABOUTME: Processes telemetry data to identify turn zones and analyze driver performance

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from sklearn.cluster import DBSCAN

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.geometry import (
    convert_gps_to_meters,
    load_centerline,
    project_points_onto_centerline,
)
from src.event_detection import detect_events, classify_zone

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, desc=None):
        return iterable


def load_telemetry_chunked(
    telemetry_paths, needed_params, chunk_size=50000, sample_fraction=1.0
):
    """
    Load telemetry from multiple files in chunks, pivot each chunk individually.

    Args:
        telemetry_paths: List of (race_name, path) tuples
        needed_params: List of parameter names to keep
        chunk_size: Number of rows per chunk
        sample_fraction: Fraction of data to sample (0.01 = 1%, 1.0 = 100%)

    Returns:
        DataFrame with columns: vehicle_number, lap, timestamp, race, [parameters], x_meters, y_meters
    """
    all_data = []

    for race_name, telemetry_path in telemetry_paths:
        print(f"\n  Loading {race_name}: {telemetry_path}")
        if sample_fraction < 1.0:
            print(f"  Sampling {sample_fraction * 100:.1f}% of data")
        sys.stdout.flush()

        # Read and pivot each chunk individually to save memory
        chunk_count = 0
        for i, chunk in enumerate(pd.read_csv(telemetry_path, chunksize=chunk_size)):
            # Sample the chunk if needed (NOT recommended - breaks timestamp alignment)
            if sample_fraction < 1.0:
                chunk = chunk.sample(frac=sample_fraction, random_state=42)

            # Filter to needed parameters
            chunk_filtered = chunk[chunk["telemetry_name"].isin(needed_params)].copy()

            if len(chunk_filtered) == 0:
                continue

            # Add sequence number to handle timestamp duplicates
            # Group by (vehicle, lap, timestamp, telemetry_name) and add row number
            chunk_filtered["seq"] = chunk_filtered.groupby(
                ["vehicle_number", "lap", "timestamp", "telemetry_name"]
            ).cumcount()

            # Pivot this chunk immediately (small pivot = low memory)
            try:
                chunk_wide = chunk_filtered.pivot_table(
                    index=["vehicle_number", "lap", "timestamp", "seq"],
                    columns="telemetry_name",
                    values="telemetry_value",
                    aggfunc="first",
                ).reset_index()
                chunk_wide.columns.name = None

                # Drop seq column
                chunk_wide = chunk_wide.drop(columns=["seq"])

                # Add race column
                chunk_wide["race"] = race_name

                # Drop rows with NaN (incomplete parameter sets)
                chunk_wide = chunk_wide.dropna()

                if len(chunk_wide) > 0:
                    all_data.append(chunk_wide)
                    chunk_count += 1

            except Exception as e:
                print(f"    Warning: Chunk {i} pivot failed: {e}")

            # Progress indicator every 10 chunks
            if (i + 1) % 10 == 0:
                print(
                    f"    Processed {(i + 1) * chunk_size:,} rows, {chunk_count} chunks with data",
                    end="\r",
                )
                sys.stdout.flush()

        print(f"\n    Finished: {chunk_count} chunks with valid data")
        sys.stdout.flush()

    # Combine all races
    print("\n  Combining all races...")
    sys.stdout.flush()
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"  Total samples: {len(df_combined):,}")
    sys.stdout.flush()

    # Convert GPS to meters
    print("  Converting GPS to meters...")
    sys.stdout.flush()
    df_combined = convert_gps_to_meters(df_combined)
    print(f"  GPS conversion complete: {len(df_combined):,} samples")
    sys.stdout.flush()

    return df_combined


def detect_turn_zones(
    telemetry_paths,
    centerline_path,
    output_path,
    eps_m=50,
    min_samples=20,
    sample_fraction=1.0,
):
    """
    Auto-detect turn zones from lateral G-force data.

    Algorithm:
    1. Load telemetry from both races
    2. Remove accy_can == 0 (straights)
    3. Keep |accy_can| > P10 (top 90% of cornering)
    4. Project GPS → track distance
    5. Cluster on 1D track distance using DBSCAN
    6. Compute zone boundaries per cluster

    Args:
        telemetry_paths: List of (race_name, path) tuples
        centerline_path: Path to track centerline CSV
        output_path: Path to save turn_zones.json
        eps_m: DBSCAN epsilon (max distance between samples in cluster)
        min_samples: DBSCAN min_samples (min points to form cluster)
    """
    print("\n=== Step 1: Load Data ===")

    # Load centerline
    centerline_x, centerline_y = load_centerline(centerline_path)

    # Load telemetry (both races)
    needed_params = [
        "VBOX_Long_Minutes",
        "VBOX_Lat_Min",
        "accy_can",
        "Laptrigger_lapdist_dls",
    ]
    df = load_telemetry_chunked(
        telemetry_paths,
        needed_params,
        chunk_size=50000,
        sample_fraction=sample_fraction,
    )

    print("\n=== Step 2: Data Summary ===")
    print(f"  Total samples: {len(df):,}")
    print(f"  Races: {df['race'].unique()}")
    print(f"  Vehicles: {df['vehicle_number'].nunique()}")
    print(f"  Laps: {df['lap'].nunique()}")

    print("\n=== Step 3: Filter Cornering Data ===")

    # Remove zero lateral G (straights)
    df_cornering = df[df["accy_can"] != 0].copy()
    print(f"  After removing accy_can == 0: {len(df_cornering):,} samples")

    # Calculate absolute lateral G
    df_cornering["abs_accy"] = np.abs(df_cornering["accy_can"])

    # Keep top 90% (above P10)
    p10_threshold = df_cornering["abs_accy"].quantile(0.10)
    df_high_g = df_cornering[df_cornering["abs_accy"] > p10_threshold].copy()
    print(f"  P10 threshold: {p10_threshold:.3f}g")
    print(f"  After keeping |accy_can| > P10: {len(df_high_g):,} samples")

    print("\n=== Step 4: Project to Track Distance ===")

    # Project GPS to track distance
    track_distances = project_points_onto_centerline(
        df_high_g["x_meters"].values,
        df_high_g["y_meters"].values,
        centerline_x,
        centerline_y,
    )
    df_high_g["track_distance"] = track_distances
    print(f"  Projected {len(track_distances):,} points to centerline")

    print("\n=== Step 5: Save High-G Points for Visualization ===")

    # Save high-G points to CSV for histogram/plotting
    output_csv = str(output_path).replace(".json", "_points.csv")
    df_high_g[
        ["track_distance", "abs_accy", "x_meters", "y_meters", "race", "vehicle_number"]
    ].to_csv(output_csv, index=False)
    print(f"  Saved {len(df_high_g)} high-G points to: {output_csv}")

    print("\n=== Step 6: Cluster Turn Zones (DBSCAN) ===")

    # Cluster on 1D track distance
    X = df_high_g["track_distance"].values.reshape(-1, 1)
    dbscan = DBSCAN(eps=eps_m, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    # Count clusters (exclude noise: label = -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    n_clusters = len(unique_labels)
    n_noise = list(labels).count(-1)

    print(f"  Found {n_clusters} turn zones")
    print(f"  Noise points: {n_noise}")

    print("\n=== Step 6: Compute Zone Boundaries ===")

    turn_zones = []

    for zone_id in sorted(unique_labels):
        # Get samples in this cluster
        cluster_mask = labels == zone_id
        cluster_data = df_high_g[cluster_mask]

        # Compute zone boundaries (2.5th to 97.5th percentile)
        distances = cluster_data["track_distance"].values
        start_distance = np.percentile(distances, 2.5)
        end_distance = np.percentile(distances, 97.5)

        # Average lateral G in zone
        avg_lateral_g = cluster_data["abs_accy"].mean()

        # Compute bounding box (for visualization)
        x_vals = cluster_data["x_meters"].values
        y_vals = cluster_data["y_meters"].values

        zone = {
            "zone_id": int(zone_id) + 1,  # 1-indexed for display
            "start_distance_m": float(start_distance),
            "end_distance_m": float(end_distance),
            "name": f"Turn {int(zone_id) + 1}",
            "avg_lateral_g": float(avg_lateral_g),
            "bounds": {
                "x_min": float(x_vals.min()),
                "x_max": float(x_vals.max()),
                "y_min": float(y_vals.min()),
                "y_max": float(y_vals.max()),
            },
        }

        turn_zones.append(zone)
        print(
            f"  Zone {zone['zone_id']}: {start_distance:.0f}-{end_distance:.0f}m, "
            f"avg |accy| = {avg_lateral_g:.2f}g"
        )

    print("\n=== Step 7: Save Turn Zones ===")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(turn_zones, f, indent=2)

    print(f"✓ Saved {len(turn_zones)} turn zones to: {output_path}")


def classify_laps_step(
    telemetry_paths,
    centerline_path,
    turn_zones_path,
    envelopes_path,
    output_path,
    sample_fraction=1.0,
):
    """
    Classify laps based on grip utilization and over-limit events.

    Args:
        telemetry_paths: List of (race_name, path) tuples
        centerline_path: Path to track centerline CSV
        turn_zones_path: Path to turn_zones.json
        envelopes_path: Path to friction_envelopes.json
        output_path: Path to save lap_classifications.csv
        sample_fraction: Fraction of data to sample (1.0 = 100%)
    """
    print("\n=== Classify Laps: Load Data ===")

    # Load centerline
    centerline_x, centerline_y = load_centerline(centerline_path)

    # Load turn zones
    with open(turn_zones_path) as f:
        turn_zones = json.load(f)
    print(f"  Loaded {len(turn_zones)} turn zones")

    # Load friction envelopes
    with open(envelopes_path) as f:
        envelopes = json.load(f)
    print(f"  Loaded envelopes for {len(envelopes)} drivers")

    # Load telemetry with all needed signals
    needed_params = [
        "VBOX_Long_Minutes",
        "VBOX_Lat_Min",
        "accx_can",
        "accy_can",
        "aps",
        "Steering_Angle",
    ]

    print("\n=== Loading telemetry ===")
    df = load_telemetry_chunked(
        telemetry_paths,
        needed_params,
        chunk_size=50000,
        sample_fraction=sample_fraction,
    )

    print(f"\n  Total samples loaded: {len(df):,}")
    print(f"  Races: {df['race'].unique()}")
    print(f"  Vehicles: {df['vehicle_number'].nunique()}")

    # Project to track distance
    print("\n=== Projecting to track distance ===")
    track_distances = project_points_onto_centerline(
        df["x_meters"].values,
        df["y_meters"].values,
        centerline_x,
        centerline_y,
    )
    df["track_distance_m"] = track_distances
    print(f"  Projected {len(track_distances):,} points to centerline")

    # Process each race separately
    all_results = []

    for race_name in df["race"].unique():
        print(f"\n=== Processing {race_name} ===")
        race_data = df[df["race"] == race_name].copy()

        # Process each driver
        drivers = race_data["vehicle_number"].unique()
        for driver in tqdm(drivers, desc=f"  {race_name} drivers"):
            driver_data = race_data[race_data["vehicle_number"] == driver].copy()

            # Detect events using rolling window
            driver_data = detect_events(driver_data, window_size=10)

            # Get envelope for this driver
            driver_envelope = envelopes.get(str(driver), {})

            # Process each lap
            laps = driver_data["lap"].unique()
            for lap_num in laps:
                lap_data = driver_data[driver_data["lap"] == lap_num]

                # Process each zone
                for zone in turn_zones:
                    zone_id = zone["zone_id"]
                    zone_start = zone["start_distance_m"]
                    zone_end = zone["end_distance_m"]

                    # Filter samples in this zone
                    zone_samples = lap_data[
                        (lap_data["track_distance_m"] >= zone_start)
                        & (lap_data["track_distance_m"] <= zone_end)
                    ]

                    # Skip if no samples in zone
                    if len(zone_samples) == 0:
                        continue

                    # Classify this zone
                    classification_result = classify_zone(
                        zone_samples, driver_envelope, zone_id
                    )

                    # Build result row
                    result = {
                        "race": race_name,
                        "vehicle_number": driver,
                        "lap": lap_num,
                        "zone_id": zone_id,
                        "zone_name": zone["name"],
                        **classification_result,
                    }
                    all_results.append(result)

    # Convert to DataFrame and save
    print("\n=== Saving Results ===")
    results_df = pd.DataFrame(all_results)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"  Saved {len(results_df)} classifications to: {output_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nClassification Distribution:")
    print(results_df["classification"].value_counts())
    print("\nEvent Distribution:")
    print(f"  Wheelspin events: {results_df['wheelspin'].sum()}")
    print(f"  Understeer events: {results_df['understeer'].sum()}")
    print(f"  Oversteer events: {results_df['oversteer'].sum()}")

    print("\nAverage Utilization by Classification:")
    print(results_df.groupby("classification")["avg_utilization"].mean())


def main():
    parser = argparse.ArgumentParser(
        description="Traction efficiency analysis pipeline"
    )
    parser.add_argument(
        "--step",
        choices=["detect_turns", "build_envelopes", "classify_laps"],
        required=True,
        help="Analysis step to run",
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
        "--output", help="Output file path (auto-generated if not specified)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=50.0,
        help="DBSCAN epsilon (meters) for turn clustering",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="DBSCAN min_samples for turn clustering",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Fraction of data to sample (0.01 = 1%, 1.0 = 100%)",
    )

    args = parser.parse_args()

    # Set script directory as working directory
    script_dir = Path(__file__).parent.parent

    # Resolve paths relative to script directory
    r1_path = script_dir / args.r1_telemetry
    r2_path = script_dir / args.r2_telemetry
    centerline_path = script_dir / args.centerline

    if args.step == "detect_turns":
        telemetry_paths = [("R1", r1_path), ("R2", r2_path)]
        output_path = args.output or script_dir / "data/processed/turn_zones.json"
        detect_turn_zones(
            telemetry_paths=telemetry_paths,
            centerline_path=centerline_path,
            output_path=output_path,
            eps_m=args.eps,
            min_samples=args.min_samples,
            sample_fraction=args.sample_fraction,
        )

    elif args.step == "build_envelopes":
        print("TODO: Implement friction envelope building")

    elif args.step == "classify_laps":
        telemetry_paths = [("R1", r1_path), ("R2", r2_path)]
        turn_zones_path = script_dir / "data/processed/turn_zones.json"
        envelopes_path = script_dir / "data/processed/friction_envelopes.json"
        output_path = (
            args.output or script_dir / "data/processed/lap_classifications.csv"
        )

        classify_laps_step(
            telemetry_paths=telemetry_paths,
            centerline_path=centerline_path,
            turn_zones_path=turn_zones_path,
            envelopes_path=envelopes_path,
            output_path=output_path,
            sample_fraction=args.sample_fraction,
        )


if __name__ == "__main__":
    main()
