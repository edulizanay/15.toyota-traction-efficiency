# ABOUTME: Generate friction circle visualization data for each driver
# ABOUTME: Calculates grip envelopes and exports samples for D3.js hexbin plotting

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.geometry import convert_gps_to_meters


def fit_polynomial_envelope(accx_values, accy_values, degree=5):
    """
    Fit a polynomial r = f(theta) to envelope points in polar coordinates.

    Args:
        accx_values: Array of longitudinal G values (envelope points)
        accy_values: Array of lateral G values (envelope points)
        degree: Polynomial degree (default 5)

    Returns:
        coefficients: Polynomial coefficients [a0, a1, ..., an]
    """
    # Convert to polar coordinates
    radii = np.sqrt(accx_values**2 + accy_values**2)
    thetas = np.arctan2(accy_values, accx_values)

    # Sort by theta
    sort_idx = np.argsort(thetas)
    thetas = thetas[sort_idx]
    radii = radii[sort_idx]

    # Fit polynomial
    coeffs = np.polyfit(thetas, radii, degree)

    return coeffs


def evaluate_envelope(coeffs, accx, accy):
    """
    Evaluate the envelope polynomial at a given (accx, accy) point.

    Args:
        coeffs: Polynomial coefficients from fit_polynomial_envelope
        accx: Longitudinal G value
        accy: Lateral G value

    Returns:
        envelope_max: Maximum total G at this angle according to envelope
    """
    theta = np.arctan2(accy, accx)
    envelope_r = np.polyval(coeffs, theta)
    return envelope_r


def load_telemetry_chunked(telemetry_paths, needed_params, chunk_size=50000):
    """
    Load telemetry from multiple files in chunks, pivot each chunk individually.

    Args:
        telemetry_paths: List of (race_name, path) tuples
        needed_params: List of parameter names to keep
        chunk_size: Number of rows per chunk

    Returns:
        DataFrame with columns: vehicle_number, lap, timestamp, race, [parameters], x_meters, y_meters
    """
    all_data = []

    for race_name, telemetry_path in telemetry_paths:
        print(f"\n  Loading {race_name}: {telemetry_path}")
        sys.stdout.flush()

        # Read and pivot each chunk individually to save memory
        chunk_count = 0
        for i, chunk in enumerate(pd.read_csv(telemetry_path, chunksize=chunk_size)):
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

    # Convert GPS to meters if GPS columns present
    if (
        "VBOX_Long_Minutes" in df_combined.columns
        and "VBOX_Lat_Min" in df_combined.columns
    ):
        print("  Converting GPS to meters...")
        sys.stdout.flush()
        df_combined = convert_gps_to_meters(df_combined)
        print(f"  GPS conversion complete: {len(df_combined):,} samples")
        sys.stdout.flush()

    return df_combined


def build_friction_circle_data(
    telemetry_paths,
    output_samples_path,
    output_envelopes_path,
    centerline_path,
    turn_zones_path,
    output_zone_stats_path,
):
    """
    Build friction circle data: raw samples and per-driver grip envelopes.

    Args:
        telemetry_paths: List of (race_name, path) tuples
        output_samples_path: Path to save friction_circle_samples.csv
        output_envelopes_path: Path to save friction_envelopes.json
        centerline_path: Path to track centerline CSV
        turn_zones_path: Path to turn zones JSON
        output_zone_stats_path: Path to save zone statistics JSON
    """
    print("\n=== Step 1: Load Telemetry Data ===")

    # Load telemetry with accx and accy
    needed_params = ["accx_can", "accy_can", "VBOX_Long_Minutes", "VBOX_Lat_Min"]
    df = load_telemetry_chunked(telemetry_paths, needed_params, chunk_size=50000)

    print("\n=== Step 2: Data Summary ===")
    print(f"  Total samples: {len(df):,}")
    print(f"  Races: {df['race'].unique()}")
    print(f"  Vehicles: {sorted(df['vehicle_number'].unique())}")
    print(f"  Laps: {df['lap'].nunique()}")

    print("\n=== Step 3: Calculate Friction Circle Values ===")

    # Calculate absolute values for quarter-circle visualization
    df["abs_accx"] = np.abs(df["accx_can"])
    df["abs_accy"] = np.abs(df["accy_can"])

    # Calculate total G
    df["total_g"] = np.sqrt(df["accx_can"] ** 2 + df["accy_can"] ** 2)

    print(f"  Max total_g across all drivers: {df['total_g'].max():.2f}g")
    print(f"  Mean total_g: {df['total_g'].mean():.2f}g")

    print("\n=== Step 4: Save Sample Data for Hexbin ===")

    # Save samples for visualization
    samples_df = df[["vehicle_number", "abs_accx", "abs_accy"]].copy()
    output_samples_path = Path(output_samples_path)
    output_samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(output_samples_path, index=False)
    print(f"  ✓ Saved {len(samples_df):,} samples to: {output_samples_path}")

    print("\n=== Step 5: Build Field-Wide Friction Envelope (all drivers) ===")

    # Build shared envelope from ALL drivers' data combined
    num_angle_bins = 72
    angle_bins = {}

    for _, row in df.iterrows():
        radius = row["total_g"]
        angle = np.arctan2(row["abs_accy"], row["abs_accx"])
        bin_key = int(np.floor(angle / (np.pi / 2) * num_angle_bins))

        if bin_key not in angle_bins:
            angle_bins[bin_key] = []
        angle_bins[bin_key].append(
            {
                "radius": radius,
                "accx": row["abs_accx"],
                "accy": row["abs_accy"],
                "angle": angle,
            }
        )

    # Extract 99.5th percentile per angular bin
    envelope_points = []
    for bin_idx, points in angle_bins.items():
        points_sorted = sorted(points, key=lambda p: p["radius"], reverse=True)
        p995_idx = int(len(points_sorted) * 0.005)  # top 0.5%
        p995_point = points_sorted[p995_idx]

        envelope_points.append(
            {
                "accx": float(p995_point["accx"]),
                "accy": float(p995_point["accy"]),
                "total_g": float(p995_point["radius"]),
            }
        )

    # Sort by angle
    envelope_points.sort(key=lambda p: np.arctan2(p["accy"], p["accx"]))

    # Fit polynomial to envelope
    accx_vals = np.array([p["accx"] for p in envelope_points])
    accy_vals = np.array([p["accy"] for p in envelope_points])
    poly_coeffs = fit_polynomial_envelope(accx_vals, accy_vals, degree=5)

    # Create field-wide envelope entry
    envelopes = {}
    envelopes["all"] = {
        "envelope_points": envelope_points,
        "poly_coeffs": poly_coeffs.tolist(),
    }

    max_g = max(point["total_g"] for point in envelope_points)
    print(
        f"  Field-wide envelope: {len(envelope_points)} points, max grip = {max_g:.2f}g"
    )

    print("\n=== Step 6: Build Per-Driver Friction Envelopes (for visualization) ===")

    # Keep adding to existing envelopes dict (field-wide "all" is already in there)
    for vehicle_num in sorted(df["vehicle_number"].unique()):
        vehicle_data = df[df["vehicle_number"] == vehicle_num].copy()

        # Calculate 99.5th percentile envelope using angular binning
        num_angle_bins = 72
        angle_bins = {}

        for _, row in vehicle_data.iterrows():
            radius = row["total_g"]
            angle = np.arctan2(row["abs_accy"], row["abs_accx"])
            bin_key = int(np.floor(angle / (np.pi / 2) * num_angle_bins))

            if bin_key not in angle_bins:
                angle_bins[bin_key] = []
            angle_bins[bin_key].append(
                {
                    "radius": radius,
                    "accx": row["abs_accx"],
                    "accy": row["abs_accy"],
                    "angle": angle,
                }
            )

        # Extract 99.5th percentile per angular bin
        envelope_points = []
        for bin_idx, points in angle_bins.items():
            points_sorted = sorted(points, key=lambda p: p["radius"], reverse=True)
            p995_idx = int(len(points_sorted) * 0.005)  # top 0.5%
            p995_point = points_sorted[p995_idx]

            envelope_points.append(
                {
                    "accx": float(p995_point["accx"]),
                    "accy": float(p995_point["accy"]),
                    "total_g": float(p995_point["radius"]),
                }
            )

        # Sort by angle
        envelope_points.sort(key=lambda p: np.arctan2(p["accy"], p["accx"]))

        # Fit polynomial to envelope
        accx_vals = np.array([p["accx"] for p in envelope_points])
        accy_vals = np.array([p["accy"] for p in envelope_points])
        poly_coeffs = fit_polynomial_envelope(accx_vals, accy_vals, degree=5)

        envelopes[str(vehicle_num)] = {
            "envelope_points": envelope_points,
            "poly_coeffs": poly_coeffs.tolist(),
        }

        max_g = max(point["total_g"] for point in envelope_points)
        print(
            f"  Vehicle {vehicle_num}: {len(envelope_points)} envelope points, max grip = {max_g:.2f}g"
        )

    print("\n=== Step 6: Save Friction Envelopes ===")

    output_envelopes_path = Path(output_envelopes_path)
    output_envelopes_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_envelopes_path, "w") as f:
        json.dump(envelopes, f, indent=2)

    print(
        f"  ✓ Saved envelopes for {len(envelopes)} drivers to: {output_envelopes_path}"
    )

    print("\n=== Step 7: Calculate Per-Zone Statistics ===")

    # Load centerline and turn zones
    from src.geometry import load_centerline, project_points_onto_centerline

    centerline_x, centerline_y = load_centerline(centerline_path)
    with open(turn_zones_path, "r") as f:
        turn_zones = json.load(f)

    # Project samples to track distance
    print("  Projecting samples to centerline...")
    track_distances = project_points_onto_centerline(
        df["x_meters"].values, df["y_meters"].values, centerline_x, centerline_y
    )
    df["track_distance"] = track_distances

    # Map samples to zones
    def get_zone_id(track_dist):
        for zone in turn_zones:
            if zone["start_distance_m"] <= track_dist <= zone["end_distance_m"]:
                return zone["zone_id"]
        return None

    df["zone_id"] = df["track_distance"].apply(get_zone_id)

    # Calculate per-driver, per-race, per-zone statistics
    zone_stats = {}

    for race in df["race"].unique():
        zone_stats[race] = {}
        race_data = df[df["race"] == race].copy()

        print(f"\n  Race {race}:")
        for vehicle_num in sorted(race_data["vehicle_number"].unique()):
            vehicle_data = race_data[race_data["vehicle_number"] == vehicle_num].copy()
            zone_stats[race][str(vehicle_num)] = []

            # Use field-wide shared envelope (same for all drivers - competitive comparison)
            poly_coeffs = np.array(envelopes["all"]["poly_coeffs"])

            for zone in turn_zones:
                zone_data = vehicle_data[vehicle_data["zone_id"] == zone["zone_id"]]

                if len(zone_data) == 0:
                    continue

                # Calculate envelope_max for each sample in this zone
                envelope_maxes = []
                utilizations = []

                for _, row in zone_data.iterrows():
                    envelope_max = evaluate_envelope(
                        poly_coeffs, row["abs_accx"], row["abs_accy"]
                    )
                    envelope_maxes.append(envelope_max)

                    # Calculate utilization (avoid division by zero)
                    if envelope_max > 0.01:
                        utilization = row["total_g"] / envelope_max
                        utilizations.append(utilization)

                avg_total_g = zone_data["total_g"].mean()
                max_total_g = zone_data["total_g"].max()
                avg_envelope_max = np.mean(envelope_maxes) if envelope_maxes else 0
                avg_utilization = np.mean(utilizations) if utilizations else 0
                sample_count = len(zone_data)

                zone_stats[race][str(vehicle_num)].append(
                    {
                        "zone_id": zone["zone_id"],
                        "zone_name": zone["name"],
                        "avg_total_g": float(avg_total_g),
                        "max_total_g": float(max_total_g),
                        "avg_envelope_max": float(avg_envelope_max),
                        "avg_utilization": float(avg_utilization),
                        "sample_count": int(sample_count),
                    }
                )

            print(
                f"    Vehicle {vehicle_num}: {len(zone_stats[race][str(vehicle_num)])} zones with data"
            )

    # Save zone statistics
    output_zone_stats_path = Path(output_zone_stats_path)
    with open(output_zone_stats_path, "w") as f:
        json.dump(zone_stats, f, indent=2)

    print(f"  ✓ Saved zone statistics to: {output_zone_stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build friction circle visualization data"
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
        "--turn-zones",
        default="data/processed/turn_zones.json",
        help="Path to turn zones JSON",
    )
    parser.add_argument(
        "--output-samples",
        default="data/processed/friction_circle_samples.csv",
        help="Output path for sample data CSV",
    )
    parser.add_argument(
        "--output-envelopes",
        default="data/processed/friction_envelopes.json",
        help="Output path for envelope JSON",
    )
    parser.add_argument(
        "--output-zone-stats",
        default="data/processed/zone_statistics.json",
        help="Output path for zone statistics JSON",
    )

    args = parser.parse_args()

    # Set script directory as working directory
    script_dir = Path(__file__).parent.parent

    # Resolve paths relative to script directory
    r1_path = script_dir / args.r1_telemetry
    r2_path = script_dir / args.r2_telemetry
    centerline_path = script_dir / args.centerline
    turn_zones_path = script_dir / args.turn_zones
    output_samples_path = script_dir / args.output_samples
    output_envelopes_path = script_dir / args.output_envelopes
    output_zone_stats_path = script_dir / args.output_zone_stats

    telemetry_paths = [("R1", r1_path), ("R2", r2_path)]

    build_friction_circle_data(
        telemetry_paths=telemetry_paths,
        output_samples_path=output_samples_path,
        output_envelopes_path=output_envelopes_path,
        centerline_path=centerline_path,
        turn_zones_path=turn_zones_path,
        output_zone_stats_path=output_zone_stats_path,
    )

    print("\n✓ Friction circle data generation complete!")


if __name__ == "__main__":
    main()
