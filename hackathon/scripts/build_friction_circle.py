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
    telemetry_paths, output_samples_path, output_envelopes_path
):
    """
    Build friction circle data: raw samples and per-driver grip envelopes.

    Args:
        telemetry_paths: List of (race_name, path) tuples
        output_samples_path: Path to save friction_circle_samples.csv
        output_envelopes_path: Path to save friction_envelopes.json
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

    print("\n=== Step 5: Build Friction Envelopes (per driver) ===")

    envelopes = {}

    for vehicle_num in sorted(df["vehicle_number"].unique()):
        vehicle_data = df[df["vehicle_number"] == vehicle_num].copy()

        # Bin by abs_accy (lateral G) into 20 bins
        accy_min, accy_max = (
            vehicle_data["abs_accy"].min(),
            vehicle_data["abs_accy"].max(),
        )
        bins = np.linspace(accy_min, accy_max, 21)  # 21 edges = 20 bins
        vehicle_data["accy_bin"] = pd.cut(
            vehicle_data["abs_accy"], bins=bins, labels=False, include_lowest=True
        )

        # For each bin, get 95th percentile of total_g
        envelope_points = []
        for bin_idx in range(20):
            bin_data = vehicle_data[vehicle_data["accy_bin"] == bin_idx]

            if len(bin_data) == 0:
                continue

            # Get representative accy for this bin (center)
            bin_center_accy = (bins[bin_idx] + bins[bin_idx + 1]) / 2

            # Get 95th percentile of total_g in this bin
            max_total_g = np.percentile(bin_data["total_g"], 95)

            envelope_points.append(
                {"accy": float(bin_center_accy), "total_g_max": float(max_total_g)}
            )

        envelopes[str(vehicle_num)] = envelope_points

        max_g = max(point["total_g_max"] for point in envelope_points)
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
        "--output-samples",
        default="data/processed/friction_circle_samples.csv",
        help="Output path for sample data CSV",
    )
    parser.add_argument(
        "--output-envelopes",
        default="data/processed/friction_envelopes.json",
        help="Output path for envelope JSON",
    )

    args = parser.parse_args()

    # Set script directory as working directory
    script_dir = Path(__file__).parent.parent

    # Resolve paths relative to script directory
    r1_path = script_dir / args.r1_telemetry
    r2_path = script_dir / args.r2_telemetry
    output_samples_path = script_dir / args.output_samples
    output_envelopes_path = script_dir / args.output_envelopes

    telemetry_paths = [("R1", r1_path), ("R2", r2_path)]

    build_friction_circle_data(
        telemetry_paths=telemetry_paths,
        output_samples_path=output_samples_path,
        output_envelopes_path=output_envelopes_path,
    )

    print("\n✓ Friction circle data generation complete!")


if __name__ == "__main__":
    main()
