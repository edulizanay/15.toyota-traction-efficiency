# ABOUTME: Corrects friction circle outliers using angular gap detection
# ABOUTME: Applies local window median method to find realistic max per angle

import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def detect_realistic_max_local_window(
    radii, window_size=5, threshold=3.0, min_samples=50, max_iterations=10
):
    """
    Detect realistic maximum radius using local window median on gaps.
    Iteratively removes outliers until no more are found.

    Args:
        radii: Sorted array of radius values (descending)
        window_size: Size of window for local median (±window_size)
        threshold: Multiplier for outlier detection (gap > threshold × local_median)
        min_samples: Minimum samples needed to apply detection
        max_iterations: Maximum number of outlier removal iterations

    Returns:
        realistic_max: The last value before the outlier jump
    """
    if len(radii) < min_samples:
        # Not enough samples, use 99th percentile as fallback
        return np.percentile(radii, 99)

    current_radii = radii.copy()

    for iteration in range(max_iterations):
        # Calculate gaps between consecutive values
        gaps = np.diff(current_radii) * -1  # Make positive (since radii is descending)

        if len(gaps) < 10:
            # Too few gaps to analyze
            break

        # Find first outlier gap using local window median
        outlier_found = False

        for i in range(len(gaps)):
            # Get local window of gaps
            window_start = max(0, i - window_size)
            window_end = min(len(gaps), i + window_size + 1)
            local_gaps = gaps[window_start:window_end]

            # Skip if we don't have enough local context
            if len(local_gaps) < 3:
                continue

            local_median = np.median(local_gaps)

            # Check if current gap is an outlier
            if local_median > 0 and gaps[i] > threshold * local_median:
                # Found outlier - remove values before this gap
                current_radii = current_radii[i + 1 :]
                outlier_found = True
                break

        # If no outlier found, we're done
        if not outlier_found:
            break

    return current_radii[0] if len(current_radii) > 0 else radii[0]


def correct_outliers_angular(
    df_vehicle, num_angle_bins=12, window_size=5, threshold=3.0
):
    """
    Correct outliers using angular binning and local window median.

    Args:
        df_vehicle: DataFrame with abs_accx, abs_accy for one vehicle
        num_angle_bins: Number of angular bins (e.g., 12 = 7.5° each)
        window_size: Window size for local median
        threshold: Gap threshold multiplier

    Returns:
        DataFrame with corrected abs_accx_corrected and abs_accy_corrected columns
    """
    df = df_vehicle.copy()

    # Convert to polar coordinates
    df["angle"] = np.arctan2(df["abs_accy"], df["abs_accx"])  # radians, 0 to π/2
    df["radius"] = np.sqrt(df["abs_accx"] ** 2 + df["abs_accy"] ** 2)

    # Create angular bins
    angle_edges = np.linspace(0, np.pi / 2, num_angle_bins + 1)
    df["angle_bin"] = pd.cut(
        df["angle"], bins=angle_edges, labels=False, include_lowest=True
    )

    # Detect realistic max for each angular bin
    realistic_max_per_bin = {}

    for bin_idx in range(num_angle_bins):
        bin_data = df[df["angle_bin"] == bin_idx]

        if len(bin_data) == 0:
            continue

        radii = np.sort(bin_data["radius"].values)[::-1]  # Descending
        realistic_max = detect_realistic_max_local_window(
            radii, window_size=window_size, threshold=threshold
        )
        realistic_max_per_bin[bin_idx] = realistic_max

    # Apply radial rescaling
    df["abs_accx_corrected"] = df["abs_accx"]
    df["abs_accy_corrected"] = df["abs_accy"]

    for idx, row in df.iterrows():
        bin_idx = row["angle_bin"]

        if pd.isna(bin_idx) or bin_idx not in realistic_max_per_bin:
            continue

        realistic_max = realistic_max_per_bin[bin_idx]
        current_radius = row["radius"]

        if current_radius > realistic_max:
            # Rescale radially
            scale_factor = realistic_max / current_radius
            df.at[idx, "abs_accx_corrected"] = row["abs_accx"] * scale_factor
            df.at[idx, "abs_accy_corrected"] = row["abs_accy"] * scale_factor

    return df[["abs_accx_corrected", "abs_accy_corrected"]], realistic_max_per_bin


def main():
    parser = argparse.ArgumentParser(
        description="Correct friction circle outliers using angular gap detection"
    )
    parser.add_argument(
        "--input",
        default="data/processed/friction_circle_samples.csv",
        help="Input samples CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed/friction_circle_samples_corrected.csv",
        help="Output corrected samples CSV",
    )
    parser.add_argument(
        "--angle-bins",
        type=int,
        default=12,
        help="Number of angular bins (default: 12)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Local window size for median (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Gap threshold multiplier (default: 3.0)",
    )
    parser.add_argument(
        "--test-vehicle", type=int, help="Test on single vehicle only (for debugging)"
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output

    print("\n=== Friction Circle Outlier Correction ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(
        f"Parameters: {args.angle_bins} angular bins, window={args.window_size}, threshold={args.threshold}x"
    )

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(input_path)
    print(f"   Total samples: {len(df):,}")
    print(f"   Vehicles: {sorted(df['vehicle_number'].unique())}")

    # Process vehicles
    vehicles = (
        [args.test_vehicle]
        if args.test_vehicle
        else sorted(df["vehicle_number"].unique())
    )

    corrected_dfs = []

    for vehicle_num in vehicles:
        print(f"\n2. Processing Vehicle {vehicle_num}...")
        vehicle_data = df[df["vehicle_number"] == vehicle_num].copy()

        print(f"   Samples: {len(vehicle_data):,}")
        print(
            f"   Original max total_g: {np.sqrt(vehicle_data['abs_accx'] ** 2 + vehicle_data['abs_accy'] ** 2).max():.3f}"
        )

        # Apply correction
        corrected_cols, realistic_max_per_bin = correct_outliers_angular(
            vehicle_data,
            num_angle_bins=args.angle_bins,
            window_size=args.window_size,
            threshold=args.threshold,
        )

        vehicle_data["abs_accx"] = corrected_cols["abs_accx_corrected"]
        vehicle_data["abs_accy"] = corrected_cols["abs_accy_corrected"]

        # Statistics
        corrected_max = np.sqrt(
            vehicle_data["abs_accx"] ** 2 + vehicle_data["abs_accy"] ** 2
        ).max()
        print(f"   Corrected max total_g: {corrected_max:.3f}")

        # Show realistic max per angular bin
        print("   Realistic max per angle:")
        for bin_idx in sorted(realistic_max_per_bin.keys()):
            angle_deg = (bin_idx + 0.5) * (90 / args.angle_bins)
            print(
                f"     Bin {bin_idx:2d} (~{angle_deg:4.1f}°): {realistic_max_per_bin[bin_idx]:.3f}g"
            )

        corrected_dfs.append(vehicle_data[["vehicle_number", "abs_accx", "abs_accy"]])

    # Save corrected data
    print("\n3. Saving corrected data...")
    df_corrected = pd.concat(corrected_dfs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_corrected.to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(df_corrected):,} samples to: {output_path}")

    print("\n✓ Outlier correction complete!")


if __name__ == "__main__":
    main()
