# ABOUTME: Build friction envelopes from corrected friction circle samples

import numpy as np
import pandas as pd
import json
from pathlib import Path


def fit_polynomial_envelope(accx_values, accy_values, degree=5):
    """
    Fit a polynomial r = f(theta) to envelope points in polar coordinates.

    Args:
        accx_values: Array of longitudinal G values (envelope points)
        accy_values: Array of lateral G values (envelope points)
        degree: Polynomial degree (default 5)

    Returns:
        coefficients: Polynomial coefficients [a0, a1, ..., an] (numpy.polyfit order)
    """
    radii = np.sqrt(accx_values ** 2 + accy_values ** 2)
    thetas = np.arctan2(accy_values, accx_values)

    sort_idx = np.argsort(thetas)
    thetas = thetas[sort_idx]
    radii = radii[sort_idx]

    coeffs = np.polyfit(thetas, radii, degree)
    return coeffs


def build_corrected_envelopes(
    corrected_samples_csv: Path,
    output_envelopes_json: Path,
    num_angle_bins: int = 72,
):
    """
    Build per-driver friction envelopes from corrected samples.

    The corrected samples should be the output of scripts/correct_outliers.py
    (columns: vehicle_number, abs_accx, abs_accy).
    """
    corrected_samples_csv = Path(corrected_samples_csv)
    output_envelopes_json = Path(output_envelopes_json)
    output_envelopes_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(corrected_samples_csv)

    envelopes = {}
    for vehicle_num in sorted(df["vehicle_number"].unique()):
        vehicle_data = df[df["vehicle_number"] == vehicle_num].copy()

        angle_bins = {}
        # Precompute per-sample radius and angle from corrected values
        vehicle_data["radius"] = np.sqrt(
            vehicle_data["abs_accx"] ** 2 + vehicle_data["abs_accy"] ** 2
        )
        vehicle_data["angle"] = np.arctan2(
            vehicle_data["abs_accy"], vehicle_data["abs_accx"]
        )

        for _, row in vehicle_data.iterrows():
            angle = row["angle"]
            bin_key = int(np.floor(angle / (np.pi / 2) * num_angle_bins))
            angle_bins.setdefault(bin_key, []).append(
                {
                    "radius": float(row["radius"]),
                    "accx": float(row["abs_accx"]),
                    "accy": float(row["abs_accy"]),
                    "angle": float(angle),
                }
            )

        envelope_points = []
        for bin_idx, points in angle_bins.items():
            points_sorted = sorted(points, key=lambda p: p["radius"], reverse=True)
            if len(points_sorted) == 0:
                continue
            # Use top 0.5% per angle bin, but ensure we pick at least the max
            p995_idx = max(0, int(len(points_sorted) * 0.005))
            p995_point = points_sorted[p995_idx]
            envelope_points.append(
                {
                    "accx": float(p995_point["accx"]),
                    "accy": float(p995_point["accy"]),
                    "total_g": float(p995_point["radius"]),
                }
            )

        # Sort points by angle so the curve is monotone in theta
        envelope_points.sort(key=lambda p: np.arctan2(p["accy"], p["accx"]))

        accx_vals = np.array([p["accx"] for p in envelope_points])
        accy_vals = np.array([p["accy"] for p in envelope_points])
        poly_coeffs = fit_polynomial_envelope(accx_vals, accy_vals, degree=5)

        envelopes[str(vehicle_num)] = {
            "envelope_points": envelope_points,
            "poly_coeffs": poly_coeffs.tolist(),
        }

    with open(output_envelopes_json, "w") as f:
        json.dump(envelopes, f, indent=2)


if __name__ == "__main__":
    # Default paths that match the rest of the repo
    script_dir = Path(__file__).parent.parent
    input_csv = script_dir / "data/processed/friction_circle_samples_corrected.csv"
    output_json = script_dir / "data/processed/friction_envelopes_corrected.json"
    build_corrected_envelopes(input_csv, output_json)
    print(f"✓ Saved corrected envelopes to: {output_json}")

