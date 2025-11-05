#!/usr/bin/env python3
# ABOUTME: Extract per-driver best lap times across R1+R2
# ABOUTME: Uses sector analysis files and rebuilds JSON if missing/empty

import csv
import json
from pathlib import Path
from typing import Optional, Dict


def parse_lap_time_str(s: str) -> Optional[float]:
    """Parse a lap time like '1:45.035' or '45.035' into seconds.

    Returns None if parsing fails.
    """
    if not s:
        return None
    s = s.strip()
    if not s:
        return None

    # Normalize and guard against non-time strings
    if any(tok in s.lower() for tok in ["pit", "dnf", "dns", "--", "nan"]):
        return None

    try:
        parts = s.split(":")
        if len(parts) == 1:
            # 'SS.mmm'
            return float(parts[0])
        elif len(parts) == 2:
            # 'M:SS.mmm' or 'MM:SS.mmm'
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # 'H:MM:SS.mmm' (unlikely for laps, but handle defensively)
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except Exception:
        return None

    return None


def format_time(seconds: float) -> str:
    """Format seconds as M:SS.mmm (no leading zero on minutes)."""
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}:{rem:06.3f}"


def best_laps_from_sector_files(input_paths) -> Dict[str, str]:
    """Compute per-driver best lap across provided sector analysis CSVs.

    The files are expected to be ';'-delimited with at least columns:
    - NUMBER (driver)
    - LAP_TIME (formatted) and/or S1_SECONDS, S2_SECONDS, S3_SECONDS (floats)
    """
    best_secs: Dict[str, float] = {}

    for path in input_paths:
        if not Path(path).exists():
            continue

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                driver = (row.get("NUMBER") or "").strip()
                if not driver:
                    continue

                # Prefer numeric sector seconds if available
                s1 = row.get("S1_SECONDS")
                s2 = row.get("S2_SECONDS")
                s3 = row.get("S3_SECONDS")
                lap_time_s: Optional[float] = None

                try:
                    if s1 is not None and s2 is not None and s3 is not None:
                        s1f = float(str(s1).strip().replace(",", "."))
                        s2f = float(str(s2).strip().replace(",", "."))
                        s3f = float(str(s3).strip().replace(",", "."))
                        lap_time_s = s1f + s2f + s3f
                except Exception:
                    lap_time_s = None

                if lap_time_s is None:
                    lap_time_s = parse_lap_time_str(row.get("LAP_TIME", ""))

                if lap_time_s is None or lap_time_s <= 0:
                    continue

                prev = best_secs.get(driver)
                if prev is None or lap_time_s < prev:
                    best_secs[driver] = lap_time_s

    # Format results
    return {drv: format_time(sec) for drv, sec in best_secs.items()}


def main():
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / "data" / "input"
    output_path = script_dir / "data" / "processed" / "driver_best_laps.json"

    # Input sector analysis files (present in repo)
    inputs = [
        input_dir / "23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV",
        input_dir / "23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV",
    ]

    # Check if output exists and is non-empty
    needs_build = True
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and len(data) > 0:
                needs_build = False
        except Exception:
            needs_build = True

    if not needs_build:
        print(f"driver_best_laps.json already populated: {output_path}")
        return

    # Build best laps
    best_laps = best_laps_from_sector_files(inputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_laps, f, indent=2)

    print(f"✓ Saved best laps for {len(best_laps)} drivers to: {output_path}")


if __name__ == "__main__":
    main()

