#!/usr/bin/env python3
# ABOUTME: Extract total race finish times for each driver from race analysis files
# ABOUTME: Outputs JSON file with driver numbers and their total race times

import csv
import json

# Load race 1 and race 2 analysis data
race1_path = "../data/input/23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV"
race2_path = "../data/input/23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV"

# Track last lap elapsed time for each driver in each race
driver_finish_times = {}

# Process both race files
for race_path in [race1_path, race2_path]:
    # Normalize to R1/R2 keys for downstream consumers
    race_name = "R1" if "Race 1" in race_path else "R2"

    with open(race_path, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        driver_laps = {}

        for row in reader:
            # Strip whitespace from column names
            row = {k.strip(): v.strip() for k, v in row.items()}

            driver_num = row["NUMBER"]
            lap_num_str = row["LAP_NUMBER"]
            elapsed_time = row["ELAPSED"]

            # Track the highest lap number and its elapsed time for each driver
            try:
                lap_num = int(lap_num_str)
                if (
                    driver_num not in driver_laps
                    or lap_num > driver_laps[driver_num]["lap"]
                ):
                    driver_laps[driver_num] = {"lap": lap_num, "elapsed": elapsed_time}
            except (ValueError, KeyError):
                continue

        # Store the finish time for each driver in this race
        for driver_num, lap_data in driver_laps.items():
            if driver_num not in driver_finish_times:
                driver_finish_times[driver_num] = {}
            driver_finish_times[driver_num][race_name] = lap_data["elapsed"]

# Save total (per-race) times to JSON for UI consumption
total_times_output_path = "../data/processed/driver_total_times.json"
with open(total_times_output_path, "w") as f:
    json.dump(driver_finish_times, f, indent=2)

print(f"Extracted race finish times for {len(driver_finish_times)} drivers")
print(f"Saved to {total_times_output_path}")
