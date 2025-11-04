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
    race_name = "Race 1" if "Race 1" in race_path else "Race 2"

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

# For display, use the better (faster) of the two races
driver_times = {}
for driver_num, races in driver_finish_times.items():
    # If driver has both races, pick the faster one (shorter time)
    # Otherwise just use whatever race they have
    if len(races) == 2:
        # Compare times - they're in format MM:SS.SSS
        time1 = races.get("Race 1", "99:99.999")
        time2 = races.get("Race 2", "99:99.999")
        driver_times[driver_num] = time1 if time1 < time2 else time2
    else:
        # Only one race available
        driver_times[driver_num] = list(races.values())[0]

# Save to JSON
output_path = "../data/processed/driver_best_laps.json"
with open(output_path, "w") as f:
    json.dump(driver_times, f, indent=2)

print(f"Extracted race finish times for {len(driver_times)} drivers")
print(f"Saved to {output_path}")
