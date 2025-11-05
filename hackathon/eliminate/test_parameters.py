# ABOUTME: Test different parameter sets to see impact on classification distribution
# ABOUTME: Simulates classification with modified thresholds before actually implementing them

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

script_dir = Path(__file__).parent.parent

# Load current classifications
print("\n" + "=" * 60)
print("CURRENT CLASSIFICATION RESULTS")
print("=" * 60)

classifications = pd.read_csv(script_dir / "data/processed/lap_classifications.csv")

total = len(classifications)
aggressive = len(classifications[classifications["classification"] == "Aggressive"])
optimal = len(classifications[classifications["classification"] == "Optimal"])
conservative = len(classifications[classifications["classification"] == "Conservative"])

print(f"\nTotal zones: {total:,}")
print(f"  Aggressive:   {aggressive:5} ({aggressive / total * 100:5.1f}%)")
print(f"  Optimal:      {optimal:5} ({optimal / total * 100:5.1f}%)")
print(f"  Conservative: {conservative:5} ({conservative / total * 100:5.1f}%)")

# Event stats
print("\nEvent Counts:")
print(f"  Wheelspin events:  {classifications['wheelspin'].sum():5}")
print(f"  Understeer events: {classifications['understeer'].sum():5}")
print(f"  Oversteer events:  {classifications['oversteer'].sum():5}")

# Analyze by fastest vs slowest drivers
print("\n" + "=" * 60)
print("CURRENT RESULTS BY DRIVER SPEED")
print("=" * 60)

# Load driver times to identify fast vs slow
driver_times = pd.read_json(script_dir / "data/processed/driver_total_times.json")

# Convert to list of (driver, time) for R1
r1_times = []
for driver, times in driver_times.items():
    if "R1" in times:
        time_str = times["R1"]
        # Parse MM:SS.mmm format
        parts = time_str.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])
        total_seconds = minutes * 60 + seconds
        r1_times.append((int(driver), total_seconds))

r1_times.sort(key=lambda x: x[1])

# Top 5 and bottom 5
fast_drivers = [d[0] for d in r1_times[:5]]
slow_drivers = [d[0] for d in r1_times[-5:]]

print(f"\nFast drivers (top 5): {fast_drivers}")
print(f"Slow drivers (bottom 5): {slow_drivers}")

for label, drivers in [("Fast", fast_drivers), ("Slow", slow_drivers)]:
    driver_data = classifications[classifications["vehicle_number"].isin(drivers)]
    if len(driver_data) == 0:
        continue

    total = len(driver_data)
    agg = len(driver_data[driver_data["classification"] == "Aggressive"])
    opt = len(driver_data[driver_data["classification"] == "Optimal"])
    cons = len(driver_data[driver_data["classification"] == "Conservative"])

    print(f"\n{label} Drivers ({len(drivers)} drivers, {total} zones):")
    print(f"  Aggressive:   {agg:5} ({agg / total * 100:5.1f}%)")
    print(f"  Optimal:      {opt:5} ({opt / total * 100:5.1f}%)")
    print(f"  Conservative: {cons:5} ({cons / total * 100:5.1f}%)")

# Now simulate with different parameter sets
print("\n" + "=" * 60)
print("SIMULATED RESULTS WITH PROPOSED PARAMETERS")
print("=" * 60)

# Define parameter sets to test
parameter_sets = {
    "Current": {
        "optimal_threshold": 0.90,
        "macro_coverage": 0.08,
        "macro_run_length": 6,
        "use_and_logic": False,
        "require_high_util": False,
    },
    "Proposed (Other LLM)": {
        "optimal_threshold": 0.85,
        "macro_coverage": 0.12,
        "macro_run_length": 8,
        "use_and_logic": True,
        "require_high_util": True,
        "high_util_threshold": 0.85,
    },
    "Conservative Alt": {
        "optimal_threshold": 0.80,
        "macro_coverage": 0.15,
        "macro_run_length": 10,
        "use_and_logic": True,
        "require_high_util": True,
        "high_util_threshold": 0.80,
    },
    "Balanced (Recommended)": {
        "optimal_threshold": 0.81,
        "macro_coverage": 0.11,
        "macro_run_length": 7,
        "use_and_logic": True,
        "require_high_util": True,
        "high_util_threshold": 0.85,
    },
}

for param_name, params in parameter_sets.items():
    print(f"\n{'-' * 60}")
    print(f"Parameter Set: {param_name}")
    print(f"{'-' * 60}")

    # Show parameters
    print("\nParameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Reclassify based on parameters
    def reclassify(row, params):
        # Check macro-event criteria
        coverage = float(row["event_coverage"])
        run_len = int(row["max_event_run_len"])
        utilization = float(row["avg_utilization"])

        if params["use_and_logic"]:
            # AND logic: both conditions must be met
            is_macro = (
                coverage >= params["macro_coverage"]
                and run_len >= params["macro_run_length"]
            )
        else:
            # OR logic: either condition
            is_macro = (
                coverage >= params["macro_coverage"]
                or run_len >= params["macro_run_length"]
            )

        # Apply high utilization requirement for aggressive
        if params.get("require_high_util", False):
            is_macro = is_macro and (utilization >= params["high_util_threshold"])

        # Classify
        if is_macro:
            return "Aggressive"
        elif utilization >= params["optimal_threshold"]:
            return "Optimal"
        else:
            return "Conservative"

    # Apply reclassification
    reclassified = classifications.copy()
    reclassified["new_classification"] = reclassified.apply(
        lambda row: reclassify(row, params), axis=1
    )

    # Calculate stats
    total = len(reclassified)
    agg = len(reclassified[reclassified["new_classification"] == "Aggressive"])
    opt = len(reclassified[reclassified["new_classification"] == "Optimal"])
    cons = len(reclassified[reclassified["new_classification"] == "Conservative"])

    print("\nOverall Distribution:")
    print(f"  Aggressive:   {agg:5} ({agg / total * 100:5.1f}%)")
    print(f"  Optimal:      {opt:5} ({opt / total * 100:5.1f}%)")
    print(f"  Conservative: {cons:5} ({cons / total * 100:5.1f}%)")

    # Fast vs slow breakdown
    for label, drivers in [("Fast", fast_drivers), ("Slow", slow_drivers)]:
        driver_data = reclassified[reclassified["vehicle_number"].isin(drivers)]
        if len(driver_data) == 0:
            continue

        total = len(driver_data)
        agg = len(driver_data[driver_data["new_classification"] == "Aggressive"])
        opt = len(driver_data[driver_data["new_classification"] == "Optimal"])
        cons = len(driver_data[driver_data["new_classification"] == "Conservative"])

        print(f"\n  {label} Drivers:")
        print(f"    Aggressive:   {agg:5} ({agg / total * 100:5.1f}%)")
        print(f"    Optimal:      {opt:5} ({opt / total * 100:5.1f}%)")
        print(f"    Conservative: {cons:5} ({cons / total * 100:5.1f}%)")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nTarget distributions (from other LLM):")
print("  Whole field: Optimal 15-25%, Conservative 65-80%, Aggressive 5-12%")
print("  Front pack:  Optimal 25-40%, Conservative 55-70%, Aggressive 3-8%")
