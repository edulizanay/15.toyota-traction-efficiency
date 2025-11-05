# ABOUTME: Check Pro vs Am driver distribution and aggressive classification rates
# ABOUTME: Validates if detection thresholds are too strict by comparing Pro and Am performance

import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent.parent

# Load sector analysis files (has CLASS column with Pro/Am)
print("\n=== Loading Pro/Am Classification from USAC Data ===")
r1_sectors = pd.read_csv(
    script_dir / "data/input/23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV",
    sep=";",
)
r2_sectors = pd.read_csv(
    script_dir / "data/input/23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV",
    sep=";",
)

# Extract unique vehicle -> class mapping
r1_mapping = r1_sectors[["NUMBER", "CLASS"]].drop_duplicates()
r2_mapping = r2_sectors[["NUMBER", "CLASS"]].drop_duplicates()
class_mapping = pd.concat([r1_mapping, r2_mapping]).drop_duplicates()

print(f"\nTotal vehicles: {len(class_mapping)}")
print("\nClass Distribution:")
print(class_mapping["CLASS"].value_counts())

print("\n=== Vehicle Numbers by Class ===")
pro_vehicles = class_mapping[class_mapping["CLASS"] == "Pro"]["NUMBER"].tolist()
am_vehicles = class_mapping[class_mapping["CLASS"] == "Am"]["NUMBER"].tolist()

print(f"\nPro vehicles ({len(pro_vehicles)}): {sorted(pro_vehicles)}")
print(f"Am vehicles ({len(am_vehicles)}): {sorted(am_vehicles)}")

# Load our lap classifications
print("\n=== Loading Our Classification Results ===")
classifications = pd.read_csv(script_dir / "data/processed/lap_classifications.csv")


# Add Pro/Am label to our classifications
def get_class(vehicle_num):
    if vehicle_num in pro_vehicles:
        return "Pro"
    elif vehicle_num in am_vehicles:
        return "Am"
    else:
        return "Unknown"


classifications["driver_class"] = classifications["vehicle_number"].apply(get_class)

print(f"\nTotal zone classifications: {len(classifications)}")
print("\nDriver Class Distribution in our data:")
print(classifications["driver_class"].value_counts())

# Compare aggressive rates between Pro and Am
print("\n=== AGGRESSIVE CLASSIFICATION RATES ===")

for driver_class in ["Pro", "Am"]:
    class_data = classifications[classifications["driver_class"] == driver_class]

    if len(class_data) == 0:
        print(f"\n{driver_class}: No data")
        continue

    total = len(class_data)
    aggressive = len(class_data[class_data["classification"] == "Aggressive"])
    optimal = len(class_data[class_data["classification"] == "Optimal"])
    conservative = len(class_data[class_data["classification"] == "Conservative"])

    print(f"\n{driver_class} Drivers:")
    print(f"  Total zones: {total}")
    print(f"  Aggressive: {aggressive} ({aggressive / total * 100:.1f}%)")
    print(f"  Optimal: {optimal} ({optimal / total * 100:.1f}%)")
    print(f"  Conservative: {conservative} ({conservative / total * 100:.1f}%)")

    # Event breakdown
    wheelspin = class_data["wheelspin"].sum()
    understeer = class_data["understeer"].sum()
    oversteer = class_data["oversteer"].sum()

    print("  Events:")
    print(f"    Wheelspin: {wheelspin} zones")
    print(f"    Understeer: {understeer} zones")
    print(f"    Oversteer: {oversteer} zones")

print("\n=== ANALYSIS ===")
pro_data = classifications[classifications["driver_class"] == "Pro"]
am_data = classifications[classifications["driver_class"] == "Am"]

if len(pro_data) > 0 and len(am_data) > 0:
    pro_agg_pct = (
        len(pro_data[pro_data["classification"] == "Aggressive"]) / len(pro_data) * 100
    )
    am_agg_pct = (
        len(am_data[am_data["classification"] == "Aggressive"]) / len(am_data) * 100
    )

    print(f"\nPro Aggressive Rate: {pro_agg_pct:.1f}%")
    print(f"Am Aggressive Rate: {am_agg_pct:.1f}%")
    print(f"Difference: {am_agg_pct - pro_agg_pct:.1f}%")

    if abs(pro_agg_pct - am_agg_pct) < 10:
        print("\n⚠️  THRESHOLDS ARE TOO STRICT!")
        print("Pro and Am drivers show similar aggressive rates.")
        print("This suggests the detection logic is flagging normal corrections,")
        print("not actual mistakes.")
    else:
        print("\n✓ Thresholds appear reasonable")
        print("Clear difference between Pro and Am aggressive rates.")
