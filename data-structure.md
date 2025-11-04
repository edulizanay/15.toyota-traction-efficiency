# Data Directory Documentation

## Overview

This repository contains telemetry and timing data from the Toyota GR Cup race at Barber Motorsports Park. The data includes two races (R1 and R2) with detailed vehicle telemetry and official USAC timing/scoring data.

---

## File Structure

### Telemetry Data Files (Toyota GR Cup ECU Data)

#### Lap Timing Files

**`R1_barber_lap_start.csv` (65KB)**
**`R2_barber_lap_start.csv` (68KB)**

Records when each lap begins for each vehicle.

**Format:** CSV (comma-separated)

| Column | Type | Description |
|--------|------|-------------|
| `expire_at` | timestamp | Data expiration timestamp (empty in dataset) |
| `lap` | integer | Lap number |
| `meta_event` | string | Event identifier (e.g., "I_R06_2025-09-07") |
| `meta_session` | string | Session identifier (R1 or R2) |
| `meta_source` | string | Data source ("kafka:gr-raw") |
| `meta_time` | timestamp | Time message was received |
| `original_vehicle_id` | string | Original vehicle identifier (e.g., "GR86-002-000") |
| `outing` | integer | Outing number |
| `timestamp` | timestamp | ECU timestamp (may not be accurate) |
| `vehicle_id` | string | Vehicle identifier |
| `vehicle_number` | integer | Car number (0 if not assigned) |

---

**`R1_barber_lap_end.csv` (65KB)**
**`R2_barber_lap_end.csv` (68KB)**

Records when each lap ends for each vehicle.

**Format:** CSV (comma-separated)

| Column | Type | Description |
|--------|------|-------------|
| `expire_at` | timestamp | Data expiration timestamp (empty in dataset) |
| `lap` | integer | Lap number |
| `meta_event` | string | Event identifier |
| `meta_session` | string | Session identifier (R1 or R2) |
| `meta_source` | string | Data source |
| `meta_time` | timestamp | Time message was received |
| `original_vehicle_id` | string | Original vehicle identifier |
| `outing` | integer | Outing number |
| `timestamp` | timestamp | ECU timestamp |
| `vehicle_id` | string | Vehicle identifier |
| `vehicle_number` | integer | Car number |

---

**`R1_barber_lap_time.csv` (65KB)**
**`R2_barber_lap_time.csv` (68KB)**

Records lap completion times for each vehicle.

**Format:** CSV (comma-separated)

| Column | Type | Description |
|--------|------|-------------|
| `expire_at` | timestamp | Data expiration timestamp (empty in dataset) |
| `lap` | integer | Lap number |
| `meta_event` | string | Event identifier |
| `meta_session` | string | Session identifier (R1 or R2) |
| `meta_source` | string | Data source |
| `meta_time` | timestamp | Time message was received |
| `original_vehicle_id` | string | Original vehicle identifier |
| `outing` | integer | Outing number |
| `timestamp` | timestamp | ECU timestamp |
| `vehicle_id` | string | Vehicle identifier |
| `vehicle_number` | integer | Car number |

**Note:** Lap numbers may be erroneously reported (sometimes as lap #32768). Use time values for accurate lap determination.

---

#### Telemetry Data Files

**`R1_barber_telemetry_data.csv` (1.5GB)**
**`R2_barber_telemetry_data.csv` (1.5GB)**

Contains all vehicle telemetry parameters recorded during the race.

**Format:** CSV (comma-separated)
**Structure:** Long format (one row per parameter per timestamp)

| Column | Type | Description |
|--------|------|-------------|
| `expire_at` | timestamp | Data expiration timestamp (empty in dataset) |
| `lap` | integer | Lap number |
| `meta_event` | string | Event identifier |
| `meta_session` | string | Session identifier (R1 or R2) |
| `meta_source` | string | Data source ("kafka:gr-raw") |
| `meta_time` | timestamp | Time message was received |
| `original_vehicle_id` | string | Original vehicle identifier |
| `outing` | integer | Outing number |
| `telemetry_name` | string | Name of the telemetry parameter (see below) |
| `telemetry_value` | float | Value of the telemetry parameter |
| `timestamp` | timestamp | ECU timestamp |
| `vehicle_id` | string | Vehicle identifier |
| `vehicle_number` | integer | Car number |

**Telemetry Parameters (`telemetry_name` values):**

| Parameter | Description | Unit |
|-----------|-------------|------|
| `speed` | Actual vehicle speed | km/h |
| `gear` | Current gear selection | integer (0-6) |
| `nmot` | Engine RPM | RPM |
| `aps` | Accelerator pedal position | % (0-100) |
| `pbrake_f` | Front brake pressure | bar |
| `pbrake_r` | Rear brake pressure | bar |
| `accx_can` | Longitudinal acceleration | G's (+ = accelerating, - = braking) |
| `accy_can` | Lateral acceleration | G's (+ = left turn, - = right turn) |
| `Steering_Angle` | Steering wheel angle | degrees (0 = straight, - = CCW, + = CW) |
| `VBOX_Long_Minutes` | GPS longitude | degrees |
| `VBOX_Lat_Min` | GPS latitude | degrees |
| `Laptrigger_lapdist_dls` | Distance from start/finish line | meters |

**Note:** The `ath` (throttle blade position) parameter is not present in this dataset, only `aps` (accelerator pedal position).

---

### USAC Official Timing Data

These files contain official race timing and scoring data from the Al Kamel timing system.

**Format:** CSV (semicolon-separated `;`)

#### Race Results Files

**`03_Provisional Results_Race 1_Anonymized.CSV` (2.4KB)**
**`03_Provisional Results_Race 2_Anonymized.CSV` (2.4KB)**
**`03_Results GR Cup Race 2 Official_Anonymized.CSV` (2.4KB)**

Official race results with finishing positions and lap times.

| Column | Description |
|--------|-------------|
| `POSITION` | Finishing position |
| `NUMBER` | Car number |
| `STATUS` | Race status (Classified, DNF, etc.) |
| `LAPS` | Total laps completed |
| `TOTAL_TIME` | Total race time |
| `GAP_FIRST` | Gap to first place |
| `GAP_PREVIOUS` | Gap to previous position |
| `FL_LAPNUM` | Fastest lap number |
| `FL_TIME` | Fastest lap time |
| `FL_KPH` | Fastest lap speed (km/h) |
| `CLASS` | Class (Pro/Am) |
| `GROUP` | Group classification |
| `DIVISION` | Division (GR Cup) |
| `VEHICLE` | Vehicle model |
| `TIRES` | Tire specification |
| Other columns | ECM system metadata (mostly empty) |

---

**`05_Provisional Results by Class_Race 1_Anonymized.CSV` (1.6KB)**
**`05_Provisional Results by Class_Race 2_Anonymized.CSV` (1.6KB)**
**`05_Results by Class GR Cup Race 1 Official_Anonymized.CSV` (1.6KB)**

Results separated by class (Pro/Am).

**Same structure as race results files above.**

---

#### Sector Analysis Files

**`23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV` (139KB)**
**`23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV` (145KB)**

Detailed lap-by-lap sector timing analysis.

| Column | Description |
|--------|-------------|
| `NUMBER` | Car number |
| `DRIVER_NUMBER` | Driver number |
| `LAP_NUMBER` | Lap number |
| `LAP_TIME` | Total lap time |
| `LAP_IMPROVEMENT` | Improvement indicator |
| `CROSSING_FINISH_LINE_IN_PIT` | Pit lane flag |
| `S1`, `S2`, `S3` | Sector times (formatted) |
| `S1_IMPROVEMENT`, `S2_IMPROVEMENT`, `S3_IMPROVEMENT` | Sector improvement flags |
| `KPH` | Average lap speed (km/h) |
| `ELAPSED` | Total elapsed time |
| `HOUR` | Time of day |
| `S1_LARGE`, `S2_LARGE`, `S3_LARGE` | Large sector time flags |
| `TOP_SPEED` | Top speed on lap (km/h) |
| `PIT_TIME` | Pit stop duration |
| `CLASS` | Class (Pro/Am) |
| `GROUP` | Group |
| `MANUFACTURER` | Manufacturer |
| `FLAG_AT_FL` | Flag condition at finish line (GF=Green, FCY=Full Course Yellow) |
| `S1_SECONDS`, `S2_SECONDS`, `S3_SECONDS` | Sector times in seconds |
| `IM1a_time` through `FL_elapsed` | Intermediate timing points and elapsed times |

**Sector Breakdown:**
- **S1**: First sector (Start/Finish to first intermediate)
- **S2**: Second sector (First intermediate to second intermediate)
- **S3**: Third sector (Second intermediate to Start/Finish)
- **IM** points: Additional intermediate timing points (IM1a, IM1, IM2a, IM2, IM3a)

---

#### Weather Data Files

**`26_Weather_Race 1_Anonymized.CSV` (2.7KB)**
**`26_Weather_Race 2_Anonymized.CSV` (2.7KB)**

Weather conditions throughout the race session.

| Column | Description | Unit |
|--------|-------------|------|
| `TIME_UTC_SECONDS` | Unix timestamp | seconds |
| `TIME_UTC_STR` | Human-readable timestamp | string |
| `AIR_TEMP` | Air temperature | °C |
| `TRACK_TEMP` | Track surface temperature | °C (0 = not measured) |
| `HUMIDITY` | Relative humidity | % |
| `PRESSURE` | Atmospheric pressure | mbar |
| `WIND_SPEED` | Wind speed | km/h |
| `WIND_DIRECTION` | Wind direction | degrees (0-360) |
| `RAIN` | Rain indicator | 0=no rain, 1=rain |

---

#### Best Laps Analysis

**`99_Best 10 Laps By Driver_Race 1_Anonymized.CSV` (3.5KB)**
**`99_Best 10 Laps By Driver_Race 2_Anonymized.CSV` (3.5KB)**

Top 10 fastest laps for each driver with consistency analysis.

| Column | Description |
|--------|-------------|
| `NUMBER` | Car number |
| `VEHICLE` | Vehicle model |
| `CLASS` | Class (Pro/Am) |
| `TOTAL_DRIVER_LAPS` | Total laps completed by driver |
| `BESTLAP_1` through `BESTLAP_10` | Best lap times (1st through 10th) |
| `BESTLAP_1_LAPNUM` through `BESTLAP_10_LAPNUM` | Lap numbers for each best lap |
| `AVERAGE` | Average of best 10 laps |

---

## Data Quality Notes

### Known Issues

1. **Vehicle Numbers**: Some vehicles show car number `000`, meaning the number wasn't assigned to the ECU yet. Use chassis number for unique identification.

2. **Lap Count Errors**: Lap numbers sometimes erroneously reported (often as #32768). Time values are reliable for determining actual lap numbers.

3. **Timestamp Accuracy**:
   - `meta_time`: Reliable - when message was received
   - `timestamp`: May not be accurate - ECU time can drift

4. **Missing Parameters**: The `ath` (throttle blade position) parameter is not in the telemetry data, only `aps` (accelerator pedal position) is available.

### File Format Differences

- **Telemetry files**: Comma-separated (`,`)
- **USAC timing files**: Semicolon-separated (`;`)

### File Sizes

- Large files (1.5GB each): Telemetry data files - **DO NOT read entire file at once**
- Medium files (65-145KB): Lap timing and sector analysis
- Small files (<10KB): Results, weather, best laps

---

## Usage Recommendations

### For Analysis

1. **Start with USAC timing data** for overview (results, sectors, weather)
2. **Use lap_start/lap_end/lap_time** to understand lap structure
3. **Sample telemetry data** carefully due to large file size (1.5GB)
4. **Join data** using `vehicle_number`, `lap`, and `timestamp` fields

### For Performance

- Read telemetry files in chunks or filter by specific laps/vehicles
- Use `meta_time` for reliable temporal ordering
- Cross-reference lap numbers between telemetry and USAC data using time values

### For Insights

- Compare sector times across drivers (USAC analysis files)
- Analyze telemetry patterns during fastest laps (combine telemetry + best laps data)
- Study racing lines using GPS coordinates
- Correlate weather conditions with performance
- Identify braking points, throttle application, and cornering techniques
