# Traction Efficiency Analysis - Implementation Plan

## 1. Objective

Build a traction efficiency analysis tool that classifies driver performance as **Conservative** (leaving time), **Aggressive** (wheelspin/sliding), or **Optimal** (using grip efficiently) using friction circle physics.

**Core insight:** `total_G = sqrt(accx² + accy²)` represents total grip usage. Compare actual grip usage against each driver's maximum envelope to detect inefficiencies.

**Deliverable:** Interactive D3.js dashboard showing:
- Track view with color-coded turn zones (green/yellow/red by performance)
- Friction circle hexbin heatmap per driver/zone
- Lap classification timeline and time lost estimates

---

## 2. What We Already Have

### From Old Repo (`14.toyota-hackathon/deliverables/`)

#### Python Code (track-agnostic utilities):
- `convert_gps_to_meters()` - GPS (lat/lon) → UTM (x/y meters)
- `compute_centerline()` - Auto-generate smooth track centerline from GPS telemetry
- `project_points_onto_centerline()` - Map GPS points → track distance (1D)
- `resample_by_distance()` - Resample GPS path to uniform spacing
- `smooth_periodic()` - Savitzky-Golay smoothing with periodic wrapping
- `rotate_coordinates()` - 2D rotation (for visualization only, NOT data storage)
- `filter_racing_laps()` - Filter to racing laps (3500-4000m heuristic)

#### Manual Assets (Barber-specific):
- `corner_labels.json` - Manually placed corner labels (C1-C17) with x/y positions
- `Barber-Motorsports-Park.png` - Reference track image
- `pit_lane.json` - Pre-extracted pit lane GPS path (from vehicle 13, lap 2)

#### Data Files:
- `telemetry.csv` (1.5GB) - R1 Barber telemetry with all parameters
- `usac.csv` - USAC race results

---

## 3. High-Level Implementation Steps

### Step 1: Setup & Data Loading

**Create repo structure:**
```
src/
├── geometry.py          # Port GPS/centerline utilities
├── data_loader.py       # Chunked telemetry loading
├── turn_detector.py     # Auto-detect turn zones from lateral G
├── friction_envelope.py # Build grip envelopes per driver/zone
├── classifier.py        # Classify laps as Conservative/Aggressive/Optimal
└── exporter.py          # Export JSON/CSV for D3.js

data/
├── input/               # Raw telemetry + USAC
├── processed/           # Generated artifacts (CSV/JSON)
└── assets/              # Manual files (corner labels, PNG)

frontend/
├── index.html           # D3.js dashboard
├── track-view.js        # Track map visualization
└── analytics.js         # Friction circle + tables
```

**Port existing functions:**
- Copy `geometry.py` utilities from old repo (GPS conversion, centerline, projection)
- Adapt `data_loader.py` to load new parameters (`accx_can`, `accy_can`)

---

### Step 2: Generate Track Geometry (Track-Agnostic)

**Input:** `telemetry.csv` (GPS coordinates from any vehicle/lap)

**Process:**
1. Load GPS coordinates (`VBOX_Long_Minutes`, `VBOX_Lat_Min`)
2. Convert to UTM meters: `x_meters, y_meters = convert_gps_to_meters(lon, lat)`
3. Generate centerline: `centerline_x, centerline_y = compute_centerline(telemetry_df)`
4. Save as `data/processed/track_centerline.csv`

**Note:** Do NOT rotate data - store in raw UTM frame. Apply `rotate_coordinates()` only in D3.js rendering layer.

**Existing functions:**
- ✅ `convert_gps_to_meters()`
- ✅ `compute_centerline()`
- ✅ `resample_by_distance()`, `smooth_periodic()`

---

### Step 3: Auto-Detect Turn Zones

**Algorithm:**
1. Filter telemetry to racing laps (3500-4000m)
2. Calculate `|accy_can|` for all samples
3. Keep samples where `|accy_can| > P75` (75th percentile = real cornering)
4. Project GPS → track distance: `track_dist = project_points_onto_centerline(x, y, centerline_x, centerline_y)`
5. Cluster on 1D track distance using DBSCAN (`eps=50m`, `min_samples=20`)
6. For each cluster: compute zone boundaries (2.5th to 97.5th percentile of track distance)
7. Save as `data/processed/turn_zones.json`

**Output format:**
```json
[
  {
    "zone_id": 1,
    "start_distance_m": 0,
    "end_distance_m": 235,
    "name": "Turn 1",
    "avg_lateral_g": 0.85,
    "bounds": {"x_min": ..., "x_max": ..., "y_min": ..., "y_max": ...}
  }
]
```

**Existing functions:**
- ✅ `project_points_onto_centerline()`
- ✅ `filter_racing_laps()`
- 🆕 DBSCAN clustering (use scikit-learn)

**Key fix:** Cluster on **track distance (1D)**, not GPS XY (2D), to avoid spatial artifacts on overlapping sections.

---

### Step 4: Build Friction Envelopes

**Per driver, per turn zone:**
1. Filter telemetry to racing laps in this zone
2. Calculate `total_G = sqrt(accx² + accy²)` for all samples
3. Bin by `accy` (lateral G) into 20 bins
4. For each bin: `max_total_G = 95th percentile(total_G)` in that bin
5. Connect bins → envelope curve
6. Save as `data/processed/friction_envelopes.json`

**Output format:**
```json
{
  "78": {
    "1": [
      {"accy": 0.0, "total_g_max": 1.15},
      {"accy": 0.2, "total_g_max": 1.25},
      {"accy": 0.5, "total_g_max": 1.35}
    ]
  }
}
```

**Existing functions:**
- 🆕 Need to implement envelope calculation
- ✅ Can use `np.percentile()` for 95th percentile

---

### Step 5: Classify Laps

**Per driver, per lap, per turn zone:**
1. Calculate average `total_G` for this lap segment
2. Look up envelope max for this `accy` range
3. `utilization = avg_total_G / envelope_max`
4. Detect over-limit events (wheelspin, understeer, oversteer - see `@traction-analysis-concept.md`)
5. Classify:
   - If `over_limit_events > 0`: **Aggressive**
   - Else if `utilization < 0.95`: **Conservative**
   - Else: **Optimal**
6. Estimate time lost (see concept doc)
7. Save as `data/processed/lap_classifications.csv`

**Output format:**
```csv
vehicle_number,lap,zone_id,classification,avg_utilization,over_limit_events,time_lost_s
78,5,1,Conservative,0.87,0,0.12
78,5,2,Optimal,0.98,0,0.0
```

**Existing functions:**
- 🆕 Need to implement classification logic
- 🆕 Need event detection (wheelspin, understeer, oversteer)

---

### Step 6: Extract Pit Lane (Optional)

**Approach:**
1. Load USAC sector data (`23_AnalysisEnduranceWithSections_*.CSV`)
2. Find laps where `CROSSING_FINISH_LINE_IN_PIT == 1`
3. **Timebase alignment:** Join USAC → telemetry via `vehicle_number` + `timestamp` (match within ±5 seconds)
4. Extract full GPS path for those laps
5. **Additional filter:** Keep only samples where `speed < 80 km/h` (pit speed limit)
6. **Geofence:** Define pit entry/exit zones from manual inspection, filter GPS to those bounds
7. Stitch together multiple laps, smooth, save as `data/processed/pit_lane.json`

**Existing functions:**
- ✅ `smooth_periodic()` for smoothing GPS path
- 🆕 Need USAC → telemetry join logic
- 🆕 Need geofence definition (can extract from existing `pit_lane.json`)

**Alternative:** Copy existing `pit_lane.json` from old repo for now, automate later.

---

### Step 7: Build D3.js Dashboard

**Tab 1: Track View**
- Load `track_centerline.csv`, `turn_zones.json`, `corner_labels.json`
- Render track with D3.js line + polygon zones
- Color zones by classification (aggregate across all drivers or selected driver)
- Apply `rotate_coordinates()` in JavaScript to match PNG orientation
- Add corner labels

**Tab 2: Analytics**
- Left panel: Table of driver/zone/utilization/events (load `lap_classifications.csv`)
- Right panel: Friction circle hexbin
  - Load `friction_envelopes.json` + raw telemetry samples
  - D3.js hexbin of (accx, |accy|) colored by density
  - Overlay envelope boundary line

**Existing functions:**
- 🆕 Need to port `rotate_coordinates()` to JavaScript
- 🆕 D3.js visualization code (new implementation)

---

## 4. Repository Structure

```
15.toyota-hackathon-traction-efficiency/
├── README.md                      # Project overview
├── context.md                     # Hackathon rules and data info
├── traction-analysis-concept.md   # Technical concept
├── implementation-plan.md         # This file
├── migration-plan.md              # Migration notes
│
├── hackathon/                     # Main working directory
│   │
│   ├── src/                       # Python processing pipeline
│   │   ├── __init__.py
│   │   ├── geometry.py            # GPS/centerline utilities (ported)
│   │   ├── data_loader.py         # Chunked telemetry loading
│   │   ├── pit_extractor.py       # Pit lane extraction from USAC + telemetry
│   │   ├── turn_detector.py       # Auto-detect turn zones (DBSCAN)
│   │   ├── friction_envelope.py   # Build grip envelopes per driver/zone
│   │   ├── classifier.py          # Lap classification logic
│   │   └── exporter.py            # Export JSON/CSV for D3.js
│   │
│   ├── data/
│   │   ├── input/                 # Raw data (symlinked, not committed)
│   │   │   ├── telemetry.csv      # → ../../14.toyota-hackathon/deliverables/data/input/telemetry.csv
│   │   │   └── usac_sectors.csv   # USAC sector analysis
│   │   ├── processed/             # Generated artifacts (committed)
│   │   │   ├── track_centerline.csv
│   │   │   ├── pit_lane.json
│   │   │   ├── turn_zones.json
│   │   │   ├── friction_envelopes.json
│   │   │   └── lap_classifications.csv
│   │   └── assets/                # Manual files (committed)
│   │       ├── corner_labels.json
│   │       └── track_reference.png
│   │
│   ├── scripts/                   # Pipeline execution scripts
│   │   ├── 1_generate_track.py    # Step 1: Generate centerline
│   │   ├── 2_extract_pitlane.py   # Step 2: Extract pit lane
│   │   ├── 3_detect_turns.py      # Step 3: Detect turn zones
│   │   ├── 4_build_envelopes.py   # Step 4: Build friction envelopes
│   │   └── 5_classify_laps.py     # Step 5: Classify laps
│   │
│   ├── frontend/                  # D3.js visualization
│   │   ├── index.html             # Main dashboard
│   │   ├── css/
│   │   │   └── styles.css         # Dashboard styles
│   │   ├── js/
│   │   │   ├── main.js            # Dashboard initialization
│   │   │   ├── track-view.js      # Track map visualization
│   │   │   ├── analytics.js       # Friction circle + tables
│   │   │   └── utils.js           # Shared utilities (rotation, data loading)
│   │   └── test/                  # Test HTML pages for incremental dev
│   │       ├── test_track.html    # Test: Track rendering only
│   │       ├── test_pitlane.html  # Test: Track + pit lane
│   │       └── test_zones.html    # Test: Track + turn zones
│   │
│   ├── requirements.txt           # Python dependencies
│   ├── .gitignore                 # Exclude data/input/, large files
│   └── README.md                  # Hackathon-specific notes
│
└── deliverables/                  # Legacy files (may delete later)
    └── ...
```

---

## 5. High-Level Implementation Steps

### Step 1: Create Folder Structure
**Action:**
```bash
cd hackathon/
mkdir -p src scripts data/{input,processed,assets} frontend/{css,js,test}
```

**Outcome:** Clean workspace ready for development

---

### Step 2: Delete Legacy Files
**Action:**
- Remove `/deliverables/` directory (old brake analysis code)
- Keep only documentation in root: `README.md`, `context.md`, `traction-analysis-concept.md`

**Outcome:** No confusion between old/new code

---

### Step 3: Install Dependencies
**Action:**
```bash
# Create requirements.txt
echo "pandas
numpy
scipy
scikit-learn
pyproj
matplotlib" > requirements.txt

# Install
pip install -r requirements.txt
```

**Outcome:** Python environment ready

---

### Step 4: Port Geometry Utilities
**Action:**
- Copy `geometry.py` functions from old repo to `hackathon/src/geometry.py`
- Copy `convert_gps_to_meters()`, `project_points_onto_centerline()` to `hackathon/src/geometry.py`

**Files created:**
- `hackathon/src/geometry.py`
- `hackathon/src/data_loader.py`

**Outcome:** Track-agnostic utilities available

---

### Step 5: Generate Track Centerline + Render in D3.js → **Show to Edu**

**Processing:**
```bash
python scripts/1_generate_track.py
```
- Load telemetry GPS coordinates
- Convert GPS → UTM meters
- Generate smooth centerline
- Save to `data/processed/track_centerline.csv`

**Rendering:**
- Create `frontend/test/test_track.html`
- Load `track_centerline.csv` with D3.js
- Render track as SVG path
- Apply rotation to match PNG orientation

**Outcome:** Interactive track map showing centerline → **Review with Edu**

---

### Step 6: Extract Pit Lane + Render on Track → **Show to Edu**

**Processing:**
```bash
python scripts/2_extract_pitlane.py
```
- Load USAC sector data
- Find laps with `CROSSING_FINISH_LINE_IN_PIT == 1`
- Join USAC → telemetry via vehicle + timestamp
- Filter to speed < 80 km/h
- Stitch GPS paths, smooth
- Save to `data/processed/pit_lane.json`

**Rendering:**
- Update `frontend/test/test_pitlane.html`
- Load `pit_lane.json`
- Render as dashed line on track

**Outcome:** Track + pit lane visualization → **Review with Edu**

---

### Step 7: Auto-Detect Turn Zones + Render → **Show to Edu**

**Processing:**
```bash
python scripts/3_detect_turns.py
```
- Filter racing laps (3500-4000m)
- Calculate `|accy_can|`, keep P75+ samples
- Project GPS → track distance
- DBSCAN clustering on 1D track distance
- Compute zone boundaries (P2.5 to P97.5)
- Save to `data/processed/turn_zones.json`

**Rendering:**
- Update `frontend/test/test_zones.html`
- Load `turn_zones.json`
- Render colored polygons for each turn zone
- Add zone labels

**Outcome:** Track with auto-detected turn zones → **Review with Edu**

---

### Step 8: Build Friction Envelopes → **Show Sample Data to Edu**

**Processing:**
```bash
python scripts/4_build_envelopes.py
```
- Calculate `total_G = sqrt(accx² + accy²)` for all samples
- Group by driver + zone
- Bin by `accy`, find 95th percentile `total_G` per bin
- Save to `data/processed/friction_envelopes.json`

**Validation:**
- Print sample envelope for one driver/zone
- Plot envelope curve with matplotlib

**Outcome:** Friction envelope JSON → **Review sample data with Edu**

---

### Step 9: Classify Laps → **Show Sample Classifications to Edu**

**Processing:**
```bash
python scripts/5_classify_laps.py
```
- For each lap/zone: calculate average `total_G`
- Compare to envelope max
- Detect over-limit events (wheelspin, understeer, oversteer)
- Classify as Conservative/Aggressive/Optimal
- Estimate time lost
- Save to `data/processed/lap_classifications.csv`

**Validation:**
- Print sample classifications for one driver
- Show distribution: % Conservative vs Aggressive vs Optimal

**Outcome:** Lap classification CSV → **Review sample with Edu**

---

### Step 10: Build Full D3.js Dashboard → **Show to Edu**

**Rendering:**
- Create `frontend/index.html` with two tabs
- **Tab 1: Track View**
  - Load centerline, pit lane, zones, classifications
  - Color zones by aggregated performance
  - Click zone → filter to that zone
- **Tab 2: Analytics**
  - Left panel: Table of driver/zone/utilization/events
  - Right panel: Friction circle hexbin heatmap
  - Interactive filtering

**Outcome:** Complete interactive dashboard → **Final review with Edu**

---

## 6. Key Implementation Fixes

### ✅ Cluster on Track Distance (1D), Not GPS (2D)
- After projecting GPS → centerline, use **track distance** for DBSCAN
- Avoids spatial artifacts on overlapping track sections

### ✅ Rotation Only in Rendering
- Store all processed data in raw UTM coordinates
- Apply rotation in D3.js visualization layer only
- Never rotate persisted artifacts

### ✅ Timebase Alignment for USAC → Telemetry
- USAC lap numbers can be unreliable
- Join via `vehicle_number` + match `ELAPSED` time to telemetry `timestamp` (within ±5s window)
- Document join logic clearly

### ✅ Improved Pit Lane Extraction
- Combine `CROSSING_FINISH_LINE_IN_PIT` flag with:
  - Speed filter (`speed < 80 km/h`)
  - Geofence near pit entry/exit
- Stitch multiple laps to recover full pit trace
