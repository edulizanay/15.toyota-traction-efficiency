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
│   ├── src/                       # Python modules
│   │   ├── geometry.py            # GPS/centerline/projection utilities
│   │   └── analysis.py            # Turn detection, envelopes, classification
│   │
│   ├── scripts/
│   │   ├── generate_geometry.py   # Generate track centerline + pit lane
│   │   └── analyze_traction.py    # Detect turns, build envelopes, classify laps
│   │
│   ├── data/
│   │   ├── input/                 # Raw data (symlinked, not committed)
│   │   │   ├── telemetry.csv      # → ../deliverables/data/input/telemetry.csv
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
│   ├── dashboard.html             # Single-page dashboard (track + analytics)
│   ├── dashboard.js               # All D3.js logic (track view + friction circle)
│   ├── dashboard.css              # All styles
│   │
│   ├── requirements.txt           # Python dependencies
│   └── .gitignore                 # Exclude data/input/, large files
│
└── deliverables/                  # Legacy files (may delete later)
    └── ...
```

---

## 5. High-Level Implementation Steps

### Step 1: Create Folder Structure
**Action:**
```bash
mkdir -p hackathon/{src,scripts,data/{input,processed,assets}}
touch hackathon/{dashboard.html,dashboard.js,dashboard.css}
```

**Outcome:** Clean workspace ready for development

---

### Step 2: Install Dependencies
**Action:**
```bash
cd hackathon/
echo "pandas
numpy
scipy
scikit-learn
pyproj" > requirements.txt

pip install -r requirements.txt
```

**Outcome:** Python environment ready

---

### Step 3: Port Geometry Utilities
**Action:**
- Copy functions from old repo to `src/geometry.py`:
  - `convert_gps_to_meters()`, `compute_centerline()`, `project_points_onto_centerline()`
  - `resample_by_distance()`, `smooth_periodic()`, `rotate_coordinates()`

**Files created:**
- `src/geometry.py` (all GPS/track utilities)

**Outcome:** Track-agnostic utilities available

---

### Step 4: Generate Track + Pit Lane → **Show to Edu**

**Processing:**
```bash
python scripts/generate_geometry.py
```
- Load telemetry GPS → convert to UTM → generate centerline
- Extract pit lane (USAC + telemetry join + speed filter + smoothing)
- Save:
  - `data/processed/track_centerline.csv`
  - `data/processed/pit_lane.json`

**Rendering:**
- Create basic `dashboard.html` with D3.js
- Load centerline + pit lane
- Render track as SVG (apply rotation to match PNG)

**Outcome:** Interactive track map → **Review with Edu**

---

### Step 5: Detect Turn Zones + Render → **Show to Edu**

**Processing:**
```bash
python scripts/analyze_traction.py --step=detect_turns
```
- Filter racing laps, keep high `|accy_can|` samples
- Project GPS → track distance, cluster with DBSCAN
- Save: `data/processed/turn_zones.json`

**Rendering:**
- Update `dashboard.html`: render turn zones as colored polygons
- Add zone labels

**Outcome:** Track with auto-detected zones → **Review with Edu**

---

### Step 6: Build Friction Envelopes → **Show Sample Data**

**Processing:**
```bash
python scripts/analyze_traction.py --step=build_envelopes
```
- Calculate `total_G = sqrt(accx² + accy²)`
- Group by driver/zone, bin by `accy`, find 95th percentile
- Save: `data/processed/friction_envelopes.json`

**Validation:**
- Print sample envelope for one driver/zone

**Outcome:** Envelope data ready → **Review with Edu**

---

### Step 7: Classify Laps → **Show Sample Classifications**

**Processing:**
```bash
python scripts/analyze_traction.py --step=classify_laps
```
- Compare actual vs envelope, detect over-limit events
- Classify as Conservative/Aggressive/Optimal
- Save: `data/processed/lap_classifications.csv`

**Validation:**
- Show distribution for one driver

**Outcome:** Classifications ready → **Review with Edu**

---

### Step 8: Build Full Dashboard → **Show to Edu**

**Rendering:**
- Complete `dashboard.html` with two views:
  - **Track View**: Color zones by performance, interactive filtering
  - **Friction Circle**: Hexbin heatmap + envelope overlay
- Load all processed data, add interactivity

**Outcome:** Complete dashboard → **Final review with Edu**

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
