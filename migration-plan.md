# Migration Plan: Brake Analysis → Traction Efficiency Analysis

## 1. Context

Pivoting from brake point consistency analysis to **traction efficiency analysis** for new hackathon submission. Building everything fresh in **D3.js** (track view + analytics). See `@traction-analysis-concept.md` for full technical concept.

---

## 2. What We Need from Old Repo

### Python Utilities (port from `src/visuals/`, `src/data_processing.py`)

**Track geometry generation** (all track-agnostic):
- `compute_centerline()` - Auto-generate from GPS telemetry
- `convert_gps_to_meters()` - GPS → UTM conversion
- `project_points_onto_centerline()` - Map GPS to track distance
- `rotate_coordinates()` - Match PNG orientation
- Smoothing/resampling functions

**Spatial assignment**:
- Use GPS coordinates (x_meters, y_meters) to project onto centerline
- Get track distance for clustering (don't use `Laptrigger_lapdist_dls` directly - it wraps per lap)

### Manually Created Assets

**Must copy** (hard to auto-generate):
- `corner_labels.json` - Manually placed corner labels (C1-C17)
- `Barber-Motorsports-Park.png` - Reference image

### Pit Lane Extraction Strategy

**Data source:** `CROSSING_FINISH_LINE_IN_PIT` column in USAC sector analysis files (`23_AnalysisEnduranceWithSections_*.CSV`)

**Approach:**
1. Find all laps where drivers crossed finish line in pit
2. Extract full GPS paths for those laps (entry → exit)
3. Stitch together multiple laps to get complete pit lane trace
4. Smooth and save as `pit_lane.json`

### Telemetry Parameters Needed

From `R1_barber_telemetry_data.csv`:
- `accx_can`, `accy_can` - G-forces (NEW for friction circle)
- `speed`, `aps`, `Steering_Angle` - Context
- `VBOX_Long_Minutes`, `VBOX_Lat_Min` - GPS coordinates
- `vehicle_number`, `lap`, `timestamp` - Identifiers

Note: `Laptrigger_lapdist_dls` NOT used for clustering (wraps per lap)

---

## 3. Data Format Strategy

### File Formats

**CSV for tabular data:**
- Lap classifications, driver summaries, telemetry samples
- Easy to inspect, debug, native D3.js support

**JSON for nested/hierarchical:**
- Turn zones, friction envelopes, corner labels, pit lane
- Natural for nested structures (driver → zone → envelope points)

**Skip:** Parquet (overkill), GeoJSON (unnecessary complexity)

### Computation Split

**Pre-compute in Python:**
- Turn zone detection (DBSCAN on GPS coordinates)
- Friction envelopes per driver/zone
- Lap classifications
- All GPS → UTM conversions
- Track distance projections
- Statistical aggregations

**Compute in browser (D3.js):**
- View filtering/sorting
- Visual scales/axes
- Hover tooltips
- Color interpolations
- Display calculations only

**Reason:** Heavy computation in Python = faster, handles 1.5GB data. Browser renders pre-processed results.

### Generated Artifact Structure

```
data/
├── input/
│   ├── telemetry.csv              # Raw 1.5GB from old repo
│   └── usac_sectors.csv           # USAC timing data
├── processed/
│   ├── track_centerline.csv       # x_meters, y_meters (generated)
│   ├── pit_lane.json              # [{x_meters, y_meters}] (extracted)
│   ├── turn_zones.json            # [{zone_id, start_dist, end_dist, bounds, avg_lateral_g}]
│   ├── friction_envelopes.json    # {driver_id: {zone_id: [{accy, total_g_max}]}}
│   ├── lap_classifications.csv    # vehicle, lap, zone, classification, util%, events, time_lost
│   └── telemetry_sample.csv       # Small subset for browser hexbin (optional)
└── assets/
    ├── corner_labels.json         # Manual corner positions (copied)
    └── track_reference.png        # PNG for orientation (copied)
```

---

## 4. Approach

1. **Port geometry utilities** from old repo
2. **Generate track** from telemetry (verify track-agnostic)
3. **Extract pit lane** from USAC + telemetry GPS paths
4. **Save generated files** to `data/processed/`
5. **Build traction analysis** per `@traction-analysis-concept.md`
6. **Build D3.js visualization** (track view + analytics)
