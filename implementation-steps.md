# Traction Efficiency Analysis — Step‑by‑Step Checklist

Use this checklist to track progress. Keep outputs in `hackathon/data/processed/`.

Legend: [ ] pending, [x] done, [~] in progress

## 0) Geometry Baseline (done)
- [x] Convert GPS to meters and compute centerline (R1)
- [x] Compute inner/outer boundaries and save artifacts
  - Geometry: `track_centerline.csv`, `track_boundaries.json`

## 1) Project Telemetry To Track Distance
- [ ] Load R1/R2 telemetry in chunks (need: `VBOX_Long_Minutes`, `VBOX_Lat_Min`, `accx_can`, `accy_can`, `vehicle_number`, `lap`, `timestamp`)
- [ ] Convert GPS to meters
- [ ] Project each sample onto the centerline → add `track_distance_m`
- [ ] Persist intermediate (in memory or parquet/CSV per chunk) as needed

## 2) Detect Turn Zones (1D Clustering)
- [ ] Filter to racing laps (e.g., lap length sanity window)
- [ ] Keep samples with |`accy_can`| above a percentile threshold (e.g., P10)
- [ ] DBSCAN on 1D `track_distance_m` (suggest: `eps=50 m`, `min_samples=20`)
- [ ] For each cluster: compute robust start/end (2.5–97.5th percentile of `track_distance_m`)
- [ ] Save `turn_zones.json`
- [ ] QA: Expect ~17 zones at Barber; inspect gaps/overlaps

## 3) Build Friction Envelopes (per driver, per zone)
- [ ] Compute `total_G = sqrt(accx_can^2 + accy_can^2)` per sample
- [ ] Bin by `accy_can` (e.g., 20 bins uniform in |accy| or quantiles)
- [ ] For each bin: 95th percentile of `total_G` → envelope point
- [ ] Save per driver/zone curves → `friction_envelopes.json`
- [ ] QA: Plot one driver/zone to visually confirm shape and smoothness

## 4) Classify Laps + Estimate Time Lost
- [ ] Aggregate samples by (driver, lap, zone)
- [ ] Compute `avg_total_G` and find `envelope_max` for that `accy` range
- [ ] Utilization = `avg_total_G / envelope_max`
- [ ] (Optional) Detect events via short rolling trends:
  - Wheelspin: throttle↑ while `accx_can` trend↓
  - Understeer: steering↑ while `accy_can` plateaus
  - Oversteer: `accy_can` spike with `accx_can` drop
- [ ] Classification rules:
  - If events > 0 → Aggressive
  - Else if utilization < 0.95 → Conservative
  - Else → Optimal
- [ ] Estimate simple time lost from grip deficit over zone duration
- [ ] Save `lap_classifications.csv`
- [ ] QA: Distribution of classes per driver; spot‑check a few laps

## 5) Dashboard Deliverable
- [ ] Track view: color zones by latest classification; show corner labels
- [ ] Analytics view: friction‑circle hexbin of (`accx_can`, |`accy_can`|) with envelope overlay
- [ ] Controls: driver selector, lap filter; hover tooltips with utilization/time lost
- [ ] Load data from `data/processed/` and ensure fast reload

## 6) Orchestration Script
- [ ] Create `hackathon/scripts/analyze_traction.py` with subcommands:
  - `--step=detect_turns`
  - `--step=build_envelopes`
  - `--step=classify_laps`
- [ ] Document CLI usage in `README.md`

## 7) Validation & QA
- [ ] Zone count and spacing sanity vs 17‑corner track
- [ ] Envelope smoothness; clamp outliers; optional light smoothing
- [ ] Classification reasonableness across drivers and laps
- [ ] Performance: chunk sizes, memory, and output file sizes

## 8) Stretch (nice‑to‑have)
- [ ] Event markers on friction circle and along zones
- [ ] Compare R1 vs R2 (toggle)
- [ ] Per‑zone leaderboard and driver deltas

## Key Constraints (do not change)
- Cluster on 1D track distance (after projecting to centerline), not raw GPS XY
- Do not rotate persisted data; any rotation is visualization‑only

