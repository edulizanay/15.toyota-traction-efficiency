# Traction Efficiency Analysis - Toyota GR Cup

Friction circle analysis to classify driver performance as Conservative, Aggressive, or Optimal using telemetry from Toyota GR Cup racing.

## Project Overview

This project analyzes driver traction efficiency using the friction circle concept (`total_G = sqrt(accx² + accy²)`). By comparing actual grip usage against each driver's maximum envelope, we identify:

- **Conservative** driving (leaving time on the table)
- **Aggressive** driving (wheelspin, understeer, oversteer)
- **Optimal** driving (95-100% grip utilization, no mistakes)

**Deliverable:** Interactive D3.js dashboard with track view and friction circle analytics.

## Repository Structure

```
├── context.md                     # Hackathon rules and data info
├── traction-analysis-concept.md   # Technical concept
├── implementation-plan.md         # Step-by-step implementation plan
├── legacy/                        # Old code and assets for porting
├── barber/                        # Raw telemetry data (not committed)
└── hackathon/                     # Main working directory (TBD)
```

## Quick Start

See [implementation-plan.md](implementation-plan.md) for detailed implementation steps.

## Data

Telemetry data from Toyota GR Cup races at Barber Motorsports Park:
- R1 (Race 1): 11.5M telemetry samples
- R2 (Race 2): 11.7M telemetry samples

Using both races for envelope construction provides better statistical confidence.

## Key Technical Decisions

- **D3.js** for all visualizations (track + analytics)
- **DBSCAN clustering on 1D track distance** for turn zone detection
- **GPS coordinates for spatial assignment** (not Laptrigger_lapdist_dls)
- **Rotation only in rendering** (store data in raw UTM)
- **Pre-compute heavy analysis in Python** (browser just renders)
- **Data Quality**: Some vehicles had duplicate timestamps across parameters. Added sequence numbering during pivot to recover 99.9% of lost data.

## Grip Envelope & Utilization

- Angle‑aware envelope: For each driver we build a friction “envelope” in the longitudinal–lateral plane (accx vs accy). We bin samples by direction (angle θ = atan2(accy, accx)) and, per angle bin, take a high percentile of net grip (total_g = sqrt(accx² + accy²)). Interpolating those points gives a boundary radius R(θ) — the driver’s observed limit for that mix of braking/turning.
- Per‑sample utilization: For every telemetry sample, compute total_g and its θ, then u = total_g / R(θ). This compares what the car did to what it could have done at that exact balance of longitudinal and lateral demand.
- Per‑zone utilization: Aggregate a corner’s samples with the median of u (robust to outliers). That median is the zone’s grip utilization.
- Classification (concept):
  - Optimal: median utilization ≥ threshold and no sustained over‑limit events.
  - Aggressive: sustained over‑limit behavior (union of wheelspin/understeer/oversteer) that meets both coverage and run‑length thresholds, while at high utilization.
  - Conservative: everything else.
- Why angle‑aware (not single‑axis): Tires share grip between braking/traction and cornering (combined‑slip), so the limit is ellipse‑like. A single “max G” or single‑axis comparison would mislabel entry/exit phases; using R(θ) makes utilization fair at every steering/throttle mix.
- Envelope percentile trade‑off: A higher percentile (e.g., 99.5%) reflects “true peak” and yields lower utilizations (stricter Optimal). A lower percentile (e.g., 95%) is more “repeatable” and raises utilizations. We pair the chosen percentile with an appropriate Optimal threshold to produce realistic distributions.

## License

Hackathon project - Toyota Gazoo Racing / Devpost
