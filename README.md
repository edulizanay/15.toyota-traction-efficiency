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

## License

Hackathon project - Toyota Gazoo Racing / Devpost
