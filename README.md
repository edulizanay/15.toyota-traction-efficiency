# Traction Efficiency Analysis

**Hack the Track 2025 - Driver Training & Insights**

Shows where drivers waste time by being too cautious or lose it by overdriving. Uses g-force data to measure how much grip each driver uses in every corner, then classifies their approach as Conservative, Aggressive, or Optimal.

> **"Too aggressive and you'll slip. Too cautious and you'll get passed. How do you find the right balance?"**


## Demo

[![Demo Video](./docs/images/dashboard-screenshot.png)](https://youtu.be/[TODO: ADD_YOUTUBE_ID])

*Click above to watch the interactive dashboard demo on YouTube*

- **Try the tool here**: [TODO: ADD_GITHUB_PAGES_URL]
- **Official Dataset**: `barber-motorsports-park.zip` from [trddev.com/hackathon-2025](https://trddev.com/hackathon-2025)

## What We Learned

- **Confidence gaps are bigger than skill gaps**: Turn 8 sees 92% of drivers at 1.35g (near maximum grip). Turn 7, one corner earlier, sees 100% conservative driving at only 1.02g. Same cars, same drivers, 0.33g difference—it's all mental.

- **Aggressive ≠ Fast**: Drivers who cross the envelope pull 1.36g but lose time correcting wheelspin and slides. Optimal drivers use 1.31g cleanly and go faster because the car does what they want.

- **78% of corners are driven scared**: Most drivers use only 70% of available grip. They could go 20% faster in those turns just by trusting the tires more—no technique changes needed.

![Friction Circle Dashboard](./docs/images/Envelope.png)
*Example: Driver friction envelope showing grip usage across all corners*

## Quick Start

### Prerequisites
- Python 3.13+ (tested on 3.13)
- Virtual environment recommended

### Installation

```bash
# Navigate to hackathon directory
cd hackathon

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Place data files:**
Extract the downloaded `barber-motorsports-park.zip` and place the telemetry CSVs in `hackathon/data/input/`:
- `R1_barber_telemetry_data.csv` (Race 1)
- `R2_barber_telemetry_data.csv` (Race 2)

Alternatively, use `--r1-telemetry` and `--r2-telemetry` flags to specify custom file paths.

### Usage

**Copy-paste to run the full pipeline:**

```bash
cd hackathon
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/generate_geometry.py
python scripts/analyze_traction.py --step detect_turns
python scripts/build_friction_circle.py
python scripts/analyze_traction.py --step classify_laps
python -m http.server 8000
```

Open http://localhost:8000/dashboard.html in your browser.

The generated dashboards show:
- Track map with color-coded zones (Conservative/Aggressive/Optimal)
- Friction circle visualization per driver
- Driver comparison against field-wide envelope

### Input Data

The tool uses:
- **GR Cup Telemetry** (Barber Motorsports Park) from `barber-motorsports-park.zip` - longitudinal acceleration (accx), lateral acceleration (accy), GPS coordinates, speed
- **USAC Timing Data** - lap times and race results

Data files are located in `hackathon/data/input/`:
- `R1_barber_telemetry_data.csv` - Race 1 telemetry (11.5M samples)
- `R2_barber_telemetry_data.csv` - Race 2 telemetry (11.7M samples)

## How It Works

### 1. Understanding the Friction Circle

Your tires have a fixed grip budget. You can spend it on accelerating, braking, or turning—but the total is limited by physics.

Think of it like a wallet with $100:
- Spend $100 on braking → 0 left for turning
- Spend $50 on braking + $50 on turning → corner while slowing down
- Spend $100 on turning → pure cornering at maximum speed

We measure this using g-forces:
- **Longitudinal (accx)**: How hard you're accelerating or braking
- **Lateral (accy)**: How hard you're cornering
- **Total grip used**: `sqrt(accx² + accy²)`

**The friction circle** plots every combination of braking/acceleration vs cornering. The outer boundary—your envelope—shows the maximum grip available.

**What happens at different points:**
- **Inside the envelope**: You have grip left over—you can take the turn harder or carry more speed
- **On the envelope**: You're at the limit—this is perfect driving
- **Try to go outside**: Physics won't let you—the tires slip, you get wheelspin/understeer/oversteer, and g-forces drop back down inside

Fast drivers live on the envelope. Average drivers stay safely inside it.

### 2. Turn Zone Detection

Before analyzing grip, we need to identify where corners actually are. Instead of using track maps, we let the data define the zones:

1. **Filter for Cornering**: Remove samples where lateral G = 0 (straight sections)
2. **High-G Threshold**: Keep only the top 90% of lateral g-force samples (above 10th percentile)—these are the real corners where drivers are actually turning hard
3. **Cluster by Location**: Use DBSCAN clustering on track distance to group nearby high-G samples into turn zones
4. **Define Boundaries**: For each cluster, set zone boundaries at 2.5th to 97.5th percentile to capture the full corner

**Why this matters**: Our turn zones aren't the same as the track map's corner numbers. We only analyze sections where drivers are actually using lateral grip. A "corner" on the map that's taken flat-out won't show up as a zone—and that's the point. We measure what drivers do, not what the map says.

**Outcome**: 11 turn zones at Barber Motorsports Park, defined by where drivers actually load the tires laterally.

### 3. Grip Envelope Construction

For each driver, we build an angle-aware envelope that captures their grip limit in every direction:

1. **Collect All Samples**: Gather every telemetry point with accx, accy, GPS, and speed
2. **Calculate Direction**: For each sample, compute angle `θ = atan2(accy, accx)` (e.g., hard braking while turning left)
3. **Bin by Direction**: Divide the 360° circle into angle bins (e.g., every 5°)
4. **Find Limit per Bin**: Within each bin, take the 99.5th percentile of `total_G`—this represents the observed limit for that specific balance of braking/turning
5. **Interpolate Boundary**: Connect these points to form a smooth envelope `R(θ)`—the driver's maximum grip at any angle

**Outcome**: A personalized grip envelope showing what each driver can do at every combination of braking and cornering.

### 4. Grip Utilization & Classification

With envelopes built, we measure how efficiently each driver uses their available grip:

1. **Per-Sample Utilization**: For every telemetry sample, compute:
   - `utilization = total_G / R(θ)`
   - This shows "what you did" ÷ "what you could have done" at that exact steering/throttle mix

2. **Per-Zone Utilization**: Within each turn zone, take the median utilization (robust to outliers)

3. **Classification Logic**:
   - **Optimal**: Median utilization ≥ 81% with no sustained over-limit events (clean, fast driving)
   - **Aggressive**: Sustained over-limit behavior (wheelspin, understeer, oversteer) that exceeds coverage/duration thresholds—trying too hard and losing time
   - **Conservative**: Everything else—leaving time on the table by not using available grip

**Why angle-aware matters**: Tires share grip between braking and cornering (combined slip). A single "max G" comparison would mislabel corner entry/exit phases. Using `R(θ)` makes utilization fair at every throttle/brake/steering combination.

**Outcome**: Every turn zone gets classified as Conservative, Aggressive, or Optimal for each driver.

### 5. Interactive Dashboard

The dashboard provides:
- **Track visualization**: Overhead track map with zones color-coded by classification
- **Friction circle plot**: Driver envelope overlaid on field-wide envelope
- **Driver comparison**: Compare any driver's grip utilization against the field
- **Zone details**: Hover over zones to see median utilization and classification

## Coaching Workflow

Here's how a coach uses this tool to improve driver performance, using real data from Vehicle #18 (back-marker):

### Step 1: Identify Problem Zones

The dashboard flags **Turn 1** as 100% conservative (red zone on track map):
- Driver is using **1.11g**
- Fastest drivers hit **1.42g**
- **0.31g left on the table** (only 63% grip utilization)

**Coaching advice**: "In Turn 1, you can accelerate 0.31g harder through the corner. You're at 63% grip—the tires have way more to give. Carry more entry speed and trust the front end."

### Step 2: Target High-Impact Zones

**Turn 2** shows consistent underperformance (100% conservative, 52/52 laps):
- Driver: **1.10g** at 72% utilization
- Optimal drivers: **1.26g** at 83% utilization
- **0.16g gap** = significant lap time loss

**Coaching advice**: "Turn 2 is costing you time every single lap. You're 0.16g below optimal. Work on corner entry confidence—brake later and get on throttle earlier."

### Step 3: Fix Overdriving

**Turn 8** shows 15% aggressive classification despite lower g-forces:
- Driver experiences wheelspin/understeer trying to push
- Actually pulling **1.24g** vs optimal drivers at **1.38g**

**Coaching advice**: "Turn 8 you're fighting the car—smooth inputs get better results. You're spinning the wheels instead of putting power down. Ease off 10% and focus on clean throttle application."

### Step 4: Track Progress

After coaching session, re-run analysis:
- Did Turn 1 utilization increase from 63% → 75%?
- Did Turn 8 aggressive laps decrease from 15% → 5%?
- Measure lap time improvement zone by zone

**Result**: Moving just 3 turns from Conservative to Optimal can gain 0.5-1.0 seconds per lap.

### Looking at the Future

By giving drivers precise, data-driven feedback on grip utilization, we transform subjective "feel" into measurable performance metrics. This enables targeted coaching: where to push harder, where to ease off, and where technique is already optimal—turning telemetry into competitive advantage.

## Project Structure

```
hackathon/
├── dashboard.html             # Main interactive dashboard (friction circle + track + classifications)
├── friction_circle.html       # Friction circle visualization
├── track.html                 # Track map visualization
├── driver-stats.html          # Driver statistics dashboard
├── requirements.txt           # Python dependencies
├── data/
│   ├── input/                 # Raw telemetry CSVs (R1, R2)
│   └── processed/             # Generated CSVs (envelopes, classifications, centerline)
├── scripts/                   # Analysis pipeline scripts
│   ├── build_friction_circle.py      # Envelope construction
│   ├── analyze_traction.py           # Grip utilization analysis
│   ├── generate_geometry.py          # Track centerline generation
│   └── [other analysis scripts]
└── src/                       # Core modules
    ├── event_detection.py     # Wheelspin/understeer/oversteer detection & classification
    └── geometry.py            # GPS math, smoothing, coordinate transforms
```

## License

This project was created for the Hack the Track 2025 hackathon by Toyota Gazoo Racing. All telemetry data is provided by Toyota Racing Development and is subject to their licensing terms.

## Authors

Built by Eduardo Lizana for Hack the Track 2025.

## Acknowledgments

- Toyota Gazoo Racing for providing GR Cup telemetry datasets
- USAC for race timing data
- Barber Motorsports Park for the incredible racing venue
