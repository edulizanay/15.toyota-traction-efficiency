# ABOUTME: Generate interactive HTML histogram of turn zone detection
# ABOUTME: Shows track distance distribution to identify cornering zones

import pandas as pd
import numpy as np

# Load high-G points
df = pd.read_csv("data/processed/turn_zones_points.csv")

# Filter to P70 (top 30% of cornering data)
p70_threshold = df["abs_accy"].quantile(0.70)
df_filtered = df[df["abs_accy"] > p70_threshold].copy()

print(f"Original points: {len(df):,}")
print(f"P70 threshold: {p70_threshold:.3f}g")
print(
    f"After P70 filter: {len(df_filtered):,} points ({len(df_filtered) / len(df) * 100:.1f}%)"
)

# Create histogram with 10m bins
bins = np.arange(0, df_filtered["track_distance"].max() + 10, 10)
hist, edges = np.histogram(df_filtered["track_distance"], bins=bins)

# Generate HTML
html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Turn Zone Histogram</title>
    <style>
        body {{ font-family: monospace; padding: 20px; background: #1e1e1e; color: #d4d4d4; }}
        h1 {{ color: #4ec9b0; }}
        .stats {{ margin: 20px 0; padding: 10px; background: #2d2d2d; border-left: 4px solid #4ec9b0; }}
        .histogram {{ margin: 20px 0; display: flex; align-items: flex-end; height: 400px; gap: 1px; }}
        .bin {{ display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 2px; position: relative; }}
        .bar {{ width: 100%; background: #4ec9b0; cursor: pointer; }}
        .bar:hover {{ background: #5edac0; }}
        .tooltip {{ position: absolute; bottom: 105%; left: 50%; transform: translateX(-50%);
                    background: #2d2d2d; color: #d4d4d4; padding: 5px 10px; border-radius: 4px;
                    white-space: nowrap; font-size: 11px; display: none; z-index: 100;
                    border: 1px solid #4ec9b0; }}
        .bin:hover .tooltip {{ display: block; }}
        .instruction {{ margin: 20px 0; padding: 15px; background: #2d2d2d; border-left: 4px solid #ce9178; }}
    </style>
</head>
<body>
    <h1>Turn Zone Detection - Track Distance Histogram</h1>
    <div class="stats">
        <div>P70 threshold: {p70_threshold:.3f}g (top 30% of cornering data)</div>
        <div>Filtered points: {len(df_filtered):,} ({len(df_filtered) / len(df) * 100:.1f}% of original)</div>
        <div>Track distance range: {df_filtered["track_distance"].min():.0f}m - {df_filtered["track_distance"].max():.0f}m</div>
        <div>Bin size: 10 meters</div>
    </div>
    <div class="instruction">
        <strong>Instructions:</strong> Identify turn zones from the histogram peaks below.
        Tell Claude where to make the cuts (e.g., "Turn 1: 50-600m, Turn 2: 900-1400m, etc.")
    </div>
    <div class="histogram">
"""

max_count = max(hist)
for i, (count, edge) in enumerate(zip(hist, edges[:-1])):
    height = int((count / max_count) * 380)  # 380px max height (400 - padding)
    end_edge = edges[i + 1]
    html += f"""        <div class="bin">
            <div class="tooltip">{edge:.0f}m - {end_edge:.0f}m<br>{count:,} points</div>
            <div class="bar" style="height: {height}px;"></div>
        </div>
"""

html += """    </div>
</body>
</html>"""

# Save HTML
with open("histogram.html", "w") as f:
    f.write(html)

print("Created histogram.html - opening in browser...")
