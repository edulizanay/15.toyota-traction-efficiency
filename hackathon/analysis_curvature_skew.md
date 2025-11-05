# Analysis: Why Classification Percentages Are Skewed By Zone

## Executive Summary

The classification distribution is heavily skewed by zone, with some zones showing 99-100% conservative classification while others show 60-90% optimal/aggressive. **This is NOT primarily due to curvature differences, but rather due to using a single field-wide friction envelope that doesn't account for zone-specific driving characteristics.**

## Key Findings

### 1. The Distribution is Extremely Skewed

**High Performance Zones (5, 8, 9):**
- Conservative: 23.7%
- Optimal: 62.6%
- Aggressive: 13.7%

**Low Performance Zones (1, 3, 4, 6, 7, 10, 11):**
- Conservative: 99.0%
- Optimal: 1.0%
- Aggressive: 0.0%

**Specific Examples:**
- Turn 7: 100% conservative, max utilization = 67.3%
- Turn 8: 30% aggressive + 61% optimal = 91% high performance
- Turn 4: 100% conservative
- Turns 1, 3, 6, 10, 11: 95-100% conservative

### 2. The Root Cause: Anisotropic Envelope + Zone Characteristics

The field-wide friction envelope is **highly anisotropic** (direction-dependent):
- 0° (pure acceleration/braking): 1.399g
- 45° (mixed): 1.390g
- 90° (pure lateral cornering): 1.816g

**This creates unfair comparisons:**
- Zones with pure cornering are compared against 1.816g boundary → utilization appears low
- Zones with mixed braking+cornering are compared against ~1.4g boundary → utilization appears high

### 3. It's NOT About Curvature

Comparing high vs low performance zones:
- **Avg total G difference**: 1.272g vs 1.124g (only 0.148g difference)
- **Event coverage**: Low-perf zones actually have MORE events (0.131 vs 0.114)
- **Understeer events**: Low-perf zones have MORE (6442 vs 2876)

The issue is that **events don't translate to "aggressive" classification** because the zones don't reach the high utilization threshold (0.81) due to envelope mismatch.

### 4. Turn 7 vs Turn 8 Comparison

**Turn 7 (100% Conservative):**
- Avg total G: 1.018g
- Avg utilization: 0.566 (56.6%)
- Max utilization: 0.673 (67.3%)
- Even the BEST lap is below 0.81 threshold
- 79% of laps have understeer events, but don't qualify as "aggressive"

**Turn 8 (91% Optimal/Aggressive):**
- Avg total G: 1.349g
- Avg utilization: 0.895 (89.5%)
- Max utilization: 1.008 (100.8%)
- 91.5% of laps exceed 0.81 threshold
- 91% of laps have understeer events

### 5. Why This Happens

The envelope is built from the **maximum G-forces across ALL zones**:
- Max pure braking: 1.402g (probably from heavy braking zones)
- Max pure cornering: 1.849g (probably from Turn 8/9)

When evaluating Turn 7 (a slower zone):
- Drivers are physically limited by track geometry
- Even perfect driving won't reach 1.849g in this zone
- Utilization calculations use the global envelope → always looks conservative

## Recommendations

### Option 1: Zone-Specific Envelopes (Most Accurate)
Build a separate friction envelope for each zone. This would give fair comparisons within each zone's physical constraints.

**Pros:**
- Most accurate representation of driver performance per zone
- Each zone judged against its own potential

**Cons:**
- More complex implementation
- Harder to compare across zones
- Requires enough data per zone

### Option 2: Percentile-Based Envelope (Simpler)
Instead of using the absolute maximum, use 95th or 98th percentile G-forces to build the envelope.

**Pros:**
- Reduces impact of outliers
- Single envelope still allows cross-zone comparison
- Simple to implement

**Cons:**
- Still won't fix the fundamental zone characteristic differences
- May still have skewed distributions

### Option 3: Normalized Utilization By Zone
Calculate utilization against the field-wide envelope, but normalize by zone-specific max observed utilization.

**Pros:**
- Preserves cross-zone comparability
- Fair within-zone comparisons
- Relatively simple

**Cons:**
- Two-step calculation is less intuitive
- Requires careful threshold tuning

### Option 4: Accept and Document (Current State)
Accept that some zones are inherently harder to optimize and document this in the visualization.

**Pros:**
- No changes needed
- Reflects physical reality

**Cons:**
- Misleading to non-technical users
- Doesn't provide actionable insights for slower zones

## My Recommendation

I recommend **Option 1: Zone-Specific Envelopes** for the following reasons:

1. **Fairness**: Each zone is judged against its own potential
2. **Actionability**: Drivers can see where they're leaving time on the table in EVERY zone, not just the fast ones
3. **Accuracy**: Reflects the physical reality that different zones have different limits
4. **Better insights**: Can identify which zones have the most room for improvement

The implementation would involve:
1. Modify `build_friction_circle.py` to build per-zone envelopes
2. Update `event_detection.py` to use zone-specific envelopes in `classify_zone()`
3. Re-run the classification pipeline

Would you like me to implement this change?
