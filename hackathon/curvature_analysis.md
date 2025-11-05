# Root Cause Analysis: Zone Classification Skew

## Key Finding: It's About Force Vector Angles + Driver Behavior

### Pattern 1: Pure Cornering Zones (85-90°) = Always Conservative

**Zones operating at pure lateral forces:**
- Turn 1, 3, 7, 11 (all at 80-90°)
- ALL are 98-100% conservative
- Boundary: ~1.7-1.8g (the maximum pure cornering capability)

These zones are physically limited - drivers can't push harder because they're already at steady-state cornering with minimal longitudinal forces.

### Pattern 2: Mixed Zones Split Into Two Groups

**Zones with longitudinal + lateral forces (65-70°):**

**HIGH PERFORMANCE (Turn 5, 8, 9):**
- Turn 5: 1.225g @ 65° → 81% util → 43% cons / 49% opt / 8% agg
- Turn 8: 1.349g @ 65° → 90% util → 8% cons / 51% opt / 40% agg
- Turn 9: 1.240g @ 70° → 80% util → 20% cons / 67% opt / 13% agg

**LOW PERFORMANCE (Turn 2, 4, 6, 10):**
- Turn 2: 1.191g @ 70° → 77% util → 95% cons / 4% opt
- Turn 4: 1.074g @ 70° → 70% util → 100% cons
- Turn 6: 1.064g @ 65° → 70% util → 100% cons
- Turn 10: 1.130g @ 65° → 75% util → 97% cons

## The Critical Question

Turns 2/4/6/10 operate at the SAME angles (65-70°) as Turns 5/8/9, so they have the SAME envelope boundary (~1.5g).

But drivers achieve:
- **Turn 8**: 1.349g (90% of boundary)
- **Turn 6**: 1.064g (70% of boundary)

**Both are at 65°, same boundary, but 0.28g difference in actual performance!**

## Hypothesis: Corner Type Matters

The 65-70° angle indicates longitudinal forces (braking or acceleration) combined with lateral.

**Possible explanations:**

1. **Exit vs Entry zones:**
   - Turns 5/8/9 might be corner EXITS (accelerating out) → drivers push hard
   - Turns 2/4/6/10 might be corner ENTRIES (braking in) → drivers conservative

2. **Strategic importance:**
   - Turns 5/8/9 might feed onto long straights → high priority for lap time
   - Turns 2/4/6/10 might be mid-sector → less critical

3. **Corner difficulty:**
   - Turns 5/8/9 might be "easy" corners with clear optimal lines
   - Turns 2/4/6/10 might be technical corners that are hard to optimize

4. **Risk/reward:**
   - Turns 5/8/9 might have good run-off → safe to push
   - Turns 2/4/6/10 might have walls/gravel → risky to push

## Next Steps to Investigate

Need to check:
1. Sign of accx (braking vs acceleration) for each zone
2. Track layout - which zones feed into straights?
3. Corner speed profiles - are these entries, apexes, or exits?
