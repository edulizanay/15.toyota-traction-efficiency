# 90-Second Video Script - Traction Efficiency Analysis

## [0-15s] Setup & Hook

- How do race drivers take their curves? What actually matters for speed?
  - (Show: Cars racing through corners at Barber)

- Turns out, it comes down to grip. Your tires have a fixed amount.
  - (Show: Close-up of tires on track)

- You can use it to brake, accelerate, or turn—but you can't max out all three at once. Physics won't let you.
  - (Show: Simple diagram of grip budget concept)

## [15-40s] The Problem & Discovery

- Push too hard? The tires slip—wheelspin, understeer. You lose time fighting the car.
  - (Show: Car sliding/losing control footage)

- Too cautious? You're leaving speed on the table. Other drivers pass you.
  - (Show: Slow conservative driving vs aggressive passing)

- Here's what's interesting: At Barber, we found three corners—Turns 5, 8, and 9—where drivers push hard. They're using 90% of available grip.
  - (Show: Track map highlighting turns 5, 8, 9 in green)

- But the other corners? Mostly conservative. Turn 7 sees only 1.02g. Next corner, Turn 8, they hit 1.35g.
  - (Show: Track map with Turn 7 red, Turn 8 green, numbers visible)

- Same cars, same drivers. The difference? Confidence. Track geometry. How the corner flows.
  - (Show: Side-by-side comparison of Turn 7 vs Turn 8)

## [40-65s] The Solution & Dashboard Demo

- So how do you find the optimal balance? We built a tool.
  - (Show: Dashboard opening/loading)

- This is the friction circle. The outer boundary is your envelope—maximum grip.
  - (Show: Friction circle with envelope boundary highlighted)

- Inside means you have room left. On the boundary? That's optimal.
  - (Show: Points inside vs on envelope, with annotations)

- Here's a real driver, Turn 1. Pulling 1.11g. The envelope shows they could hit 1.42g.
  - (Show: Driver friction circle with Turn 1 data, 1.11g vs 1.42g labels)

- That's 0.31g left on the table—only 63% grip usage.
  - (Show: Track map with Turn 1 highlighted red, "63%" visible)

- Turn 8, different problem. Fighting the car, wheels spinning. Actually slower than smooth drivers who stay on the envelope.
  - (Show: Turn 8 aggressive classification, yellow zone on map)

## [65-90s] Impact & Call to Action

- Drivers use this to dial in every corner—how hard to brake, when to throttle, how tight to turn.
  - (Show: Dashboard switching between different drivers)

- The goal? Stay close to the envelope without crossing it.
  - (Show: Optimal driver overlay on friction circle, nearly touching envelope)

- Moving just 3 turns from conservative to optimal? Gains 0.5 to 1 second per lap.
  - (Show: Before/after comparison, lap time improvement)

- That's the difference between podium and back-marker.
  - (Show: Final results/standings)

- Built for Hack the Track 2025. Link below.
  - (Show: GitHub repo URL, demo link on screen)
