# Wall Designer – Improved Greedy Strategy

**Algorithm file:** `student_algorithms/wall_greedy_v2.py`

---

## Score Summary

| Wall | v1 (baseline) | v2 (improved) | Gain |
|------|--------------|--------------|------|
| R2-N | 0.4959 | **0.9222** | +0.43 |
| R7-S | 0.5134 | **0.8201** | +0.31 |

---

## What Was Wrong with v1

The baseline algorithm sorted artworks by focal weight and placed them
left-to-right starting at the wall's left margin, always using the minimum
gap (0.25 ft).  This caused three zero-scoring criteria on R2-N:

| Criterion (weight) | v1 raw | v2 raw | Why v1 failed |
|---|---|---|---|
| visual_balance (0.8) | 0.0 | 89.3 | All art packed on the left half |
| focal_point (0.4) | 0.0 | 80.0 | Highest-focal work placed at x=0.12, far from centre |
| spacing_regularity (0.9) | 91.1 | 75.0 | Min gaps give raw=91; preferred_value=75 |

---

## Four Improvements

### 1 · Gap optimisation (targets `spacing_regularity`)

The scoring formula for spacing is:

```
raw = 100 – avg_deviation_from_ideal × 30
```

The scorer's `preferred_value` is **75**, so the target deviation is
`25/30 ≈ 0.833 ft`, meaning the optimal uniform gap is
`0.55 + 0.833 ≈ 1.38 ft` — much larger than v1's 0.25 ft.

Using this gap raises `spacing_regularity` score from **0.80 → 1.00**
(raw 91 → 75 = exact preferred value) and also spreads artworks further
apart, improving wall utilisation as a side-effect.

### 2 · Focal-anchor placement (targets `focal_point` + `visual_balance`)

`featured_work_near_preferred_zone` peaks at `raw = preferred_value = 80`,
which happens when the featured work's centre is exactly
`zone_half + (100 − 80)/20 = 1.5 + 1.0 = 2.5 ft` to the right of the
wall's centreline.

The algorithm:
1. Identifies the **focal anchor** (highest `focal_weight` artwork).
2. Computes `anchor_x = wall_centre + 2.5 ft − anchor_width/2`.
3. Places all other artworks to the **left** of the anchor with the
   optimal gap, computing `start_x` so the gap to the anchor is also 1.38 ft.

By-product on `visual_balance`: the anchor sits on the right half of the
wall while all lighter works sit on the left half.  For R2-N this gives a
left/right mass ratio of 10.82 vs 13.41, yielding `balance_raw = 89.3`
(was 0.0).

### 3 · Theme-greedy ordering (targets `thematic_cohesion`)

A nearest-neighbour chain algorithm orders artworks to maximise the sum of
adjacent theme-similarity scores:

- Start from the "loneliest" artwork (lowest max-similarity to any other).
- Repeatedly append the most thematically similar remaining artwork.

This ensures the most cohesive sub-cluster (e.g., the portrait/figure pair
A8 + A9 with similarity 0.95) is placed together in the centre of the
group, while works with weaker thematic ties sit at the edges.

### 4 · Multi-start scoring + local swap improvement

Five candidate arrangements are generated:

| # | Strategy |
|---|---|
| S1 | Theme-greedy others → focal anchor on right |
| S2 | Intensity-sorted others → focal anchor on right |
| S3 | Full theme-greedy chain, uniform gap, from left |
| S4 | Focal-weight-sorted, optimal gap, from left |
| S5 | v1 sort with optimal gap (baseline for comparison) |

Each is scored with `evaluate()`.  The best-scoring candidate is then
refined by a **pairwise-swap local search**: all pairs of artworks in the
winning ordering are tried as swaps; if any swap increases the total score
by > 0.0001, it is accepted and the process repeats until stable.

---

## What Each Improvement Contributes (R2-N)

| Criterion | v1 score | After gap fix | After anchor | After local swap |
|---|---|---|---|---|
| spacing_regularity (×0.9) | 0.80 | **1.00** | 1.00 | 1.00 |
| visual_balance (×0.8) | 0.00 | 0.00 | **0.86** | 0.86 |
| wall_utilisation (×0.8) | 0.65 | 0.90 | **0.88** | 0.88 |
| thematic_cohesion (×0.5) | 0.89 | 0.89 | 0.89 | 0.89 |
| focal_point (×0.4) | 0.00 | 0.00 | **1.00** | 1.00 |
| **Total** | **0.50** | 0.74 | **0.92** | **0.92** |

---

## Ideas Tried But Not Adopted

- **Placing works on both sides of the anchor**: putting some artworks
  to the right of the anchor improved utilisation slightly but hurt visual
  balance because the anchor then shared the right half with other works,
  reducing the clean left-heavy / right-anchor split.

- **Gap tuning per criterion**: instead of one optimal gap, trying
  separate gaps for spacing vs. utilisation independently.  The scoring
  function's tolerance bands are wide enough that the single-gap
  optimisation already saturates both criteria.

- **Exhaustive permutation search**: for n ≤ 8 artworks this would be
  feasible, but the pairwise-swap local search finds the same result faster
  because the theme-greedy initialisation is already near-optimal.
