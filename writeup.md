# Wall Designer – Improved Greedy Strategy Writeup

**Algorithm:** `student_algorithms/wall_greedy_v2.py`

---

## 1. Description of Improvements

### Starting Point: What v1 Did Wrong

The baseline (`wall_greedy_v1.py`) sorted artworks by focal weight and placed
them left-to-right starting at the wall's left edge, always using the
**minimum gap of 0.25 ft**. Running it on wall R2-N produced a total score of
**0.4959**, with three criteria scoring zero:

| Criterion | v1 raw | Problem |
|---|---|---|
| `visual_balance` | 0.0 | All five artworks crammed into the left 13 ft of a 40 ft wall |
| `focal_point` | 0.0 | The most important artwork (A26) placed at x = 0.12 ft, 20 ft from centre |
| `spacing_regularity` | 91.1 | Min gaps give raw = 91; scorer *prefers* raw = 75 |

Understanding why `spacing_regularity` scored poorly even though the gaps were
consistent was the first important insight: the scorer has a `preferred_value`
of 75, not 100. A raw score of 91 (too regular, too tight) is actually *outside*
the preferred range.

---

### Idea 1 – Gap Optimisation

**What I tried:** Instead of always using the minimum gap, compute the gap that
targets the scorer's preferred raw value of 75.

**How it works:** The `gap_variance_vs_ideal` formula is:

```
raw = 100 – average_deviation_from_ideal × 30
```

Setting `raw = 75` gives `average_deviation = 25/30 ≈ 0.833 ft`, so the
optimal uniform gap is `ideal_gap + 0.833 = 0.55 + 0.833 ≈ 1.38 ft`.

**Result:** Spacing score jumped from **0.80 → 1.00**. The wider gaps also
spread the artworks further apart, which improved wall utilisation from 32% to
44% as a free side-effect.

---

### Idea 2 – Focal-Anchor Placement

**What I tried:** Place the highest focal-weight artwork at the specific x
position where the `featured_work_near_preferred_zone` scorer returns its
preferred raw value (80), then pack the remaining artworks to the left.

**How it works:** The focal-point scorer gives raw = 100 when the featured
work's centre is within 1.5 ft of the wall centre, and decreases by 20 points
per foot beyond that. Its `preferred_value` is 80, which corresponds to a
distance of `1.5 + (100 − 80)/20 = 2.5 ft` from the wall centre. So the
algorithm computes:

```
anchor_x = wall_centre + 2.5 ft – anchor_width / 2
```

All other artworks are then placed to the left of the anchor with the 1.38 ft
optimal gap, with `start_x` calculated so the gap between the last left-side
work and the anchor is also 1.38 ft.

**Result:** `focal_point` score went from **0.0 → 1.00**. An unintended but
welcome side-effect: because the heavy anchor (A26, mass = 13.4) sits alone on
the right half of the wall while the four lighter works (combined mass = 10.8)
sit on the left, `visual_balance` raw rose from 0 to **89.3** (score
**0.0 → 0.86**).

---

### Idea 3 – Theme-Greedy Ordering

**What I tried:** Order the artworks using a nearest-neighbour chain on the
theme-similarity table, instead of purely by focal weight.

**How it works:**
1. Start from the "loneliest" artwork — the one with the lowest maximum
   theme-similarity to any other artwork. This puts the odd-one-out at the far
   end of the group, away from the focal anchor.
2. Repeatedly append the artwork most similar to the current chain tail.

For R2-N, this produced the order `[A1 → A5 → A8 → A9]` before the anchor
A26, giving adjacent pairs: `animal↔landscape (0.78)`, `landscape↔self_portrait
(0.20)`, `self_portrait↔portrait/figure (0.95)`. The most cohesive pair
(A8–A9, similarity 0.95) ends up adjacent in the middle of the group.

**Result:** Thematic cohesion stayed at its maximum achievable value of 50.75
(raw) for this artwork set — no combination can score higher because A26
(abstraction) has a default similarity of 0.1 with all other themes. The
improvement here is that the ordering is *principled* rather than arbitrary.

---

### Idea 4 – Multi-Start Scoring

**What I tried:** Generate five different candidate arrangements using different
orderings and placement strategies, score every one with `evaluate()`, and keep
the best.

**The five candidates:**

| # | Strategy |
|---|---|
| S1 | Theme-greedy others → focal anchor on right *(primary)* |
| S2 | Intensity-sorted others → focal anchor on right |
| S3 | Full theme-greedy chain, optimal gap, from left edge |
| S4 | Focal-weight-sorted, optimal gap, from left edge |
| S5 | v1 baseline sort, minimum gap, from left edge |

This costs five calls to `evaluate()` but guarantees we never accidentally keep
a weaker arrangement just because it was generated first.

---

### Idea 5 – Local Swap Improvement

**What I tried:** After picking the best candidate, try all pairwise swaps of
artworks in the winning ordering, re-place with the same gap, and re-score.
Accept any swap that strictly improves the total. Repeat until no improvement is
found.

For n = 5 artworks this is 10 pairs per pass — fast enough to run to
convergence in milliseconds. This step catches ordering mistakes that the
greedy initialisation might introduce.

---

## 2. Results

### What Worked Well

| Idea | Impact |
|---|---|
| Gap optimisation | Largest single gain — spacing 0.80 → 1.00 |
| Focal-anchor placement | Fixed two zero-scoring criteria at once (focal_point + visual_balance) |
| Multi-start scoring | Guarantees the best of five strategies is kept |
| Local swap | Polishes the final ordering without restructuring the layout |

### What Did Not Work Well

**Placing artworks on both sides of the anchor.** My first attempt put 2
artworks left and 2 right of A26. This hurt `visual_balance` significantly —
with A26 sharing the right half with other works, the mass distribution became
worse than putting everything on the left. The clean split (4 works left, anchor
alone right) turned out to be optimal for these candidates.

**Targeting wall utilisation directly.** I tried starting the arrangement from
x = 0 (instead of x ≈ 7) to stretch the span. This violated the `max_gap`
constraint (1.5 ft) between the left group and the anchor, causing the entire
arrangement to score 0. The gap constraint is hard, so utilisation is limited by
physics: with 5 artworks totalling 12 ft on a 40 ft wall, the maximum reachable
span with valid gaps is about 18 ft (45% utilisation), well below the 70%
target.

**Per-criterion gap tuning.** Trying different gaps for spacing vs. utilisation
added complexity without improving the score. The tolerance bands in the scoring
function are wide enough that the single 1.38 ft gap already sits in the
preferred zone for both criteria.

---

## 3. Final Algorithm

The final algorithm (`wall_greedy_v2.py`) works as follows:

1. **Filter** artworks to eligible only.
2. **Compute the optimal gap** by reading `spacing_regularity.preferred_value`
   from the scoring config and solving for the uniform gap that targets that
   raw score.
3. **Identify the focal anchor** — the artwork with the highest `focal_weight`.
4. **Compute the anchor's target x position** from the `focal_point` scoring
   parameters so its centre lands at the preferred distance from the wall
   centre.
5. **Generate five candidate orderings** using different strategies (theme-greedy,
   intensity-sorted, focal-sorted, baseline, full-theme).
6. **Score all five** with `evaluate()` and keep the best.
7. **Local swap improvement**: try all pairwise artwork swaps in the winning
   ordering; accept any that raise the score.
8. **Return** the highest-scoring placement.

### Why It Is Better Than v1

v1 made three implicit assumptions that all turned out to be wrong:

| v1 assumption | Reality |
|---|---|
| Tighter gaps = better spacing | Scorer *prefers* raw = 75, not 100 |
| Highest focal work should go first (leftmost) | It should go near the centre-right |
| Left-to-right placement is always fine | All art on the left half → balance = 0 |

v2 reads the scoring configuration to derive each placement decision rather
than hard-coding minimum-gap, left-aligned behaviour. It also uses the scorer
itself (`evaluate()`) to compare candidates, so the algorithm improves in
lockstep with any changes to the scoring config.

**Score comparison:**

| Wall | v1 | v2 | Gain |
|---|---|---|---|
| R2-N | 0.4959 | **0.9222** | +86% |
| R7-S | 0.5134 | **0.8201** | +60% |

---

## 4. Reflection

### What I Learned About Greedy Algorithms

**Greedy algorithms are only as good as their heuristic.** v1 was greedy —
it made locally sensible decisions (high focal weight first, pack left) — but
those decisions were not aligned with the actual scoring function. The biggest
lesson was to *read the scorer* before designing the heuristic. Once I computed
what gap produces preferred raw = 75, and what x position produces
preferred focal raw = 80, the algorithm essentially solved itself.

**Local optimality is fragile.** The theme-greedy ordering found the best
3-pair sum for the left group in one pass, but the final swap step still caught
improvements in some wall configurations. A greedy chain can get "stuck" in a
good-but-not-best ordering, and a cheap local search is worth adding.

**Constraints and objectives interact.** Wall utilisation seemed like a simple
"spread the art out" objective, but the `max_gap` hard constraint (1.5 ft)
creates a ceiling on how far artworks can be spread. Ignoring this interaction
caused constraint failures that zeroed the entire score. Any improvement to one
criterion has to be checked against all hard constraints first.

**Multi-start is cheap insurance.** Running five candidates costs only five
extra calls to `evaluate()`. For problems this small it costs almost nothing
and eliminates the risk of the first strategy being a bad fit for a particular
wall or artwork set.

### What I Would Try Next With More Time

1. **Exhaustive permutation search for small n.** With ≤ 8 artworks there are
   at most 40,320 orderings. Scoring all of them would guarantee the globally
   optimal ordering for any fixed gap — a useful upper bound to compare against.

2. **Joint gap + ordering optimisation.** Currently the gap is fixed before
   orderings are tried. A small grid search over gap values (e.g., 0.25 to 1.5
   in steps of 0.1) combined with the best ordering for each gap would find a
   better joint optimum, especially on walls where the utilisation score is very
   sensitive to gap size.

3. **Smarter anchor-side assignment.** For R7-S, the focal anchor (A21) has
   lower visual mass than the combined mass of the other four works, so placing
   it alone on the right gives `visual_balance` = 27.6 rather than the ~89
   achieved on R2-N. A better strategy would check both sides (anchor left vs.
   anchor right) and pick the one with better predicted balance before committing
   to an arrangement.

4. **Dynamic programming on theme chains.** The nearest-neighbour theme ordering
   is a greedy heuristic for a problem that has an exact O(n² × 2ⁿ) DP solution
   (Held-Karp for shortest Hamiltonian path). For n ≤ 8 this is fast enough to
   run exactly and would guarantee the maximum-cohesion ordering.
