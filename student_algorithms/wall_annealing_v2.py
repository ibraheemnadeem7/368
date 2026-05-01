"""
wall_annealing_v2.py - Simulated annealing that optimises the raw evaluator
score directly.

v1 used the steering score (real_score + coverage bonus + duplicate / underfill /
tiny-work penalties) for both acceptance and tracking. On well-behaved walls
that bias pushed metrics past their preferred values: the coverage bonus
rewarded wall_utilization above preferred=70, raw spacing crossed past
preferred=75 into the diminishing-return tail, and target-shaped scores
dropped while raw values kept climbing.

v2 fixes this by:
1. Annealing on the raw `evaluate(...).total` directly. No coverage bonus,
   no auxiliary penalties. The target-shaped scoring curves are honoured.
2. Cooling more slowly with a lower starting temperature, so the search
   doesn't escape good basins it has already found.
3. Running more independent restarts (5 vs 3) with different RNG seeds to
   widen basin coverage.
4. Doing a strict hill-climb polish (raw score) after each anneal so the
   best basin is exploited fully.

Move types and placement strategies are reused from `_search_moves`.
"""

import math
import random

from wall_designer.scorer import evaluate

from student_algorithms import wall_greedy_v7 as v7
from student_algorithms._search_moves import (
    evaluate_state,
    fallback_seed,
    initial_state_from_placements,
    pick_move,
)


T0 = 0.02
ALPHA = 0.998
STEPS = 5000
T_MIN = 1e-5
POLISH_ITERS = 6000
POLISH_STALL = 1500
RESTARTS = 5
SEED = 42


def _real_score(state, wall, eligible, scoring_data):
    """Return the raw evaluator score and the best-strategy placements."""
    _, placements, real = evaluate_state(state, wall, eligible, scoring_data, evaluate)
    return real, placements


def _anneal(initial_state, wall, eligible, scoring_data, rng):
    cur = initial_state
    cur_score, cur_placements = _real_score(cur, wall, eligible, scoring_data)
    best = cur
    best_score = cur_score
    best_placements = cur_placements

    T = T0
    for _ in range(STEPS):
        if T < T_MIN:
            break
        _, move_fn = pick_move(cur, scoring_data, rng)
        cand = move_fn(cur, eligible, wall, scoring_data, rng)
        if cand is None:
            T *= ALPHA
            continue
        cand_score, cand_placements = _real_score(cand, wall, eligible, scoring_data)
        delta = cand_score - cur_score
        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-9)):
            cur, cur_score, cur_placements = cand, cand_score, cand_placements
            if cur_score > best_score:
                best = cur
                best_score = cur_score
                best_placements = cur_placements
        T *= ALPHA

    return best, best_score, best_placements


def _polish(state, wall, eligible, scoring_data, rng):
    """Strict hill-climb on the raw score until a stall budget is exhausted."""
    cur = state
    cur_score, cur_placements = _real_score(cur, wall, eligible, scoring_data)
    best_score = cur_score
    best_placements = cur_placements
    stall = 0

    for _ in range(POLISH_ITERS):
        if stall >= POLISH_STALL:
            break
        _, move_fn = pick_move(cur, scoring_data, rng)
        cand = move_fn(cur, eligible, wall, scoring_data, rng)
        if cand is None:
            stall += 1
            continue
        cand_score, cand_placements = _real_score(cand, wall, eligible, scoring_data)
        if cand_score > cur_score + 1e-6:
            cur, cur_score, cur_placements = cand, cand_score, cand_placements
            if cur_score > best_score:
                best_score = cur_score
                best_placements = cur_placements
            stall = 0
        else:
            stall += 1

    return best_score, best_placements


def generate(wall, artworks, scoring_data):
    seed_placements = v7.generate(wall, artworks, scoring_data)
    if seed_placements:
        initial = initial_state_from_placements(seed_placements, artworks, scoring_data)
    else:
        initial = fallback_seed(artworks, wall, scoring_data)
        if initial is None:
            return []
    if len(initial['ordering']) < 2:
        return seed_placements or []

    seed_real = (
        evaluate(wall, seed_placements, artworks, scoring_data)['total']
        if seed_placements else 0.0
    )

    best_score = seed_real
    best_placements = seed_placements

    # Polish the seed first (raw-score HC pass).
    seed_score, seed_polish_placements = _polish(
        initial, wall, artworks, scoring_data, random.Random(SEED)
    )
    if seed_score > best_score:
        best_score = seed_score
        best_placements = seed_polish_placements

    # Independent anneal + polish restarts.
    for k in range(RESTARTS):
        anneal_rng = random.Random(SEED + 17 * (k + 1))
        polish_rng = random.Random(SEED + 31 * (k + 1))
        anneal_state, _, _ = _anneal(initial, wall, artworks, scoring_data, anneal_rng)
        polish_score, polish_placements = _polish(
            anneal_state, wall, artworks, scoring_data, polish_rng
        )
        if polish_score > best_score:
            best_score = polish_score
            best_placements = polish_placements

    return best_placements
