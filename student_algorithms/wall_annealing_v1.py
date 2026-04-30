"""
wall_annealing_v1.py - Simulated-annealing wall designer.

Starts from the v7 greedy result and explores the solution space using a
geometric cooling schedule. Worse moves are accepted with probability
exp(-delta / T). After cooling, a final hill-climb pass (T = 0) polishes the
best state seen.

Move types: swap, reposition (nudge gap), replace, remove, add.

Temperature schedule:
    T_0   = 0.05      (about 5% of the score range)
    alpha = 0.9965    (geometric decay)
    steps = 4000
    T_min = 1e-4
"""

import math
import random

from wall_designer.scorer import evaluate

from student_algorithms import wall_greedy_v7 as v7
from student_algorithms._search_moves import (
    MOVES,
    evaluate_state,
    fallback_seed,
    initial_state_from_placements,
)

T0 = 0.05
ALPHA = 0.9965
STEPS = 4000
T_MIN = 1e-4
POLISH_ITERS = 4000
POLISH_STALL = 800
SEED = 42


def _anneal(initial_state, wall, eligible, scoring_data, rng):
    cur = initial_state
    cur_score, cur_placements = evaluate_state(cur, wall, eligible, scoring_data, evaluate)
    best, best_score, best_placements = cur, cur_score, cur_placements

    T = T0
    for _ in range(STEPS):
        if T < T_MIN:
            break
        _, move_fn = rng.choice(MOVES)
        cand = move_fn(cur, eligible, wall, scoring_data, rng)
        if cand is None:
            T *= ALPHA
            continue
        cand_score, cand_placements = evaluate_state(
            cand, wall, eligible, scoring_data, evaluate
        )
        delta = cand_score - cur_score
        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-9)):
            cur, cur_score, cur_placements = cand, cand_score, cand_placements
            if cur_score > best_score:
                best, best_score, best_placements = cur, cur_score, cur_placements
        T *= ALPHA

    return best, best_score, best_placements


def _polish(state, wall, eligible, scoring_data, rng):
    cur = state
    cur_score, cur_placements = evaluate_state(cur, wall, eligible, scoring_data, evaluate)
    stall = 0
    for _ in range(POLISH_ITERS):
        if stall >= POLISH_STALL:
            break
        _, move_fn = rng.choice(MOVES)
        cand = move_fn(cur, eligible, wall, scoring_data, rng)
        if cand is None:
            stall += 1
            continue
        cand_score, cand_placements = evaluate_state(
            cand, wall, eligible, scoring_data, evaluate
        )
        if cand_score > cur_score + 1e-6:
            cur, cur_score, cur_placements = cand, cand_score, cand_placements
            stall = 0
        else:
            stall += 1
    return cur_score, cur_placements


def generate(wall, artworks, scoring_data):
    rng = random.Random(SEED)

    seed_placements = v7.generate(wall, artworks, scoring_data)
    if seed_placements:
        initial = initial_state_from_placements(seed_placements, artworks, scoring_data)
    else:
        initial = fallback_seed(artworks, wall, scoring_data)
        if initial is None:
            return []
    if len(initial['ordering']) < 2:
        return seed_placements or []

    best_score = -1.0
    best_placements = []

    # Polish from the seed first (gives the same baseline a HC pass would).
    seed_polish_score, seed_polish_placements = _polish(
        initial, wall, artworks, scoring_data, random.Random(SEED)
    )
    if seed_polish_score > best_score:
        best_score, best_placements = seed_polish_score, seed_polish_placements

    # Three independent anneal + polish restarts.
    for k in range(3):
        anneal_rng = random.Random(SEED + 17 * (k + 1))
        polish_rng = random.Random(SEED + 31 * (k + 1))
        anneal_state, _, _ = _anneal(initial, wall, artworks, scoring_data, anneal_rng)
        polish_score, polish_placements = _polish(
            anneal_state, wall, artworks, scoring_data, polish_rng
        )
        if polish_score > best_score:
            best_score, best_placements = polish_score, polish_placements

    seed_score = (
        evaluate(wall, seed_placements, artworks, scoring_data)['total']
        if seed_placements else 0.0
    )
    return best_placements if best_score >= seed_score else seed_placements
