"""
wall_annealing_v1.py - Simulated-annealing wall designer.

Starts from the v7 greedy result and explores the solution space using a
geometric cooling schedule. Worse moves are accepted with probability
exp(-delta / T) on the steering score (see _search_moves.evaluate_state).
After cooling, an HC polish pass refines the best state seen. Three
independent restarts keep the best of all polishes.

Move types: swap, reposition (gap nudge), replace, remove, add.
Moves are sampled with weights favouring `add` when the wall is underfilled.

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
    evaluate_state,
    fallback_seed,
    initial_state_from_placements,
    pick_move,
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
    cur_steer, cur_placements, _ = evaluate_state(
        cur, wall, eligible, scoring_data, evaluate
    )
    best = cur
    best_steer = cur_steer
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
        cand_steer, cand_placements, _ = evaluate_state(
            cand, wall, eligible, scoring_data, evaluate
        )
        delta = cand_steer - cur_steer
        if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-9)):
            cur, cur_steer, cur_placements = cand, cand_steer, cand_placements
            if cur_steer > best_steer:
                best = cur
                best_steer = cur_steer
                best_placements = cur_placements
        T *= ALPHA

    return best, best_steer, best_placements


def _polish(state, wall, eligible, scoring_data, rng):
    cur = state
    cur_steer, cur_placements, _ = evaluate_state(
        cur, wall, eligible, scoring_data, evaluate
    )
    best_steer = cur_steer
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
        cand_steer, cand_placements, _ = evaluate_state(
            cand, wall, eligible, scoring_data, evaluate
        )
        if cand_steer > cur_steer + 1e-6:
            cur, cur_steer, cur_placements = cand, cand_steer, cand_placements
            if cur_steer > best_steer:
                best_steer = cur_steer
                best_placements = cur_placements
            stall = 0
        else:
            stall += 1
    return best_steer, best_placements


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

    best_score = -1.0
    best_placements = []

    # Polish from the seed first (matches an HC pass).
    seed_score, seed_polish_placements = _polish(
        initial, wall, artworks, scoring_data, random.Random(SEED)
    )
    if seed_score > best_score:
        best_score, best_placements = seed_score, seed_polish_placements

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

    seed_real = (
        evaluate(wall, seed_placements, artworks, scoring_data)['total']
        if seed_placements else 0.0
    )
    return best_placements if best_score >= seed_real else seed_placements
