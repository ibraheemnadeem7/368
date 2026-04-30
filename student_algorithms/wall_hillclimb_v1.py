"""
wall_hillclimb_v1.py - Hill-climbing wall designer.

Starts from the v7 greedy result and improves it by repeatedly applying small
moves, accepting only changes that strictly increase the score.

Move types implemented (5):
    swap, reposition (nudge gap), replace, remove, add

Stopping criterion: STALL_LIMIT consecutive non-improving proposals.
"""

import random

from wall_designer.scorer import evaluate

from student_algorithms import wall_greedy_v7 as v7
from student_algorithms._search_moves import (
    MOVES,
    evaluate_state,
    fallback_seed,
    initial_state_from_placements,
)

MAX_ITERS = 4000
STALL_LIMIT = 600
SEED = 42


def _hill_climb(initial_state, wall, eligible, scoring_data, rng):
    cur_state = initial_state
    cur_score, cur_placements = evaluate_state(
        cur_state, wall, eligible, scoring_data, evaluate
    )
    best_state, best_score, best_placements = cur_state, cur_score, cur_placements

    stall = 0
    for _ in range(MAX_ITERS):
        if stall >= STALL_LIMIT:
            break
        _, move_fn = rng.choice(MOVES)
        cand = move_fn(cur_state, eligible, wall, scoring_data, rng)
        if cand is None:
            stall += 1
            continue
        cand_score, cand_placements = evaluate_state(
            cand, wall, eligible, scoring_data, evaluate
        )
        if cand_score > cur_score + 1e-6:
            cur_state, cur_score, cur_placements = cand, cand_score, cand_placements
            if cur_score > best_score:
                best_state, best_score, best_placements = cur_state, cur_score, cur_placements
            stall = 0
        else:
            stall += 1

    return best_state, best_score, best_placements


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

    _, _, best_placements = _hill_climb(initial, wall, artworks, scoring_data, rng)

    if not best_placements:
        return seed_placements or []

    seed_score = (
        evaluate(wall, seed_placements, artworks, scoring_data)['total']
        if seed_placements else 0.0
    )
    final_score = evaluate(wall, best_placements, artworks, scoring_data)['total']
    return best_placements if final_score >= seed_score else seed_placements
