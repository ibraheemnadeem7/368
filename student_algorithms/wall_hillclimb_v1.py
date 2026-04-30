"""
wall_hillclimb_v1.py - Hill-climbing wall designer.

Starts from the v7 greedy result and improves it by repeatedly applying small
moves, accepting only changes that strictly increase the steering score.

The steering score is `evaluate(...).total + coverage_bonus -
near_duplicate_penalty` (see _search_moves.evaluate_state). The optimizer
hill-climbs on the steering score; the returned placements are the ones whose
real `evaluate` score is highest among everything visited.

Move types implemented (5):
    swap, reposition (gap nudge), replace, remove, add
Moves are sampled with weights that favour `add` while the wall is
underfilled.

Stopping criterion: STALL_LIMIT consecutive non-improving proposals.
"""

import random

from wall_designer.scorer import evaluate

from student_algorithms import wall_greedy_v7 as v7
from student_algorithms._search_moves import (
    evaluate_state,
    fallback_seed,
    initial_state_from_placements,
    pick_move,
)

MAX_ITERS = 4000
STALL_LIMIT = 600
SEED = 42


def _hill_climb(initial_state, wall, eligible, scoring_data, rng):
    cur = initial_state
    cur_steer, cur_placements, _ = evaluate_state(
        cur, wall, eligible, scoring_data, evaluate
    )
    best_steer = cur_steer
    best_placements = cur_placements

    stall = 0
    for _ in range(MAX_ITERS):
        if stall >= STALL_LIMIT:
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

    _, best_placements = _hill_climb(initial, wall, artworks, scoring_data, rng)

    if not best_placements:
        return seed_placements or []

    seed_score = (
        evaluate(wall, seed_placements, artworks, scoring_data)['total']
        if seed_placements else 0.0
    )
    final_score = evaluate(wall, best_placements, artworks, scoring_data)['total']
    return best_placements if final_score >= seed_score else seed_placements
