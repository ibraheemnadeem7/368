"""
Shared move primitives for hill-climbing and simulated-annealing wall designers.

A search state is represented by:
    {
        'ordering': [artwork dicts in left-to-right order],
        'gap':      float,
    }

`evaluate_state` builds placements (trying both uniform and anchor-right
placement strategies) and returns the best `(score, placements)` pair.

All moves return a NEW state (or None if the move is not applicable).
"""

import random

from student_algorithms import wall_greedy_v5 as v5


def _f(v, default=0.0):
    try:
        return float(v or default)
    except (TypeError, ValueError):
        return default


def evaluate_state(state, wall, eligible_pool, scoring_data, evaluate_fn):
    ordering = state['ordering']
    gap = state['gap']
    if len(ordering) < 2:
        return 0.0, []

    candidates = []
    p_uniform = v5._place_uniform(wall, ordering, gap, scoring_data)
    if p_uniform:
        candidates.append(p_uniform)

    anchor = max(ordering, key=lambda a: _f(a.get('focal_weight')))
    others = [a for a in ordering if a['id'] != anchor['id']]
    p_anchor = v5._place_anchor_right(wall, others, anchor, gap, scoring_data)
    if p_anchor:
        candidates.append(p_anchor)

    best_score = -1.0
    best_placements = []
    for placements in candidates:
        result = evaluate_fn(wall, placements, eligible_pool, scoring_data)
        if result['failed_constraints']:
            continue
        if result['total'] > best_score:
            best_score = result['total']
            best_placements = placements
    return max(best_score, 0.0), best_placements


def _used_ids(state):
    return {a['id'] for a in state['ordering']}


def _unused(state, full_pool):
    used = _used_ids(state)
    return [a for a in full_pool if a['id'] not in used]


# ---- moves ---------------------------------------------------------------

def move_swap(state, full_pool, wall, scoring_data, rng):
    n = len(state['ordering'])
    if n < 2:
        return None
    i, j = rng.sample(range(n), 2)
    new_order = list(state['ordering'])
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return {'ordering': new_order, 'gap': state['gap']}


def move_reposition(state, full_pool, wall, scoring_data, rng):
    """Nudge the uniform gap up or down -> re-flows every artwork."""
    mn, mx = v5._gap_bounds(scoring_data)
    delta = rng.choice([-0.15, -0.07, -0.03, 0.03, 0.07, 0.15])
    new_gap = max(mn, min(mx, state['gap'] + delta))
    if abs(new_gap - state['gap']) < 1e-6:
        return None
    return {'ordering': list(state['ordering']), 'gap': new_gap}


def move_replace(state, full_pool, wall, scoring_data, rng):
    n = len(state['ordering'])
    if n == 0:
        return None
    pool = _unused(state, full_pool)
    pool = [a for a in pool if v5._fits_on_wall(a, wall, scoring_data)]
    if not pool:
        return None
    i = rng.randrange(n)
    new_art = rng.choice(pool)
    new_order = list(state['ordering'])
    new_order[i] = new_art
    return {'ordering': new_order, 'gap': state['gap']}


def move_remove(state, full_pool, wall, scoring_data, rng):
    n = len(state['ordering'])
    if n <= 2:
        return None
    i = rng.randrange(n)
    new_order = state['ordering'][:i] + state['ordering'][i + 1:]
    return {'ordering': new_order, 'gap': state['gap']}


def move_add(state, full_pool, wall, scoring_data, rng):
    limit = v5._max_artworks(scoring_data)
    if len(state['ordering']) >= limit:
        return None
    pool = _unused(state, full_pool)
    pool = [a for a in pool if v5._fits_on_wall(a, wall, scoring_data)]
    if not pool:
        return None

    wall_w = _f(wall.get('width_ft'))
    mn, _ = v5._gap_bounds(scoring_data)
    used_width = sum(_f(a.get('width_ft')) for a in state['ordering'])
    span_with = used_width + (len(state['ordering']) + 1) * state['gap']
    headroom = wall_w - mn - span_with
    pool = [a for a in pool if _f(a.get('width_ft')) <= max(headroom, 0.0)]
    if not pool:
        return None

    new_art = rng.choice(pool)
    insert_at = rng.randrange(len(state['ordering']) + 1)
    new_order = list(state['ordering'])
    new_order.insert(insert_at, new_art)
    return {'ordering': new_order, 'gap': state['gap']}


MOVES = [
    ('swap', move_swap),
    ('reposition', move_reposition),
    ('replace', move_replace),
    ('remove', move_remove),
    ('add', move_add),
]


def fallback_seed(eligible_pool, wall, scoring_data):
    """Pick the smallest feasible artworks that fit min_artworks at min_gap."""
    feas = [a for a in eligible_pool if v5._fits_on_wall(a, wall, scoring_data)]
    feas.sort(key=lambda a: _f(a.get('width_ft')))
    mn, _ = v5._gap_bounds(scoring_data)
    wall_w = _f(wall.get('width_ft'))
    hc = scoring_data.get('scoring', {}).get('hard_constraints', {})
    min_n = int(hc.get('min_artworks', {}).get('value', 2) or 2)
    max_n = int(hc.get('max_artworks', {}).get('value', 8) or 8)

    chosen = []
    width_sum = 0.0
    used_titles = set()
    for a in feas:
        title = ''.join(ch.lower() for ch in (a.get('title') or '') if ch.isalnum())
        if title and title in used_titles:
            continue
        w = _f(a.get('width_ft'))
        next_n = len(chosen) + 1
        span = width_sum + w + (next_n - 1) * mn + mn  # margins
        if span > wall_w:
            continue
        chosen.append(a)
        width_sum += w
        if title:
            used_titles.add(title)
        if len(chosen) >= max_n:
            break
    if len(chosen) < min_n:
        return None
    return {'ordering': chosen[:max_n], 'gap': mn}


def initial_state_from_placements(placements, eligible_pool, scoring_data):
    """Recover a search state from a placement list (sorted left-to-right)."""
    sorted_p = sorted(placements, key=lambda p: p.get('x_ft', 0.0))
    lookup = {a['id']: a for a in eligible_pool}
    ordering = [lookup[p['artwork_id']] for p in sorted_p if p['artwork_id'] in lookup]
    gap = v5._optimal_gap(scoring_data)
    if len(sorted_p) >= 2:
        gaps = []
        for a, b in zip(sorted_p, sorted_p[1:]):
            wa = _f(lookup.get(a['artwork_id'], {}).get('width_ft'))
            gaps.append(b['x_ft'] - (a['x_ft'] + wa))
        gaps = [g for g in gaps if g > 0]
        if gaps:
            gap = sum(gaps) / len(gaps)
    return {'ordering': ordering, 'gap': gap}
