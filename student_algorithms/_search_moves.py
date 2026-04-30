"""
Shared move primitives for hill-climbing and simulated-annealing wall designers.

A search state is represented by:
    {
        'ordering': [artwork dicts in left-to-right order],
        'gap':      float,
    }

`evaluate_state` builds placements (trying uniform, anchor-right, and
anchor-centered placement strategies) and returns the best
`(score, placements)` pair, with a small coverage bonus so the optimizer
prefers layouts that span more of the wall.

All moves return a NEW state (or None if the move is not applicable).
"""

import random

from student_algorithms import wall_greedy_v5 as v5


COVERAGE_WEIGHT = 0.10         # bonus for using more of the wall
NEAR_DUPLICATE_PENALTY = 0.25  # subtracted per detected near-duplicate pair
THEME_OVERLAP_THRESHOLD = 0.5  # jaccard threshold for "near duplicate"
UNDERFILL_WEIGHT = 0.05        # per-work shortfall vs. target count for the wall
TINY_WORK_PENALTY = 0.04       # per artwork whose width < TINY_RATIO * wall_w
TINY_RATIO = 0.04
ORIENTATION_BONUS = 0.04       # awarded when >= 80% share orientation, else penalty
SCALE_SPREAD_PENALTY = 0.05    # applied when max_width / median_width > SCALE_RATIO
SCALE_RATIO = 4.0




def _f(v, default=0.0):
    try:
        return float(v or default)
    except (TypeError, ValueError):
        return default


def _norm_str(s):
    return ''.join(ch.lower() for ch in (s or '') if ch.isalnum())


def _target_count(wall, scoring_data):
    wall_w = _f(wall.get('width_ft'))
    hc = scoring_data.get('scoring', {}).get('hard_constraints', {})
    min_n = int(hc.get('min_artworks', {}).get('value', 2) or 2)
    max_n = int(hc.get('max_artworks', {}).get('value', 8) or 8)
    target = round(wall_w / 6.0)
    return max(min_n, min(max_n, target))


def _underfill_penalty(ordering, wall, scoring_data):
    target = _target_count(wall, scoring_data)
    short = max(0, target - len(ordering))
    return UNDERFILL_WEIGHT * short


def _tiny_work_penalty(ordering, wall):
    wall_w = _f(wall.get('width_ft'))
    if wall_w <= 0:
        return 0.0
    threshold = wall_w * TINY_RATIO
    tiny = sum(1 for a in ordering if _f(a.get('width_ft')) < threshold)
    return TINY_WORK_PENALTY * tiny


def _orientation_term(ordering):
    if len(ordering) < 3:
        return 0.0
    counts = {}
    for a in ordering:
        o = a.get('orientation') or 'unknown'
        counts[o] = counts.get(o, 0) + 1
    dominant = max(counts.values()) / len(ordering)
    if dominant >= 0.8:
        return ORIENTATION_BONUS
    if dominant < 0.6:
        return -ORIENTATION_BONUS
    return 0.0


def _scale_spread_penalty(ordering):
    widths = sorted(_f(a.get('width_ft')) for a in ordering if _f(a.get('width_ft')) > 0)
    if len(widths) < 3:
        return 0.0
    median = widths[len(widths) // 2]
    if median <= 0:
        return 0.0
    if widths[-1] / median > SCALE_RATIO:
        return SCALE_SPREAD_PENALTY
    return 0.0


def _theme_overlap(a, b):
    ta = {t for t in (a.get('theme_tags') or []) if t}
    tb = {t for t in (b.get('theme_tags') or []) if t}
    if a.get('primary_theme'):
        ta.add(a['primary_theme'])
    if b.get('primary_theme'):
        tb.add(b['primary_theme'])
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _near_duplicate_pairs(ordering):
    """Count pairs that share artist and have high theme overlap, or share title."""
    count = 0
    n = len(ordering)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ordering[i], ordering[j]
            ta, tb = _norm_str(a.get('title')), _norm_str(b.get('title'))
            if ta and ta == tb:
                count += 1
                continue
            aa, ab = _norm_str(a.get('artist')), _norm_str(b.get('artist'))
            if aa and aa == ab and _theme_overlap(a, b) >= THEME_OVERLAP_THRESHOLD:
                count += 1
    return count


def _coverage(placements, wall, ordering):
    if not placements:
        return 0.0
    wall_w = _f(wall.get('width_ft'))
    if wall_w <= 0:
        return 0.0
    lookup = {a['id']: a for a in ordering}
    leftmost = min(p['x_ft'] for p in placements)
    rightmost = max(p['x_ft'] + _f(lookup.get(p['artwork_id'], {}).get('width_ft'))
                    for p in placements)
    return max(0.0, min(1.0, (rightmost - leftmost) / wall_w))


def _place_anchor_centered(wall, ordering, anchor, gap, scoring_data):
    """
    Symmetric placement: anchor sits at wall centre, the rest split around it
    (alternating right then left of the anchor by ordering index).

    The split prevents the all-left clustering produced by `_place_anchor_right`
    on wide walls.
    """
    mn, mx = v5._gap_bounds(scoring_data)
    gap = max(mn, min(mx, gap))
    wall_w = _f(wall.get('width_ft'))
    if wall_w <= 0:
        return []

    others = [a for a in ordering if a['id'] != anchor['id']]
    if not others:
        return []

    # Alternate: even indices go to right of anchor, odd go to left.
    right_side, left_side = [], []
    for idx, art in enumerate(others):
        (right_side if idx % 2 == 0 else left_side).append(art)

    margin = mn / 2.0
    anchor_w = _f(anchor.get('width_ft'))
    if anchor_w <= 0:
        return []

    wall_center = _f(wall.get('centerline_ft'), wall_w / 2.0)
    anchor_x = wall_center - anchor_w / 2.0

    # Build right span
    right_widths = [_f(a.get('width_ft')) for a in right_side]
    right_span = sum(right_widths) + len(right_side) * gap
    # Build left span (read left_side in reverse so the closest-to-anchor first)
    left_widths = [_f(a.get('width_ft')) for a in left_side]
    left_span = sum(left_widths) + len(left_side) * gap

    # Adjust anchor_x so both sides fit within margins
    needed_left = anchor_x - margin
    needed_right = (wall_w - margin) - (anchor_x + anchor_w)
    if left_span > needed_left:
        anchor_x += (left_span - needed_left)
    if right_span > needed_right:
        anchor_x -= (right_span - needed_right)
    anchor_x = max(margin, min(anchor_x, wall_w - anchor_w - margin))
    # If still doesn't fit on either side, give up and let other strategies handle it.
    if anchor_x - margin + 1e-6 < left_span:
        return []
    if (wall_w - margin) - (anchor_x + anchor_w) + 1e-6 < right_span:
        return []

    placements = []
    # Place left items, working outward from anchor
    x = anchor_x - gap
    for art in left_side:  # nearest-to-anchor first
        w = _f(art.get('width_ft'))
        x -= w
        placements.append({
            'artwork_id': art['id'],
            'x_ft': round(x, 2),
            'y_ft': v5._clamp_y(wall, art),
            'locked': bool(art.get('locked', False)),
            'required': bool(art.get('required', False)),
            'notes': '',
        })
        x -= gap

    # Place anchor
    placements.append({
        'artwork_id': anchor['id'],
        'x_ft': round(anchor_x, 2),
        'y_ft': v5._clamp_y(wall, anchor),
        'locked': bool(anchor.get('locked', False)),
        'required': bool(anchor.get('required', False)),
        'notes': '',
    })

    # Place right items
    x = anchor_x + anchor_w + gap
    for art in right_side:
        w = _f(art.get('width_ft'))
        placements.append({
            'artwork_id': art['id'],
            'x_ft': round(x, 2),
            'y_ft': v5._clamp_y(wall, art),
            'locked': bool(art.get('locked', False)),
            'required': bool(art.get('required', False)),
            'notes': '',
        })
        x += w + gap

    return placements


def evaluate_state(state, wall, eligible_pool, scoring_data, evaluate_fn):
    ordering = state['ordering']
    gap = state['gap']
    if len(ordering) < 2:
        return 0.0, [], 0.0

    candidates = []
    p_uniform = v5._place_uniform(wall, ordering, gap, scoring_data)
    if p_uniform:
        candidates.append(p_uniform)

    anchor = max(ordering, key=lambda a: _f(a.get('focal_weight')))
    others = [a for a in ordering if a['id'] != anchor['id']]
    p_anchor_right = v5._place_anchor_right(wall, others, anchor, gap, scoring_data)
    if p_anchor_right:
        candidates.append(p_anchor_right)
    p_anchor_centered = _place_anchor_centered(wall, ordering, anchor, gap, scoring_data)
    if p_anchor_centered:
        candidates.append(p_anchor_centered)

    dup_pen = NEAR_DUPLICATE_PENALTY * _near_duplicate_pairs(ordering)
    underfill_pen = _underfill_penalty(ordering, wall, scoring_data)
    tiny_pen = _tiny_work_penalty(ordering, wall)
    orient_term = _orientation_term(ordering)
    spread_pen = _scale_spread_penalty(ordering)

    best_total = -1.0
    best_placements = []
    for placements in candidates:
        result = evaluate_fn(wall, placements, eligible_pool, scoring_data)
        if result['failed_constraints']:
            continue
        coverage = _coverage(placements, wall, ordering)
        adjusted = (
            result['total']
            + COVERAGE_WEIGHT * coverage
            + orient_term
            - dup_pen
            - underfill_pen
            - tiny_pen
            - spread_pen
        )
        if adjusted > best_total:
            best_total = adjusted
            best_placements = placements

    if not best_placements:
        return 0.0, [], 0.0

    real_result = evaluate_fn(wall, best_placements, eligible_pool, scoring_data)
    real_score = 0.0 if real_result['failed_constraints'] else real_result['total']
    return best_total, best_placements, real_score


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

    # Try the current gap; if nothing fits, fall back to the minimum gap so
    # `add` can still fire when a wide gap was eating all the headroom.
    new_gap = state['gap']
    span_with = used_width + (len(state['ordering']) + 1) * new_gap
    headroom = wall_w - mn - span_with
    fitting = [a for a in pool if _f(a.get('width_ft')) <= max(headroom, 0.0)]
    if not fitting and new_gap > mn + 1e-6:
        new_gap = mn
        span_with = used_width + (len(state['ordering']) + 1) * new_gap
        headroom = wall_w - mn - span_with
        fitting = [a for a in pool if _f(a.get('width_ft')) <= max(headroom, 0.0)]
    if not fitting:
        return None

    new_art = rng.choice(fitting)
    insert_at = rng.randrange(len(state['ordering']) + 1)
    new_order = list(state['ordering'])
    new_order.insert(insert_at, new_art)
    return {'ordering': new_order, 'gap': new_gap}


_MOVE_FNS = {
    'swap': move_swap,
    'reposition': move_reposition,
    'replace': move_replace,
    'remove': move_remove,
    'add': move_add,
}

# Kept for any external import; iteration order doesn't matter since callers
# now use pick_move() for weighted sampling.
MOVES = list(_MOVE_FNS.items())


def pick_move(state, scoring_data, rng):
    """
    Weighted move sampling.

    `add` is upweighted when the wall is underfilled (count < max - 1), so the
    optimizer is pushed to fill wide walls instead of settling on a sparse
    layout.
    """
    n = len(state['ordering'])
    limit = v5._max_artworks(scoring_data)
    weights = {
        'swap': 2.0,
        'reposition': 1.5,
        'replace': 2.0,
        'remove': 1.0 if n > 3 else 0.3,
        'add': 3.0 if n < limit - 1 else (1.0 if n < limit else 0.0),
    }
    names = list(weights.keys())
    ws = [weights[k] for k in names]
    name = rng.choices(names, weights=ws, k=1)[0]
    return name, _MOVE_FNS[name]


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
        title = _norm_str(a.get('title'))
        if title and title in used_titles:
            continue
        w = _f(a.get('width_ft'))
        next_n = len(chosen) + 1
        span = width_sum + w + (next_n - 1) * mn + mn
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
