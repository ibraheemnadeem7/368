"""
wall_greedy_v3.py  –  Improved greedy wall designer (margin-aware)

Improvements over v1
---------------------
1. Gap optimisation        – targets spacing_raw = preferred_value (≈ 1.38 ft gap)
2. Focal-anchor placement  – anchor centre 2.5 ft right of wall centre → focal_point = 1.0
3. Theme-greedy ordering   – nearest-neighbour chain on theme-similarity table
4. Multi-start + local swap – 5 candidates scored; pairwise-swap polish on winner
5. Wall-edge margins       – both placement helpers respect margin = min_gap/2 on
                             each side, so the leftmost artwork is never flush against
                             the wall edge (matches place_left_to_right baseline).
"""

from wall_designer.placer import place_left_to_right
from wall_designer.scorer import evaluate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _f(v, default=0.0):
    """Safe float cast."""
    try:
        return float(v or default)
    except (TypeError, ValueError):
        return default


def _hard(scoring_data):
    return scoring_data.get('scoring', {}).get('hard_constraints', {})


def _gap_bounds(scoring_data):
    h = _hard(scoring_data)
    return (
        _f(h.get('min_gap_ft', {}).get('value'), 0.25),
        _f(h.get('max_gap_ft', {}).get('value'), 1.50),
    )


def _focal_params(scoring_data):
    """Return (center_zone_half_ft, preferred_focal_raw, penalty_per_ft)."""
    fp = (scoring_data.get('scoring', {})
          .get('criteria', {})
          .get('focal_point', {}))
    alg = fp.get('algorithm', {}).get('params', {})
    return (
        _f(alg.get('center_zone_half_width_ft'), 1.5),
        _f(fp.get('preferred_value'), 80.0),
        20.0,   # hard-coded in scoring_methods.py
    )


def _optimal_gap(scoring_data):
    """
    Return a gap that targets spacing_raw = preferred_value (75 by default).

    gap_variance_vs_ideal raw = 100 − avg_deviation × 30
    Setting raw = preferred_value (75) → avg_deviation = 25/30 ≈ 0.833 ft
    → gap = ideal_gap + 0.833 ≈ 0.55 + 0.833 = 1.383 ft
    Clamped to [min_gap, max_gap].
    """
    criteria = scoring_data.get('scoring', {}).get('criteria', {})
    sr = criteria.get('spacing_regularity', {})
    preferred_raw = _f(sr.get('preferred_value'), 75.0)
    alg_params = sr.get('algorithm', {}).get('params', {})
    ideal_gap = _f(alg_params.get('ideal_gap_ft'), 0.55)

    deviation = (100.0 - preferred_raw) / 30.0
    gap = ideal_gap + deviation
    mn, mx = _gap_bounds(scoring_data)
    return max(mn, min(mx, gap))


def _theme_sim(a, b, pairwise_tables, default=0.1):
    """Best theme-similarity score between two artworks."""
    table = pairwise_tables.get('theme_similarity', {})
    pairs = table.get('pairs', {})
    default_sim = table.get('default_similarity', default)

    keys_a = ([a['primary_theme']] if a.get('primary_theme') else []) + list(a.get('theme_tags', []))
    keys_b = ([b['primary_theme']] if b.get('primary_theme') else []) + list(b.get('theme_tags', []))

    best = default_sim
    for ka in keys_a:
        for kb in keys_b:
            best = max(best, pairs.get(f'{ka}|{kb}', pairs.get(f'{kb}|{ka}', default_sim)))
    return best


def _greedy_theme_order(artworks, pairwise_tables):
    """
    Nearest-neighbour chain to maximise sum of adjacent theme similarities.
    Starts from the 'loneliest' artwork (lowest max-similarity to any other).
    """
    if len(artworks) <= 1:
        return list(artworks)

    remaining = list(artworks)

    def max_sim(art):
        others = [o for o in remaining if o['id'] != art['id']]
        return max((_theme_sim(art, o, pairwise_tables) for o in others), default=0.0)

    start = min(remaining, key=max_sim)
    chain = [start]
    remaining.remove(start)

    while remaining:
        last = chain[-1]
        nxt = max(remaining, key=lambda a: _theme_sim(last, a, pairwise_tables))
        chain.append(nxt)
        remaining.remove(nxt)

    return chain


# ---------------------------------------------------------------------------
# placement builders
# ---------------------------------------------------------------------------

def _place_uniform(wall, ordering, gap, scoring_data):
    """
    Place artworks left-to-right with a uniform gap, respecting a left/right
    margin of min_gap/2 on each side.
    """
    mn, mx = _gap_bounds(scoring_data)
    gap = max(mn, min(mx, gap))
    wall_width = _f(wall.get('width_ft'))
    target_y = _f(wall.get('default_hang_y_ft'), 5.5)
    margin = mn / 2.0

    x = margin
    placements = []
    for art in ordering:
        w = _f(art.get('width_ft'))
        h = _f(art.get('height_ft'))
        if w <= 0:
            continue
        if x + w > wall_width - margin:
            break
        placements.append({
            'artwork_id': art['id'],
            'x_ft': round(x, 2),
            'y_ft': round(target_y - h / 2.0, 2),
            'locked': bool(art.get('locked', False)),
            'required': bool(art.get('required', False)),
            'notes': '',
        })
        x += w + gap
    return placements


def _place_anchor_right(wall, others_order, anchor, gap, scoring_data):
    """
    Place the focal anchor so its centre is wall_centre + optimal_offset
    (targeting focal_point raw = preferred_value).  Other artworks are packed
    to the left with the same gap, never closer than margin to the left edge.
    """
    mn, mx = _gap_bounds(scoring_data)
    gap = max(mn, min(mx, gap))
    margin = mn / 2.0

    wall_width = _f(wall.get('width_ft'))
    wall_center = _f(wall.get('centerline_ft'), wall_width / 2.0)
    target_y = _f(wall.get('default_hang_y_ft'), 5.5)

    zone_half, preferred_raw, penalty = _focal_params(scoring_data)
    optimal_dist = zone_half + (100.0 - preferred_raw) / penalty
    anchor_w = _f(anchor.get('width_ft'))
    anchor_x = (wall_center + optimal_dist) - anchor_w / 2.0
    anchor_x = max(margin, min(anchor_x, wall_width - anchor_w - margin))

    n = len(others_order)
    sum_w = sum(_f(a.get('width_ft')) for a in others_order)

    start_x = anchor_x - sum_w - n * gap

    if start_x < margin:
        if n > 0 and anchor_x - margin > sum_w:
            gap = max(mn, min(mx, (anchor_x - margin - sum_w) / n))
        start_x = max(margin, anchor_x - sum_w - n * gap)

    x = start_x
    placements = []
    for art in others_order:
        w = _f(art.get('width_ft'))
        h = _f(art.get('height_ft'))
        if w <= 0:
            continue
        placements.append({
            'artwork_id': art['id'],
            'x_ft': round(x, 2),
            'y_ft': round(target_y - h / 2.0, 2),
            'locked': bool(art.get('locked', False)),
            'required': bool(art.get('required', False)),
            'notes': '',
        })
        x += w + gap

    anchor_h = _f(anchor.get('height_ft'))
    placements.append({
        'artwork_id': anchor['id'],
        'x_ft': round(anchor_x, 2),
        'y_ft': round(target_y - anchor_h / 2.0, 2),
        'locked': bool(anchor.get('locked', False)),
        'required': bool(anchor.get('required', False)),
        'notes': '',
    })
    return placements


# ---------------------------------------------------------------------------
# local improvement
# ---------------------------------------------------------------------------

def _local_swap(wall, ordering, gap, eligible, scoring_data, current_score):
    """
    Try all pairwise swaps in the ordering, re-place, re-score.
    Accepts a swap only if it strictly improves the total score.
    Repeats until no improving swap is found.
    """
    best_order = list(ordering)
    best_score = current_score
    improved = True

    while improved:
        improved = False
        for i in range(len(best_order)):
            for j in range(i + 1, len(best_order)):
                candidate = list(best_order)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                placements = _place_uniform(wall, candidate, gap, scoring_data)
                if not placements:
                    continue
                result = evaluate(wall, placements, eligible, scoring_data)
                if result['total'] > best_score + 1e-4:
                    best_order = candidate
                    best_score = result['total']
                    improved = True
    return best_order, best_score


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def generate(wall, artworks, scoring_data):
    eligible = [a for a in artworks if a.get('eligible', True)]
    if not eligible:
        return []

    pairwise = scoring_data.get('scoring', {}).get('pairwise_tables', {})
    gap = _optimal_gap(scoring_data)

    anchor = max(eligible, key=lambda a: _f(a.get('focal_weight')))
    others = [a for a in eligible if a['id'] != anchor['id']]

    theme_others = _greedy_theme_order(others, pairwise)
    s1 = _place_anchor_right(wall, theme_others, anchor, gap, scoring_data)

    intensity_others = sorted(others, key=lambda a: _f(a.get('visual_intensity')))
    s2 = _place_anchor_right(wall, intensity_others, anchor, gap, scoring_data)

    theme_full = _greedy_theme_order(eligible, pairwise)
    s3 = _place_uniform(wall, theme_full, gap, scoring_data)

    focal_sorted = sorted(
        eligible,
        key=lambda a: (_f(a.get('focal_weight')), _f(a.get('visual_intensity'))),
        reverse=True,
    )
    s4 = _place_uniform(wall, focal_sorted, gap, scoring_data)

    s5 = place_left_to_right(wall, focal_sorted, scoring_data)

    best_score = -1.0
    best_placements = []
    best_ordering = theme_full

    for ordering, placements in [
        (theme_others + [anchor], s1),
        (intensity_others + [anchor], s2),
        (theme_full, s3),
        (focal_sorted, s4),
        (focal_sorted, s5),
    ]:
        if not placements:
            continue
        result = evaluate(wall, placements, eligible, scoring_data)
        if result['total'] > best_score:
            best_score = result['total']
            best_placements = placements
            best_ordering = ordering

    if not best_placements:
        return []

    improved_order, _ = _local_swap(
        wall, best_ordering, gap, eligible, scoring_data, best_score
    )
    improved_placements = _place_uniform(wall, improved_order, gap, scoring_data)
    if improved_placements:
        result = evaluate(wall, improved_placements, eligible, scoring_data)
        if result['total'] > best_score:
            best_placements = improved_placements

    return best_placements
