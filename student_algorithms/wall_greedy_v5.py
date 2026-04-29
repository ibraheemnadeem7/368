"""
wall_greedy_v5.py  -  Fit-aware greedy wall designer

This version fixes two failure modes in v4:
1. It filters out artworks that cannot physically fit on the target wall.
2. It never builds or returns arrangements that violate hard constraints
   such as max_artworks or stay_within_wall.
"""

from wall_designer.placer import place_left_to_right
from wall_designer.scorer import evaluate


def _f(v, default=0.0):
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


def _max_artworks(scoring_data, default=8):
    hard = _hard(scoring_data)
    cfg = hard.get('max_artworks', {})
    if cfg.get('enabled'):
        return max(1, int(_f(cfg.get('value'), default)))
    return default


def _focal_params(scoring_data):
    fp = (
        scoring_data.get('scoring', {})
        .get('criteria', {})
        .get('focal_point', {})
    )
    alg = fp.get('algorithm', {}).get('params', {})
    return (
        _f(alg.get('center_zone_half_width_ft'), 1.5),
        _f(fp.get('preferred_value'), 80.0),
        20.0,
    )


def _optimal_gap(scoring_data):
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


def _greedy_theme_order_anchored(others, anchor, pairwise_tables):
    if not others:
        return []
    if len(others) == 1:
        return list(others)

    remaining = list(others)
    end = max(remaining, key=lambda a: _theme_sim(a, anchor, pairwise_tables))
    chain = [end]
    remaining.remove(end)

    while remaining:
        first = chain[0]
        nxt = max(remaining, key=lambda a: _theme_sim(a, first, pairwise_tables))
        chain.insert(0, nxt)
        remaining.remove(nxt)

    return chain


def _clamp_y(wall, art):
    target_y = _f(wall.get('default_hang_y_ft'), 5.5)
    wall_h = _f(wall.get('height_ft'), 12.0)
    art_h = _f(art.get('height_ft'))
    return round(min(max(target_y - art_h / 2.0, 0.0), max(0.0, wall_h - art_h)), 2)


def _fits_on_wall(art, wall, scoring_data):
    mn, _ = _gap_bounds(scoring_data)
    width = _f(art.get('width_ft'))
    height = _f(art.get('height_ft'))
    wall_w = _f(wall.get('width_ft'))
    wall_h = _f(wall.get('height_ft'), 12.0)
    margin = mn / 2.0

    if width <= 0 or height <= 0:
        return False
    if width > max(0.0, wall_w - 2.0 * margin):
        return False
    if height > wall_h:
        return False
    return True


def _score_for_subset(art, anchor, pairwise_tables, wall_width):
    width = _f(art.get('width_ft'))
    focal = _f(art.get('focal_weight'))
    intensity = _f(art.get('visual_intensity'))
    theme = 0.0 if anchor is None else _theme_sim(art, anchor, pairwise_tables)
    width_bonus = 0.0 if wall_width <= 0 else max(0.0, 1.0 - (width / wall_width))
    return (0.45 * theme) + (0.25 * focal) + (0.20 * intensity) + (0.10 * width_bonus)


def _subset_by_rank(anchor, others, rank_fn, wall, scoring_data, limit):
    mn, _ = _gap_bounds(scoring_data)
    margin = mn / 2.0
    wall_w = _f(wall.get('width_ft'))
    usable = max(0.0, wall_w - 2.0 * margin)

    chosen = []
    total_width = 0.0

    if anchor is not None:
        chosen.append(anchor)
        total_width += _f(anchor.get('width_ft'))

    for art in sorted(others, key=rank_fn, reverse=True):
        if len(chosen) >= limit:
            break
        trial_count = len(chosen) + 1
        trial_width = total_width + _f(art.get('width_ft'))
        trial_span = trial_width + max(0, trial_count - 1) * mn
        if trial_span <= usable + 1e-6:
            chosen.append(art)
            total_width = trial_width

    return chosen


def _candidate_subsets(feasible, wall, scoring_data, pairwise_tables):
    limit = _max_artworks(scoring_data)
    if not feasible:
        return []

    wall_w = _f(wall.get('width_ft'))
    anchor = max(feasible, key=lambda a: (_f(a.get('focal_weight')), -_f(a.get('width_ft'))))
    others = [a for a in feasible if a['id'] != anchor['id']]

    subsets = []
    seen = set()

    def add_subset(items):
        ids = tuple(a['id'] for a in items)
        if len(items) >= 2 and ids not in seen:
            seen.add(ids)
            subsets.append(list(items))

    add_subset(_subset_by_rank(
        anchor,
        others,
        rank_fn=lambda a: _score_for_subset(a, anchor, pairwise_tables, wall_w),
        wall=wall,
        scoring_data=scoring_data,
        limit=limit,
    ))
    add_subset(_subset_by_rank(
        anchor,
        others,
        rank_fn=lambda a: _f(a.get('visual_intensity')),
        wall=wall,
        scoring_data=scoring_data,
        limit=limit,
    ))
    add_subset(_subset_by_rank(
        anchor,
        others,
        rank_fn=lambda a: _f(a.get('focal_weight')),
        wall=wall,
        scoring_data=scoring_data,
        limit=limit,
    ))
    add_subset(_subset_by_rank(
        anchor,
        others,
        rank_fn=lambda a: -_f(a.get('width_ft')),
        wall=wall,
        scoring_data=scoring_data,
        limit=limit,
    ))

    full_theme = _greedy_theme_order(feasible, pairwise_tables)
    theme_subset = []
    total_width = 0.0
    mn, _ = _gap_bounds(scoring_data)
    usable = max(0.0, wall_w - mn)
    for art in full_theme:
        trial_count = len(theme_subset) + 1
        trial_width = total_width + _f(art.get('width_ft'))
        trial_span = trial_width + max(0, trial_count - 1) * mn
        if trial_count <= limit and trial_span <= usable + 1e-6:
            theme_subset.append(art)
            total_width = trial_width
    add_subset(theme_subset)

    return subsets


def _place_uniform(wall, ordering, gap, scoring_data):
    mn, mx = _gap_bounds(scoring_data)
    gap = max(mn, min(mx, gap))
    wall_width = _f(wall.get('width_ft'))
    min_margin = mn / 2.0

    to_place = []
    span = 0.0
    for art in ordering:
        w = _f(art.get('width_ft'))
        h = _f(art.get('height_ft'))
        if w <= 0 or h <= 0:
            continue
        extra = (gap if to_place else 0.0) + w
        if 2.0 * min_margin + span + extra > wall_width:
            break
        to_place.append(art)
        span += extra

    if len(to_place) < 2:
        return []

    x = max(min_margin, (wall_width - span) / 2.0)
    placements = []
    for art in to_place:
        w = _f(art.get('width_ft'))
        placements.append({
            'artwork_id': art['id'],
            'x_ft': round(x, 2),
            'y_ft': _clamp_y(wall, art),
            'locked': bool(art.get('locked', False)),
            'required': bool(art.get('required', False)),
            'notes': '',
        })
        x += w + gap
    return placements


def _place_anchor_right(wall, others_order, anchor, gap, scoring_data):
    mn, mx = _gap_bounds(scoring_data)
    gap = max(mn, min(mx, gap))
    wall_width = _f(wall.get('width_ft'))
    wall_center = _f(wall.get('centerline_ft'), wall_width / 2.0)

    zone_half, preferred_raw, penalty = _focal_params(scoring_data)
    optimal_dist = zone_half + (100.0 - preferred_raw) / penalty
    anchor_w = _f(anchor.get('width_ft'))
    if anchor_w <= 0:
        return []

    margin = mn / 2.0
    anchor_x = (wall_center + optimal_dist) - anchor_w / 2.0
    anchor_x = max(margin, min(anchor_x, wall_width - anchor_w - margin))

    chosen = []
    used_width = 0.0
    for art in others_order:
        w = _f(art.get('width_ft'))
        next_count = len(chosen) + 1
        needed = used_width + w + next_count * gap
        if anchor_x - margin >= needed - 1e-6:
            chosen.append(art)
            used_width += w

    if not chosen:
        return []

    start_x = anchor_x - used_width - len(chosen) * gap
    start_x = max(margin, start_x)

    placements = []
    x = start_x
    for art in chosen:
        w = _f(art.get('width_ft'))
        placements.append({
            'artwork_id': art['id'],
            'x_ft': round(x, 2),
            'y_ft': _clamp_y(wall, art),
            'locked': bool(art.get('locked', False)),
            'required': bool(art.get('required', False)),
            'notes': '',
        })
        x += w + gap

    placements.append({
        'artwork_id': anchor['id'],
        'x_ft': round(anchor_x, 2),
        'y_ft': _clamp_y(wall, anchor),
        'locked': bool(anchor.get('locked', False)),
        'required': bool(anchor.get('required', False)),
        'notes': '',
    })
    return placements


def _local_swap(wall, ordering, gap, eligible, scoring_data, current_score):
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
                if result['failed_constraints']:
                    continue
                if result['total'] > best_score + 1e-4:
                    best_order = candidate
                    best_score = result['total']
                    improved = True
    return best_order, best_score


def generate(wall, artworks, scoring_data):
    eligible = [a for a in artworks if a.get('eligible', True)]
    feasible = [a for a in eligible if _fits_on_wall(a, wall, scoring_data)]
    if len(feasible) < 2:
        return []

    pairwise = scoring_data.get('scoring', {}).get('pairwise_tables', {})
    gap = _optimal_gap(scoring_data)

    best_score = -1.0
    best_placements = []
    best_ordering = []
    best_subset = []

    for subset in _candidate_subsets(feasible, wall, scoring_data, pairwise):
        if len(subset) < 2:
            continue

        anchor = max(subset, key=lambda a: _f(a.get('focal_weight')))
        others = [a for a in subset if a['id'] != anchor['id']]
        theme_others = _greedy_theme_order_anchored(others, anchor, pairwise)
        intensity_others = sorted(others, key=lambda a: _f(a.get('visual_intensity')))
        theme_full = _greedy_theme_order(subset, pairwise)
        focal_sorted = sorted(
            subset,
            key=lambda a: (_f(a.get('focal_weight')), _f(a.get('visual_intensity'))),
            reverse=True,
        )

        candidates = [
            (theme_others + [anchor], _place_anchor_right(wall, theme_others, anchor, gap, scoring_data)),
            (intensity_others + [anchor], _place_anchor_right(wall, intensity_others, anchor, gap, scoring_data)),
            (theme_full, _place_uniform(wall, theme_full, gap, scoring_data)),
            (focal_sorted, _place_uniform(wall, focal_sorted, gap, scoring_data)),
            (focal_sorted, place_left_to_right(wall, focal_sorted, scoring_data)),
        ]

        for ordering, placements in candidates:
            if not placements:
                continue
            result = evaluate(wall, placements, subset, scoring_data)
            if result['failed_constraints']:
                continue
            if result['total'] > best_score:
                best_score = result['total']
                best_placements = placements
                best_ordering = ordering
                best_subset = subset

    if not best_placements:
        return []

    improved_order, improved_score = _local_swap(
        wall, best_ordering, gap, best_subset, scoring_data, best_score
    )
    improved_placements = _place_uniform(wall, improved_order, gap, scoring_data)
    if improved_placements:
        result = evaluate(wall, improved_placements, best_subset, scoring_data)
        if not result['failed_constraints'] and result['total'] > improved_score:
            best_placements = improved_placements

    return best_placements
