"""
wall_greedy_v6.py  -  Scale-aware curatorial greedy wall designer

Builds on v5, but adds a strong preference for better scale on wider walls:
- discourages long chains of tiny works on large walls
- rewards subsets with one or two anchor-scale works
- prefers stronger average artwork size and wall utilization rhythm
"""

from wall_designer.placer import place_left_to_right
from wall_designer.scorer import evaluate

from student_algorithms import wall_greedy_v5 as v5


def _f(v, default=0.0):
    try:
        return float(v or default)
    except (TypeError, ValueError):
        return default


def _wall_scale_profile(wall):
    wall_w = _f(wall.get('width_ft'))
    if wall_w >= 36.0:
        return {
            'large_wall': True,
            'tiny_ratio': 0.055,
            'anchor_ratio': 0.14,
            'target_avg_ratio': 0.11,
            'target_util': 0.68,
        }
    if wall_w >= 24.0:
        return {
            'large_wall': False,
            'tiny_ratio': 0.045,
            'anchor_ratio': 0.11,
            'target_avg_ratio': 0.085,
            'target_util': 0.64,
        }
    return {
        'large_wall': False,
        'tiny_ratio': 0.0,
        'anchor_ratio': 0.0,
        'target_avg_ratio': 0.06,
        'target_util': 0.58,
    }


def _scale_stats(ordering, wall):
    wall_w = max(_f(wall.get('width_ft')), 1.0)
    profile = _wall_scale_profile(wall)
    widths = [_f(a.get('width_ft')) for a in ordering if _f(a.get('width_ft')) > 0]
    ratios = [w / wall_w for w in widths]
    avg_ratio = (sum(ratios) / len(ratios)) if ratios else 0.0
    tiny_count = sum(1 for r in ratios if r < profile['tiny_ratio'])
    anchor_count = sum(1 for r in ratios if r >= profile['anchor_ratio'])
    return profile, avg_ratio, tiny_count, anchor_count


def _scale_bonus(ordering, wall):
    if len(ordering) < 2:
        return 0.0

    wall_w = max(_f(wall.get('width_ft')), 1.0)
    widths = [_f(a.get('width_ft')) for a in ordering if _f(a.get('width_ft')) > 0]
    if not widths:
        return 0.0

    profile, avg_ratio, tiny_count, anchor_count = _scale_stats(ordering, wall)
    occupied = sum(widths)

    mn, _ = v5._gap_bounds({'scoring': {'hard_constraints': {'min_gap_ft': {'value': 0.25}}}})
    occupied += max(0, len(widths) - 1) * mn
    util_ratio = occupied / wall_w

    avg_score = max(0.0, 1.0 - min(1.0, abs(avg_ratio - profile['target_avg_ratio']) / max(profile['target_avg_ratio'], 0.01)))
    util_score = max(0.0, 1.0 - min(1.0, abs(util_ratio - profile['target_util']) / 0.22))

    anchor_score = 0.0
    if profile['large_wall']:
        if anchor_count >= 2:
            anchor_score = 1.0
        elif anchor_count == 1:
            anchor_score = 0.65
    else:
        anchor_score = min(1.0, anchor_count)

    tiny_penalty = 0.0
    if profile['large_wall']:
        tiny_penalty = min(1.0, tiny_count / 2.5)

    return (
        0.40 * avg_score +
        0.35 * util_score +
        0.25 * anchor_score -
        0.35 * tiny_penalty
    )


def _passes_wide_wall_gate(ordering, wall):
    profile, avg_ratio, tiny_count, anchor_count = _scale_stats(ordering, wall)
    if not profile['large_wall']:
        return True

    if anchor_count < 1:
        return False
    if tiny_count > 1:
        return False
    if avg_ratio < profile['target_avg_ratio'] * 0.68:
        return False
    return True


def _placed_ordering(ordering, placements):
    placed_ids = {p['artwork_id'] for p in placements}
    return [art for art in ordering if art['id'] in placed_ids]


def _subset_rank(ordering, wall, pairwise_tables):
    base = v5._cluster_cohesion(ordering, pairwise_tables)
    scale = _scale_bonus(ordering, wall)
    return (0.58 * base) + (0.42 * scale)


def _wide_wall_subsets(feasible, wall, scoring_data, pairwise_tables):
    limit = v5._max_artworks(scoring_data)
    wall_w = _f(wall.get('width_ft'))
    mn, _ = v5._gap_bounds(scoring_data)
    usable = max(0.0, wall_w - mn)

    anchors = sorted(
        feasible,
        key=lambda a: (
            (_f(a.get('width_ft')) / max(wall_w, 1.0)) * 0.45 +
            _f(a.get('focal_weight')) * 0.35 +
            _f(a.get('visual_intensity')) * 0.20
        ),
        reverse=True,
    )[:6]

    subsets = []
    seen = set()

    def add_subset(items):
        ids = tuple(a['id'] for a in items)
        if len(items) >= 2 and ids not in seen:
            seen.add(ids)
            subsets.append(list(items))

    for anchor in anchors:
        others = [a for a in feasible if a['id'] != anchor['id']]
        ranked = sorted(
            others,
            key=lambda a: (
                0.40 * v5._theme_sim(a, anchor, pairwise_tables) +
                0.18 * v5._palette_sim(a, anchor) +
                0.10 * v5._mood_sim(a, anchor) +
                0.20 * min(1.0, _f(a.get('width_ft')) / max(wall_w * 0.18, 1.0)) +
                0.12 * _f(a.get('focal_weight'))
            ),
            reverse=True,
        )

        chosen = [anchor]
        total_width = _f(anchor.get('width_ft'))
        for art in ranked:
            if len(chosen) >= limit:
                break
            trial_count = len(chosen) + 1
            trial_width = total_width + _f(art.get('width_ft'))
            trial_span = trial_width + max(0, trial_count - 1) * mn
            if trial_span <= usable + 1e-6:
                chosen.append(art)
                total_width = trial_width

        add_subset(chosen)

    return sorted(subsets, key=lambda s: _subset_rank(s, wall, pairwise_tables), reverse=True)


def _candidate_subsets(feasible, wall, scoring_data, pairwise_tables):
    base = v5._candidate_subsets(feasible, wall, scoring_data, pairwise_tables)
    if not _wall_scale_profile(wall)['large_wall']:
        return base

    merged = []
    seen = set()
    for subset in _wide_wall_subsets(feasible, wall, scoring_data, pairwise_tables) + base:
        ids = tuple(a['id'] for a in subset)
        if ids not in seen:
            seen.add(ids)
            merged.append(subset)
    gated = [subset for subset in merged if _passes_wide_wall_gate(subset, wall)]
    return gated or merged


def _curatorial_bonus(ordering, wall, pairwise_tables):
    return v5._curatorial_bonus(ordering, wall, pairwise_tables) + (0.55 * _scale_bonus(ordering, wall))


def generate(wall, artworks, scoring_data):
    eligible = [a for a in artworks if a.get('eligible', True)]
    feasible = [a for a in eligible if v5._fits_on_wall(a, wall, scoring_data)]
    if len(feasible) < 2:
        return []

    pairwise = scoring_data.get('scoring', {}).get('pairwise_tables', {})
    gap = v5._optimal_gap(scoring_data)

    best_score = -1.0
    best_total_value = -1.0
    best_placements = []
    best_ordering = []
    best_subset = []

    for subset in _candidate_subsets(feasible, wall, scoring_data, pairwise):
        if len(subset) < 2:
            continue

        anchor = max(
            subset,
            key=lambda a: (
                _f(a.get('focal_weight')) +
                0.30 * min(1.0, _f(a.get('width_ft')) / max(_f(wall.get('width_ft')) * 0.18, 1.0))
            ),
        )
        others = [a for a in subset if a['id'] != anchor['id']]
        theme_others = v5._greedy_theme_order_anchored(others, anchor, pairwise)
        intensity_others = sorted(others, key=lambda a: _f(a.get('visual_intensity')))
        theme_full = v5._greedy_theme_order(subset, pairwise)
        focal_sorted = sorted(
            subset,
            key=lambda a: (_f(a.get('focal_weight')), _f(a.get('visual_intensity')), _f(a.get('width_ft'))),
            reverse=True,
        )

        candidates = [
            (theme_others + [anchor], v5._place_anchor_right(wall, theme_others, anchor, gap, scoring_data)),
            (intensity_others + [anchor], v5._place_anchor_right(wall, intensity_others, anchor, gap, scoring_data)),
            (theme_full, v5._place_uniform(wall, theme_full, gap, scoring_data)),
            (focal_sorted, v5._place_uniform(wall, focal_sorted, gap, scoring_data)),
            (focal_sorted, place_left_to_right(wall, focal_sorted, scoring_data)),
        ]

        for ordering, placements in candidates:
            if not placements:
                continue
            placed = _placed_ordering(ordering, placements)
            if not _passes_wide_wall_gate(placed, wall):
                continue
            result = evaluate(wall, placements, subset, scoring_data)
            if result['failed_constraints']:
                continue
            total_value = result['total'] + (0.34 * _curatorial_bonus(placed, wall, pairwise))
            if total_value > best_total_value:
                best_total_value = total_value
                best_score = result['total']
                best_placements = placements
                best_ordering = placed
                best_subset = subset

    if not best_placements:
        return []

    improved_order, _ = v5._local_swap(
        wall, best_ordering, gap, best_subset, scoring_data, best_score
    )
    improved_placements = v5._place_uniform(wall, improved_order, gap, scoring_data)
    if improved_placements:
        placed = _placed_ordering(improved_order, improved_placements)
        result = evaluate(wall, improved_placements, best_subset, scoring_data)
        improved_total_value = result['total'] + (0.34 * _curatorial_bonus(placed, wall, pairwise))
        if not result['failed_constraints'] and improved_total_value > best_total_value:
            best_placements = improved_placements

    return best_placements
