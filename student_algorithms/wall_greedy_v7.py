"""
wall_greedy_v7.py  -  Duplicate-aware curatorial greedy wall designer

Targets medium walls such as R2-W where the previous versions could select:
- duplicate or near-duplicate works
- too many tiny accent pieces
- visually scattered mixtures around one larger anchor
"""

from wall_designer.placer import place_left_to_right
from wall_designer.scorer import evaluate

from student_algorithms import wall_greedy_v5 as v5


def _f(v, default=0.0):
    try:
        return float(v or default)
    except (TypeError, ValueError):
        return default


def _norm_title(title):
    return ''.join(ch.lower() for ch in (title or '') if ch.isalnum() or ch.isspace()).strip()


def _wall_profile(wall):
    wall_w = _f(wall.get('width_ft'))
    if wall_w >= 20.0 and wall_w <= 30.0:
        return {
            'medium_wall': True,
            'tiny_ratio': 0.055,
            'target_avg_width': 2.7,
            'min_medium_count': 3,
        }
    return {
        'medium_wall': False,
        'tiny_ratio': 0.05,
        'target_avg_width': 2.2,
        'min_medium_count': 2,
    }


def _tag_jaccard(values_a, values_b):
    a = {x for x in values_a if x}
    b = {x for x in values_b if x}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _palette_sim(a, b):
    return _tag_jaccard(a.get('palette_tags', []), b.get('palette_tags', []))


def _mood_sim(a, b):
    return _tag_jaccard(a.get('mood_tags', []), b.get('mood_tags', []))


def _has_duplicate_titles(artworks):
    titles = [_norm_title(a.get('title')) for a in artworks]
    titles = [t for t in titles if t]
    return len(set(titles)) != len(titles)


def _duplicate_penalty(artworks):
    return 1.0 if _has_duplicate_titles(artworks) else 0.0


def _scale_stats(artworks, wall):
    wall_w = max(_f(wall.get('width_ft')), 1.0)
    profile = _wall_profile(wall)
    widths = [_f(a.get('width_ft')) for a in artworks if _f(a.get('width_ft')) > 0]
    avg_width = (sum(widths) / len(widths)) if widths else 0.0
    tiny_count = sum(1 for w in widths if (w / wall_w) < profile['tiny_ratio'])
    medium_count = sum(1 for w in widths if w >= 2.4)
    return profile, avg_width, tiny_count, medium_count


def _cluster_score(artworks, pairwise_tables):
    if len(artworks) < 2:
        return 0.0

    theme = []
    palette = []
    mood = []
    for i in range(len(artworks)):
        for j in range(i + 1, len(artworks)):
            a = artworks[i]
            b = artworks[j]
            theme.append(v5._theme_sim(a, b, pairwise_tables))
            palette.append(_palette_sim(a, b))
            mood.append(_mood_sim(a, b))

    return (
        0.50 * (sum(theme) / len(theme)) +
        0.32 * (sum(palette) / len(palette)) +
        0.18 * (sum(mood) / len(mood))
    )


def _scale_bonus(artworks, wall):
    profile, avg_width, tiny_count, medium_count = _scale_stats(artworks, wall)
    avg_score = max(0.0, 1.0 - min(1.0, abs(avg_width - profile['target_avg_width']) / max(profile['target_avg_width'], 0.5)))
    medium_score = min(1.0, medium_count / max(profile['min_medium_count'], 1))
    tiny_penalty = min(1.0, tiny_count / 2.0)
    return (0.58 * avg_score) + (0.42 * medium_score) - (0.40 * tiny_penalty)


def _subset_value(artworks, wall, pairwise_tables):
    return (
        0.62 * _cluster_score(artworks, pairwise_tables) +
        0.38 * _scale_bonus(artworks, wall) -
        0.80 * _duplicate_penalty(artworks)
    )


def _passes_gate(artworks, wall):
    profile, avg_width, tiny_count, medium_count = _scale_stats(artworks, wall)
    if _has_duplicate_titles(artworks):
        return False
    if profile['medium_wall']:
        if tiny_count > 1:
            return False
        if medium_count < profile['min_medium_count']:
            return False
        if avg_width < 2.1:
            return False
    return True


def _rank_against_anchor(art, anchor, wall):
    wall_w = max(_f(wall.get('width_ft')), 1.0)
    width = _f(art.get('width_ft'))
    return (
        0.30 * v5._theme_sim(art, anchor, {}) +
        0.22 * _palette_sim(art, anchor) +
        0.10 * _mood_sim(art, anchor) +
        0.18 * _f(art.get('focal_weight')) +
        0.10 * _f(art.get('visual_intensity')) +
        0.10 * min(1.0, width / (wall_w * 0.16))
    )


def _candidate_subsets(feasible, wall, scoring_data, pairwise_tables):
    limit = v5._max_artworks(scoring_data)
    wall_w = _f(wall.get('width_ft'))
    mn, _ = v5._gap_bounds(scoring_data)
    usable = max(0.0, wall_w - mn)

    anchors = sorted(
        feasible,
        key=lambda a: (
            _f(a.get('focal_weight')) +
            0.20 * _f(a.get('visual_intensity')) +
            0.20 * min(1.0, _f(a.get('width_ft')) / max(wall_w * 0.16, 1.0))
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
        others = [a for a in feasible if a['id'] != anchor['id'] and _norm_title(a.get('title')) != _norm_title(anchor.get('title'))]
        ranked = sorted(
            others,
            key=lambda a: (
                0.35 * v5._theme_sim(a, anchor, pairwise_tables) +
                0.20 * _palette_sim(a, anchor) +
                0.10 * _mood_sim(a, anchor) +
                0.15 * _f(a.get('focal_weight')) +
                0.10 * _f(a.get('visual_intensity')) +
                0.10 * min(1.0, _f(a.get('width_ft')) / max(wall_w * 0.16, 1.0))
            ),
            reverse=True,
        )

        chosen = [anchor]
        total_width = _f(anchor.get('width_ft'))
        used_titles = {_norm_title(anchor.get('title'))}
        for art in ranked:
            title = _norm_title(art.get('title'))
            if title and title in used_titles:
                continue
            if len(chosen) >= limit:
                break
            trial_count = len(chosen) + 1
            trial_width = total_width + _f(art.get('width_ft'))
            trial_span = trial_width + max(0, trial_count - 1) * mn
            if trial_span <= usable + 1e-6:
                chosen.append(art)
                total_width = trial_width
                if title:
                    used_titles.add(title)

        add_subset(chosen)

    base = v5._candidate_subsets(feasible, wall, scoring_data, pairwise_tables)
    for subset in base:
        if not _has_duplicate_titles(subset):
            add_subset(subset)

    gated = [s for s in subsets if _passes_gate(s, wall)]
    ranked = gated or subsets
    return sorted(ranked, key=lambda s: _subset_value(s, wall, pairwise_tables), reverse=True)


def _placed_ordering(ordering, placements):
    placed_ids = {p['artwork_id'] for p in placements}
    return [a for a in ordering if a['id'] in placed_ids]


def _curatorial_bonus(artworks, wall, pairwise_tables):
    if not artworks:
        return 0.0
    return _subset_value(artworks, wall, pairwise_tables)


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
                _f(a.get('focal_weight')),
                _f(a.get('visual_intensity')),
                _f(a.get('width_ft')),
            ),
        )
        others = [a for a in subset if a['id'] != anchor['id']]
        theme_others = v5._greedy_theme_order_anchored(others, anchor, pairwise)
        theme_full = v5._greedy_theme_order(subset, pairwise)
        focal_sorted = sorted(
            subset,
            key=lambda a: (
                _f(a.get('focal_weight')),
                _f(a.get('visual_intensity')),
                _f(a.get('width_ft')),
            ),
            reverse=True,
        )

        candidates = [
            (theme_others + [anchor], v5._place_anchor_right(wall, theme_others, anchor, gap, scoring_data)),
            (theme_full, v5._place_uniform(wall, theme_full, gap, scoring_data)),
            (focal_sorted, v5._place_uniform(wall, focal_sorted, gap, scoring_data)),
            (focal_sorted, place_left_to_right(wall, focal_sorted, scoring_data)),
        ]

        for ordering, placements in candidates:
            if not placements:
                continue
            placed = _placed_ordering(ordering, placements)
            if not _passes_gate(placed, wall):
                continue
            result = evaluate(wall, placements, subset, scoring_data)
            if result['failed_constraints']:
                continue
            total_value = result['total'] + (0.28 * _curatorial_bonus(placed, wall, pairwise))
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
        improved_total_value = result['total'] + (0.28 * _curatorial_bonus(placed, wall, pairwise))
        if not result['failed_constraints'] and _passes_gate(placed, wall) and improved_total_value > best_total_value:
            best_placements = improved_placements

    return best_placements
