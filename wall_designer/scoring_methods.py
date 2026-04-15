from statistics import pstdev


def target_score(raw, preferred, tolerance, in_tolerance_floor=0.7, out_of_tolerance_decay=0.02, min_score=0.0):
    diff = abs(raw - preferred)
    if diff <= tolerance:
        if tolerance == 0:
            return 1.0
        closeness = 1.0 - (diff / tolerance)
        return max(in_tolerance_floor, in_tolerance_floor + (1.0 - in_tolerance_floor) * closeness)
    value = 1.0 - ((diff - tolerance) * out_of_tolerance_decay)
    return max(min_score, min(1.0, value))


def _ordered(placements, artwork_lookup):
    return sorted(placements, key=lambda p: p['x_ft'])


def adjacent_theme_similarity_average(wall, placements, artwork_lookup, params, pairwise_tables):
    ordered = _ordered(placements, artwork_lookup)
    if len(ordered) < 2:
        return 100.0
    theme_table = pairwise_tables.get('theme_similarity', {})
    default_sim = params.get('default_similarity', theme_table.get('default_similarity', 0.1))
    pair_scores = theme_table.get('pairs', {})
    vals = []
    for i in range(len(ordered) - 1):
        a = artwork_lookup[ordered[i]['artwork_id']]
        b = artwork_lookup[ordered[i + 1]['artwork_id']]
        keys_a = []
        keys_b = []
        if params.get('use_primary_theme_first', True):
            if a.get('primary_theme'): keys_a.append(a['primary_theme'])
            if b.get('primary_theme'): keys_b.append(b['primary_theme'])
        if params.get('allow_theme_tag_fallback', True):
            keys_a.extend(a.get('theme_tags', []))
            keys_b.extend(b.get('theme_tags', []))
        best = default_sim
        for ka in keys_a:
            for kb in keys_b:
                k1=f'{ka}|{kb}'
                k2=f'{kb}|{ka}'
                best=max(best, pair_scores.get(k1, pair_scores.get(k2, default_sim)))
        vals.append(best)
    return 100.0 * (sum(vals) / len(vals))


def distinct_artist_ratio(wall, placements, artwork_lookup, params, pairwise_tables):
    ordered = _ordered(placements, artwork_lookup)
    if not ordered:
        return 0.0
    artists = [artwork_lookup[p['artwork_id']].get('artist', '') for p in ordered]
    ratio = len(set(artists)) / len(artists)
    raw = ratio * 100.0
    penalty = params.get('adjacency_penalty_same_artist', 0)
    for i in range(len(artists)-1):
        if artists[i] and artists[i] == artists[i+1]:
            raw -= penalty
    return max(0.0, raw)


def gap_variance_vs_ideal(wall, placements, artwork_lookup, params, pairwise_tables):
    ordered = _ordered(placements, artwork_lookup)
    if len(ordered) < 2:
        return 100.0
    ideal = params.get('ideal_gap_ft', 0.55)
    min_gap = params.get('min_gap_ft', 0.25)
    max_gap = params.get('max_gap_ft', 1.5)
    out_penalty = params.get('out_of_range_gap_penalty', 25)
    gaps = []
    raw = 100.0
    for i in range(len(ordered)-1):
        a = ordered[i]
        b = ordered[i+1]
        width_a = float(artwork_lookup[a['artwork_id']].get('width_ft', 0.0) or 0.0)
        gap = b['x_ft'] - (a['x_ft'] + width_a)
        gaps.append(gap)
        if gap < min_gap or gap > max_gap:
            raw -= out_penalty
    dev = sum(abs(g - ideal) for g in gaps) / len(gaps)
    raw -= dev * 30.0
    return max(0.0, min(100.0, raw))


def left_right_visual_mass_balance(wall, placements, artwork_lookup, params, pairwise_tables):
    center = wall.get('centerline_ft', wall.get('width_ft', 0.0)/2.0)
    if center <= 0 or not placements:
        return 0.0
    left_mass = 0.0
    right_mass = 0.0
    floor = params.get('intensity_floor', 0.5)
    for p in placements:
        art = artwork_lookup[p['artwork_id']]
        width = float(art.get('width_ft', 0.0) or 0.0)
        height = float(art.get('height_ft', 0.0) or 0.0)
        intensity = max(float(art.get('visual_intensity', floor) or floor), floor)
        mass = width * height * intensity
        art_center = p['x_ft'] + width / 2.0
        if art_center <= center:
            left_mass += mass
        else:
            right_mass += mass
    total = left_mass + right_mass
    if total <= 0:
        return 0.0
    diff_ratio = abs(left_mass - right_mass) / total
    return max(0.0, 100.0 * (1.0 - diff_ratio))


def occupied_width_ratio(wall, placements, artwork_lookup, params, pairwise_tables):
    if not placements:
        return 0.0
    wall_width = float(wall.get('width_ft', 0.0) or 0.0)
    if wall_width <= 0:
        return 0.0
    ordered = _ordered(placements, artwork_lookup)
    first = ordered[0]
    last = ordered[-1]
    last_width = float(artwork_lookup[last['artwork_id']].get('width_ft', 0.0) or 0.0)
    occupied = (last['x_ft'] + last_width) - first['x_ft']
    ratio = occupied / wall_width
    return max(0.0, min(100.0, ratio * 100.0))


def featured_work_near_preferred_zone(wall, placements, artwork_lookup, params, pairwise_tables):
    if not placements:
        return 0.0
    featured = max(placements, key=lambda p: float(artwork_lookup[p['artwork_id']].get('focal_weight', 0.0) or 0.0))
    art = artwork_lookup[featured['artwork_id']]
    width = float(art.get('width_ft', 0.0) or 0.0)
    center = featured['x_ft'] + width / 2.0
    wall_center = wall.get('centerline_ft', wall.get('width_ft', 0.0) / 2.0)
    half = params.get('center_zone_half_width_ft', 1.5)
    diff = abs(center - wall_center)
    if diff <= half:
        return 100.0
    return max(0.0, 100.0 - ((diff - half) * 20.0))


METHODS = {
    'adjacent_theme_similarity_average': adjacent_theme_similarity_average,
    'distinct_artist_ratio': distinct_artist_ratio,
    'gap_variance_vs_ideal': gap_variance_vs_ideal,
    'left_right_visual_mass_balance': left_right_visual_mass_balance,
    'occupied_width_ratio': occupied_width_ratio,
    'featured_work_near_preferred_zone': featured_work_near_preferred_zone,
}
