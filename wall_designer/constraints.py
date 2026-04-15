def _sorted_with_right_edges(placements, artwork_lookup):
    ordered = sorted(placements, key=lambda p: p['x_ft'])
    enriched = []
    for p in ordered:
        art = artwork_lookup[p['artwork_id']]
        width = float(art.get('width_ft', 0.0) or 0.0)
        enriched.append((p, p['x_ft'], p['x_ft'] + width))
    return enriched


def evaluate_hard_constraints(wall, placements, artworks, scoring_data):
    constraints = scoring_data.get('scoring', {}).get('hard_constraints', {})
    artwork_lookup = {a['id']: a for a in artworks}
    failures = []

    count = len(placements)
    if constraints.get('min_artworks', {}).get('enabled') and count < constraints['min_artworks']['value']:
        failures.append('min_artworks')
    if constraints.get('max_artworks', {}).get('enabled') and count > constraints['max_artworks']['value']:
        failures.append('max_artworks')

    wall_width = wall.get('width_ft', 0.0)
    ordered = _sorted_with_right_edges(placements, artwork_lookup)

    if constraints.get('stay_within_wall', {}).get('enabled'):
        for _, left, right in ordered:
            if left < 0 or right > wall_width:
                failures.append('stay_within_wall')
                break

    if constraints.get('no_overlap', {}).get('enabled'):
        for i in range(len(ordered) - 1):
            if ordered[i][2] > ordered[i + 1][1]:
                failures.append('no_overlap')
                break

    min_gap = constraints.get('min_gap_ft', {}).get('value')
    if constraints.get('min_gap_ft', {}).get('enabled') and min_gap is not None:
        for i in range(len(ordered) - 1):
            gap = ordered[i + 1][1] - ordered[i][2]
            if gap < min_gap:
                failures.append('min_gap_ft')
                break

    max_gap = constraints.get('max_gap_ft', {}).get('value')
    if constraints.get('max_gap_ft', {}).get('enabled') and max_gap is not None:
        for i in range(len(ordered) - 1):
            gap = ordered[i + 1][1] - ordered[i][2]
            if gap > max_gap:
                failures.append('max_gap_ft')
                break

    if constraints.get('respect_locked_positions', {}).get('enabled'):
        for p in placements:
            art = artwork_lookup[p['artwork_id']]
            if art.get('locked') and art.get('locked_position', {}).get('x_ft') is not None:
                expected_x = float(art['locked_position']['x_ft'])
                expected_y = float(art['locked_position'].get('y_ft', p['y_ft']))
                if abs(p['x_ft'] - expected_x) > 0.01 or abs(p['y_ft'] - expected_y) > 0.01:
                    failures.append('respect_locked_positions')
                    break

    if constraints.get('require_eligible', {}).get('enabled'):
        for art in artworks:
            if art.get('eligible') is False:
                failures.append('require_eligible')
                break

    return sorted(set(failures))
