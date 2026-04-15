def place_left_to_right(wall, artworks, scoring_data):
    constraints = scoring_data['scoring']['hard_constraints']
    wall_width = wall.get('width_ft', 0.0)
    margin = max(0.0, constraints.get('min_gap_ft', {}).get('value', 0.25) / 2.0)
    target_y = wall.get('default_hang_y_ft', 5.5)
    min_gap = constraints.get('min_gap_ft', {}).get('value', 0.25)

    x = margin
    placements = []

    for art in artworks:
        width = float(art.get('width_ft', 0.0) or 0.0)
        if width <= 0:
            continue

        if art.get('locked') and art.get('locked_position', {}).get('x_ft') is not None:
            px = float(art['locked_position']['x_ft'])
            py = float(art['locked_position'].get('y_ft', target_y))
        else:
            px = x
            py = target_y - (float(art.get('height_ft', 0.0) or 0.0) / 2.0)

        if px + width > wall_width - margin:
            continue

        placements.append({
            'artwork_id': art['id'],
            'x_ft': round(px, 2),
            'y_ft': round(py, 2),
            'locked': bool(art.get('locked', False)),
            'required': bool(art.get('required', False)),
            'notes': ''
        })

        x = px + width + min_gap

    return placements
