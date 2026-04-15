import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def get_wall(gallery_data, wall_id):
    gallery = gallery_data.get('gallery', {})
    for room in gallery.get('rooms', []):
        room_id = room.get('id') or room.get('room_id')
        for wall in room.get('walls', []):
            current_id = wall.get('id') or wall.get('wall_id')
            if current_id == wall_id:
                normalized = dict(wall)
                normalized['wall_id'] = current_id
                normalized['room_id'] = room_id
                normalized['width_ft'] = wall.get('width_ft', wall.get('length_ft', 0.0))
                normalized['height_ft'] = wall.get('height_ft', room.get('ceiling_height_ft', 12.0))
                normalized['centerline_ft'] = normalized['width_ft'] / 2.0 if normalized['width_ft'] else 0.0
                return normalized
    raise ValueError(f"Wall '{wall_id}' not found")


def build_artwork_lookup(art_data):
    artworks = art_data.get('art', {}).get('artworks', [])
    lookup = {}
    duplicates = []
    for art in artworks:
        art_id = art.get('id')
        if not art_id:
            continue
        if art_id in lookup:
            duplicates.append(art_id)
            continue
        lookup[art_id] = art
    return lookup, duplicates


def get_candidate_artworks(art_data, candidate_ids, scoring_data=None):
    lookup, duplicates = build_artwork_lookup(art_data)
    constraints = (scoring_data or {}).get('scoring', {}).get('hard_constraints', {})
    require_eligible = constraints.get('require_eligible', {}).get('enabled', False)

    selected = []
    missing = []
    skipped = []

    for art_id in candidate_ids:
        art = lookup.get(art_id)
        if not art:
            missing.append(art_id)
            continue
        if require_eligible and art.get('eligible') is False:
            skipped.append((art_id, 'eligible=false'))
            continue
        selected.append(art)

    return {
        'artworks': selected,
        'missing_ids': missing,
        'skipped': skipped,
        'duplicate_ids_in_catalog': duplicates,
    }


def get_wall_design(show_data, wall_id):
    wall_designs = show_data.get('show', {}).get('wall_designs', [])
    for entry in wall_designs:
        if entry.get('wall_id') == wall_id:
            return entry
    return None
