from wall_designer.placer import place_left_to_right


def generate(wall, artworks, scoring_data):
    # Simple baseline heuristic:
    # 1. keep eligible wall-friendly works
    # 2. prioritize higher focal weight, then higher visual intensity, then wider works
    filtered = [a for a in artworks if a.get('eligible', True)]
    filtered = sorted(
        filtered,
        key=lambda a: (
            float(a.get('focal_weight', 0.0) or 0.0),
            float(a.get('visual_intensity', 0.0) or 0.0),
            float(a.get('width_ft', 0.0) or 0.0),
        ),
        reverse=True,
    )
    return place_left_to_right(wall, filtered, scoring_data)
