from .constraints import evaluate_hard_constraints
from .scoring_methods import METHODS, target_score


def evaluate(wall, placements, artworks, scoring_data):
    artwork_lookup = {a['id']: a for a in artworks}
    failures = evaluate_hard_constraints(wall, placements, artworks, scoring_data)
    criteria = scoring_data.get('scoring', {}).get('criteria', {})
    pairwise_tables = scoring_data.get('scoring', {}).get('pairwise_tables', {})

    used = {}
    warnings = []
    weighted_sum = 0.0
    total_importance = 0.0

    for crit_key, crit in criteria.items():
        method_name = crit.get('algorithm', {}).get('method')
        fn = METHODS.get(method_name)
        if not fn:
            warnings.append(f'Unsupported criterion method: {method_name}')
            continue
        raw = fn(wall, placements, artwork_lookup, crit.get('algorithm', {}).get('params', {}), pairwise_tables)
        curve = crit.get('scoring_curve', {})
        score = target_score(
            raw=raw,
            preferred=crit.get('preferred_value', 50),
            tolerance=crit.get('tolerance', 10),
            in_tolerance_floor=curve.get('in_tolerance_floor', 0.7),
            out_of_tolerance_decay=curve.get('out_of_tolerance_decay', 0.02),
            min_score=curve.get('min_score', 0.0),
        )
        importance = float(crit.get('importance', 0.0) or 0.0)
        used[crit_key] = {
            'raw': round(raw, 2),
            'score': round(score, 4),
            'importance': importance,
        }
        weighted_sum += score * importance
        total_importance += importance

    total = 0.0 if total_importance == 0 else weighted_sum / total_importance
    if failures:
        total = 0.0

    return {
        'total': round(total, 4),
        'failed_constraints': failures,
        'criteria': used,
        'warnings': warnings,
    }
