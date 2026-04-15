from .loader import save_yaml


def _next_arrangement_id(show_data, wall_id):
    prefix = f'ARR_{wall_id.replace("-", "")}_'
    existing = show_data.get('show', {}).get('arrangements', [])
    nums = []
    for arr in existing:
        arr_id = arr.get('id', '')
        if arr_id.startswith(prefix):
            try:
                nums.append(int(arr_id.split('_')[-1]))
            except Exception:
                pass
    next_num = max(nums, default=0) + 1
    return f'{prefix}{next_num:03d}'


def upsert_arrangement(show_data, wall_id, placements, score, title=None, status='generated', notes=''):
    show = show_data.setdefault('show', {})
    arrangements = show.setdefault('arrangements', [])
    new_arr = {
        'id': _next_arrangement_id(show_data, wall_id),
        'title': title or f'{wall_id} Generated Arrangement',
        'space_id': wall_id,
        'status': status,
        'notes': notes,
        'score': score,
        'placements': placements,
    }

    # replace existing generated arrangement for this wall if one exists, otherwise append
    for i, arr in enumerate(arrangements):
        if arr.get('space_id') == wall_id and arr.get('status') == 'generated':
            new_arr['id'] = arr.get('id', new_arr['id'])
            arrangements[i] = new_arr
            return show_data

    arrangements.append(new_arr)
    return show_data


def save_show(path, show_data):
    save_yaml(path, show_data)
