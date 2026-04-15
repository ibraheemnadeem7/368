import argparse
import importlib.util
import json

from .loader import load_yaml, get_wall, get_candidate_artworks, get_wall_design
from .scorer import evaluate
from .show_io import upsert_arrangement, save_show


def load_algorithm(path):
    spec = importlib.util.spec_from_file_location('student_algorithm', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'generate'):
        raise ValueError('Algorithm file must define generate(wall, artworks, scoring_data)')
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery', required=True)
    parser.add_argument('--art', required=True)
    parser.add_argument('--scoring', required=True)
    parser.add_argument('--show', required=True)
    parser.add_argument('--wall', required=True)
    parser.add_argument('--algorithm', required=True)
    parser.add_argument('--candidates', default='')
    args = parser.parse_args()

    gallery_data = load_yaml(args.gallery)
    art_data = load_yaml(args.art)
    scoring_data = load_yaml(args.scoring)
    show_data = load_yaml(args.show)

    wall = get_wall(gallery_data, args.wall)
    candidate_ids = []
    if args.candidates.strip():
        candidate_ids = [x.strip() for x in args.candidates.split(',') if x.strip()]
    else:
        design = get_wall_design(show_data, args.wall)
        if design:
            candidate_ids = list(design.get('candidate_artwork_ids', []))

    if not candidate_ids:
        raise ValueError(f'No candidate artworks supplied for wall {args.wall}')

    loaded = get_candidate_artworks(art_data, candidate_ids, scoring_data=scoring_data)
    artworks = loaded['artworks']

    algo = load_algorithm(args.algorithm)
    placements = algo.generate(wall, artworks, scoring_data)
    score = evaluate(wall, placements, artworks, scoring_data)

    title = f'{args.wall} Generated Arrangement'
    show_data = upsert_arrangement(show_data, args.wall, placements, score, title=title)
    save_show(args.show, show_data)

    result = {
        'wall_id': args.wall,
        'candidate_ids': candidate_ids,
        'selected_artworks': [a['id'] for a in artworks],
        'missing_ids': loaded['missing_ids'],
        'skipped': loaded['skipped'],
        'duplicate_ids_in_catalog': loaded['duplicate_ids_in_catalog'],
        'placements': placements,
        'score': score,
        'show_written_to': args.show,
    }

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
