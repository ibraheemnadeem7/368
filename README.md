# Wall Designer Starter

A small Python development environment for designing **one wall at a time** using the same general data shapes as the Gallery YAML Manager.

This starter is designed to help students:
- read the gallery, art, scoring, and show files
- generate placements for a single wall
- score those placements
- write the result back into `show.arrangements`
- upload the show file into the current React visualizer

## Current design choices

- **Art format:** same schema as the React system (`schema_version: 2.0`, `art.artworks`)
- **Gallery format:** same schema as the React system (`gallery.rooms[].walls[]`)
- **Scoring format:** same general `scoring:` schema as the server, but with a simpler default profile
- **Show format:** same visualizer-compatible `show.arrangements` format, extended with optional `show.wall_designs`

## What this starter supports now

- single-wall generation
- a greedy baseline algorithm
- a wall-focused subset of the scoring schema
- writing/updating one arrangement in `show.yaml`

## What it does not try to do yet

- full multi-wall optimization
- room-level or gallery-level scoring
- running Python inside the React app
- editing all scoring methods from the schema

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m wall_designer.main   --gallery data/gallery.yaml   --art data/art.yaml   --scoring data/scoring_basic.yaml   --show data/show.yaml   --wall R2-N   --algorithm student_algorithms/wall_greedy_v1.py
```

This uses the candidate artwork ids already stored under `show.wall_designs` for the selected wall.

You can also override candidates explicitly:

```bash
python -m wall_designer.main   --gallery data/gallery.yaml   --art data/art.yaml   --scoring data/scoring_basic.yaml   --show data/show.yaml   --wall R2-N   --candidates A1,A5,A8,A26,A9   --algorithm student_algorithms/wall_greedy_v1.py
```

## Recommended student workflow

1. Start with the provided greedy algorithm.
2. Run the Python tool locally.
3. Inspect the updated `data/show.yaml`.
4. Upload the show into the current React visualizer.
5. Improve the algorithm and repeat.

## Important note about the art file

This starter includes a **real-schema starter subset** of the art catalog so the project runs immediately.
The code is written to support the full server art catalog directly. To switch to the full catalog, replace `data/art.yaml` with your exported app art file.

## File overview

- `data/gallery.yaml` — current gallery schema
- `data/art.yaml` — starter subset of the real art schema
- `data/scoring_basic.yaml` — slimmed-down default scoring profile
- `data/show.yaml` — visualizer-compatible show file with `wall_designs`
- `student_algorithms/wall_greedy_v1.py` — baseline algorithm students can modify
- `wall_designer/main.py` — command-line runner
- `wall_designer/scorer.py` — evaluates arrangements using a subset of the general scoring schema
- `wall_designer/show_io.py` — writes/upserts arrangements into the show file

## Supported scoring pieces

### Hard constraints
- `min_artworks`
- `max_artworks`
- `no_overlap`
- `stay_within_wall`
- `min_gap_ft`
- `max_gap_ft`
- `respect_locked_positions`
- `require_eligible`

### Criteria methods currently implemented
- `gap_variance_vs_ideal`
- `left_right_visual_mass_balance`
- `occupied_width_ratio`
- `adjacent_theme_similarity_average`
- `featured_work_near_preferred_zone`

Unsupported criteria are skipped with a warning.
