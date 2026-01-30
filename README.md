KAGGLECLIP
==========

Output structure (final)
------------------------
All pipeline outputs are written under a single base directory (default: `outputs/`).

```
outputs/
  00_raw_segments/
  01_scored_segments/
  02_selected_clips/      # final clips only (no duplicates)
  03_thumbnails/
  04_metadata/
    ranking.csv
    selected_ranking.csv
    selection_audit.json
    agent_meta.json
  logs/
    pipeline.log
```

Run the pipeline
----------------
Example:

```
python -c "from src.pipeline import run_pipeline; run_pipeline('path/to/video.mp4', out_dir='outputs')"
```

Cleanup / migration (legacy outputs)
------------------------------------
If you have old runs with duplicate folders (e.g. `public/` or `clips/`),
use the cleanup script to move the best copies into the new structure:

```
python cleanup_outputs.py --base outputs
```

The cleanup script will:
- move all `.mp4` into `outputs/02_selected_clips/` (dedupe by size/newest)
- move legacy metadata files into `outputs/04_metadata/`
- remove legacy folders (`public/`, `clips/`, `thumbnails/`, `artifacts/`, `cache/`)
