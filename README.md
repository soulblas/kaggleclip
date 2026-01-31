KAGGLECLIP
==========

Output structure (run-scoped by default)
---------------------------------------
Default outputs are written per run:

```
outputs/
  runs/
    <RUN_ID>/
      00_raw_segments/
      01_scored_segments/
      02_selected_clips/
      03_thumbnails/
      04_metadata/
        ranking.csv
        selected_ranking.csv
        selection_audit.json
        bucket_stats.json
        agent_meta.json
        export_manifest.json
        run_manifest.json
        run_complete.json
      logs/
        pipeline.log
```

If you want a single base output directory (no run subfolder), set `RUN_SCOPED_OUTPUT=0` or pass `run_scoped=False`.

Run the pipeline
----------------
Example:

```
python -c "from src.pipeline import run_pipeline; run_pipeline('path/to/video.mp4', out_dir='outputs')"
```

Base output (no run subfolder):

```
python -c "from src.pipeline import run_pipeline; run_pipeline('path/to/video.mp4', out_dir='outputs', run_scoped=False)"
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
- remove legacy folders (`public/`, `clips/`, `thumbnails/`, `artifacts/`, `cache`)
