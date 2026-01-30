from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_output_paths(base_dir: str | Path) -> Dict[str, Path]:
    base = Path(base_dir)
    paths = {
        "base_dir": base,
        "raw_segments_dir": base / "00_raw_segments",
        "scored_segments_dir": base / "01_scored_segments",
        "selected_clips_dir": base / "02_selected_clips",
        "thumbnails_dir": base / "03_thumbnails",
        "metadata_dir": base / "04_metadata",
        "logs_dir": base / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
