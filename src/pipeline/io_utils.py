from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
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


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_meta_path(path: Path) -> Path:
    return Path(str(path) + ".meta.json")


def write_cache_meta(path: Path, meta: Dict[str, Any]) -> None:
    meta = dict(meta)
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    write_json(_cache_meta_path(path), meta)


def load_cached_json(path: Path, expected_meta: Dict[str, Any] | None) -> tuple[Any | None, bool, str]:
    path = Path(path)
    if not path.exists():
        return None, False, "missing"
    meta_path = _cache_meta_path(path)
    if not meta_path.exists():
        return None, False, "meta_missing"
    try:
        meta = read_json(meta_path) or {}
    except Exception:
        return None, False, "meta_unreadable"
    if expected_meta:
        for k in ("video_fingerprint", "config_hash"):
            if meta.get(k) != expected_meta.get(k):
                return None, False, f"meta_mismatch:{k}"
    try:
        return read_json(path), True, "hit"
    except Exception:
        return None, False, "cache_unreadable"


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
