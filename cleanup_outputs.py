from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from src.pipeline.io_utils import resolve_output_paths


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("cleanup_outputs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _pick_preferred(a: Path, b: Path) -> Path:
    try:
        size_a = a.stat().st_size
        size_b = b.stat().st_size
    except Exception:
        size_a = size_b = 0
    if size_a != size_b:
        return a if size_a > size_b else b
    try:
        mtime_a = a.stat().st_mtime
        mtime_b = b.stat().st_mtime
    except Exception:
        mtime_a = mtime_b = 0
    return a if mtime_a >= mtime_b else b


def _move_with_dedupe(src: Path, dst_dir: Path, logger: logging.Logger) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.move(str(src), str(dst))
        logger.info(f"Moved: {src} -> {dst}")
        return

    keep = _pick_preferred(dst, src)
    drop = src if keep == dst else dst
    if keep != dst:
        shutil.move(str(src), str(dst))
        logger.info(f"Replaced with larger/newer: {dst}")
    if drop.exists():
        drop.unlink(missing_ok=True)
        logger.info(f"Removed duplicate: {drop}")


def cleanup_outputs(base_dir: str | Path) -> None:
    paths = resolve_output_paths(base_dir)
    logger = _setup_logger(paths["logs_dir"] / "cleanup.log")

    base = paths["base_dir"]
    selected_dir = paths["selected_clips_dir"]
    metadata_dir = paths["metadata_dir"]
    logs_dir = paths["logs_dir"]

    logger.info(f"Cleaning outputs under: {base}")

    for mp4 in base.rglob("*.mp4"):
        if selected_dir in mp4.parents:
            continue
        _move_with_dedupe(mp4, selected_dir, logger)

    legacy_meta = {
        "ranking.csv": metadata_dir / "ranking.csv",
        "selected_ranking.csv": metadata_dir / "selected_ranking.csv",
        "selection_audit.json": metadata_dir / "selection_audit.json",
        "agent_meta.json": metadata_dir / "agent_meta.json",
        "bucket_stats.json": metadata_dir / "bucket_stats.json",
        "pipeline_log.txt": logs_dir / "pipeline.log",
        "pipeline.log": logs_dir / "pipeline.log",
    }
    for name, dst in legacy_meta.items():
        for src in base.rglob(name):
            if dst == src:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                keep = _pick_preferred(dst, src)
                drop = src if keep == dst else dst
                if keep != dst:
                    shutil.move(str(src), str(dst))
                    logger.info(f"Replaced with larger/newer: {dst}")
                if drop.exists():
                    drop.unlink(missing_ok=True)
                    logger.info(f"Removed duplicate: {drop}")
            else:
                shutil.move(str(src), str(dst))
                logger.info(f"Moved: {src} -> {dst}")

    for legacy_dir in [base / "public", base / "clips", base / "thumbnails", base / "artifacts", base / "cache"]:
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir, ignore_errors=True)
            logger.info(f"Removed legacy dir: {legacy_dir}")

    logger.info("Cleanup complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="outputs", help="Base output directory")
    args = parser.parse_args()
    cleanup_outputs(args.base)


if __name__ == "__main__":
    main()
