from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .agent_contract import load_agent_contract
from .asr_light import run_asr
from .candidates import run_candidate_mining, run_context_expansion, add_marker_candidates
from .export import ensure_outdir, run_export, write_agent_meta
from .features import run_feature_extraction
from .io_utils import write_json, load_cached_json, write_cache_meta, resolve_output_paths, stable_hash
from .logging_utils import (
    StageTimer,
    log_flush,
    setup_pipeline_logger,
    log_i,
    log_ok,
    log_warn,
    log_warning_panel,
)
from .scoring import run_scoring
from .segmentation import run_segmentation
from .selection import run_selection, snap_selected

logger = logging.getLogger("viralshort")


def _resolve_input_video(input_video: str | None) -> Path:
    if input_video:
        p = Path(input_video)
        if p.exists():
            if p.is_dir():
                exts = ("*.mp4", "*.mkv", "*.mov", "*.webm", "*.m4v")
                candidates = []
                for ext in exts:
                    candidates += list(p.rglob(ext))
                if candidates:
                    return sorted(candidates)[0]
            else:
                return p

    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates = []
        for ext in ("*.mp4", "*.mkv", "*.mov", "*.webm", "*.m4v"):
            candidates += list(input_root.rglob(ext))
        if candidates:
            return sorted(candidates)[0]

    raise RuntimeError("No input video found. Provide input_video or ensure /kaggle/input has a video.")


def run_pipeline(input_video: str, out_dir: str = "outputs", **kwargs):
    state: Dict[str, Any] = {}

    seed = int(kwargs.get("seed", os.getenv("PIPELINE_SEED", "1337")))
    random.seed(seed)
    try:
        import numpy as _np  # type: ignore

        _np.random.seed(seed)
    except Exception:
        pass

    def _truthy(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if val is None:
            return False
        return str(val).strip().lower() not in ("0", "false", "no", "off")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_suffix = f"{os.getpid() % 10000:04d}"
    run_id = f"{run_ts}_{run_suffix}"

    out_dir_path = ensure_outdir(out_dir)
    run_scoped = _truthy(kwargs.get("run_scoped", os.getenv("RUN_SCOPED_OUTPUT", "1")))
    run_dir = Path(out_dir_path) / "runs" / run_id if run_scoped else Path(out_dir_path)
    output_paths = resolve_output_paths(run_dir)

    agent = load_agent_contract("AGENTS.md")
    write_agent_meta(output_paths["metadata_dir"], agent)

    log_file = output_paths["logs_dir"] / "pipeline.log"
    setup_pipeline_logger(log_file)

    state["LOCK_NO_FACE"] = True
    state["LOCK_NO_CROP_ZOOM"] = True
    state["LOCK_NO_SUBTITLES"] = True

    state["EXPORT_W"] = 1080
    state["EXPORT_H"] = 1920
    state["EXPORT_FPS"] = 30
    state["AUDIO_BITRATE"] = "160k"
    state["EXPORT_MODE"] = "FIT_PAD_BLACK"

    state["MIN_CLIP_SEC"] = 15.0
    state["MAX_CLIP_SEC"] = 120.0
    state["HOOK_WINDOW_SEC"] = 5.0

    state["MAX_FINAL_CLIPS"] = 6
    state["MIN_GAP_SEC"] = 30.0
    state["SEGMENT_DURATION_SEC"] = 600.0
    state["MAX_PER_SEGMENT"] = 2

    state["ASR_LANGUAGE"] = "id"
    state["ASR_ENABLED"] = True
    state["ASR_TOP_PERCENT"] = 0.40
    state["ASR_TOPN_PER_BUCKET"] = int(kwargs.get("asr_topn_per_bucket", 8))
    state["ASR_MODEL_NAME"] = kwargs.get("asr_model_name", "tiny")
    state["ASR_BEAM_SIZE"] = int(kwargs.get("asr_beam_size", 1))
    state["ASR_LIGHT_MODE"] = bool(kwargs.get("asr_light_mode", True))
    state["ASR_LIGHT_SEC"] = float(kwargs.get("asr_light_sec", 12.0))
    state["ASR_LIGHT_OFFSET"] = float(kwargs.get("asr_light_offset", 2.0))
    state["MAX_ASR_BLOCK_SEC"] = 28.0
    state["MAX_ASR_BLOCK_WALL_SEC"] = 45.0
    state["ASR_BLOCK_OVERLAP_SEC"] = 0.25

    state["TRIGGER_WORDS"] = [
        "anjir",
        "anjay",
        "gila",
        "serius",
        "beneran",
        "parah",
        "lucuu",
        "ngakak",
        "ketawa",
        "kok",
        "loh",
        "hah",
        "apaan",
        "buset",
        "astaga",
        "waduh",
        "wkwk",
        "wkwkwk",
        "yaampun",
        "kaget",
        "plot",
        "twist",
        "tapi",
        "ternyata",
        "eh",
        "coba",
        "sumpah",
    ]

    state["FFMPEG_BIN"] = kwargs.get("ffmpeg_bin", os.getenv("FFMPEG_BIN", "ffmpeg"))
    state["FFPROBE_BIN"] = kwargs.get("ffprobe_bin", os.getenv("FFPROBE_BIN", "ffprobe"))

    state["RUN_TIMESTAMP"] = run_ts
    state["RUN_ID"] = run_id
    state["RUN_SCOPED_OUTPUT"] = run_scoped
    state["BASE_OUT_DIR"] = str(out_dir_path)
    state["SEED"] = seed
    state["TIME_BUCKETS"] = 5
    state["OUTPUT_PATHS"] = output_paths
    state["OUT_DIR"] = output_paths["base_dir"]
    state["RAW_SEGMENTS_DIR"] = output_paths["raw_segments_dir"]
    state["SCORED_SEGMENTS_DIR"] = output_paths["scored_segments_dir"]
    state["SELECTED_CLIPS_DIR"] = output_paths["selected_clips_dir"]
    state["THUMBS_DIR"] = output_paths["thumbnails_dir"]
    state["METADATA_DIR"] = output_paths["metadata_dir"]
    state["LOG_DIR"] = output_paths["logs_dir"]
    state["CACHE_DIR"] = output_paths["raw_segments_dir"] / "_cache"
    state["CACHE_DIR"].mkdir(parents=True, exist_ok=True)

    state["TAIL_EXT_SEC"] = float(kwargs.get("tail_ext_sec", 1.6))
    state["TAIL_MAX_SEC"] = float(kwargs.get("tail_max_sec", 3.0))
    state["LEAD_SILENCE_TRIM_SEC"] = float(kwargs.get("lead_silence_trim_sec", 0.8))
    state["START_EXPAND_MIN"] = float(kwargs.get("start_expand_min", 2.0))
    state["START_EXPAND_MAX"] = float(kwargs.get("start_expand_max", 6.0))
    state["START_SPEECH_WINDOW"] = float(kwargs.get("start_speech_window", 0.6))
    state["SNAP_WORD_RADIUS"] = float(kwargs.get("snap_word_radius", 1.8))
    state["SNAP_SILENCE_RADIUS"] = float(kwargs.get("snap_silence_radius", 1.2))

    env_keys = [
        "RUN_SCOPED_OUTPUT",
        "USE_FLEXIBLE_DUR",
        "MIN_DUR_SEC",
        "MAX_DUR_SEC",
        "IDEAL_DUR_MIN",
        "IDEAL_DUR_MAX",
        "DUR_SHORT_MIN",
        "DUR_SHORT_MAX",
        "DUR_MID_MIN",
        "DUR_MID_MAX",
        "DUR_LONG_MIN",
        "DUR_LONG_MAX",
        "END_LOOKBACK",
        "END_LOOKAHEAD",
        "ASR_TOPN_PER_BUCKET",
        "ASR_MODEL_NAME",
        "ASR_BEAM_SIZE",
        "ASR_LIGHT_MODE",
        "ASR_LIGHT_SEC",
        "ASR_LIGHT_OFFSET",
        "FFMPEG_BIN",
        "FFPROBE_BIN",
    ]
    env_overrides = {k: os.getenv(k) for k in env_keys if os.getenv(k) is not None}
    state["ENV_OVERRIDES"] = env_overrides

    config_snapshot = {
        "EXPORT_W": state["EXPORT_W"],
        "EXPORT_H": state["EXPORT_H"],
        "EXPORT_FPS": state["EXPORT_FPS"],
        "AUDIO_BITRATE": state["AUDIO_BITRATE"],
        "EXPORT_MODE": state["EXPORT_MODE"],
        "RUN_SCOPED_OUTPUT": state["RUN_SCOPED_OUTPUT"],
        "FFMPEG_BIN": state["FFMPEG_BIN"],
        "FFPROBE_BIN": state["FFPROBE_BIN"],
        "MIN_CLIP_SEC": state["MIN_CLIP_SEC"],
        "MAX_CLIP_SEC": state["MAX_CLIP_SEC"],
        "HOOK_WINDOW_SEC": state["HOOK_WINDOW_SEC"],
        "MAX_FINAL_CLIPS": state["MAX_FINAL_CLIPS"],
        "MIN_GAP_SEC": state["MIN_GAP_SEC"],
        "TIME_BUCKETS": state["TIME_BUCKETS"],
        "ASR_LANGUAGE": state["ASR_LANGUAGE"],
        "ASR_ENABLED": state["ASR_ENABLED"],
        "ASR_TOPN_PER_BUCKET": state["ASR_TOPN_PER_BUCKET"],
        "ASR_MODEL_NAME": state["ASR_MODEL_NAME"],
        "ASR_BEAM_SIZE": state["ASR_BEAM_SIZE"],
        "ASR_LIGHT_MODE": state["ASR_LIGHT_MODE"],
        "ASR_LIGHT_SEC": state["ASR_LIGHT_SEC"],
        "ASR_LIGHT_OFFSET": state["ASR_LIGHT_OFFSET"],
        "MAX_ASR_BLOCK_SEC": state["MAX_ASR_BLOCK_SEC"],
        "ASR_BLOCK_OVERLAP_SEC": state["ASR_BLOCK_OVERLAP_SEC"],
        "TAIL_EXT_SEC": state["TAIL_EXT_SEC"],
        "TAIL_MAX_SEC": state["TAIL_MAX_SEC"],
        "LEAD_SILENCE_TRIM_SEC": state["LEAD_SILENCE_TRIM_SEC"],
        "START_EXPAND_MIN": state["START_EXPAND_MIN"],
        "START_EXPAND_MAX": state["START_EXPAND_MAX"],
        "START_SPEECH_WINDOW": state["START_SPEECH_WINDOW"],
        "SNAP_WORD_RADIUS": state["SNAP_WORD_RADIUS"],
        "SNAP_SILENCE_RADIUS": state["SNAP_SILENCE_RADIUS"],
        "TRIGGER_WORDS": state["TRIGGER_WORDS"],
    }
    config_hash = stable_hash({"config": config_snapshot, "env": env_overrides})
    state["CONFIG_HASH"] = config_hash

    run_manifest = {
        "run_id": run_id,
        "run_timestamp_utc": run_ts,
        "status": "started",
        "seed": seed,
        "config_hash": config_hash,
        "config_snapshot": config_snapshot,
        "env_overrides": env_overrides,
        "agent": agent,
        "run_scoped_output": run_scoped,
        "base_out_dir": str(out_dir_path),
        "output_paths": {k: str(v) for k, v in output_paths.items()},
    }
    state["RUN_MANIFEST"] = run_manifest
    state["RUN_MANIFEST_PATH"] = Path(state["METADATA_DIR"]) / "run_manifest.json"
    state["RUN_COMPLETE_PATH"] = Path(state["METADATA_DIR"]) / "run_complete.json"

    with StageTimer(0, "Initialization"):
        def _clear_dir_contents(path: Path) -> None:
            if not path.exists():
                return
            for child in path.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except Exception:
                    pass

        def _bin_version(bin_name: str) -> str:
            try:
                out = subprocess.check_output([bin_name, "-version"], stderr=subprocess.STDOUT).decode("utf-8")
                return out.splitlines()[0].strip()
            except Exception as e:
                return f"unavailable({e})"

        def _git_sha() -> str:
            try:
                out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(Path.cwd()))
                return out.decode("utf-8").strip()
            except Exception:
                return "unknown"

        versions = {
            "python": sys.version.split()[0],
            "ffmpeg": _bin_version(state["FFMPEG_BIN"]),
            "ffprobe": _bin_version(state["FFPROBE_BIN"]),
            "git_sha": _git_sha(),
            "asr_model": state["ASR_MODEL_NAME"],
        }
        state["VERSIONS"] = versions
        run_manifest["versions"] = versions

        write_json(state["RUN_MANIFEST_PATH"], run_manifest)
        log_i(logger, f"RUN_ID: {run_id}")
        log_i(logger, f"OUT_DIR: {run_dir}")
        log_i(logger, f"CONFIG_HASH: {config_hash[:12]}")
        log_i(logger, f"SEED: {seed}")
        if env_overrides:
            log_i(logger, f"ENV_OVERRIDES: {env_overrides}")
        log_i(logger, f"VERSIONS: {versions}")
        log_i(logger, f"LOG_FILE: {log_file}")
        if not run_scoped:
            log_warn(logger, "RUN_SCOPED_OUTPUT=0: clearing base output stage dirs to avoid mixed runs")
            for p in (
                output_paths["raw_segments_dir"],
                output_paths["scored_segments_dir"],
                output_paths["selected_clips_dir"],
                output_paths["thumbnails_dir"],
                output_paths["metadata_dir"],
            ):
                _clear_dir_contents(p)
        log_flush()

    with StageTimer(1, "Ingest Video"):
        video_path = _resolve_input_video(input_video)
        state["VIDEO_PATH"] = video_path
        logger.info(f"Selected video: {video_path}")

        cmd = [
            state["FFPROBE_BIN"],
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(video_path),
        ]
        meta = json.loads(subprocess.check_output(cmd).decode("utf-8"))
        duration = float(meta["format"]["duration"])
        state["ANALYZED_DURATION"] = duration
        logger.info(f"Duration: {duration:.2f}s")

        try:
            st = video_path.stat()
            video_fingerprint = {
                "basename": video_path.name,
                "size_bytes": int(st.st_size),
                "mtime": float(st.st_mtime),
                "duration_sec": float(duration),
            }
        except Exception:
            video_fingerprint = {
                "basename": video_path.name,
                "size_bytes": -1,
                "mtime": 0.0,
                "duration_sec": float(duration),
            }
        video_hash = stable_hash(video_fingerprint)
        state["VIDEO_FINGERPRINT"] = video_fingerprint
        state["VIDEO_HASH"] = video_hash
        state["CACHE_META"] = {"video_fingerprint": video_fingerprint, "config_hash": config_hash}
        state["RUN_MANIFEST"]["video_fingerprint"] = video_fingerprint
        state["RUN_MANIFEST"]["video_hash"] = video_hash
        state["RUN_MANIFEST"]["analyzed_duration_sec"] = float(duration)
        write_json(state["RUN_MANIFEST_PATH"], state["RUN_MANIFEST"])

        write_json(
            Path(state["RAW_SEGMENTS_DIR"]) / "video_meta.json",
            {
                "video_path": str(video_path),
                "duration_sec": duration,
                "run_timestamp_utc": state["RUN_TIMESTAMP"],
                "run_id": state["RUN_ID"],
            },
        )

    run_segmentation(state)

    with StageTimer(4, "Visual Sampling + Shot Detection"):
        random.seed(state.get("SEED", 1337))
        duration = float(state["ANALYZED_DURATION"])
        n_thumbs = min(24, max(8, int(duration // 30)))
        ts = sorted({min(duration - 0.1, (i + 1) * duration / (n_thumbs + 1)) for i in range(n_thumbs)})
        thumb_paths = []
        for i, t in enumerate(ts, 1):
            outp = Path(state["THUMBS_DIR"]) / f"sample_{i:02d}_{t:.2f}.jpg"
            cmd = [
                state["FFMPEG_BIN"],
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(t),
                "-i",
                str(state["VIDEO_PATH"]),
                "-vframes",
                "1",
                "-q:v",
                "2",
                str(outp),
            ]
            subprocess.run(cmd, check=True)
            thumb_paths.append(str(outp))
        write_json(
            Path(state["RAW_SEGMENTS_DIR"]) / "thumbnail_samples.json",
            {"timestamps": ts, "paths": thumb_paths},
        )
        logger.info(f"Sample thumbnails: {len(thumb_paths)}")

        shot_cuts = []
        shot_path = Path(state["RAW_SEGMENTS_DIR"]) / "shot_cuts.json"
        cache_meta = state.get("CACHE_META") or {}
        cached, hit, reason = load_cached_json(shot_path, cache_meta)
        if hit and isinstance(cached, list):
            shot_cuts = cached
            log_i(logger, "CACHE_HIT shot_cuts")
        else:
            if reason != "missing":
                log_warn(logger, f"CACHE_MISS shot_cuts ({reason}) -> recompute")
            try:
                from scenedetect import SceneManager, open_video  # type: ignore[import-not-found]
                from scenedetect.detectors import ContentDetector  # type: ignore[import-not-found]

                video = open_video(str(state["VIDEO_PATH"]))
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(threshold=27.0))
                scene_manager.detect_scenes(video, show_progress=False)
                scene_list = scene_manager.get_scene_list()
                for (start, end) in scene_list[1:]:
                    shot_cuts.append(float(start.get_seconds()))
                write_json(shot_path, shot_cuts)
                if cache_meta:
                    write_cache_meta(shot_path, cache_meta)
                logger.info(f"Shot cuts: {len(shot_cuts)}")
            except Exception as e:
                logger.warning(f"Shot detection unavailable; continuing without. ({e})")
                shot_cuts = []
        state["SHOT_CUTS"] = shot_cuts

    with StageTimer(5, "Hard Lock: NO FACE / NO CROP / NO ZOOM"):
        logger.info("This pipeline does not perform face detection/tracking/crop/zoom. Export uses FIT+PAD only.")
        logger.info("This pipeline does not generate SRT/subtitles or burn-in captions.")

    run_candidate_mining(state)
    run_context_expansion(state)
    run_feature_extraction(state)
    run_asr(state)
    add_marker_candidates(state)
    run_scoring(state)
    run_selection(state)
    snap_selected(state)
    run_export(state)

    log_warning_panel(logger)

    run_manifest = state.get("RUN_MANIFEST", {})
    run_manifest["status"] = "complete"
    run_manifest["selected_count"] = len(state.get("SELECTED", []))
    write_json(state["RUN_MANIFEST_PATH"], run_manifest)
    write_json(
        state["RUN_COMPLETE_PATH"],
        {
            "run_id": state.get("RUN_ID"),
            "status": "ok",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_count": len(state.get("SELECTED", [])),
        },
    )
    log_ok(logger, f"Artifacts: {state['RUN_MANIFEST_PATH']}")
    log_ok(logger, f"Artifacts: {Path(state['METADATA_DIR']) / 'selection_audit.json'}")
    log_ok(logger, f"Artifacts: {Path(state['METADATA_DIR']) / 'selected.json'}")
    log_ok(logger, f"Artifacts: {Path(state['METADATA_DIR']) / 'ranking.csv'}")
    log_ok(logger, f"Artifacts: {Path(state['METADATA_DIR']) / 'export_manifest.json'}")

    return {"status": "ok", "out_dir": str(run_dir), "agent": agent}
