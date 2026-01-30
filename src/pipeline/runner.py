from __future__ import annotations

import json
import logging
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .agent_contract import load_agent_contract
from .asr_light import run_asr
from .candidates import run_candidate_mining
from .export import ensure_outdir, run_export, write_agent_meta
from .features import run_feature_extraction
from .io_utils import write_json, read_json, resolve_output_paths
from .logging_utils import StageTimer, log_flush, setup_pipeline_logger
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
    out_dir_path = ensure_outdir(out_dir)
    output_paths = resolve_output_paths(out_dir_path)

    agent = load_agent_contract("AGENTS.md")
    write_agent_meta(output_paths["metadata_dir"], agent)

    log_file = output_paths["logs_dir"] / "pipeline.log"
    setup_pipeline_logger(log_file)

    state: Dict[str, Any] = {}

    state["LOCK_NO_FACE"] = True
    state["LOCK_NO_CROP_ZOOM"] = True
    state["LOCK_NO_SUBTITLES"] = True

    state["EXPORT_W"] = 1080
    state["EXPORT_H"] = 1920
    state["EXPORT_FPS"] = 30
    state["AUDIO_BITRATE"] = "160k"
    state["EXPORT_MODE"] = "FIT_PAD_BLACK"

    state["MIN_CLIP_SEC"] = 18.0
    state["MAX_CLIP_SEC"] = 60.0
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

    state["FFMPEG_BIN"] = "ffmpeg"
    state["FFPROBE_BIN"] = "ffprobe"

    state["RUN_TIMESTAMP"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
    state["START_EXPAND_MIN"] = float(kwargs.get("start_expand_min", 3.0))
    state["START_EXPAND_MAX"] = float(kwargs.get("start_expand_max", 8.0))
    state["START_SPEECH_WINDOW"] = float(kwargs.get("start_speech_window", 0.6))
    state["SNAP_WORD_RADIUS"] = float(kwargs.get("snap_word_radius", 1.8))
    state["SNAP_SILENCE_RADIUS"] = float(kwargs.get("snap_silence_radius", 1.2))

    # resolve_output_paths already ensures directory creation

    logger.info(f"OUT_DIR: {out_dir_path}")
    logger.info(
        "Hard Locks: NO_FACE=%s NO_CROP_ZOOM=%s NO_SUBTITLES=%s",
        state["LOCK_NO_FACE"],
        state["LOCK_NO_CROP_ZOOM"],
        state["LOCK_NO_SUBTITLES"],
    )
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

        write_json(
            Path(state["RAW_SEGMENTS_DIR"]) / "video_meta.json",
            {
                "video_path": str(video_path),
                "duration_sec": duration,
                "run_timestamp_utc": state["RUN_TIMESTAMP"],
            },
        )

    run_segmentation(state)

    with StageTimer(3, "Thumbnails Sampling (No AI)"):
        random.seed(1337)
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

    with StageTimer(4, "Shot Detection (Optional)"):
        shot_cuts = []
        shot_path = Path(state["RAW_SEGMENTS_DIR"]) / "shot_cuts.json"
        if shot_path.exists():
            shot_cuts = read_json(shot_path)
            logger.info("Loaded cached shot cuts")
        else:
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
                logger.info(f"Shot cuts: {len(shot_cuts)}")
            except Exception as e:
                logger.warning(f"Shot detection unavailable; continuing without. ({e})")
                shot_cuts = []
        state["SHOT_CUTS"] = shot_cuts

    with StageTimer(5, "Hard Lock: NO FACE / NO CROP / NO ZOOM"):
        logger.info("This pipeline does not perform face detection/tracking/crop/zoom. Export uses FIT+PAD only.")
        logger.info("This pipeline does not generate SRT/subtitles or burn-in captions.")

    run_candidate_mining(state)
    run_feature_extraction(state)
    run_asr(state)
    run_scoring(state)
    run_selection(state)
    snap_selected(state)
    run_export(state)

    return {"status": "ok", "out_dir": str(out_dir_path), "agent": agent}
