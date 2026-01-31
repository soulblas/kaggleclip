from __future__ import annotations

import csv
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import write_json
from .logging_utils import StageTimer, log_flush, log_ok, log_warn, log_err

logger = logging.getLogger("viralshort")


def ensure_outdir(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_agent_meta(out_dir: str | Path, agent: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "agent_meta.json"
    write_json(path, agent)
    return path


def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=check,
    )


def _portrait_vf_nozoom(export_w: int, export_h: int) -> str:
    return (
        f"scale={export_w}:{export_h}:force_original_aspect_ratio=decrease,"
        f"pad={export_w}:{export_h}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1"
    )


def export_clip(
    video_path: Path,
    out_path: Path,
    start: float,
    end: float,
    export_w: int,
    export_h: int,
    fps_export: int,
    audio_bitrate: str,
    ffmpeg_bin: str,
) -> subprocess.CompletedProcess:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-vf",
        _portrait_vf_nozoom(export_w, export_h),
        "-r",
        str(fps_export),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    return run_cmd(cmd, check=False)


def _ffprobe_duration(ffprobe_bin: str, path: Path) -> float:
    try:
        out = subprocess.check_output(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        ).decode("utf-8").strip()
        return float(out)
    except Exception:
        return -1.0


def _validate_clip_file(ffprobe_bin: str, path: Path, expected_dur: float, tol: float = 1.2) -> None:
    assert path.exists(), f"Missing clip file: {path}"
    size = path.stat().st_size
    assert size > 50_000, f"Clip too small / likely failed: {path} ({size} bytes)"
    dur = _ffprobe_duration(ffprobe_bin, path)
    assert dur > 0, f"ffprobe duration failed: {path}"
    assert abs(dur - expected_dur) <= tol, (
        f"Bad duration {dur:.2f}s (expected ~{expected_dur:.2f}s) for {path}"
    )


def export_thumb(
    video_path: Path, out_path: Path, t: float, export_w: int, export_h: int, ffmpeg_bin: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(video_path),
        "-vf",
        _portrait_vf_nozoom(export_w, export_h),
        "-vframes",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    run_cmd(cmd, check=True)


def run_export(state: Dict[str, Any]) -> Dict[str, Any]:
    export_w = int(state.get("EXPORT_W", 1080))
    export_h = int(state.get("EXPORT_H", 1920))
    fps_export = int(state.get("FPS_EXPORT", state.get("EXPORT_FPS", 30)))
    audio_bitrate = state.get("AUDIO_BITRATE", "160k")

    video_path = Path(state["VIDEO_PATH"])
    clips_dir = Path(state["SELECTED_CLIPS_DIR"])
    thumbs_dir = Path(state["THUMBS_DIR"])
    metadata_dir = Path(state["METADATA_DIR"])
    ffprobe_bin = state.get("FFPROBE_BIN", "ffprobe")
    ffmpeg_bin = state.get("FFMPEG_BIN", "ffmpeg")

    with StageTimer(
        12,
        f"Finalize + Export (NO ZOOM pad {export_w}x{export_h}@{fps_export}) + Thumbs + Ranking",
    ):
        assert "SELECTED" in state and isinstance(state["SELECTED"], list), "SELECTED missing"

        clips_dir.mkdir(parents=True, exist_ok=True)
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        for p in clips_dir.glob("*.mp4"):
            p.unlink()
        for p in thumbs_dir.glob("*.jpg"):
            p.unlink()

        ranking_rows = []
        export_manifest = []
        expected_files = []
        for rank, c in enumerate(state["SELECTED"], 1):
            cid = str(c.get("id"))
            st = float(c.get("start", 0.0))
            en = float(c.get("end", st))
            dur = float(c.get("duration", en - st))
            vs = float((c.get("scores", {}) or {}).get("viral_score", 0.0))

            clip_out = clips_dir / f"clip_{rank:02d}_{cid}.mp4"
            thumb_out = thumbs_dir / f"thumb_{rank:02d}_{cid}.jpg"

            proc = export_clip(
                video_path,
                clip_out,
                st,
                en,
                export_w,
                export_h,
                fps_export,
                audio_bitrate,
                ffmpeg_bin,
            )
            expected_files.append(clip_out)
            if proc.returncode != 0:
                out = (proc.stdout or "").strip()
                snippet = out[-400:] if out else ""
                log_err(
                    logger,
                    f"EXPORT_FAIL id={cid} rc={proc.returncode} stderr_snip={snippet}",
                )
                raise RuntimeError(f"ffmpeg export failed for {cid} (rc={proc.returncode})")

            _validate_clip_file(ffprobe_bin, clip_out, expected_dur=(en - st))
            size_kb = max(1, int(clip_out.stat().st_size / 1024))
            log_ok(
                logger,
                f"EXPORT clip_{rank:02d} id={cid} start={st:.2f} end={en:.2f} dur={dur:.2f} -> {clip_out} size={size_kb}KB",
            )

            t_thumb = st + min(0.5, max(0.0, dur * 0.10))
            try:
                export_thumb(video_path, thumb_out, t_thumb, export_w, export_h, ffmpeg_bin)
            except Exception as e:
                log_warn(logger, f"EXPORT_THUMB_FAIL id={cid} err={e}")

            er = c.get("editorial_reason", [])
            er = " | ".join(er) if isinstance(er, list) else str(er)

            ranking_rows.append(
                {
                    "rank": int(rank),
                    "id": cid,
                    "start": float(st),
                    "end": float(en),
                    "duration": float(dur),
                    "viral_score": float(vs),
                    "clip_path": str(clip_out),
                    "thumbnail_path": str(thumb_out),
                    "editorial_reason": er,
                }
            )
            export_manifest.append(
                {
                    "rank": int(rank),
                    "id": cid,
                    "start": float(st),
                    "end": float(en),
                    "duration": float(dur),
                    "clip_path": str(clip_out),
                    "thumbnail_path": str(thumb_out),
                    "size_bytes": int(clip_out.stat().st_size),
                    "validated_duration_sec": float(_ffprobe_duration(ffprobe_bin, clip_out)),
                }
            )

        ranking_csv = metadata_dir / "selected_ranking.csv"
        with ranking_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "rank",
                "id",
                "start",
                "end",
                "duration",
                "viral_score",
                "clip_path",
                "thumbnail_path",
                "editorial_reason",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(ranking_rows)

        manifest_path = metadata_dir / "export_manifest.json"
        write_json(manifest_path, export_manifest)

        selected_count = len(state.get("SELECTED", []))
        mp4_files = list(clips_dir.glob("*.mp4"))
        if selected_count > 0 and len(mp4_files) == 0:
            raise RuntimeError("Export failed: zero mp4 files produced")
        if len(mp4_files) < selected_count:
            missing = [p for p in expected_files if not p.exists()]
            raise RuntimeError(f"Export incomplete: missing clips: {[str(p) for p in missing]}")

        log_flush()

    return state
