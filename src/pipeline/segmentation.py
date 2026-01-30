from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .io_utils import read_json, write_json
from .logging_utils import StageTimer, log_flush

logger = logging.getLogger("viralshort")


def _speech_blocks_from_silence(
    total_dur: float, silence_segments: List[List[float]], pad: float = 0.05
) -> List[List[float]]:
    sil = sorted(
        [[max(0.0, s - pad), min(total_dur, e + pad)] for s, e in (silence_segments or [])],
        key=lambda x: x[0],
    )
    out = []
    cur = 0.0
    for s, e in sil:
        if s > cur:
            out.append([cur, s])
        cur = max(cur, e)
    if cur < total_dur:
        out.append([cur, total_dur])
    out = [[s, e] for s, e in out if (e - s) >= 0.6]
    return out


def _speech_blocks_from_energy(
    energy_curve: List[Dict[str, Any]],
    total_dur: float,
    thr_pct: int = 60,
    min_len: float = 0.8,
    max_gap: float = 0.35,
) -> List[List[float]]:
    if not energy_curve:
        return []
    vals = [float(p.get("rms", 0.0)) for p in energy_curve]
    if not vals:
        return []
    thr = float(np.percentile(vals, thr_pct))
    blocks = []
    in_speech = False
    s = None
    last_t = None
    for p in energy_curve:
        t = float(p.get("time", 0.0))
        rms = float(p.get("rms", 0.0))
        if rms >= thr:
            if not in_speech:
                in_speech = True
                s = t
            last_t = t
        else:
            if in_speech and last_t is not None:
                if (t - last_t) > max_gap:
                    e = last_t
                    if (e - s) >= min_len:
                        blocks.append([s, min(e, total_dur)])
                    in_speech = False
                    s = None
                    last_t = None
    if in_speech and s is not None and last_t is not None:
        if (last_t - s) >= min_len:
            blocks.append([s, min(last_t, total_dur)])
    return blocks


def run_segmentation(state: Dict[str, Any]) -> Dict[str, Any]:
    art_dir = Path(state["ART_DIR"])
    video_path = Path(state["VIDEO_PATH"])
    analyzed_duration = float(state["ANALYZED_DURATION"])
    ffmpeg_bin = state.get("FFMPEG_BIN", "ffmpeg")

    with StageTimer(2, "Audio Extract + Global Analysis"):
        audio_wav = Path(art_dir) / "audio.wav"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(audio_wav),
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"Audio extracted: {audio_wav}")

        sil_path = Path(art_dir) / "silence_segments.json"
        if sil_path.exists():
            silence_segments = read_json(sil_path)
            logger.info("Loaded cached silence_segments.json")
        else:
            silence_cmd = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "info",
                "-i",
                str(audio_wav),
                "-af",
                "silencedetect=noise=-35dB:d=0.35",
                "-f",
                "null",
                "-",
            ]
            p = subprocess.Popen(silence_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            _, err = p.communicate()

            if p.returncode != 0:
                logger.warning("silencedetect failed; continuing with empty silence list")
                silence_segments = []
            else:
                starts, ends = [], []
                for line in (err or "").splitlines():
                    if "silence_start" in line:
                        m = re.search(r"silence_start: ([0-9\\.]+)", line)
                        if m:
                            starts.append(float(m.group(1)))
                    if "silence_end" in line:
                        m = re.search(r"silence_end: ([0-9\\.]+)", line)
                        if m:
                            ends.append(float(m.group(1)))

                silence_segments = []
                j = 0
                for s in starts:
                    while j < len(ends) and ends[j] < s:
                        j += 1
                    if j < len(ends):
                        silence_segments.append([s, ends[j]])
                        j += 1

            write_json(sil_path, silence_segments)
            logger.info("Wrote silence_segments.json")

        logger.info(f"Silence segments: {len(silence_segments)}")

        energy_curve = []
        try:
            import soundfile as sf

            y, sr = sf.read(str(audio_wav))
        except Exception as e:
            logger.warning(f"soundfile read failed; falling back to scipy.io.wavfile ({e})")
            from scipy.io import wavfile

            sr, y = wavfile.read(str(audio_wav))
            if hasattr(y, "dtype") and y.dtype != np.float32:
                if np.issubdtype(y.dtype, np.integer):
                    y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
                else:
                    y = y.astype(np.float32)

        if y.ndim > 1:
            y = y.mean(axis=1)

        hop = int(0.02 * sr)
        win = int(0.04 * sr)

        for i in range(0, len(y) - win, hop):
            frame = y[i : i + win]
            val = float(np.sqrt(np.mean(frame * frame)) + 1e-12)
            t = float(i / sr)
            energy_curve.append({"time": t, "rms": val})

        write_json(Path(art_dir) / "energy_curve.json", energy_curve)
        logger.info(f"Energy points: {len(energy_curve)} (saved to energy_curve.json)")

    with StageTimer(2, "Speech Blocks (silence complement + VAD fallback)"):
        total = float(analyzed_duration)
        speech_blocks = _speech_blocks_from_silence(total, silence_segments, pad=0.05)
        if not speech_blocks:
            logger.warning("No speech blocks from silence; using energy VAD fallback")
            speech_blocks = _speech_blocks_from_energy(energy_curve, total)

        write_json(Path(art_dir) / "speech_blocks.json", speech_blocks)
        logger.info(f"Speech blocks: {len(speech_blocks)}")

        if not speech_blocks:
            raise RuntimeError("SPEECH_BLOCKS missing/empty without fallback")

    state["AUDIO_WAV"] = audio_wav
    state["SILENCE_SEGMENTS"] = silence_segments
    state["ENERGY_CURVE"] = energy_curve
    state["SPEECH_BLOCKS"] = speech_blocks

    log_flush()
    return state
