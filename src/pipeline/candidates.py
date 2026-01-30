from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .io_utils import write_json
from .logging_utils import StageTimer, log_flush

logger = logging.getLogger("viralshort")


def complement_intervals(
    total_end: float, silence_segments: List[List[float]], pad: float = 0.0
) -> List[List[float]]:
    sil = sorted(
        [[max(0.0, s - pad), min(total_end, e + pad)] for s, e in (silence_segments or [])],
        key=lambda x: x[0],
    )
    out = []
    cur = 0.0
    for s, e in sil:
        if s > cur:
            out.append([cur, s])
        cur = max(cur, e)
    if cur < total_end:
        out.append([cur, total_end])
    return out


def get_peak_time(energy_curve, start, end):
    pts = [p for p in energy_curve if start <= p["time"] <= end]
    if not pts:
        return None, None
    m = max(pts, key=lambda x: x["rms"])
    return float(m["time"]), float(m["rms"])


def _best_silence_start_in_window(
    lo: float, hi: float, target: float, silence_segments: List[List[float]]
):
    best = None
    best_d = None
    for s, e in (silence_segments or []):
        s = float(s)
        if lo <= s <= hi:
            d = abs(s - target)
            if best_d is None or d < best_d:
                best_d = d
                best = s
    return best


def _best_block_end_in_window(
    lo: float, hi: float, target: float, speech_blocks: List[List[float]]
):
    best = None
    best_d = None
    for s, e in (speech_blocks or []):
        e = float(e)
        if lo <= e <= hi:
            d = abs(e - target)
            if best_d is None or d < best_d:
                best_d = d
                best = e
    return best


def pick_flexible_end(
    start: float,
    hard_end: float,
    silence_segments: List[List[float]],
    speech_blocks: List[List[float]],
    min_dur_sec: float,
    ideal_dur_min: float,
    ideal_dur_max: float,
    end_lookback: float,
    end_lookahead: float,
):
    start = float(start)
    hard_end = float(hard_end)

    min_end = min(start + min_dur_sec, hard_end)
    if hard_end <= min_end + 0.05:
        return hard_end, "end:video_short"

    target = min(start + (ideal_dur_min + ideal_dur_max) * 0.5, hard_end)

    win_lo = max(min_end, target - end_lookback)
    win_hi = min(hard_end, target + end_lookahead)

    sil = _best_silence_start_in_window(win_lo, win_hi, target, silence_segments)
    if sil is not None:
        return float(max(sil, min_end)), "end:silence_seek"

    blk = _best_block_end_in_window(win_lo, win_hi, target, speech_blocks)
    if blk is not None:
        return float(max(blk, min_end)), "end:block_seek"

    return hard_end, "end:hardcap"


def _compute_rms_percentile(energy_curve, q=97):
    if not energy_curve:
        return None
    vals = []
    for p in energy_curve:
        if isinstance(p, dict) and "rms" in p:
            try:
                vals.append(float(p["rms"]))
            except Exception:
                pass
    if not vals:
        return None
    try:
        return float(np.percentile(vals, q))
    except Exception:
        return None


def run_candidate_mining(state: Dict[str, Any]) -> Dict[str, Any]:
    min_clip_sec = float(state.get("MIN_CLIP_SEC", 18.0))
    max_clip_sec = float(state.get("MAX_CLIP_SEC", 60.0))

    use_flexible_dur = os.getenv("USE_FLEXIBLE_DUR", "1") == "1"
    min_dur_sec = float(os.getenv("MIN_DUR_SEC", str(min_clip_sec)))
    max_dur_sec = float(os.getenv("MAX_DUR_SEC", str(max_clip_sec)))

    ideal_dur_min = float(os.getenv("IDEAL_DUR_MIN", "25"))
    ideal_dur_max = float(os.getenv("IDEAL_DUR_MAX", "45"))

    end_lookback = float(os.getenv("END_LOOKBACK", "3.0"))
    end_lookahead = float(os.getenv("END_LOOKAHEAD", "2.0"))

    energy_curve = state.get("ENERGY_CURVE", [])
    silence_segments = state.get("SILENCE_SEGMENTS", [])
    speech_blocks = state.get("SPEECH_BLOCKS", [])
    analyzed_duration = float(state.get("ANALYZED_DURATION", 0.0))

    if "P97" in state:
        try:
            p97_val = float(state.get("P97"))
        except Exception:
            p97_val = _compute_rms_percentile(energy_curve, q=97)
    else:
        p97_val = _compute_rms_percentile(energy_curve, q=97)

    enable_peak_windows = p97_val is not None

    with StageTimer(6, "Segment Proposal (Candidate Mining)"):
        candidates = []
        cid = 0

        for bs, be in speech_blocks:
            bs = float(bs)
            be = float(be)
            dur = be - bs
            if dur < min_clip_sec:
                continue

            win = min(max_dur_sec, max(min_dur_sec, dur))
            step = 6.0
            t = bs
            while t + min_dur_sec <= be:
                hard_end = min(be, t + win)

                if use_flexible_dur:
                    end, end_reason = pick_flexible_end(
                        start=t,
                        hard_end=hard_end,
                        silence_segments=silence_segments,
                        speech_blocks=speech_blocks,
                        min_dur_sec=min_dur_sec,
                        ideal_dur_min=ideal_dur_min,
                        ideal_dur_max=ideal_dur_max,
                        end_lookback=end_lookback,
                        end_lookahead=end_lookahead,
                    )
                else:
                    end, end_reason = hard_end, "end:fixed"

                if (end - t) >= min_dur_sec:
                    cid += 1
                    candidates.append(
                        {
                            "id": f"cand_{cid:04d}",
                            "type": "speech_block",
                            "start": float(t),
                            "end": float(end),
                            "duration": float(end - t),
                            "end_reason": end_reason,
                        }
                    )
                t += step

        if not enable_peak_windows:
            logger.warning("[STAGE 06] Peak detection disabled (P97 unavailable) - skipping peak-centric mining")
        else:
            peaks = [p for p in energy_curve if float(p["rms"]) >= float(p97_val)]
            peaks = peaks[:: max(1, len(peaks) // 200)] if peaks else []

            for p in peaks:
                peak_time = float(p["time"])
                ideal = 40.0
                setup = 6.0
                target_peak_offset = 15.0
                min_peak_offset = 8.0
                max_peak_offset = 25.0

                wstart = peak_time - target_peak_offset
                wend_hard = wstart + ideal

                wstart = max(0.0, float(wstart))
                wend_hard = min(float(analyzed_duration), float(wend_hard))

                if wend_hard - wstart < min_dur_sec:
                    continue

                peak_abs, _ = get_peak_time(energy_curve, wstart, wend_hard)
                if peak_abs is None:
                    continue

                peak_offset = peak_abs - wstart
                if not (min_peak_offset <= peak_offset <= max_peak_offset):
                    continue
                if peak_offset < (setup + 1.0):
                    continue

                hard_end = min(wstart + max_dur_sec, wend_hard)
                if use_flexible_dur:
                    wend, end_reason = pick_flexible_end(
                        start=wstart,
                        hard_end=hard_end,
                        silence_segments=silence_segments,
                        speech_blocks=speech_blocks,
                        min_dur_sec=min_dur_sec,
                        ideal_dur_min=ideal_dur_min,
                        ideal_dur_max=ideal_dur_max,
                        end_lookback=end_lookback,
                        end_lookahead=end_lookahead,
                    )
                else:
                    wend, end_reason = hard_end, "end:fixed"

                if wend - wstart < min_dur_sec:
                    continue

                cid += 1
                candidates.append(
                    {
                        "id": f"cand_{cid:04d}",
                        "type": "peak",
                        "start": float(wstart),
                        "end": float(wend),
                        "duration": float(wend - wstart),
                        "peak_time_abs": float(peak_abs),
                        "peak_offset_in_clip": float(peak_offset),
                        "end_reason": end_reason,
                    }
                )

        final = []
        for c in candidates:
            d = float(c["end"] - c["start"])
            if d < min_dur_sec:
                continue
            if d > max_dur_sec:
                c["end"] = float(c["start"] + max_dur_sec)
                c["duration"] = float(c["end"] - c["start"])
                c["end_reason"] = (c.get("end_reason", "") + "|clamp:max").strip("|")
            final.append(c)

        candidates = sorted(final, key=lambda x: (x["start"], x["end"], x["id"]))
        write_json(Path(state["RAW_SEGMENTS_DIR"]) / "candidates.json", candidates)

        logger.info(f"Candidates: {len(candidates)}")
        log_flush()

    state["CANDIDATES"] = candidates
    return state
