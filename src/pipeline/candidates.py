from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .io_utils import write_json
from .logging_utils import StageTimer, log_flush, log_i, log_warn, log_bar_chart, log_duration_hist
from .features import compute_features

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


def _bucket_index(t: float, analyzed_duration: float, time_buckets: int) -> int:
    if analyzed_duration <= 0:
        return 0
    idx = int((t / analyzed_duration) * time_buckets)
    if idx < 0:
        idx = 0
    if idx >= time_buckets:
        idx = time_buckets - 1
    return idx


def _silence_ratio(start: float, end: float, silence_segments: List[List[float]]) -> float:
    dur = max(0.001, end - start)
    sil = 0.0
    for ss, se in silence_segments or []:
        sil += max(0.0, min(end, se) - max(start, ss))
    return float(sil / dur)


def _find_silence_end_before(
    t: float, silence_segments: List[List[float]], min_before: float, max_before: float
):
    cand = []
    for s, e in silence_segments or []:
        if e <= t:
            d = t - e
            if min_before <= d <= max_before:
                cand.append((d, e))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])
    return float(cand[0][1])


def _find_silence_start_after(t: float, silence_segments: List[List[float]], max_after: float):
    best = None
    for s, e in silence_segments or []:
        if s >= t and s <= (t + max_after):
            if best is None or s < best:
                best = s
    return best


def _find_speech_block_for_time(t: float, speech_blocks: List[List[float]]):
    for s, e in speech_blocks or []:
        if s <= t <= e:
            return float(s), float(e)
    return None, None


def _select_duration_band(
    start: float,
    speech_blocks: List[List[float]],
    silence_segments: List[List[float]],
    analyzed_duration: float,
    short_min: float,
    short_max: float,
    mid_min: float,
    mid_max: float,
    long_min: float,
    long_max: float,
):
    bs, be = _find_speech_block_for_time(start, speech_blocks)
    if bs is None or be is None:
        be = min(analyzed_duration, start + long_max)
    avail = max(0.0, float(be - start))
    probe_end = min(float(be), float(start + long_max))
    density = 1.0 - _silence_ratio(start, probe_end, silence_segments)
    if avail >= 90.0 and density >= 0.82:
        return "long", long_min, long_max
    if avail >= 55.0 and density >= 0.65:
        return "mid", mid_min, mid_max
    return "short", short_min, short_max


def run_candidate_mining(state: Dict[str, Any]) -> Dict[str, Any]:
    min_clip_sec = float(state.get("MIN_CLIP_SEC", 15.0))
    max_clip_sec = float(state.get("MAX_CLIP_SEC", 120.0))

    use_flexible_dur = os.getenv("USE_FLEXIBLE_DUR", "1") == "1"
    min_dur_sec = float(os.getenv("MIN_DUR_SEC", str(min_clip_sec)))
    max_dur_sec = float(os.getenv("MAX_DUR_SEC", str(max_clip_sec)))

    short_min = float(os.getenv("DUR_SHORT_MIN", "15"))
    short_max = float(os.getenv("DUR_SHORT_MAX", "35"))
    mid_min = float(os.getenv("DUR_MID_MIN", "45"))
    mid_max = float(os.getenv("DUR_MID_MAX", "75"))
    long_min = float(os.getenv("DUR_LONG_MIN", "75"))
    long_max = float(os.getenv("DUR_LONG_MAX", "120"))

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
                band, ideal_dur_min, ideal_dur_max = _select_duration_band(
                    t,
                    speech_blocks,
                    silence_segments,
                    analyzed_duration,
                    short_min,
                    short_max,
                    mid_min,
                    mid_max,
                    long_min,
                    long_max,
                )
                band_max = ideal_dur_max
                hard_end = min(be, t + min(win, band_max))

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
                            "dur_band": band,
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

                band, ideal_dur_min, ideal_dur_max = _select_duration_band(
                    wstart,
                    speech_blocks,
                    silence_segments,
                    analyzed_duration,
                    short_min,
                    short_max,
                    mid_min,
                    mid_max,
                    long_min,
                    long_max,
                )
                band_max = ideal_dur_max
                hard_end = min(wstart + min(max_dur_sec, band_max), wend_hard)
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
                        "dur_band": band,
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
        time_buckets = int(state.get("TIME_BUCKETS", 5))
        bucket_counts = {b: 0 for b in range(time_buckets)}
        for c in candidates:
            b = _bucket_index(float(c.get("start", 0.0)), analyzed_duration, time_buckets)
            bucket_counts[b] += 1
        log_bar_chart(
            logger,
            "bucket_counts",
            [(f"b{b}", int(bucket_counts[b])) for b in range(time_buckets)],
        )
        log_duration_hist(logger, [float(c.get("duration", 0.0)) for c in candidates])
        log_flush()

    state["CANDIDATES"] = candidates
    return state


def run_context_expansion(state: Dict[str, Any]) -> Dict[str, Any]:
    silence_segments = state.get("SILENCE_SEGMENTS", [])
    speech_blocks = state.get("SPEECH_BLOCKS", [])
    analyzed_duration = float(state.get("ANALYZED_DURATION", 0.0))
    min_clip_sec = float(state.get("MIN_CLIP_SEC", 15.0))
    max_clip_sec = float(state.get("MAX_CLIP_SEC", 120.0))
    start_expand_min = float(state.get("START_EXPAND_MIN", 2.0))
    start_expand_max = float(state.get("START_EXPAND_MAX", 6.0))
    tail_max_sec = float(state.get("TAIL_MAX_SEC", 2.5))

    with StageTimer(7, "Context Expansion (pre-score)"):
        adjust_start = 0
        adjust_end = 0
        for c in state.get("CANDIDATES", []):
            s = float(c.get("start", 0.0))
            e = float(c.get("end", s))
            s_before, e_before = s, e

            s_new = _find_silence_end_before(s, silence_segments, start_expand_min, start_expand_max)
            if s_new is not None:
                bs, be = _find_speech_block_for_time(s_new, speech_blocks)
                if bs is not None and be is not None and s_new >= bs:
                    s = float(max(0.0, s_new))

            if (e - s) < max_clip_sec:
                e_new = _find_silence_start_after(
                    e,
                    silence_segments,
                    max_after=min(tail_max_sec, max(0.0, max_clip_sec - (e - s))),
                )
                if e_new is not None:
                    e = float(min(analyzed_duration, e_new))

            s = max(0.0, float(s))
            e = min(float(analyzed_duration), float(e))
            if (e - s) < min_clip_sec:
                s, e = s_before, e_before

            if s != s_before:
                adjust_start += 1
                c["start_adjustment_sec"] = float(s_before - s)
                c["start_adjust_reason"] = "context:expand_silence"
            if e != e_before:
                adjust_end += 1
                c["end_adjustment_sec"] = float(e - e_before)
                c["end_adjust_reason"] = "context:extend_silence"

            c["start"] = float(s)
            c["end"] = float(e)
            c["duration"] = float(e - s)

        log_i(logger, f"context_expansion: start_adj={adjust_start} end_adj={adjust_end}")
        log_flush()

    return state


def add_marker_candidates(state: Dict[str, Any]) -> Dict[str, Any]:
    transcripts = state.get("TRANSCRIPTS", {})
    if not isinstance(transcripts, dict) or not transcripts:
        log_warn(logger, "marker_candidates: no transcripts available")
        return state

    analyzed_duration = float(state.get("ANALYZED_DURATION", 0.0))
    min_clip_sec = float(state.get("MIN_CLIP_SEC", 15.0))
    max_clip_sec = float(state.get("MAX_CLIP_SEC", 120.0))
    silence_segments = state.get("SILENCE_SEGMENTS", [])
    speech_blocks = state.get("SPEECH_BLOCKS", [])

    pre_sec = float(os.getenv("MARKER_PRE_SEC", "4.0"))
    post_sec = float(os.getenv("MARKER_POST_SEC", "45.0"))
    end_lookback = float(os.getenv("END_LOOKBACK", "3.0"))
    end_lookahead = float(os.getenv("END_LOOKAHEAD", "2.0"))

    candidates = list(state.get("CANDIDATES", []))
    existing = [(float(c["start"]), float(c["end"])) for c in candidates]

    def is_dup(s: float, e: float) -> bool:
        dur = max(0.001, e - s)
        for s2, e2 in existing:
            inter = max(0.0, min(e, e2) - max(s, s2))
            if inter / min(dur, max(0.001, e2 - s2)) >= 0.85:
                return True
        return False

    next_id = 0
    for c in candidates:
        cid = c.get("id", "")
        if isinstance(cid, str) and cid.startswith("cand_"):
            try:
                next_id = max(next_id, int(cid.split("_")[1]))
            except Exception:
                pass

    added = []
    for cid, tinfo in transcripts.items():
        markers = tinfo.get("markers_abs", []) or []
        for m in markers:
            m = float(m)
            s = max(0.0, m - pre_sec)
            hard_end = min(analyzed_duration, m + post_sec)
            if hard_end - s < min_clip_sec:
                continue
            end, end_reason = pick_flexible_end(
                start=s,
                hard_end=min(hard_end, s + max_clip_sec),
                silence_segments=silence_segments,
                speech_blocks=speech_blocks,
                min_dur_sec=min_clip_sec,
                ideal_dur_min=45.0,
                ideal_dur_max=min(75.0, max_clip_sec),
                end_lookback=end_lookback,
                end_lookahead=end_lookahead,
            )
            if (end - s) < min_clip_sec:
                continue
            if is_dup(s, end):
                continue
            next_id += 1
            new_c = {
                "id": f"cand_{next_id:04d}",
                "type": "marker_window",
                "start": float(s),
                "end": float(end),
                "duration": float(end - s),
                "marker_time_abs": float(m),
                "end_reason": f"{end_reason}|marker",
            }
            new_c["features"] = compute_features(new_c, state)
            added.append(new_c)
            existing.append((float(s), float(end)))

    if added:
        candidates.extend(added)
        candidates = sorted(candidates, key=lambda x: (x["start"], x["end"], x["id"]))
        state["CANDIDATES"] = candidates
        write_json(Path(state["RAW_SEGMENTS_DIR"]) / "candidates.json", candidates)
        log_i(logger, f"marker_candidates_added: {len(added)}")
        log_duration_hist(logger, [float(c.get("duration", 0.0)) for c in candidates])
    else:
        log_warn(logger, "marker_candidates_added: 0")

    log_flush()
    return state
