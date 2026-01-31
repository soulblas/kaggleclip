from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import write_json
from .logging_utils import StageTimer, log_flush, log_i, log_warn, log_bar_chart, log_duration_hist
from .scoring import score_candidate
from .features import compute_features

logger = logging.getLogger("viralshort")

TIME_BUCKETS = 5

# Normalize end_reason to allowed enumeration (Clause 9)
_ALLOWED_END_REASONS = {
    "IDEA_COMPLETE",
    "LONG_PAUSE",
    "TOPIC_SHIFT",
    "DENSITY_DROP",
    "HARDCAP_FALLBACK",
}

def _normalize_end_reason(end_reason: str) -> str:
    """Map internal end_reason strings to allowed enums."""
    if not end_reason:
        return "IDEA_COMPLETE"
    s = str(end_reason).lower()
    # Hard cap or fixed endings -> HARDCAP_FALLBACK
    if "hardcap" in s or "hard_end" in s or "fixed" in s or "video_short" in s or "clamp" in s:
        return "HARDCAP_FALLBACK"
    # Silence or pause trimmed -> LONG_PAUSE
    if "silence" in s or "pause" in s or "trim" in s:
        return "LONG_PAUSE"
    # Block/topic shift indicators
    if "block" in s or "topic" in s:
        return "TOPIC_SHIFT"
    # Density drop or energy fallback
    if "density" in s or "energy" in s:
        return "DENSITY_DROP"
    # Default to IDEA_COMPLETE for context expansion or end-of-idea
    return "IDEA_COMPLETE"


def overlaps(a_start, a_end, b_start, b_end, gap):
    return not (a_end + gap <= b_start or b_end + gap <= a_start)


def bucket_index(t, analyzed_duration, time_buckets: int):
    if analyzed_duration <= 0:
        return 0
    idx = int((t / analyzed_duration) * time_buckets)
    if idx < 0:
        idx = 0
    if idx >= time_buckets:
        idx = time_buckets - 1
    return idx


def percentile(values, q):
    if not values:
        return 0.0
    vs = sorted(values)
    k = int(round((q / 100.0) * (len(vs) - 1)))
    return float(vs[max(0, min(len(vs) - 1, k))])


def run_selection(state: Dict[str, Any]) -> Dict[str, Any]:
    analyzed_duration = float(state.get("ANALYZED_DURATION", 0.0))
    time_buckets = int(state.get("TIME_BUCKETS", TIME_BUCKETS))
    min_gap_sec = float(state.get("MIN_GAP_SEC", 30.0))
    max_final_clips = int(state.get("MAX_FINAL_CLIPS", 6))
    min_clip_sec = float(state.get("MIN_CLIP_SEC", 15.0))
    candidates = state.get("CANDIDATES", [])
    metadata_dir = Path(state["METADATA_DIR"])

    with StageTimer(11, "Selection (bucket quota + fairness)"):
        audit_map: Dict[str, Dict[str, Any]] = {}

        buckets = {b: [] for b in range(time_buckets)}
        for c in candidates:
            b = bucket_index(float(c.get("start", 0.0)), analyzed_duration, time_buckets)
            c["_bucket"] = b
            buckets[b].append(c)

        for b in buckets:
            buckets[b].sort(
                key=lambda c: (
                    -c.get("scores", {}).get("viral_score", 0.0),
                    c.get("start", 0.0),
                    c.get("id", ""),
                )
            )

        bucket_stats = {}
        meaning_min = {}
        score_min = {}
        for b, arr in buckets.items():
            meaning_vals = [c.get("scores", {}).get("meaning", 0.0) for c in arr]
            score_vals = [c.get("scores", {}).get("viral_score", 0.0) for c in arr]
            meaning_min[b] = percentile(meaning_vals, 35) if meaning_vals else 0.0
            score_min[b] = percentile(score_vals, 40) if score_vals else 0.0
            bucket_stats[b] = {
                "count": len(arr),
                "meaning_min": float(meaning_min[b]),
                "score_min": float(score_min[b]),
                "selected": 0,
                "quota": 0,
                "meaning_candidates": int(
                    sum(1 for c in arr if c.get("scores", {}).get("meaning", 0.0) >= meaning_min[b])
                ),
            }

        base_quota = 1
        quotas = {b: 0 for b in buckets}
        for b in buckets:
            quotas[b] = base_quota if buckets[b] else 0

        remaining = max_final_clips - sum(quotas.values())
        if remaining > 0:
            order = sorted(buckets.keys(), key=lambda b: len(buckets[b]), reverse=True)
            i = 0
            while remaining > 0 and order:
                quotas[order[i % len(order)]] += 1
                remaining -= 1
                i += 1

        late_bucket = time_buckets - 1
        if bucket_stats[late_bucket]["meaning_candidates"] > 0 and quotas[late_bucket] < 1:
            quotas[late_bucket] = 1

        for b in quotas:
            bucket_stats[b]["quota"] = int(quotas[b])

        selected = []

        def score_snapshot(c):
            s = c.get("scores", {})
            return {
                "viral_score": float(s.get("viral_score", 0.0)),
                "hook": float(s.get("hook", 0.0)),
                "meaning": float(s.get("meaning", 0.0)),
                "clarity": float(s.get("clarity", 0.0)),
                "finishability": float(s.get("finishability", 0.0)),
                "bucket": int(c.get("_bucket", 0)),
            }

        def record(c, decision, phase, reason_codes, reason):
            cid = c.get("id")
            if not cid:
                return
            rec = audit_map.get(cid)
            if rec is None:
                rec = {
                    "id": cid,
                    "decision": decision,
                    "phase": phase,
                    "bucket": int(c.get("_bucket", 0)),
                    "start": float(c.get("start", 0.0)),
                    "end": float(c.get("end", 0.0)),
                    "duration": float(c.get("duration", 0.0)),
                    "score_snapshot": score_snapshot(c),
                    "reason_codes": [],
                    "reason": "",
                    "editorial_reason": c.get("editorial_reason", []),
                    "end_reason": c.get("end_reason", ""),
                }
            rec["decision"] = decision
            rec["phase"] = phase
            rec["reason"] = reason
            rec["reason_codes"] = list(dict.fromkeys([str(x) for x in (reason_codes or [])]))
            # capture current score snapshot and semantics
            snap = score_snapshot(c)
            rec["score_snapshot"] = snap
            # boolean selected flag
            rec["selected"] = bool(decision == "selected")
            # semantic details
            rec["semantic_mode"] = str(c.get("scores", {}).get("semantic_mode", ""))
            rec["meaning"] = float(c.get("scores", {}).get("meaning", 0.0))
            # normalize end reason in audit as per Clause 9
            rec["end_reason"] = _normalize_end_reason(rec.get("end_reason") or c.get("end_reason", ""))
            audit_map[cid] = rec

        def can_add(c, gap_override=None, relax_score=False, relax_meaning=False):
            codes = []
            b = int(c.get("_bucket", 0))
            if float(c.get("end", 0.0)) <= float(c.get("start", 0.0)) or float(c.get("duration", 0.0)) < min_clip_sec:
                codes.append("REJ_INVALID_INTERVAL")
                return False, codes, "invalid_interval"
            if not relax_meaning and c.get("scores", {}).get("meaning", 0.0) < meaning_min[b]:
                codes.append("REJ_MEANING_BELOW_MIN")
            if not relax_score and c.get("scores", {}).get("viral_score", 0.0) < score_min[b]:
                codes.append("REJ_SCORE_BELOW_MIN")
            if codes:
                return False, codes, "below_threshold"
            gap = min_gap_sec if gap_override is None else gap_override
            for s in selected:
                if overlaps(c["start"], c["end"], s["start"], s["end"], gap=gap):
                    actual_overlap = not (c["end"] <= s["start"] or s["end"] <= c["start"])
                    return False, [("REJ_OVERLAP" if actual_overlap else "REJ_MIN_GAP")], f"overlap_or_gap_with_{s['id']}"
            return True, [], "ok"

        for b in range(time_buckets):
            need = quotas[b]
            if need <= 0:
                continue
            pool = list(buckets[b])
            while need > 0 and pool:
                c = pool.pop(0)
                ok, codes, why = can_add(c)
                if ok:
                    selected.append(c)
                    bucket_stats[b]["selected"] += 1
                    record(c, "selected", "quota", ["SEL_QUOTA"], why)
                    need -= 1
                else:
                    record(c, "rejected", "quota", codes, why)

        if len(selected) < max_final_clips:
            remaining = sorted(
                [c for c in candidates if c not in selected],
                key=lambda c: (-c.get("scores", {}).get("viral_score", 0.0), c.get("start", 0.0), c.get("id", "")),
            )
            for c in remaining:
                if len(selected) >= max_final_clips:
                    break
                ok, codes, why = can_add(c)
                if not ok:
                    record(c, "rejected", "fill", codes, why)
                    continue
                selected.append(c)
                bucket_stats[int(c.get("_bucket", 0))]["selected"] += 1
                record(c, "selected", "fill", ["SEL_TOP_SCORE"], why)

        # Late bucket rescue (relax thresholds + min_gap) if still empty
        if bucket_stats[late_bucket]["meaning_candidates"] > 0:
            has_late = any(int(s.get("_bucket", -1)) == late_bucket for s in selected)
            if not has_late:
                rescue = None
                for c in buckets[late_bucket]:
                    ok, codes, why = can_add(c, gap_override=0.0, relax_score=True, relax_meaning=True)
                    if ok:
                        rescue = c
                        break
                if rescue is not None:
                    # append rescue candidate or replace lowest non-late bucket candidate
                    if len(selected) < max_final_clips:
                        selected.append(rescue)
                    else:
                        worst = min(
                            (s for s in selected if int(s.get("_bucket", -1)) != late_bucket),
                            key=lambda s: s.get("scores", {}).get("viral_score", 0.0),
                            default=None,
                        )
                        if worst is not None:
                            selected.remove(worst)
                            record(
                                worst,
                                "rejected",
                                "rescue_replace",
                                ["REJ_BUCKET_QUOTA_FULL"],
                                "replaced_by_late_bucket",
                            )
                            selected.append(rescue)
                    record(rescue, "selected", "rescue", ["SEL_RESCUE_LATE"], "late_bucket_rescue")
                    # Log fairness enforcement (Clause G)
                    logger.info(f"TIME_FAIRNESS_ENFORCED: selected {rescue.get('id')} from last bucket")
                else:
                    log_warn(logger, "late_bucket_rescue: no eligible candidate")

        # Ensure every candidate has an audit record
        for c in candidates:
            cid = c.get("id")
            if cid in audit_map:
                continue
            ok, codes, why = can_add(c)
            if ok:
                codes = ["REJ_BUCKET_QUOTA_FULL"]
                why = "quota_or_limit_full"
            record(c, "rejected", "post", codes, why)

        # Normalize end_reason for selected clips
        for c in selected:
            c["end_reason"] = _normalize_end_reason(c.get("end_reason", ""))
        selected_sorted = sorted(selected, key=lambda c: (c["start"], c["id"]))
        band_counts = {"short": 0, "mid": 0, "long": 0}

        def choose_band(c):
            s = c.get("scores", {})
            f = c.get("features", {})
            wps = float(f.get("words_per_sec", 0.0))
            semantic_mode = s.get("semantic_mode", "")
            if semantic_mode in ("no_text", "fallback"):
                if s.get("hook", 0.0) >= 75.0 and s.get("energy", 0.0) >= 70.0 and s.get("finishability", 0.0) >= 70.0:
                    return "short"
                if s.get("hook", 0.0) >= 60.0 and s.get("finishability", 0.0) >= 60.0:
                    return "mid"
                return "mid"
            if s.get("meaning", 0.0) >= 65.0 and s.get("clarity", 0.0) >= 60.0 and s.get("finishability", 0.0) >= 60.0 and wps >= 1.4:
                return "long"
            if s.get("hook", 0.0) >= 70.0 and s.get("finishability", 0.0) >= 70.0 and (s.get("meaning", 0.0) >= 45.0 or wps >= 2.0):
                return "short"
            return "mid"

        for c in selected_sorted:
            band = choose_band(c)
            c["dur_band_target"] = band
            if band in band_counts:
                band_counts[band] += 1
        # Recount selections per bucket
        for b in bucket_stats:
            bucket_stats[b]["selected"] = int(sum(1 for s in selected_sorted if int(s.get("_bucket", -1)) == b))

        selection_audit = sorted(
            audit_map.values(),
            key=lambda r: (r.get("decision") != "selected", r.get("bucket", 0), r.get("start", 0.0), r.get("id", "")),
        )

        write_json(metadata_dir / "selected.json", selected_sorted)
        write_json(metadata_dir / "selection_audit.json", selection_audit)
        write_json(metadata_dir / "bucket_stats.json", bucket_stats)

        log_i(logger, "selected_summary")
        for j, c in enumerate(selected_sorted, 1):
            rs = c.get("scores", {})
            pct = (float(c.get("start", 0.0)) / analyzed_duration * 100.0) if analyzed_duration > 0 else 0.0
            logger.info(
                f"    #{j:02d} b{int(c.get('_bucket',0))} {c['id']} start={c['start']:.2f}s ({pct:.0f}%) dur={c['duration']:.1f}s VS={rs.get('viral_score',0):.1f}"
            )
        log_i(logger, f"Selected clips: {len(selected_sorted)}")
        log_duration_hist(logger, [float(c.get("duration", 0.0)) for c in selected_sorted])
        log_bar_chart(
            logger,
            "selected_dur_band",
            [("short", band_counts["short"]), ("mid", band_counts["mid"]), ("long", band_counts["long"])],
        )
        log_bar_chart(
            logger,
            "selected_bucket_counts",
            [(f"b{b}", int(bucket_stats[b]["selected"])) for b in range(time_buckets)],
        )

    state["SELECTED"] = selected_sorted
    log_flush()
    return state


def snap_selected(state: Dict[str, Any]) -> Dict[str, Any]:
    silence_segments = state.get("SILENCE_SEGMENTS", [])
    shot_cuts = state.get("SHOT_CUTS", [])
    analyzed_duration = float(state.get("ANALYZED_DURATION", 0.0))
    min_clip_sec = float(state.get("MIN_CLIP_SEC", 15.0))
    max_clip_sec = float(state.get("MAX_CLIP_SEC", 120.0))
    metadata_dir = Path(state["METADATA_DIR"])

    def snap_to_words(start: float, end: float, words: List[Dict[str, Any]], radius: float = 1.4):
        if not words:
            return start, end
        ws = [w for w in words if (start - radius) <= w["start"] <= (start + radius)]
        if ws:
            prior = [w for w in ws if w["start"] <= start]
            if prior:
                start = max(prior, key=lambda w: w["start"])["start"]
            else:
                start = min(ws, key=lambda w: abs(w["start"] - start))["start"]

        we = [w for w in words if (end - radius) <= w["end"] <= (end + radius)]
        if we:
            after = [w for w in we if w["end"] >= end]
            if after:
                end = min(after, key=lambda w: w["end"])["end"]
            else:
                end = min(we, key=lambda w: abs(w["end"] - end))["end"]
        return start, end

    def snap_to_silence_edges(start: float, end: float, silence_segments: List[List[float]], radius: float = 1.0):
        best_s = start
        cand = [(s, e) for s, e in silence_segments if abs(e - start) <= radius and e <= start]
        if cand:
            best_s = max(cand, key=lambda x: x[1])[1]

        best_e = end
        cand2 = [(s, e) for s, e in silence_segments if abs(s - end) <= radius and s >= end]
        if cand2:
            best_e = min(cand2, key=lambda x: x[0])[0]
        return best_s, best_e

    def avoid_shot_cut(t: float, shot_cuts: List[float], min_dist=0.30, shift=0.35, direction=+1):
        if not shot_cuts:
            return t
        for c in shot_cuts:
            if abs(c - t) < min_dist:
                return t + direction * shift
        return t

    tail_ext_sec = float(state.get("TAIL_EXT_SEC", 1.2))
    tail_max_sec = float(state.get("TAIL_MAX_SEC", 2.0))
    lead_silence_trim_sec = float(state.get("LEAD_SILENCE_TRIM_SEC", 0.8))
    start_expand_min = float(state.get("START_EXPAND_MIN", 2.0))
    start_expand_max = float(state.get("START_EXPAND_MAX", 6.0))
    start_speech_window = float(state.get("START_SPEECH_WINDOW", 0.6))
    snap_word_radius = float(state.get("SNAP_WORD_RADIUS", 1.4))
    snap_silence_radius = float(state.get("SNAP_SILENCE_RADIUS", 1.0))

    def _find_silence_start_after(t: float, silence_segments: List[List[float]], max_after: float):
        best = None
        for s, e in silence_segments:
            if s >= t and s <= (t + max_after):
                if best is None or s < best:
                    best = s
        return best

    def _find_silence_end_before(
        t: float, silence_segments: List[List[float]], min_before: float, max_before: float
    ):
        cand = []
        for s, e in silence_segments:
            if e <= t:
                d = t - e
                if min_before <= d <= max_before:
                    cand.append((d, e))
        if not cand:
            return None
        cand.sort(key=lambda x: x[0])
        return float(cand[0][1])

    def _next_word_end_after(t: float, words: List[Dict[str, Any]], max_after: float):
        cand = []
        for w in words or []:
            if "end" not in w:
                continue
            we = float(w["end"])
            if we >= t and we <= (t + max_after):
                cand.append(we)
        return min(cand) if cand else None

    def _prev_word_start_before(
        t: float, words: List[Dict[str, Any]], min_before: float, max_before: float
    ):
        cand = []
        for w in words or []:
            if "start" not in w:
                continue
            ws = float(w["start"])
            if ws <= t:
                d = t - ws
                if min_before <= d <= max_before:
                    cand.append((d, ws))
        if not cand:
            return None
        cand.sort(key=lambda x: x[0])
        return float(cand[0][1])

    def _speech_active_near_end(t: float, words: List[Dict[str, Any]], window: float = 0.6):
        if not words:
            return False
        lo = t - window
        hi = t + 0.2
        for w in words:
            if "end" not in w:
                continue
            we = float(w["end"])
            if lo <= we <= hi:
                return True
        return False

    def _speech_active_near_start(t: float, words: List[Dict[str, Any]], window: float = 0.6):
        if not words:
            return False
        lo = t - window
        hi = t + 0.2
        for w in words:
            if "start" not in w:
                continue
            ws = float(w["start"])
            if lo <= ws <= hi:
                return True
        return False

    def _add_reason(c: Dict[str, Any], msg: str):
        er = c.get("editorial_reason", [])
        if not isinstance(er, list):
            er = [str(er)]
        if msg not in er:
            er.append(msg)
        c["editorial_reason"] = er

    if state.get("_SNAP_DONE"):
        log_i(logger, "Snap already completed; skipping duplicate run")
        log_flush()
        return state

    log_i(logger, "Snap boundaries + end-of-idea polish")

    transcripts = state.get("TRANSCRIPTS", {})
    if not isinstance(transcripts, dict):
        transcripts = {}
    transcript_cache = state.get("TRANSCRIPT_CACHE", {})
    if not isinstance(transcript_cache, dict):
        transcript_cache = {}
    key_by_id = state.get("TRANSCRIPT_KEY_BY_ID", {}) or {}

    asr_available = bool(state.get("ASR_AVAILABLE", False)) and ("transcribe_candidate_abs" in state)
    if asr_available:
        def transcribe_candidate_abs_local(candidate):
            return state["transcribe_candidate_abs"](candidate, mode="full")
    else:
        def transcribe_candidate_abs_local(candidate):
            return {
                "text": "",
                "words": [],
                "markers_abs": [],
                "words_per_sec": 0.0,
                "trigger_count": 0,
                "word_count": 0,
            }

    for c in state.get("SELECTED", []):
        cid = c.get("id")
        tinfo = transcripts.get(cid, {}) if isinstance(transcripts, dict) else {}
        if cid and (cid not in transcripts or tinfo.get("mode") != "full"):
            try:
                logger.info(f"ASR (selected-only full) {cid} for snapping")
                out = transcribe_candidate_abs_local(c)
                transcripts[cid] = out
                if cid:
                    video_hash = (state.get("VIDEO_HASH") or "").strip()
                    params = f"{state.get('ASR_MODEL_NAME')}|{state.get('ASR_BEAM_SIZE')}|{state.get('ASR_LIGHT_MODE')}|{state.get('ASR_LIGHT_SEC')}|{state.get('ASR_LIGHT_OFFSET')}|{state.get('ASR_LANGUAGE')}"
                    cache_key = key_by_id.get(cid) or f"{video_hash}:{float(c.get('start',0.0)):.2f}-{float(c.get('end',0.0)):.2f}:{params}"
                    out["id"] = cid
                    out["start"] = float(c.get("start", 0.0))
                    out["end"] = float(c.get("end", 0.0))
                    out["video_hash"] = video_hash
                    transcript_cache[cache_key] = out
                if out.get("words"):
                    _add_reason(c, "asr:selected_full")
                else:
                    _add_reason(c, "asr:selected_full_empty")
            except Exception as e:
                logger.warning(f"Selected-only ASR failed for {cid}: {e}")
                _add_reason(c, "asr:selected_full_failed")

    energy_curve = state.get("ENERGY_CURVE", [])
    bucket_vals = state.get("SCORING_BUCKET_VALS")
    semantic_available = bool(state.get("SEMANTIC_AVAILABLE", False))
    time_buckets = int(state.get("TIME_BUCKETS", TIME_BUCKETS))

    def snap_to_energy_trough(start: float, end: float, energy_curve, radius: float = 0.8):
        if not energy_curve:
            return start, end
        s_window = [p for p in energy_curve if (start - radius) <= p["time"] <= (start + radius)]
        e_window = [p for p in energy_curve if (end - radius) <= p["time"] <= (end + radius)]
        if s_window:
            s_best = min(s_window, key=lambda p: p.get("rms", 0.0))
            start = float(s_best.get("time", start))
        if e_window:
            e_best = min(e_window, key=lambda p: p.get("rms", 0.0))
            end = float(e_best.get("time", end))
        return start, end

    for c in state.get("SELECTED", []):
        cid = c.get("id")
        s = float(c["start"])
        e = float(c["end"])
        band = c.get("dur_band_target", "mid")
        band_max = {"short": 35.0, "mid": 75.0, "long": 120.0}.get(band, max_clip_sec)
        band_min = {"short": 15.0, "mid": 45.0, "long": 75.0}.get(band, min_clip_sec)
        max_cap = min(max_clip_sec, band_max)

        s_before, e_before = s, e
        s = avoid_shot_cut(s, shot_cuts, direction=+1)
        if s != s_before:
            _add_reason(c, f"avoid_shot_cut:start(+{s - s_before:.2f}s)")

        e = avoid_shot_cut(e, shot_cuts, direction=+1)
        if e != e_before:
            _add_reason(c, f"avoid_shot_cut:end(+{e - e_before:.2f}s)")

        words = (transcripts.get(cid, {}) or {}).get("words", []) or []

        for (ss, se) in silence_segments:
            if se <= s and (s - se) <= lead_silence_trim_sec:
                s = float(se)
                _add_reason(c, "trim:leading_silence")
                break

        if _speech_active_near_start(s, words, window=start_speech_window):
            sil_prev = _find_silence_end_before(
                s, silence_segments, start_expand_min, start_expand_max
            )
            if sil_prev is not None:
                s = float(sil_prev)
                _add_reason(c, f"context:expand_silence(-{s_before - s:.2f}s)")
            else:
                ws_prev = _prev_word_start_before(s, words, start_expand_min, start_expand_max)
                if ws_prev is not None:
                    s = float(ws_prev)
                    _add_reason(c, f"context:expand_word(-{s_before - s:.2f}s)")

        s2, e2 = snap_to_words(s, e, words, radius=snap_word_radius)
        _add_reason(c, "snap:word" if words else "snap:nowords")

        if (e2 - s2) < min_clip_sec:
            s2, e2 = snap_to_silence_edges(s, e, silence_segments, radius=snap_silence_radius)
            _add_reason(c, "snap:silence_fallback")

        if (e2 - s2) < min_clip_sec and not words:
            s2, e2 = snap_to_energy_trough(s, e, energy_curve, radius=0.8)
            _add_reason(c, "snap:energy_fallback")

        s2 = max(0.0, float(s2))
        e2 = min(float(analyzed_duration), float(e2))
        if (e2 - s2) < min_clip_sec:
            logger.warning(f"Snap too short for {cid}; keeping original bounds")
            _add_reason(c, "snap:too_short_keep_original")
            continue

        hard_cap = min(float(analyzed_duration), float(s2 + max_cap))
        still_speaking = _speech_active_near_end(e2, words, window=0.6)

        if still_speaking and e2 < hard_cap:
            sil = _find_silence_start_after(e2, silence_segments, max_after=min(tail_max_sec, hard_cap - e2))
            if sil is not None:
                e2 = min(float(sil), hard_cap)
                _add_reason(c, f"end:seek_silence(+{e2 - e_before:.2f}s)")
            else:
                wend = _next_word_end_after(e2, words, max_after=min(tail_ext_sec, hard_cap - e2))
                if wend is not None:
                    e2 = min(float(wend), hard_cap)
                    _add_reason(c, f"end:seek_word(+{e2 - e_before:.2f}s)")
                else:
                    e2 = min(float(e2 + min(tail_ext_sec, hard_cap - e2)), hard_cap)
                    _add_reason(c, f"end:extend(+{e2 - e_before:.2f}s)")
        elif not still_speaking:
            for (ss, se) in silence_segments:
                if ss <= e2 and (e2 - ss) <= 1.2:
                    new_e = float(ss)
                    if (new_e - s2) >= min_clip_sec:
                        e2 = new_e
                        _add_reason(c, "end:trim_to_silence")
                    break

        dur = e2 - s2
        if dur > max_cap:
            _add_reason(c, f"clamp:max_sec({max_cap:.1f})")
            hard_end = float(s2 + max_cap)
            e_limit = min(float(e2), float(analyzed_duration))

            still_speaking = _speech_active_near_end(hard_end, words, window=0.6)
            if still_speaking:
                _add_reason(c, "end:still_speaking")
                sil = _find_silence_start_after(hard_end, silence_segments, max_after=tail_max_sec)
                if sil is not None:
                    e2 = min(float(sil), e_limit)
                    _add_reason(c, f"end:silence(+{e2 - hard_end:.2f}s)")
                else:
                    wend = _next_word_end_after(hard_end, words, max_after=min(tail_ext_sec, tail_max_sec))
                    if wend is not None:
                        e2 = min(float(wend), e_limit)
                        _add_reason(c, f"end:word(+{e2 - hard_end:.2f}s)")
                    else:
                        e2 = min(float(hard_end + min(tail_ext_sec, tail_max_sec)), e_limit)
                        _add_reason(c, f"end:extend(+{e2 - hard_end:.2f}s)")
            else:
                e2 = min(float(hard_end), e_limit)
                _add_reason(c, "end:hard_end")

        e2 = min(float(analyzed_duration), float(e2))
        if e2 <= s2:
            logger.warning(f"Bad bounds after clamp for {cid}; keeping original bounds")
            _add_reason(c, "bounds:bad_keep_original")
            continue
        if (e2 - s2) < min_clip_sec:
            logger.warning(f"Too short after clamp for {cid}; keeping original bounds")
            _add_reason(c, "bounds:too_short_keep_original")
            continue

        if dur < band_min:
            _add_reason(c, f"dur:below_band_min({band_min:.1f})")

        c["start"] = float(s2)
        c["end"] = float(e2)
        c["duration"] = float(e2 - s2)

        # Recompute features + scores after snap
        new_features = compute_features(c, state)
        if words:
            words_in = [w for w in words if s2 <= float(w.get("start", 0.0)) <= e2]
            word_count = len(words_in)
            wps = float(word_count / max(0.1, e2 - s2))
            markers_abs = list((transcripts.get(cid, {}) or {}).get("markers_abs", []) or [])
            trigger_count = sum(1 for m in markers_abs if s2 <= m <= e2)
            new_features["words_per_sec"] = wps
            new_features["word_count"] = word_count
            new_features["trigger_count"] = trigger_count
            new_features["markers_abs"] = markers_abs
            new_features["has_text"] = True if word_count > 0 else False
        c["features"] = new_features

        if bucket_vals:
            prev_novelty = float((c.get("scores", {}) or {}).get("novelty", 0.0))
            c.setdefault("scores", {})["novelty"] = prev_novelty
            c["scores"] = score_candidate(
                c,
                analyzed_duration=analyzed_duration,
                time_buckets=time_buckets,
                bucket_vals=bucket_vals,
                transcripts=transcripts,
                silence_segments=silence_segments,
                speech_blocks=state.get("SPEECH_BLOCKS", []),
                hook_window_sec=float(state.get("HOOK_WINDOW_SEC", 5.0)),
                semantic_available=semantic_available,
            )

            s = c.get("scores", {})
            c["editorial_reason"] = [
                f"Meaning {s.get('meaning',0):.1f} ({s.get('semantic_mode','')})",
                f"Hook {s.get('hook',0):.1f}",
                f"Clarity {s.get('clarity',0):.1f}",
                f"Energy {s.get('energy',0):.1f}",
                f"Novelty {s.get('novelty',0):.1f}",
            ]

    write_json(metadata_dir / "selected_snapped.json", state.get("SELECTED", []))
    write_json(metadata_dir / "selected.json", state.get("SELECTED", []))
    log_i(logger, "Snapping completed (end-of-idea polish enabled).")

    try:
        write_json(Path(state["SCORED_SEGMENTS_DIR"]) / "transcript.json", transcript_cache)
        logger.info(f"Saved transcript cache after snapping (entries={len(transcript_cache)})")
    except Exception as e:
        logger.warning(f"Failed to save transcript cache after snapping: {e}")

    state["TRANSCRIPTS"] = transcripts
    state["TRANSCRIPT_CACHE"] = transcript_cache
    state["_SNAP_DONE"] = True
    # Re-score all candidates after snapping (Stage 11.5) to ensure semantic consistency
    try:
        from .scoring import run_scoring  # local import to avoid circular
        state = run_scoring(state)
        # Update selection_audit with new semantic_mode and meaning for consistency
        metadata_dir = Path(state["METADATA_DIR"])
        audit_path = metadata_dir / "selection_audit.json"
        try:
            import json
            audit_entries = json.loads(audit_path.read_text(encoding="utf-8"))
        except Exception:
            audit_entries = []
        # Build mapping from candidate id to current scores
        score_map = {}
        for cand in state.get("CANDIDATES", []):
            cid = cand.get("id")
            if cid:
                score_map[cid] = cand.get("scores", {}) or {}
        for rec in audit_entries:
            cid = rec.get("id")
            if cid and cid in score_map:
                s = score_map[cid]
                rec["semantic_mode"] = str(s.get("semantic_mode", rec.get("semantic_mode", "")))
                rec["meaning"] = float(s.get("meaning", rec.get("meaning", 0.0)))
            # Ensure 'selected' flag exists
            if "selected" not in rec:
                rec["selected"] = bool(rec.get("decision") == "selected")
            # Normalize end_reason values
            rec["end_reason"] = _normalize_end_reason(rec.get("end_reason", ""))
        try:
            write_json(audit_path, audit_entries)
        except Exception:
            pass
        # Update selected.json with normalized end_reason and latest scores
        sel_path = metadata_dir / "selected.json"
        try:
            sel_entries = json.loads(sel_path.read_text(encoding="utf-8"))
        except Exception:
            sel_entries = []
        changed = False
        for c in sel_entries:
            cid = c.get("id")
            if cid and cid in score_map:
                c["scores"] = score_map[cid]
                c["end_reason"] = _normalize_end_reason(c.get("end_reason", ""))
                changed = True
        if changed:
            try:
                write_json(sel_path, sel_entries)
                # also update selected_snapped.json to match
                write_json(metadata_dir / "selected_snapped.json", sel_entries)
            except Exception:
                pass
        # Log rescore status
        logger.info("RESCORE_AFTER_SNAP: PASS")
    except Exception as e:
        logger.warning(f"RESCORE_AFTER_SNAP: FAILED due to {e}")
    log_flush()
    return state
