from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import write_json
from .logging_utils import StageTimer, log_flush

logger = logging.getLogger("viralshort")

TIME_BUCKETS = 5


def overlaps(a_start, a_end, b_start, b_end, gap):
    return not (a_end + gap <= b_start or b_end + gap <= a_start)


def bucket_index(t, analyzed_duration):
    if analyzed_duration <= 0:
        return 0
    idx = int((t / analyzed_duration) * TIME_BUCKETS)
    if idx < 0:
        idx = 0
    if idx >= TIME_BUCKETS:
        idx = TIME_BUCKETS - 1
    return idx


def percentile(values, q):
    if not values:
        return 0.0
    vs = sorted(values)
    k = int(round((q / 100.0) * (len(vs) - 1)))
    return float(vs[max(0, min(len(vs) - 1, k))])


def run_selection(state: Dict[str, Any]) -> Dict[str, Any]:
    analyzed_duration = float(state.get("ANALYZED_DURATION", 0.0))
    min_gap_sec = float(state.get("MIN_GAP_SEC", 30.0))
    max_final_clips = int(state.get("MAX_FINAL_CLIPS", 6))
    candidates = state.get("CANDIDATES", [])
    metadata_dir = Path(state["METADATA_DIR"])

    with StageTimer(10, "Selection (bucket quota + fairness)"):
        selection_audit = []

        buckets = {b: [] for b in range(TIME_BUCKETS)}
        for c in candidates:
            b = bucket_index(float(c.get("start", 0.0)), analyzed_duration)
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

        late_bucket = TIME_BUCKETS - 1
        if bucket_stats[late_bucket]["meaning_candidates"] > 0 and quotas[late_bucket] < 1:
            quotas[late_bucket] = 1

        for b in quotas:
            bucket_stats[b]["quota"] = int(quotas[b])

        selected = []

        def can_add(c):
            b = bucket_index(float(c.get("start", 0.0)), analyzed_duration)
            if c.get("scores", {}).get("meaning", 0.0) < meaning_min[b]:
                return False, "below_meaning_min"
            if c.get("scores", {}).get("viral_score", 0.0) < score_min[b]:
                return False, "below_bucket_score_min"
            for s in selected:
                if overlaps(c["start"], c["end"], s["start"], s["end"], gap=min_gap_sec):
                    return False, f"overlap_or_gap_with_{s['id']}"
            return True, "ok"

        for b in range(TIME_BUCKETS):
            need = quotas[b]
            if need <= 0:
                continue
            pool = list(buckets[b])
            while need > 0 and pool:
                c = pool.pop(0)
                ok, why = can_add(c)
                if ok:
                    selected.append(c)
                    bucket_stats[b]["selected"] += 1
                    selection_audit.append(
                        {
                            "id": c["id"],
                            "decision": "selected",
                            "phase": "quota",
                            "bucket": b,
                            "start": c["start"],
                            "score": c["scores"]["viral_score"],
                            "reason": why,
                            "editorial_reason": c.get("editorial_reason", []),
                        }
                    )
                    need -= 1
                else:
                    selection_audit.append(
                        {
                            "id": c["id"],
                            "decision": "rejected",
                            "phase": "quota",
                            "bucket": b,
                            "start": c["start"],
                            "score": c["scores"]["viral_score"],
                            "reason": why,
                            "editorial_reason": c.get("editorial_reason", []),
                        }
                    )

        if len(selected) < max_final_clips:
            remaining = sorted(
                [c for c in candidates if c not in selected],
                key=lambda c: (-c.get("scores", {}).get("viral_score", 0.0), c.get("start", 0.0), c.get("id", "")),
            )
            for c in remaining:
                if len(selected) >= max_final_clips:
                    break
                ok, why = can_add(c)
                b = bucket_index(float(c.get("start", 0.0)), analyzed_duration)
                if not ok:
                    selection_audit.append(
                        {
                            "id": c["id"],
                            "decision": "rejected",
                            "phase": "fill",
                            "bucket": b,
                            "start": c["start"],
                            "score": c["scores"]["viral_score"],
                            "reason": why,
                            "editorial_reason": c.get("editorial_reason", []),
                        }
                    )
                    continue
                selected.append(c)
                bucket_stats[b]["selected"] += 1
                selection_audit.append(
                    {
                        "id": c["id"],
                        "decision": "selected",
                        "phase": "fill",
                        "bucket": b,
                        "start": c["start"],
                        "score": c["scores"]["viral_score"],
                        "reason": "ok",
                        "editorial_reason": c.get("editorial_reason", []),
                    }
                )

        selected_sorted = sorted(selected, key=lambda c: (c["start"], c["id"]))
        write_json(metadata_dir / "selected.json", selected_sorted)
        write_json(metadata_dir / "selection_audit.json", selection_audit)
        write_json(metadata_dir / "bucket_stats.json", bucket_stats)

        logger.info("=== SELECTED CLIPS (chronological) ===")
        for j, c in enumerate(selected_sorted, 1):
            rs = c.get("scores", {})
            logger.info(
                f"#{j:02d} {c['id']} | VS={rs.get('viral_score',0):.1f} | {c['start']:.2f}-{c['end']:.2f} ({c['duration']:.1f}s)"
            )
        logger.info(f"Selected clips: {len(selected_sorted)}")

    state["SELECTED"] = selected_sorted
    log_flush()
    return state


def snap_selected(state: Dict[str, Any]) -> Dict[str, Any]:
    silence_segments = state.get("SILENCE_SEGMENTS", [])
    shot_cuts = state.get("SHOT_CUTS", [])
    analyzed_duration = float(state.get("ANALYZED_DURATION", 0.0))
    min_clip_sec = float(state.get("MIN_CLIP_SEC", 18.0))
    max_clip_sec = float(state.get("MAX_CLIP_SEC", 60.0))
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

    with StageTimer(11, "Snap boundaries + end-of-idea polish"):
        if state.get("_SNAP_DONE"):
            logger.info("Stage 11 already completed; skipping duplicate run")
            log_flush()
            return state

        transcripts = state.get("TRANSCRIPTS", {})
        if not isinstance(transcripts, dict):
            transcripts = {}

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
                    if out.get("words"):
                        _add_reason(c, "asr:selected_full")
                    else:
                        _add_reason(c, "asr:selected_full_empty")
                except Exception as e:
                    logger.warning(f"Selected-only ASR failed for {cid}: {e}")
                    _add_reason(c, "asr:selected_full_failed")

        for c in state.get("SELECTED", []):
            cid = c.get("id")
            s = float(c["start"])
            e = float(c["end"])

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

            s2 = max(0.0, float(s2))
            e2 = min(float(analyzed_duration), float(e2))
            if (e2 - s2) < min_clip_sec:
                logger.warning(f"Snap too short for {cid}; keeping original bounds")
                _add_reason(c, "snap:too_short_keep_original")
                continue

            hard_cap = min(float(analyzed_duration), float(s2 + max_clip_sec))
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
            if dur > max_clip_sec:
                _add_reason(c, f"clamp:max_sec({max_clip_sec:.1f})")
                hard_end = float(s2 + max_clip_sec)
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

            c["start"] = float(s2)
            c["end"] = float(e2)
            c["duration"] = float(e2 - s2)

        write_json(metadata_dir / "selected_snapped.json", state.get("SELECTED", []))
        logger.info("Snapping completed (end-of-idea polish enabled).")

        try:
            write_json(Path(state["SCORED_SEGMENTS_DIR"]) / "transcript.json", transcripts)
            logger.info(f"Saved transcript cache after snapping (entries={len(transcripts)})")
        except Exception as e:
            logger.warning(f"Failed to save transcript cache after snapping: {e}")

        state["TRANSCRIPTS"] = transcripts
        state["_SNAP_DONE"] = True
        log_flush()

    return state
