from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict

from .io_utils import write_json
from .logging_utils import StageTimer, log_flush

logger = logging.getLogger("viralshort")

TIME_BUCKETS = 5


def token_set(text: str):
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def clip100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def percentile_rank(values, x):
    if not values:
        return 0.0
    vs = sorted(values)
    lo, hi = 0, len(vs)
    while lo < hi:
        mid = (lo + hi) // 2
        if vs[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo / len(vs)


def bucket_index(t, analyzed_duration):
    if analyzed_duration <= 0:
        return 0
    idx = int((t / analyzed_duration) * TIME_BUCKETS)
    if idx < 0:
        idx = 0
    if idx >= TIME_BUCKETS:
        idx = TIME_BUCKETS - 1
    return idx


def finishability_score(end_t: float, silence_segments, speech_blocks) -> float:
    if silence_segments:
        for s, e in (silence_segments or []):
            s = float(s)
            if end_t <= s <= (end_t + 1.6):
                return 100.0
    if speech_blocks:
        best = None
        for s, e in (speech_blocks or []):
            e = float(e)
            if e >= end_t:
                d = e - end_t
                if best is None or d < best:
                    best = d
        if best is not None:
            if best <= 1.0:
                return 85.0
            if best <= 2.5:
                return 55.0
    return 20.0


def run_scoring(state: Dict[str, Any]) -> Dict[str, Any]:
    analyzed_duration = float(state.get("ANALYZED_DURATION", 0.0))
    metadata_dir = Path(state["METADATA_DIR"])
    scored_dir = Path(state["SCORED_SEGMENTS_DIR"])
    candidates = state.get("CANDIDATES", [])
    transcripts = state.get("TRANSCRIPTS", {})
    silence_segments = state.get("SILENCE_SEGMENTS", [])
    speech_blocks = state.get("SPEECH_BLOCKS", [])
    hook_window_sec = float(state.get("HOOK_WINDOW_SEC", 5.0))

    with StageTimer(9, "Scoring (bucket-normalized)"):
        buckets = {b: [] for b in range(TIME_BUCKETS)}
        for c in candidates:
            b = bucket_index(float(c.get("start", 0.0)), analyzed_duration)
            buckets[b].append(c)

        def build_vals(key, transform=lambda x: x):
            vals = {b: [] for b in buckets}
            for b, arr in buckets.items():
                for c in arr:
                    v = transform(c.get("features", {}).get(key, 0.0))
                    vals[b].append(float(v))
            return vals

        hook_vals = build_vals("hook_energy")
        ptm_vals = build_vals("peak_to_mean")
        std_vals = build_vals("energy_stddev")
        spk_vals = build_vals("spike_rate")
        wps_vals = build_vals("words_per_sec")
        sil_vals = build_vals("silence_ratio")
        trig_vals = build_vals("trigger_count")

        novelty_vals = {b: [] for b in buckets}
        for b, arr in buckets.items():
            seen = []
            arr_sorted = sorted(arr, key=lambda c: -c.get("features", {}).get("hook_energy", 0.0))
            for c in arr_sorted:
                txt = ""
                if isinstance(transcripts, dict):
                    txt = transcripts.get(c.get("id"), {}).get("text", "")
                tokens = token_set(txt)
                max_sim = 0.0
                for prev in seen[:8]:
                    if tokens or prev:
                        max_sim = max(max_sim, len(tokens & prev) / max(1, len(tokens | prev)))
                novelty = clip100((1.0 - max_sim) * 100) if tokens else 50.0
                c.setdefault("scores", {})
                c["scores"]["novelty"] = novelty
                novelty_vals[b].append(float(novelty))
                seen.append(tokens)

        semantic_available = False
        if isinstance(transcripts, dict):
            semantic_available = any(
                (transcripts.get(c.get("id"), {}).get("text", "") or "").strip()
                for c in candidates
            )

        if not semantic_available:
            logger.warning("SEMANTIC_FALLBACK: no transcripts for candidates")

        for c in candidates:
            f = c.get("features", {})
            b = bucket_index(float(c.get("start", 0.0)), analyzed_duration)

            p_hook = percentile_rank(hook_vals[b], f.get("hook_energy", 0.0))
            p_ptm = percentile_rank(ptm_vals[b], f.get("peak_to_mean", 0.0))
            p_std = percentile_rank(std_vals[b], f.get("energy_stddev", 0.0))
            p_spk = percentile_rank(spk_vals[b], f.get("spike_rate", 0.0))
            p_wps = percentile_rank(wps_vals[b], f.get("words_per_sec", 0.0))
            p_trig = percentile_rank(trig_vals[b], f.get("trigger_count", 0.0))
            p_sil = percentile_rank(sil_vals[b], f.get("silence_ratio", 0.0))
            p_nov = percentile_rank(novelty_vals[b], c.get("scores", {}).get("novelty", 50.0))

            if semantic_available and f.get("has_text", False):
                meaning = clip100(100 * (0.45 * p_trig + 0.35 * p_wps + 0.20 * p_nov))
                semantic_mode = "semantic"
            elif semantic_available:
                meaning = 20.0
                semantic_mode = "no_text"
            else:
                meaning = clip100(100 * (0.60 * p_hook + 0.40 * p_spk))
                semantic_mode = "fallback"

            early_marker = 0.0
            start = f.get("start_abs_for_scoring")
            if start is not None:
                ms = f.get("markers_abs", []) or []
                early_marker = 1.0 if any(start <= m <= start + hook_window_sec for m in ms) else 0.0
            hook = clip100(100 * (0.60 * p_hook + 0.25 * early_marker + 0.15 * p_wps))

            energy = clip100(100 * (0.40 * p_ptm + 0.35 * p_std + 0.25 * p_spk))

            fin = finishability_score(float(c.get("end", 0.0)), silence_segments, speech_blocks)
            fin_n = fin / 100.0
            clarity = clip100(100 * (0.55 * fin_n + 0.30 * (1.0 - p_sil) + 0.15 * p_wps))

            if semantic_available:
                viral = (0.35 * meaning) + (0.25 * hook) + (0.20 * clarity) + (0.20 * energy)
            else:
                viral = (0.20 * meaning) + (0.30 * hook) + (0.25 * clarity) + (0.25 * energy)

            c["scores"] = {
                "meaning": meaning,
                "hook": hook,
                "clarity": clarity,
                "energy": energy,
                "novelty": c.get("scores", {}).get("novelty", 50.0),
                "finishability": fin,
                "viral_score": clip100(viral),
                "semantic_mode": semantic_mode,
            }

        def editorial_reason(c):
            s = c.get("scores", {})
            return [
                f"Meaning {s.get('meaning',0):.1f} ({s.get('semantic_mode','')})",
                f"Hook {s.get('hook',0):.1f}",
                f"Clarity {s.get('clarity',0):.1f}",
                f"Energy {s.get('energy',0):.1f}",
                f"Novelty {s.get('novelty',0):.1f}",
            ]

        for c in candidates:
            c["editorial_reason"] = editorial_reason(c)

        ranking_csv = Path(metadata_dir) / "ranking.csv"
        with ranking_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "id",
                "start",
                "end",
                "duration",
                "bucket",
                "meaning",
                "hook",
                "clarity",
                "energy",
                "novelty",
                "finishability",
                "viral_score",
                "semantic_mode",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for c in sorted(
                candidates,
                key=lambda x: (-x.get("scores", {}).get("viral_score", 0.0), x.get("start", 0.0)),
            ):
                s = c.get("scores", {})
                w.writerow(
                    {
                        "id": c.get("id"),
                        "start": float(c.get("start", 0.0)),
                        "end": float(c.get("end", 0.0)),
                        "duration": float(c.get("duration", 0.0)),
                        "bucket": bucket_index(float(c.get("start", 0.0)), analyzed_duration),
                        "meaning": float(s.get("meaning", 0.0)),
                        "hook": float(s.get("hook", 0.0)),
                        "clarity": float(s.get("clarity", 0.0)),
                        "energy": float(s.get("energy", 0.0)),
                        "novelty": float(s.get("novelty", 0.0)),
                        "finishability": float(s.get("finishability", 0.0)),
                        "viral_score": float(s.get("viral_score", 0.0)),
                        "semantic_mode": s.get("semantic_mode", ""),
                    }
                )

        write_json(scored_dir / "scored_candidates.json", candidates)
        logger.info(f"Wrote ranking.csv (all candidates): {ranking_csv}")
        log_flush()

    state["CANDIDATES"] = candidates
    return state
