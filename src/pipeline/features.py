from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .logging_utils import StageTimer

logger = logging.getLogger("viralshort")


def percentile_value(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = int(round(q * (len(sorted_vals) - 1)))
    return float(sorted_vals[max(0, min(len(sorted_vals) - 1, idx))])


def window_points(energy_curve, start, end):
    return [e for e in energy_curve if start <= e["time"] <= end]


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def compute_features(candidate: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    s = candidate["start"]
    e = candidate["end"]
    dur = e - s

    energy_curve = state.get("ENERGY_CURVE", [])
    silence_segments = state.get("SILENCE_SEGMENTS", [])
    shot_cuts = state.get("SHOT_CUTS", [])
    hook_window_sec = float(state.get("HOOK_WINDOW_SEC", 5.0))

    pts = window_points(energy_curve, s, e)
    if not pts:
        pts = [{"time": s, "rms": 0.0}]
    rms = np.array([p["rms"] for p in pts], dtype=np.float32)
    mean = float(rms.mean())
    peak = float(rms.max())
    std = float(rms.std())
    ptm = float(peak / (mean + 1e-9))

    hook_pts = window_points(energy_curve, s, min(e, s + hook_window_sec))
    hook_rms = (
        np.array([p["rms"] for p in hook_pts], dtype=np.float32)
        if hook_pts
        else np.array([0.0], dtype=np.float32)
    )
    hook_energy = float(hook_rms.mean())

    early_pts = window_points(energy_curve, s, min(e, s + dur * 0.25))
    late_pts = window_points(energy_curve, max(s, e - dur * 0.25), e)
    early_energy = float(np.mean([p["rms"] for p in early_pts])) if early_pts else mean
    late_energy = float(np.mean([p["rms"] for p in late_pts])) if late_pts else mean

    peak_pt = max(pts, key=lambda x: x["rms"])
    peak_time_abs = float(peak_pt["time"])
    peak_offset_in_clip = float(peak_time_abs - s)

    all_rms_sorted = sorted([p["rms"] for p in energy_curve] or [0.0])
    p90 = percentile_value(all_rms_sorted, 0.90)
    spikes = 0
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        if b["rms"] >= p90 and b["rms"] >= a["rms"] and b["rms"] >= c["rms"]:
            spikes += 1
    spike_rate = float(spikes / (dur + 1e-6))

    sil = 0.0
    for (ss, se) in silence_segments:
        inter = max(0.0, min(e, se) - max(s, ss))
        sil += inter
    silence_ratio = float(sil / (dur + 1e-6))

    if shot_cuts:
        near_cut = float(min(abs(c - s) for c in shot_cuts + [s]) if shot_cuts else 9999.0)
        near_cut_end = float(min(abs(c - e) for c in shot_cuts + [e]) if shot_cuts else 9999.0)
        near_cut_dist = float(min(near_cut, near_cut_end))
    else:
        near_cut_dist = 9999.0

    return {
        "mean_energy": mean,
        "peak_energy": peak,
        "energy_stddev": std,
        "peak_to_mean": ptm,
        "hook_energy": hook_energy,
        "early_energy": early_energy,
        "late_energy": late_energy,
        "peak_time_abs": peak_time_abs,
        "peak_offset_in_clip": peak_offset_in_clip,
        "spike_rate": spike_rate,
        "silence_ratio": silence_ratio,
        "near_cut_dist": near_cut_dist,
        "words_per_sec": 2.0,
        "trigger_count": 0,
        "markers_abs": [],
        "start_abs_for_scoring": s,
        "end_abs_for_scoring": e,
    }


def run_feature_extraction(state: Dict[str, Any]) -> Dict[str, Any]:
    with StageTimer(7, "Feature Extraction"):
        for c in state.get("CANDIDATES", []):
            c["features"] = compute_features(c, state)
        logger.info("Features computed for all candidates.")
    return state
