from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import load_cached_json, write_cache_meta, write_json
from .logging_utils import StageTimer, log_flush, log_i, log_warn

logger = logging.getLogger("viralshort")


TIME_BUCKETS = 5


def split_ranges(start, end, max_len, overlap):
    ranges = []
    t = start
    while t < end - 0.1:
        t2 = min(end, t + max_len)
        ranges.append((t, t2))
        if t2 >= end:
            break
        t = t2 - overlap if overlap > 0 else t2
    return ranges


def safe_unlink(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _bucket_index(t, analyzed_duration, time_buckets: int):
    if analyzed_duration <= 0:
        return 0
    idx = int((t / analyzed_duration) * time_buckets)
    if idx < 0:
        idx = 0
    if idx >= time_buckets:
        idx = time_buckets - 1
    return idx


def _percentile_rank(values, x):
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


def _pre_score(c, hook_vals, ptm_vals, spk_vals, sil_vals):
    f = c.get("features", {})
    p_hook = _percentile_rank(hook_vals, f.get("hook_energy", 0.0))
    p_ptm = _percentile_rank(ptm_vals, f.get("peak_to_mean", 0.0))
    p_spk = _percentile_rank(spk_vals, f.get("spike_rate", 0.0))
    p_sil = _percentile_rank(sil_vals, f.get("silence_ratio", 0.0))
    return (0.40 * p_hook) + (0.30 * p_ptm) + (0.20 * p_spk) - (0.20 * p_sil)


def _light_window(start, end, asr_light_sec, asr_light_offset):
    dur = max(0.0, end - start)
    if dur <= asr_light_sec:
        return start, end
    s = min(end - 0.1, start + asr_light_offset)
    e = min(end, s + asr_light_sec)
    if e - s < 4.0:
        mid = start + dur * 0.5
        s = max(start, mid - asr_light_sec * 0.5)
        e = min(end, s + asr_light_sec)
    return s, e


def run_asr(state: Dict[str, Any]) -> Dict[str, Any]:
    analyzed_duration = float(state.get("ANALYZED_DURATION", 0.0))
    time_buckets = int(state.get("TIME_BUCKETS", TIME_BUCKETS))
    asr_topn_per_bucket = int(state.get("ASR_TOPN_PER_BUCKET", os.getenv("ASR_TOPN_PER_BUCKET", "4")))
    asr_model_name = state.get("ASR_MODEL_NAME", os.getenv("ASR_MODEL_NAME", "tiny"))
    asr_beam_size = int(state.get("ASR_BEAM_SIZE", os.getenv("ASR_BEAM_SIZE", "1")))
    asr_light_mode = bool(state.get("ASR_LIGHT_MODE", os.getenv("ASR_LIGHT_MODE", "1") == "1"))
    asr_light_sec = float(state.get("ASR_LIGHT_SEC", os.getenv("ASR_LIGHT_SEC", "10.0")))
    asr_light_offset = float(state.get("ASR_LIGHT_OFFSET", os.getenv("ASR_LIGHT_OFFSET", "2.0")))

    max_asr_block_sec = float(state.get("MAX_ASR_BLOCK_SEC", 28.0))
    asr_block_overlap_sec = float(state.get("ASR_BLOCK_OVERLAP_SEC", 0.25))
    asr_enabled = bool(state.get("ASR_ENABLED", True))
    asr_language = state.get("ASR_LANGUAGE", "id")
    ffmpeg_bin = state.get("FFMPEG_BIN", "ffmpeg")
    cache_dir = Path(state["CACHE_DIR"])
    scored_dir = Path(state["SCORED_SEGMENTS_DIR"])
    audio_wav = Path(state["AUDIO_WAV"])
    trigger_words = state.get("TRIGGER_WORDS", [])

    with StageTimer(9, "ASR (top-N per bucket, cache, fail-safe)"):
        transcripts_by_id: Dict[str, Any] = {}
        transcript_cache: Dict[str, Any] = {}
        transcript_cache_path = scored_dir / "transcript.json"
        cache_meta = state.get("CACHE_META") or {}

        cached, hit, reason = load_cached_json(transcript_cache_path, cache_meta)
        if hit and isinstance(cached, dict):
            transcript_cache = cached
            log_i(logger, f"CACHE_HIT transcript ({len(transcript_cache)} items)")
        else:
            log_warn(logger, f"CACHE_MISS transcript ({reason}) -> recompute")

        asr_available = False
        model = None
        max_retries = int(state.get("ASR_MODEL_MAX_RETRIES", 1))
        if asr_enabled:
            for attempt in range(max_retries + 1):
                try:
                    try:
                        from faster_whisper import WhisperModel
                    except Exception as e:
                        logger.warning(f"faster-whisper not available ({e}).")
                        raise

                    model = WhisperModel(asr_model_name, device="cpu", compute_type="int8")
                    asr_available = True
                    logger.info(f"ASR model ready: faster-whisper {asr_model_name} (cpu int8)")
                    break
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"ASR init failed (attempt {attempt+1}/{max_retries+1}): {e}")
                        time.sleep(0.2)
                    else:
                        logger.warning(f"ASR unavailable; continuing without transcript. ({e})")
                        asr_available = False

        def transcribe_candidate_abs(candidate: Dict[str, Any], mode: str = "full") -> Dict[str, Any]:
            if not asr_available or model is None:
                return {
                    "text": "",
                    "words": [],
                    "markers_abs": [],
                    "words_per_sec": 0.0,
                    "trigger_count": 0,
                    "word_count": 0,
                    "mode": mode,
                }

            cid = candidate["id"]
            start = float(candidate["start"])
            end = float(candidate["end"])

            if mode == "light":
                ls, le = _light_window(start, end, asr_light_sec, asr_light_offset)
                ranges = split_ranges(ls, le, max_asr_block_sec, asr_block_overlap_sec)
            else:
                ranges = split_ranges(start, end, max_asr_block_sec, asr_block_overlap_sec)

            all_words = []
            full_text = []
            markers_abs = []

            for bi, (bs, be) in enumerate(ranges, 1):
                block_wav = cache_dir / f"asrblock_{cid}_{bi:02d}.wav"
                subprocess.run(
                    [
                        ffmpeg_bin,
                        "-y",
                        "-loglevel",
                        "error",
                        "-ss",
                        str(bs),
                        "-t",
                        str(be - bs),
                        "-i",
                        str(audio_wav),
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(block_wav),
                    ],
                    check=True,
                )

                t0 = time.monotonic()
                logger.info(f"ASR [{cid}] block {bi}/{len(ranges)} START {bs:.2f}-{be:.2f} (mode={mode})")
                log_flush()

                segments, info = model.transcribe(
                    str(block_wav), language=asr_language, beam_size=asr_beam_size
                )
                for seg in segments:
                    if seg.text:
                        full_text.append(seg.text)
                    for w in (seg.words or []):
                        all_words.append(
                            {"word": w.word, "start": float(w.start + bs), "end": float(w.end + bs)}
                        )

                dt = time.monotonic() - t0
                logger.info(f"ASR [{cid}] block {bi} DONE in {dt:.2f}s")
                log_flush()

                safe_unlink(block_wav)

            trigger_set = set([t.lower() for t in (trigger_words or [])])
            for w in all_words:
                tok = (w.get("word") or "").strip().lower()
                if tok in trigger_set:
                    markers_abs.append(float(w.get("start", 0.0)))

            text = " ".join(full_text).strip()
            word_count = len(all_words)
            if mode == "full":
                dur = max(0.1, float(end - start))
            else:
                dur = max(0.1, float(ranges[-1][1] - ranges[0][0]))
            wps = float(word_count / dur)
            trigger_count = len(markers_abs)

            return {
                "text": text,
                "words": all_words,
                "markers_abs": markers_abs,
                "words_per_sec": wps,
                "trigger_count": trigger_count,
                "word_count": word_count,
                "mode": mode,
            }

        candidates = state.get("CANDIDATES", [])
        hook_vals = [c.get("features", {}).get("hook_energy", 0.0) for c in candidates]
        ptm_vals = [c.get("features", {}).get("peak_to_mean", 0.0) for c in candidates]
        spk_vals = [c.get("features", {}).get("spike_rate", 0.0) for c in candidates]
        sil_vals = [c.get("features", {}).get("silence_ratio", 0.0) for c in candidates]

        bucket_map = {b: [] for b in range(time_buckets)}
        for c in candidates:
            b = _bucket_index(float(c.get("start", 0.0)), analyzed_duration, time_buckets)
            bucket_map[b].append(c)

        top_for_asr = []
        for b, arr in bucket_map.items():
            arr_sorted = sorted(
                arr,
                key=lambda c: _pre_score(c, hook_vals, ptm_vals, spk_vals, sil_vals),
                reverse=True,
            )
            top_for_asr.extend(arr_sorted[:asr_topn_per_bucket])

        if not asr_available:
            logger.warning("SEMANTIC_FALLBACK: ASR unavailable")
        else:
            logger.info(
                f"ASR top-N per bucket: {len(top_for_asr)} candidates (mode={'light' if asr_light_mode else 'full'})"
            )

        video_hash = (state.get("VIDEO_HASH") or "").strip()
        key_by_id: Dict[str, str] = {}

        def _transcript_cache_key(candidate: Dict[str, Any]) -> str:
            s = float(candidate.get("start", 0.0))
            e = float(candidate.get("end", 0.0))
            params = f"{asr_model_name}|{asr_beam_size}|{asr_light_mode}|{asr_light_sec}|{asr_light_offset}|{asr_language}"
            return f"{video_hash}:{s:.2f}-{e:.2f}:{params}"

        for c in top_for_asr:
            cid = c["id"]
            cache_key = _transcript_cache_key(c)
            key_by_id[cid] = cache_key
            existing = transcript_cache.get(cache_key, {}) if isinstance(transcript_cache, dict) else {}
            if existing.get("text") and existing.get("mode") == "full":
                out = existing
            else:
                try:
                    mode = "light" if asr_light_mode else "full"
                    out = transcribe_candidate_abs(c, mode=mode)
                except Exception as e:
                    logger.warning(f"ASR failed for {cid}: {e}")
                    out = {
                        "text": "",
                        "words": [],
                        "markers_abs": [],
                        "words_per_sec": 0.0,
                        "trigger_count": 0,
                        "word_count": 0,
                        "mode": "light",
                    }
            out["id"] = cid
            out["start"] = float(c.get("start", 0.0))
            out["end"] = float(c.get("end", 0.0))
            out["video_hash"] = video_hash
            transcript_cache[cache_key] = out
            transcripts_by_id[cid] = out

            f = c.get("features", {})
            f["words_per_sec"] = float(out.get("words_per_sec", 0.0))
            f["trigger_count"] = int(out.get("trigger_count", 0))
            f["markers_abs"] = list(out.get("markers_abs", []))
            f["word_count"] = int(out.get("word_count", 0))
            # Mark has_text only if there are words (word_count > 0)
            f["has_text"] = True if int(out.get("word_count", 0)) > 0 else False
            c["features"] = f

        # Ensure every candidate has ASR feature fields populated (Clause E)
        # Candidates outside top_for_asr may not have been transcribed; set defaults
        try:
            all_candidates = state.get("CANDIDATES", [])
            for cand in all_candidates:
                cid2 = cand.get("id")
                # if transcript missing, set dummy entry and zero-valued features
                if cid2 not in transcripts_by_id:
                    # create empty transcript entry
                    transcripts_by_id[cid2] = {
                        "id": cid2,
                        "start": float(cand.get("start", 0.0)),
                        "end": float(cand.get("end", 0.0)),
                        "video_hash": (state.get("VIDEO_HASH") or "").strip(),
                        "text": "",
                        "words": [],
                        "markers_abs": [],
                        "words_per_sec": 0.0,
                        "trigger_count": 0,
                        "word_count": 0,
                        "mode": "light",
                    }
                f2 = cand.get("features", {}) or {}
                # assign default feature values if not present
                f2.setdefault("words_per_sec", 0.0)
                f2.setdefault("word_count", 0)
                f2.setdefault("trigger_count", 0)
                f2.setdefault("markers_abs", [])
                f2.setdefault("has_text", False)
                cand["features"] = f2
        except Exception:
            pass

        try:
            write_json(transcript_cache_path, transcript_cache)
            if cache_meta:
                write_cache_meta(transcript_cache_path, cache_meta)
            logger.info(f"Saved transcript cache (entries={len(transcript_cache)})")
        except Exception as e:
            logger.warning(f"Failed to save transcript cache: {e}")

        state["TRANSCRIPTS"] = transcripts_by_id
        state["TRANSCRIPT_CACHE"] = transcript_cache
        state["TRANSCRIPT_KEY_BY_ID"] = key_by_id
        state["ASR_AVAILABLE"] = asr_available
        state["transcribe_candidate_abs"] = transcribe_candidate_abs

    return state
