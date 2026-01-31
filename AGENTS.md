# AGENTS.md
Agent Version: v2-contestgrade-editorgrade

## 0) Mission
Produce short clips optimized for recommendation systems (TikTok/Reels/Shorts) by maximizing:
- Average Watch Time (AWT)
- Retention curve stability (minimize early drop + mid drop)
- Completion and rewatch probability
- Viewer-perceived value (clarity, meaning, payoff)

Duration is an OUTCOME of retention optimization, not a fixed target.

## 1) Non-Negotiable Invariants (Violation = ERROR)
1) CAPTION & HASHTAGS ISOLATION
   - Captions/hashtags/metadata MUST NOT affect candidate mining, scoring, or selection.
   - They may be generated only AFTER final selection as a separate step.

2) SPEECH_BLOCKS FIRST
   - SPEECH_BLOCKS MUST be constructed BEFORE candidate mining.
   - Primary: silence complement (speech between detected silences) with padding.
   - Fallback: energy/VAD speech segmentation if silence detection is unreliable.
   - If both fail: abort with explicit ERROR and diagnostics.

3) SEMANTIC BEFORE FINAL SCORING
   - The pipeline MUST run lightweight ASR for Top-N candidates PER time bucket BEFORE final scoring.
   - Selected-only ASR is NOT allowed.
   - If ASR unavailable: switch to SEMANTIC_FALLBACK mode with explicit log + reduced confidence.

4) TIME FAIRNESS (ANTI EARLY-ENERGY BIAS)
   - Split video into 5 buckets by time: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%.
   - Must select >= 1 clip from 80-100% IF any candidate in that bucket passes meaning_min.
   - Scoring MUST be bucket-normalized (percentile/z-score within bucket).

5) RETENTION-FIRST DURATION
   - No fixed duration forcing.
   - Prefer end-of-idea / natural stopping points:
     - long pause, punctuation boundary, topic shift, payoff delivered
   - Hard caps are LAST fallback only.

6) THRESHOLDS MUST BE RELATIVE, NOT ABSOLUTE GLOBAL
   - Any gating MUST be per-bucket percentile/z-score based.

7) DETERMINISTIC & AUDITABLE
   - Deterministic given same inputs + seed.
   - Every selected/rejected candidate MUST have reason codes logged.

## 2) Macro Pipeline Order (Must Match Logs)
Stage 00: Initialization (paths, versions, seed, agent contract)
Stage 01: Ingest Video (resolve + duration)
Stage 02: Audio Extract + Global Analysis (silence + energy)
Stage 03: Speech Blocks (silence complement + VAD fallback)
Stage 04: Visual Sampling + Shot Detection (optional)
Stage 05: Hard Lock (NO FACE / NO CROP / NO ZOOM; no subtitles)
Stage 06: Candidate Mining (multi-source)
Stage 07: Context Expansion (anti contextless start)
Stage 08: Feature Extraction
Stage 09: Lightweight ASR Top-N per bucket (or SEMANTIC_FALLBACK)
Stage 10: Bucket-Normalized Scoring
Stage 11: Selection (quota + fairness) + audit
Stage 11.5: Snap boundaries + end-of-idea polish (sub-step, not a StageTimer id)
Stage 12: Export clips + thumbs + final metadata

Logs MUST appear in stage order.

## 3) Candidate Discovery (Multi-Source, No Peak-Only Dominance)
Candidates MUST be generated from multiple sources:
A) Speech-block anchored candidates (primary)
B) Peak-centric candidates (secondary; energy spikes)
C) Structure candidates (if ASR available): Q/A, punchline markers, discourse cues
D) Topic drift candidates (optional)

A candidate pool must exist per bucket. If near-zero candidates in a bucket, log WHY.

## 4) Context Expansion (Anti Contextless Start)
For each candidate:
- Expand start backward by ~2-8 seconds IF within speech and improves comprehension.
- Do not expand into silence-only segments.
- Log start_adjustment_sec and rationale.

## 5) Editorial Quality Rules (World-Class Cut)
A clip is "editorially valid" ONLY if:
- Starts at a point that makes sense without missing premise
- Avoids mid-word/mid-sentence cut
- Has a clear mini-arc: hook -> development -> payoff/point
- Ends cleanly (end-of-idea or natural pause)

Loud but incoherent clips MUST be penalized.

## 6) Feature Requirements (Minimum Set)
Final score MUST combine:
A) Meaning Score (semantic; requires transcript if available)
B) Hook Score (retention early)
C) Clarity Score (editorial continuity + cut cleanliness)
D) Energy Score (secondary; must not dominate)

If transcript missing:
- Meaning features MUST NOT be faked.
- Switch to SEMANTIC_FALLBACK and downweight semantic components with explicit logs.

## 7) Scoring & Normalization
- Normalize within each bucket (z-score/percentile).
- Global score computed AFTER bucket normalization.
- Any gating uses bucket-normalized metrics.

## 8) Selection Logic (Deterministic Quota -> Fill)
Step 1: Per-bucket quota selection
Step 2: Global fill (after quota)
Step 3: Overlap/min-gap enforcement
Step 4: Boundary snapping + end-of-idea polish (log end_reason)

## 9) Duration Strategy (Retention-First Soft Preferences)
No hard target durations.
Soft preference only if clip stays dense & coherent and ends naturally between ~45-90s.

End reason MUST be logged:
IDEA_COMPLETE / LONG_PAUSE / TOPIC_SHIFT / DENSITY_DROP / HARDCAP_FALLBACK

## 10) Required Outputs (Audit Contract)
Default output is run-scoped:
- OUT_DIR = outputs/runs/<RUN_ID>
If RUN_SCOPED_OUTPUT=0, OUT_DIR = outputs

MUST write under OUT_DIR:
- OUT_DIR/00_raw_segments/
  - audio.wav, silence_segments.json, energy_curve.json, speech_blocks.json, candidates.json, video_meta.json, shot_cuts.json (if enabled)
- OUT_DIR/01_scored_segments/
  - transcript.json (ASR cache; may be partial), scored_candidates.json
- OUT_DIR/02_selected_clips/
  - exported .mp4 clips (MUST NOT be empty if run succeeded)
- OUT_DIR/03_thumbnails/
  - thumbnails for each selected clip
- OUT_DIR/04_metadata/
  - agent_meta.json
  - ranking.csv (all candidates, component scores, bucket id, normalization stats)
  - selected_ranking.csv
  - selection_audit.json (each candidate: selected?, reason_codes[], end_reason, key scores)
  - bucket_stats.json
  - selected.json, selected_snapped.json (if snapping)
  - export_manifest.json
  - run_manifest.json
  - run_complete.json
- OUT_DIR/logs/pipeline.log
  - stage-ordered logs, timings, warnings

## 11) Reason Codes (Minimum Set)
Selection/rejection MUST include reason codes (multiple allowed):
- SEL_QUOTA
- SEL_TOP_SCORE
- SEL_RESCUE_LATE
- REJ_INVALID_INTERVAL
- REJ_MEANING_BELOW_MIN
- REJ_SCORE_BELOW_MIN
- REJ_OVERLAP
- REJ_MIN_GAP
- REJ_BUCKET_QUOTA_FULL
- REJ_ASR_UNAVAILABLE
- REJ_SEGMENTATION_FAILED

## 12) Quality Gates (Automatic Fail Conditions)
Mark run FAILED if:
- SPEECH_BLOCKS missing/empty without fallback
- feature faking detected (e.g., constant novelty when transcript missing)
- bucket 80-100% quota missed while meaning candidates exist
- selection_audit missing reason codes
- export produced zero mp4 clips in OUT_DIR/02_selected_clips/

## 13) Reproducibility
Log:
- seed, versions (python, ffmpeg, ffprobe, git_sha), ASR model name
- deterministic tie-break rules (seeded)
- parameters used for snapping and selection
