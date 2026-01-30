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
   - Selected-only ASR is NOT allowed (semantic features cannot be dummy).
   - If ASR unavailable: switch to SEMANTIC_FALLBACK mode with reduced confidence and explicit log.

4) TIME FAIRNESS (ANTI EARLY-ENERGY BIAS)
   - Split video into 5 buckets by time: 0–20%, 20–40%, 40–60%, 60–80%, 80–100%.
   - Must select >= 1 clip from 80–100% IF any candidate in that bucket passes meaning_min.
   - Scoring MUST be bucket-normalized (percentile/z-score within bucket).

5) RETENTION-FIRST DURATION
   - No fixed duration forcing (no hard “must be 30/60/120”).
   - Prefer end-of-idea / natural stopping points:
     - long pause, punctuation boundary, topic shift, payoff delivered
   - Hard caps are LAST fallback only.

6) THRESHOLDS MUST BE RELATIVE, NOT ABSOLUTE GLOBAL
   - Any “strong_enough” gating MUST be based on per-bucket percentiles/z-scores, not absolute global thresholds.

7) DETERMINISTIC & AUDITABLE
   - Selection MUST be deterministic given the same inputs + seed.
   - Every selected/rejected candidate MUST have reason codes logged.

## 2) Macro Pipeline Order (Must Match Logs)
Stage 0: Initialization (paths, versions, seed, agent contract)
Stage 1: Audio extraction + normalization (log loudness stats)
Stage 2: Silence/VAD analysis
Stage 3: SPEECH_BLOCKS construction (required)
Stage 4: Candidate discovery (multi-source, multi-scale)
Stage 5: Context expansion (anti contextless start)
Stage 6: Lightweight ASR for Top-N per bucket
Stage 7: Feature extraction (energy + semantic + structure)
Stage 8: Bucket-normalized scoring
Stage 9: Quota selection (Top-k per bucket, deterministic)
Stage 10: Global fill + overlap/MIN_GAP
Stage 11: Boundary snap + end-of-idea polish
Stage 12: Export outputs + audit artifacts

Logs MUST appear in stage order. Parallel steps must be explicitly labeled.

## 3) Candidate Discovery (Multi-Source, No Peak-Only Dominance)
Candidates MUST be generated from multiple sources:
A) Speech-block anchored candidates (primary)
B) Peak-centric candidates (secondary, energy spikes)
C) Structure candidates (if ASR available): Q/A, punchline markers, “intinya…” / “jadi…” / “kesimpulannya…”
D) Topic drift candidates (optional): keyword drift within bucket

A candidate pool must exist per bucket. If a bucket has near-zero candidates, log WHY.

## 4) Context Expansion (Anti Contextless Start)
For each candidate:
- Expand start backward by 2–6 seconds IF within speech and improves comprehension.
- Do not expand into silence-only segments.
- Log start_adjustment_sec and rationale.

## 5) Editorial Quality Rules (World-Class Cut)
A clip is “editorially valid” ONLY if:
- Starts at a point that makes sense without missing premise
- Avoids mid-word/mid-sentence cut
- Has a clear mini-arc: hook → development → payoff/point
- Ends cleanly (end-of-idea or natural pause)

Clips that are loud but incoherent MUST be penalized.

## 6) Feature Requirements (Minimum Set)
Final score MUST combine these components:

A) Meaning Score (semantic; requires transcript if available)
- words_per_sec (WPS) and content density
- marker/punchline detection (discourse markers; Q/A patterns)
- novelty (text-based similarity vs other candidates)
- “payoff likelihood” features (resolution cues)

B) Hook Score (retention early)
- semantic hook in first 1–3 seconds (not only energy spike)
- early_marker presence

C) Clarity Score (editorial)
- speech continuity
- cut cleanliness (boundary confidence)
- low filler ratio (uh/um, repeated phrases) if transcript available

D) Energy Score (secondary)
- RMS/peak/spike
- BUT energy MUST NOT dominate without meaning/clarity

If transcript missing:
- Meaning-related features must not be faked (no constant novelty=100).
- Switch to SEMANTIC_FALLBACK and downweight meaning features with explicit logs.

## 7) Scoring & Normalization
- Each bucket computes local statistics and normalizes candidate scores (z-score or percentile).
- Global score is computed AFTER bucket normalization to prevent early energy bias.
- Any gating thresholds must reference bucket-normalized metrics.

## 8) Selection Logic (Deterministic Quota → Fill)
Step 1: Per-bucket selection
- Select Top-K candidates per bucket (K depends on desired total).
- Enforce: 80–100% bucket >=1 clip if meaning candidates exist.

Step 2: Global fill
- Fill remaining slots with highest global scores.
- Apply overlap/min-gap constraints AFTER quota selection.

Step 3: Final polish
- Boundary snapping + end-of-idea detection
- Re-score clarity if boundaries moved

## 9) Duration Strategy (Retention-First Soft Preferences)
No hard target durations.
However, the system may apply a SOFT preference if:
- The clip remains dense and coherent, and
- The clip naturally completes between ~45–90 seconds.

Rules:
- Never pad to reach a duration.
- Never cut early just to fit a range.
- End reason MUST be logged:
  IDEA_COMPLETE / LONG_PAUSE / TOPIC_SHIFT / DENSITY_DROP / HARDCAP_FALLBACK

## 10) Required Outputs (Audit Contract)
The pipeline MUST write:
- outputs/04_metadata/agent_meta.json
  - agent_file_present, agent_sha256, agent_version
- outputs/04_metadata/ranking.csv
  - all candidates with component scores + bucket id + normalization stats
- outputs/04_metadata/selection_audit.json
  - for every candidate: selected(bool), reason_codes[], key scores, end_reason
- outputs/04_metadata/bucket_stats.json
  - candidates per bucket, thresholds used, quota met or not + why
- outputs/logs/pipeline.log
  - stage-ordered logs, timings, warnings

## 11) Reason Codes (Minimum Set)
Selection/rejection MUST include reason codes (multiple allowed):
- SEL_BUCKET_TOP
- SEL_LATE_BUCKET_QUOTA
- SEL_GLOBAL_FILL
- REJ_LOW_MEANING
- REJ_LOW_HOOK
- REJ_LOW_CLARITY
- REJ_LOW_DENSITY
- REJ_OVERLAP
- REJ_MIN_GAP
- REJ_BUCKET_EMPTY
- REJ_ASR_UNAVAILABLE
- REJ_SEGMENTATION_FAILED

## 12) Quality Gates (Automatic Fail Conditions)
Mark run as FAILED if:
- SPEECH_BLOCKS missing/empty without fallback
- novelty is constant across candidates due to missing transcripts (feature is invalid)
- bucket 80–100% quota missed while meaning candidates exist
- selection_audit missing reason codes
- logs not stage-ordered

## 13) Reproducibility (Kaggle/Contest Discipline)
- Log: seed, versions (python, ffmpeg, key libs), model versions (ASR)
- Deterministic selection with fixed seed if randomness is used (tie-breakers only)
- Any heuristic must be logged with parameters used
