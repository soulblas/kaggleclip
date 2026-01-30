# AGENTS.md
Agent Version: v1-retention-tiktok

## Core Objective
Select clips optimized for TikTok recommendation signals:
- average watch time (AWT)
- retention curve stability
- completion / rewatch probability

Duration is an outcome of retention optimization, not a fixed target.

## Hard Invariants (Violation = ERROR)
1) Caption/hashtags MUST NOT influence candidate mining, scoring, or selection.
2) SPEECH_BLOCKS MUST exist before candidate mining:
   - primary: silence complement (speech between silences)
   - fallback: energy/VAD segmentation if silence fails
3) Minimal semantic signal MUST exist before final scoring:
   - run light ASR for Top-N candidates per time bucket BEFORE final scoring
4) Time fairness MUST be enforced:
   - Split video into 5 buckets: 0–20%, 20–40%, 40–60%, 60–80%, 80–100%
   - Must select >=1 clip from 80–100% IF any candidate passes meaning_min
5) Duration MUST be retention-first:
   - do not force fixed durations (e.g., 30/60/120)
   - prefer end-of-idea (pause / punctuation / topic shift) over hard caps
6) strong_enough MUST NOT be an absolute global threshold:
   - use per-bucket percentile/z-score thresholds

## Required Audit Outputs
- outputs/ranking.csv (all candidates + component scores)
- outputs/selection_audit.json (reason codes per candidate)
- outputs/bucket_stats.json (candidate counts, thresholds, quota status)
- outputs/agent_meta.json (agent_file_present, agent_sha256, agent_version)
- outputs/pipeline_log.txt (stage order + warnings)
