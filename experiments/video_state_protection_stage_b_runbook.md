# Stage B GPU0 Runbook

Run from the repository root on `elves-01` with the CUDA-capable venv active.
Only GPU0 is used. Each raw recovery variant and each noise-floor repeat is
persisted immediately and skipped after validation on restart.

## 1. Complete Five Seeds

```bash
source .venv-vllm-cu12/bin/activate
bash experiments/run_video_state_protection_killtest_gpu0.sh full
```

The command reuses all 432 valid Stage A rows and adds seed indexes 2, 3, and
4. It must finish with `validated frontier rows: 1080`. At this point the
embedded `frontier_summary.json` intentionally says `ANALYSIS-PENDING`; it is
not a GO/NO-GO result.

If a shorter maintenance window is required, use non-overlapping prompt
ranges against the same output directory. Run them sequentially on GPU0:

```bash
bash experiments/run_video_state_protection_killtest_gpu0.sh full \
  results/video_state_protection_killtest_gpu0 0 4 2 5
bash experiments/run_video_state_protection_killtest_gpu0.sh full \
  results/video_state_protection_killtest_gpu0 4 8 2 5
bash experiments/run_video_state_protection_killtest_gpu0.sh full \
  results/video_state_protection_killtest_gpu0 8 12 2 5
```

Range runs validate uniqueness and assigned-session completeness without
requiring 1080 global rows. Re-running the full command afterward validates
the merged 1080-row result without recomputing completed variants.

## 2. Measure Recovery Noise

```bash
bash experiments/run_video_state_protection_killtest_gpu0.sh noise
```

This uses `recovery_000`, `recovery_004`, and `recovery_006`, all three
checkpoint steps, full/FP16/INT8, and five repeated resumes from fixed
serialized state. It also writes nine fixed side-by-side sanity videos.

## 3. Produce Corrected Final Analysis

```bash
bash experiments/run_video_state_protection_killtest_gpu0.sh analyze
```

The analyzer refuses to produce an n=5 report unless exactly 1080 raw rows and
the noise-floor CSV are present. The legacy `frontier_summary.json`, old
crossing percentage, and old budget CSV are not final evidence.

## 4. Validate CPU Logic

```bash
python -m pytest -q tests/diffusion/test_video_state_protection_analysis.py
```

Expected final deliverables are under
`results/video_state_protection_killtest_gpu0/run/` and include
`frontier_raw_n5.csv`, `checkpoint_sizes_n5.csv`,
`minimum_safe_representation_n5.csv`, `interaction_crossings_n5.csv`,
`separability_results_n5.csv`, `budget_simulation_fixed.csv`,
`noise_floor_results.csv`, `final_frontier_summary.json`, and
`video_state_protection_final_killtest.md`.
