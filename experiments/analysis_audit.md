# Video State Protection Analysis Audit

This audit applies to the legacy analysis embedded in
`experiments/video_state_protection_killtest.py` before the Stage B analysis
fix. The raw `frontier_raw.csv` rows are not modified by this audit.

## Dataset Provenance

- Stage A contains 432 unique recovery rows: 12 prompts x 2 seeds x 3
  checkpoint steps x 6 representations.
- The per-prompt seed values are prompt-specific, so the CSV contains 24
  distinct numeric seed values but exactly 2 `seed_index` values per prompt.
- Prompt IDs are `recovery_000` through `recovery_011`; no `preflight_...`
  prompt IDs are present.
- `experiments/video_recovery_prompt_set.json` contains 14 entries. The strict
  loader slices the first `--num-prompts` entries, so the registered 12-prompt
  experiment excludes `recovery_012` and `recovery_013`. Those two excluded
  prompts are the only entries labeled `scene_transition_occlusion`. The
  Stage B prompt set must not be changed after Stage A; this is a limitation
  on content-generalization claims, not a reason to alter the data.

## Result-Path Audit

| Result | Legacy function | Input/grouping | Effective sample unit | CI method | Pooling issue |
|---|---|---|---|---|---|
| `frontier_summary.json/frontier_summary` | `_aggregate_frontier_summary` | `checkpoint_step, variant` | Each prompt-seed row | Percentile bootstrap of row means | Yes. Prompt heterogeneity and seed variability are pooled as IID. |
| `iso_storage_frontier.csv` | `_build_iso_storage_rows` | Pair rows by `prompt_id, seed, checkpoint_step, variant`, then pool all paired deltas by step/pair | Each prompt-seed delta | Percentile paired bootstrap over the pooled delta list | Yes. The pairing is correct, but the resampling level is not hierarchical. |
| `minimum_safe_representation.csv` | `_build_minimum_safe_rows` | `prompt_id, category, checkpoint_step, variant` | Seeds within one prompt/step/variant cell | Percentile bootstrap over seeds | Structurally correct for within-prompt uncertainty, but Stage A has only n=2 and therefore cannot support an n=5 label or stable 99% decisions. |
| `separability_analysis.csv` | `_oracle_session_table`, `_fit_simple_policies`, `_evaluate_simple_policies` | Oracle is built per prompt-seed-step; policies are fit and evaluated on the same rows | Prompt-seed session | In-sample point estimates; no CI | Yes. Seeds and prompts are treated as exchangeable sessions, and evaluation has training leakage. |
| `interaction_crossings.csv` | `_interaction_crossings` | Uses prompt-level minimum-safe rows and compares selected checkpoint bytes at steps 10/20/30 | Prompt pair | No CI or seed-stability analysis | It inherits unstable n=2 minimum-safe decisions and has no noise-floor filter. |
| `budget_simulation.csv` | `_budget_simulation` | Samples prompt-seed oracle sessions with replacement | Prompt-seed session | Repeated simulation trials only | Yes. It consumes in-sample fitted policies and seed-level sessions instead of prompt-level n-seed frontiers. |
| `resolved_at_n5` | `_build_iso_storage_rows` | Same pooled iso-storage deltas as above | 24 prompt-seed deltas at Stage A | CI excludes zero | Both a naming and an inference bug. No seed-count assertion exists. |

## Required Questions

### 1. Is `resolved_at_n5` only a naming bug?

No. The field is hard-coded to `resolved_at_n5`, but Stage A contains two
seeds per prompt. In addition, the code bootstraps 24 pooled prompt-seed
deltas for each step/pair as if they were IID. It therefore both mislabels
the seed count and uses the wrong population unit for the claimed aggregate
resolution.

### 2. Does iso-storage CI use the same faulty pooling path?

Yes. `_build_iso_storage_rows` correctly constructs paired representation
deltas within each prompt/seed/step, but passes the entire pooled list to
`_paired_bootstrap_ci`, which is only an alias for the flat `_bootstrap_ci`.
The corrected path must average paired seed effects within each prompt first
and bootstrap over prompt effects.

### 3. Are prompt differences treated as IID within-prompt replication?

Yes in aggregate frontier summaries, iso-storage CIs, separability fitting,
and the simulator input. The minimum-safe calculation itself groups by
prompt and uses seeds within the cell, but its downstream analyses do not
preserve that hierarchy consistently.

### 4. Does the budget simulator duplicate `progress_only` rows?

Yes. `_budget_simulation` always includes the literal `progress_only` and
then appends `best_simple_policy`. When the global best row is
`progress_only`, the policy loop emits it twice. This explains the doubled
legacy row count.

### 5. Does the simulator evaluate the strongest simple separable policy?

No. It selects one globally best row across all quality targets using
in-sample violation and excess-byte statistics. That selected name may be
`progress_only`; it is not selected per target and is not guaranteed to be
the content+progress policy. Consequently the legacy simulator cannot
support a simple-separable-versus-oracle claim.

## Shared Helpers and Contamination

- `_bootstrap_ci` and `_metric_stats` are shared by aggregate summaries,
  screening, minimum-safe selection, and iso-storage analysis. Their flat
  resampling behavior is appropriate only when the caller has already
  supplied values at the desired independent sampling level.
- `_safe_for_target` is shared by oracle construction, policy evaluation,
  and budget simulation. Its all-three-metrics rule matches the
  preregistration and should be retained.
- `_fit_simple_policies` and `_evaluate_simple_policies` use the same data for
  fitting and evaluation. Corrected policy results require prompt-held-out
  evaluation.
- `_budget_simulation` reimplements policy prediction rather than consuming
  a validated policy-decision table, creating the duplication and policy
  mismatch paths above.

## Invalidated Previous Results

Until the corrected n=5 analysis is complete, the following legacy outputs
must not be cited:

- every `resolved_at_n5` value produced from the two-seed Stage A data;
- all legacy iso-storage confidence intervals (point estimates remain useful
  only as descriptive Stage A screening values);
- the reported interaction-crossing percentage;
- every session-count, oracle-gap, or policy comparison from the legacy
  budget simulator;
- in-sample separability accuracy and violation rates as estimates of
  out-of-prompt generalization.

## Corrected Analysis Contract

The replacement analyzer must:

1. use seeds only as repeated measurements inside a prompt/step cell;
2. compute prompt-level effects before population aggregation;
3. bootstrap prompts for population-level CIs;
4. expose `seed_count` and `resolved_at_seed_count` without hard-coded n=5
   naming, and assert the expected count;
5. evaluate simple policies with prompt-held-out predictions;
6. emit each of `full`, `uniform_int8`, `progress_only`, `content_only`,
   `simple_separable`, and `joint_oracle` exactly once per simulator cell;
7. report compression gaps separately from target-crossing stability measured
   by fixed-state repeated recovery;
8. resolve a target-specific minimum-representation decision only when the
   selected tier passes all metrics in every paired repeat and each cheaper
   rejected high-fidelity tier has a stable target failure;
9. sample each base `(prompt, checkpoint_step)` session at most once in a
   mixed-target budget trial, then assign exactly one target to that session.

## Stage B Fix Verification

The replacement paths now enforce the following separation:

- `build_policy_predictions` is used only for leave-one-prompt-out
  separability evaluation.
- `build_global_policy_predictions` fits one common progress-only,
  content-only, and simple-separable model per quality target and applies that
  same model to every session in the finite-budget simulation.
- The global content policies use complexity measured from the final baseline
  video. They are explicitly labeled privileged, non-deployable upper bounds;
  this makes the simple-policy comparison conservative but does not establish
  a deployable content predictor.
- Resume validation reads and reconstructs every serialized representation,
  checks actual payload and metadata byte counts, validates metadata/component
  layout, verifies or backfills SHA-256 hashes, checks all absolute/relative
  quality metrics, and rejects sessions whose baseline/probe/checkpoint source
  artifacts cannot be loaded.
- All GO/NO-GO numeric thresholds and span counts are stored in
  `video_state_protection_stage_b_analysis_config.yaml` before Stage B. The
  analyzer refuses CLI values that differ from this preregistration.
- Interaction rows include per-seed crossing fractions and bootstrap CIs in
  addition to the aggregate minimum-representation crossing indicator.
