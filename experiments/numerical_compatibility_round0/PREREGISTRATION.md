# Numerical Compatibility Round 0 — Preregistration (rev 0.2f, 2026-09-07)

Status: design FROZEN by the user (rev 0.2b, 2026-09-07), implemented as rev 0.2c, and corrected to rev 0.2d after the
final hostile audit (Section 13, item 14: trajectory-wide invariance and the NO_GO_STATE_PORTABLE outcome). The machine-readable form is
`config.json` (+ `core.frozen_design()`); `--phase preregister` seals it. GPU execution requires a separate
explicit approval after the final hostile audit.

## 0. One-sentence question

Does an intermediate Wan2.2 denoising state created under one serving execution
configuration remain the *same sample* when its denoising is continued under a different
execution configuration on the same GPU?

## 1. Motivation and the assumption under test

Stateful diffusion systems increasingly checkpoint, reuse, or move intermediate latents
(checkpoint/resume, request migration, elastic instance rebalancing, inter-request latent
reuse). Doing so implicitly requires the consumer execution to interpret the saved state
compatibly with its producer. We test whether that compatibility holds across ordinary
single-GPU execution-configuration changes, i.e. whether the effective state is `C = (x_t, t)`
or `C = (x_t, t, phi)` with `phi` the numerical execution provenance. The observable is

    H_{A->B,t} := F_B^{t:40}(x_t^A)   compared with   F_A^{0:40}   and   F_B^{0:40}.

Motivating evidence (frozen, `results/regional_recompute_round0/rev4_1`): a synthetic
perturbation below the bf16 half-ulp confined to 3.8-11% of the tokens at t in {12,20,28}
produced a strictly incompatible final state in 24/24 cells (final rel-L2 0.04-0.16, SSIM down to
0.87), and final divergence was nearly independent of injected magnitude (GAUSS 0.30 sigma gave
0.10-0.31). Whether *real* configuration differences seed such a perturbation is exactly what
Round 0 measures. The synthetic result is motivation, not evidence.

## 2. What is fixed (identical to `regional_recompute_round0` rev 4.1)

Model `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (resolved revision recorded), WanEulerScheduler,
40 steps, 480x832x33, guidance 4.0, flow shift 12, fps 16, bf16 runtime, latent
`[1,16,9,60,104]`, exact resume via `sampling_params.latents` + `step_index`, trajectory
probe capturing float32 latents, decode to video for SSIM.

Step-boundary indexing (frozen): `x_k` is the latent ENTERING step k, k = 0..39, and `x_40` is
the final latent after step 39. `x_0` is the initial noise latent, drawn from the CPU generator
seeded with the request seed BEFORE any model forward; therefore `x_0` must be bit-identical
across configurations for the same seed (Section 6.3, anchor 0). The first model forward can
first change `x_1`.

Trajectories (3): prompts p0 (static), p2 (articulated), p3 (complex city) x seed 9101.
p1 (object motion) x seed 9202 is used only as the co-batch partner. Rationale: p3 showed the
largest amplification, p0 the smallest; three trajectories is the minimum for the 3/3 rule.

## 3. Compatibility vocabulary (frozen)

For two final states with decoded videos, computed from saved arrays:

- `STATE_DIFFERENT`  := final latent rel-L2 > 0.02 (else `STATE_SAME`).
- `OUTPUT_DIFFERENT` := mean frame SSIM < 0.95 (else `OUTPUT_SAME`).
- `STRICT_COMPATIBLE` := `STATE_SAME` and `OUTPUT_SAME`; `STRICT_INCOMPATIBLE` otherwise.
- `BIT_EXACT` := identical float32 latent sha256 (and identical video sha256 where decoded).

Round 0 gates on strict compatibility. The phrase "different sample" is reserved for cases where
BOTH `STATE_DIFFERENT` and `OUTPUT_DIFFERENT` hold; a state-only or output-only difference is
reported by its own name. Both flags are reported for every comparison. Thresholds carried over
unchanged from rev 4.1 (normalised per-position error uses the reference channel std at the
same step; material-position threshold 0.02).

## 4. Execution configurations

### 4.1 Canonical phi_0

`enable_cpu_offload=True`, `enforce_eager=False` (torch.compile / inductor), attention
backend = platform default, which on this host resolves to **FLASH_ATTN** (rev 0.2e correction: the FA2
package `flash_attn` is absent but an FA3 provider, `fa3_fwd_interface` / `flash_attn_interface`, is
present, so the selector picks FLASH_ATTN; observed in the rev0_2 canonical run 1, which stopped
fail-closed against the earlier SDPA expectation), sm_89 L40S, flashinfer 0.6.12, cuDNN 9.19,
torch 2.11.0+cu128; resolved class recorded at runtime. `step_execution=False`, `max_num_seqs=1`,
TP=SP=CFG-parallel=1, bf16. This is the
configuration of every frozen prior round, so the rev 4.1 reference finals serve as a
descriptive cross-session anchor (Section 6.4).

### 4.2 Axes

Each core axis isolates ONE intended execution-regime change relative to phi_0. AX_BATCH is a
composite serving-mode configuration (step execution, max_num_seqs, and co-batch composition all
change) and is sanity-only.

| axis | change | why it changes at runtime in real serving | class |
|---|---|---|---|
| AX_EAGER | `enforce_eager=True` | `setup_compile()` failure falls back to eager with only a warning (`diffusion_model_runner.py`); compile availability differs across images | core |
| AX_OFFLOAD | `enable_cpu_offload=False` | memory-pressure policy toggled per worker/pool; outside the vLLM-Omni #5512 contract. A `FULL_TRAJECTORY_BIT_INVARIANT` result here is a valid compatibility-matrix cell ("offload preserves the domain"), not a failure | core |
| AX_ATTN_SDPA (rev 0.2f) | `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` | platform/package-dependent backend resolution: the selector falls back to TORCH_SDPA when no flash-attention provider is importable; on this host the default is FLASH_ATTN, so FLASH_ATTN <-> SDPA is the availability-driven edge | core |
| AX_ATTN_FLASHINFER | `DIFFUSION_ATTENTION_BACKEND=FLASHINFER_ATTN` | platform/package-dependent backend resolution: different hosts/images of the same stack resolve to different implementations (this Ada host: FLASH_ATTN via an FA3 provider; other platforms default to CUDNN_ATTN or FLASHINFER_ATTN) | core |
| AX_ATTN_CUDNN | `DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN` (pins `sdpa_kernel([CUDNN_ATTENTION])`) | same as above | core |
| AX_BATCH | `step_execution=True`, `max_num_seqs=2`, target co-batched with the partner | continuous batching; duplicates #5512 Part 2 | **NOT_MEASURED (declared 2026-09-07)**: the Wan2.2 pipeline does not implement `SupportsStepExecution`, so `step_execution=True` is rejected at model-runner init and `max_num_seqs>1` is forced to 1 (`diffusion_engine.py:143`). Co-batching Wan2.2 would require implementing step execution in the trusted pipeline, which is out of Round 0 scope. |
| AX_PARALLEL | USP or CFG-parallel degree 2 | elastic parallelism / disaggregation | **NOT_MEASURED (declared 2026-09-07)**: the host has two L40S but GPU 1 is allocated to another user; only GPU 0 is available for this round. Multi-GPU layout stays outside the kill scope. |

Excluded: SAGE_ATTN / SAGE_ATTN_3 (quantised), cache backends, LoRA, quantisation (they change
the computation, not only its numerics). TF32 and reduced-precision-reduction flags are recorded
in the fingerprint but not varied.

### 4.3 Fingerprint phi (recorded for every run from the worker; equality compared exactly)

`{enforce_eager, inductor_active (runtime evidence: compiled-module marker / dynamo state of the
transformer, not the flag), enable_cpu_offload, offload_backend_class, attention_backend_resolved
(class name from the selector, not the requested string), step_execution, max_num_seqs,
batch_slot, parallel: {tp, sp, cfg}, dtype, torch, cuda, cudnn, flashinfer version or null,
flash_attn version or null, gpu_model, model_revision, cublas_workspace_config, tf32_matmul,
tf32_cudnn}`.
Rule: a run whose recorded fingerprint does not equal the fingerprint declared for its axis
(silent attention fallback, compile silently disabled) is INVALID for that axis, never re-labelled.

## 5. Run matrix

`F_X` = full 40-step run under phi_X from seed. `H_{A->X,t}` = resume under phi_X from the phi_0
checkpoint `x_t` of `F_A` run 1 (t = 0 is a full-length resume). Every run captures the float32
latent at all 41 boundaries and decodes the video; run 2 of each reproducibility pair stores
per-boundary sha256 plus the final latent only.

| block | runs per trajectory | 3 trajectories |
|---|---|---|
| canonical | `F_A` x2 (2 full) | 6 full |
| exact-resume anchor | `H_{A->A,12}` (1 resume) | 3 resume |
| per core axis X (5 axes) | `F_X` x2 (2 full); `H_{A->X,0}` anchor, `H_{A->X,12}`, `H_{A->X,20}`, `H_{A->X,28}` (4 resume) | 6 full + 12 resume each -> 30 full + 60 resume |
| fresh-engine control (descriptive) | `F_X` for p0 only, in a NEW process/engine, for phi_0 and each core axis | 6 full |
| AX_BATCH | NOT_MEASURED (see 4.2) | 0 |
| AX_PARALLEL | NOT_MEASURED (see 4.2) | 0 |

Totals: 42 full + 63 resume = 105 runs (AX_BATCH and AX_PARALLEL not measured).
Wall-time model from rev 4.1 (full 236 s; per denoising step ~4.5 s plus ~55 s fixed encode/decode,
so resume(t) ~ 55 + 4.5 (40 - t) s: t=0 235 s, t=12 181 s, t=20 145 s, t=28 109 s):
canonical 0.54 h; each core axis 0.95 h (x5 = 4.75 h); fresh-engine controls 6 x (~300 s engine +
236 s) = 0.9 h; 6 engine constructions ~0.5 h -> **~6.6 GPU-h**. Storage ~12.5 GB. Storage: ~70 all-latent runs x ~150 MB -> ~10 GB (per-boundary hashes only for run 2s).

## 6. Measurements and classification

### 6.1 Measurements (all recomputable from saved arrays)

For every pair of runs at aligned boundaries k = 0..40: `bit_exact_k` (sha256), `k*` = first k
with any difference, `delta_k` = rel-L2 and normalised RMS against the reference at k,
`k_material` = first k at which > 2% of positions exceed the material-position threshold. Final:
rel-L2, mean frame SSIM, `STATE_/OUTPUT_DIFFERENT`, `STRICT_COMPATIBLE`, `BIT_EXACT`. For hybrids:
all of the above against BOTH `F_A` and `F_B`.

### 6.2 Axis classes (each over the 3 trajectories)

Prerequisite:
- `WITHIN_REPRODUCIBLE`: `F_X` run 1 and run 2 `BIT_EXACT` (latent and video) in 3/3. Otherwise
  `WITHIN_NONDETERMINISTIC` (reported with `k*`; the axis is excluded from the decision because
  no compatibility domain can be defined for it).

Full-run relation (`F_X` vs `F_A`, 3/3 required for a class; else `FULL_MIXED`):
- `FULL_TRAJECTORY_BIT_INVARIANT` (rev 0.2d): `x_k^X == x_k^A` bit-exact at ALL 41 boundaries AND the videos
  bit-exact, in 3/3. Final-only equality with differing intermediates is NOT invariance: the question is
  intermediate-state portability, so such an axis is `FULL_SAMPLE_EQUIVALENT` and enters D. Hybrids are
  measured for every axis; no hybrid outcome is declared trivial.
- `FULL_SAMPLE_EQUIVALENT`: not trajectory-bit-invariant but `STRICT_COMPATIBLE` in 3/3.
- `FULL_SAMPLE_DIFFERENT`: `STRICT_INCOMPATIBLE` in 3/3.

Hybrid class at t = 12 (`H_{A->X,12}` vs `F_A` and vs `F_X`, 3/3 required; else `HYBRID_MIXED`), precedence in order:
- `HYBRID_COMPATIBLE_BOTH`: `STRICT_COMPATIBLE` vs `F_A` AND vs `F_X` in 3/3 (rev 0.2c; no incompatibility).
- `HYBRID_INCOMPATIBLE`: `STRICT_INCOMPATIBLE` vs `F_A` AND vs `F_X` in 3/3.
- `HYBRID_REPLAY_LIKE`: `STRICT_COMPATIBLE` vs `F_A` in 3/3 (and not vs `F_X`).
- `HYBRID_FRESH_LIKE`: `STRICT_COMPATIBLE` vs `F_X` in 3/3 (and not vs `F_A`).

Hybrids are run and classified for EVERY core axis regardless of its full-run relation; the
full-run relation is never a qualification condition. The result ranking, fixed in advance:
1. `FULL_SAMPLE_EQUIVALENT` + `HYBRID_INCOMPATIBLE` (neither regime changes the sample when used
   consistently; incompatibility arises only when state crosses regimes) — strongest;
2. `FULL_SAMPLE_DIFFERENT` + `HYBRID_INCOMPATIBLE`;
3. everything else.

Descriptive per axis: `CROSS_ENGINE_REPRODUCIBLE` (fresh-engine `F_X` for p0 `BIT_EXACT` with run 1)
or `CROSS_ENGINE_NONREPRODUCIBLE` (evidence of hidden provenance not captured by phi); and
`t_portable` = min t in {12, 20, 28} at which `H_{A->X,t}` is `STRICT_COMPATIBLE` with `F_A` in 3/3
(none if never) — the earliest safe switch point.

### 6.3 INVALID_EXPERIMENT conditions (any one)

0. `x_0` not `BIT_EXACT` across phi_0 and every measured core axis for the same seed (the
   divergence would not be numerical).
1. `F_A` not `WITHIN_REPRODUCIBLE`.
2. Exact-resume anchor `H_{A->A,12}` not `BIT_EXACT` with `F_A` run 1 in 3/3.
3. Any `H_{A->X,0}` anchor not `BIT_EXACT` with `F_X` run 1 in 3/3 for a measured axis
   (resume plumbing would confound the hybrid result).
4. Any recorded fingerprint differs from the declared one for its axis (Section 4.3).
5. Fewer than 2 core axes measured (core: EAGER, OFFLOAD, ATTN_SDPA, ATTN_FLASHINFER, ATTN_CUDNN).
6. Any Engine gate error (non-Euler path, wrong step count, probe mismatch), as in prior rounds.
7. (rev 0.2c) Non-axis environment fingerprint fields (torch, cuda, cuDNN, GPU model, model revision, dtype,
   TF32 flags, cuBLAS workspace, flashinfer/flash_attn versions, parallel degrees) differ between any two runs.

### 6.4 Descriptive controls (never gate)

- Cross-session reproducibility: `F_A` run 1 final sha256 vs the rev 4.1 reference `final.npy`
  sha256 for p0/p2/p3 x 9101 (p0: `fef35330b8bd7d45...`). Provenance differs (different commit), so a
  mismatch is reported as `CROSS_SESSION_NONREPRODUCIBLE`, not INVALID.
- Fresh-engine controls (Section 6.2) — recommended by the audit; not a kill condition in Round 0.
- `delta_k` curves of configuration axes against the frozen synthetic W1 / L4 curves at their
  captured steps.

## 7. Decision (frozen)

Let D = set of core axes that are `WITHIN_REPRODUCIBLE` and NOT `FULL_TRAJECTORY_BIT_INVARIANT`
(i.e. the regime change is numerically visible somewhere along the trajectory). AX_BATCH never enters D.

- `GO_STRONG_REGIME_CROSSING`: |D| >= 2 AND at least one axis in D is `FULL_SAMPLE_EQUIVALENT`
  and `HYBRID_INCOMPATIBLE`.
- `GO_STATE_NOT_PORTABLE`: |D| >= 2 AND at least one axis in D is `HYBRID_INCOMPATIBLE`
  (necessarily `FULL_SAMPLE_DIFFERENT` or `FULL_MIXED`).
- `WEAK`: |D| == 1 and that axis is `HYBRID_INCOMPATIBLE` (phenomenon present, generality not shown).
  Reported with all classes; no follow-up run is preregistered; the user decides whether the branch continues.
- `NO_GO_STATE_PORTABLE` (rev 0.2d): |D| >= 1 and no axis in D is `HYBRID_INCOMPATIBLE`. Regimes change
  the numerics, but crossing regimes creates no third semantics: intermediate state is portable for
  this round's axes. This is a negative result for the branch question; generic configuration
  sensitivity alone never keeps the branch alive.
- `NO_GO_INVARIANT_SINGLE_GPU`: |D| == 0 and every measured core axis is `FULL_TRAJECTORY_BIT_INVARIANT`
  (the problem, if it exists, is confined to axes this round cannot vary: parallel layout, batch).
- `NO_GO`: otherwise (for example all visible axes are `WITHIN_NONDETERMINISTIC`).

The branch question, stated with these outcomes: do execution-regime crossings create a
state-specific incompatibility? Only a GO tier answers yes.

Both GO tiers are GO under the frozen rule; the tier is a strength label. A GO claims only: "on
this model and host, intermediate state created under phi_0 and continued under at least one
other single-GPU configuration is strictly incompatible with both the replay and the fresh run".
"Different sample" is added only where Section 3 permits it. No frequency, throughput, mechanism or
cross-model claim.

## 8. Preregistered hypotheses (reported, never gating)

- H1 amplification: for every axis in D, final rel-L2 vs `F_A` >= 10 x the rel-L2 at boundary
  k*+4 (synthetic band was 10-80x).
- H2 earliest safe switch point: `t_portable` exists (<= 28) for at least one axis in D.
  Prediction from rev 4.1 data: **expected to FAIL** (a sub-ulp residue at t = 28 still gave final
  rel-L2 0.04). Note the structural effect: later t leaves fewer steps under B, so approach to
  `F_A` with t is expected; only the existence of a compatible t is informative.
- H3 first divergence: k* = 1 for every axis in D (the first model forward differs; `x_0` is
  identical by construction).
- H4 prompt ordering of final divergence matches rev 4.1 (p3 > p2 > p0).
- H5 cross-engine: every core axis is `CROSS_ENGINE_REPRODUCIBLE` (phi captures the provenance).

## 9. Kill scope and claim boundary

Kill scope: Wan2.2-T2V-A14B, 40-step Euler, bf16, single L40S, the five listed axes. The matrix is final:
no further axes will be added after this revision (more axes would turn Round 0 into a characterisation sweep). A NO_GO closes
"single-GPU configuration-induced state incompatibility on Wan2.2" only; it says nothing about
multi-GPU layouts (AX_PARALLEL not measured: GPU 1 occupied), other models, or throughput. No mechanism
(compatibility domain enforcement, scheduler, canonicalisation) is built or evaluated in Round 0;
Round 1 would have to measure how often phi changes in a serving trace and the cost of enforcing
compatibility.

## 10. Prior work and novelty boundary (delta audit 2026-09-07)

- vLLM-Omni issue #5512 "Batch-Invariant Diffusion Rollouts" (2026-07-28): contract
  `output(A alone at BS1) == output(A inside a real BSN batch)` for RL rollouts; SD3.5, one
  certified configuration, eager only; explicitly excludes offload, cache backends, quantisation
  and externally injected latents; no amplification measurement. AX_BATCH duplicates its Part 2.
- verl-omni RFC #203: batch invariance + E2E determinism on an identical hardware/library stack.
- ComfyUI Deterministic Toolkit: fixed reduction strategies (incl. LTX Video); same-seed =
  identical output; no per-step amplification curve, no cross-configuration resume semantics.
- CoRun (arXiv 2608.14376): LLM-only; position invariance + fixed-shape decode instead of
  batch-invariant kernels. A later canonicalisation mechanism, not a collision.
- Chorus (arXiv 2604.04451): inter-request latent reuse without a producer/consumer numerical
  compatibility contract. DisagFusion (arXiv 2605.25550): stage disaggregation and elastic
  rebalancing; mid-request state movement to be confirmed from the full text.

Boundary sentence: prior work asks whether the same request stays deterministic under a fixed
regime or across batch compositions; Round 0 asks whether intermediate state remains the same
sample when the execution regime changes after the state was created.

## 11. Phases and artifacts

`cpu | preregister | canonical | axis --axis {EAGER,OFFLOAD,ATTN_SDPA,ATTN_FLASHINFER,ATTN_CUDNN}
| freshcheck --axis {CANONICAL,EAGER,OFFLOAD,ATTN_SDPA,ATTN_FLASHINFER,ATTN_CUDNN} | analyze`
(AX_BATCH and AX_PARALLEL NOT_MEASURED; no phases for them).
Each `axis` invocation constructs one engine with that configuration and runs its 18 runs;
`freshcheck` is a separate process (hence a fresh engine) running `F_X` for p0 once. The worker
records the resolved fingerprint into every run.json. Output namespace
`results/numerical_compatibility_round0/rev0_3/` (rev0_2 is the frozen instrument-mismatch record, pinned as a prior run; old namespaces rejected). Trusted-source set for the
provenance hash additionally covers the axis mechanisms: attention selector/layer, CUDA platform selector,
compile.py, sequential offload backend, diffusion model runner. Provenance hash over
trusted sources + config + commit + relevant dirty status, sealed preregistration sha256,
`require_sealed` on every GPU phase, `summary.json` written exactly once, as in prior rounds.
`analyze` recomputes every class in Sections 6-7 from saved arrays and hashes, never from
declared booleans (trajectory-wide invariance comes from the stored per-boundary sha256 of every run at zero GPU
cost), and emits the compatibility matrix (axis x {within, cross-engine, full
relation, hybrid class at 12/20/28, t_portable, k*, final rel-L2, SSIM}).

## 12. Open items before freezing

1. RESOLVED: host has 2x L40S, GPU 1 in use by another user -> AX_PARALLEL NOT_MEASURED.
2. RESOLVED: AX_BATCH is NOT_MEASURED (Wan2.2 pipeline lacks step execution; Section 4.2).
3. Implementation requirement: the trajectory-probe metadata written by `pipeline_wan2_2.py` does
   not currently record the attention backend class, compile state, offload backend, or library
   versions. The fingerprint (Section 4.3) therefore needs an additive `runtime_fingerprint` block
   in the probe metadata, i.e. a change to a trusted source file. Prior rounds stay sealed under
   their own hashes; the change must land before `preregister` so the provenance hash covers it.
   Requires user approval.

## 13. Changes from rev 0.1 (first hostile audit, 2026-09-07)

1. Full-run materiality removed from the qualification condition. Axes are classified by
   `WITHIN_REPRODUCIBLE` (prerequisite) and a separate full-run relation; hybrids run for every
   axis; `FULL_SAMPLE_EQUIVALENT` + `HYBRID_INCOMPATIBLE` is now the strongest GO tier instead of
   being discarded as PARTIAL.
2. Boundary indexing defined (`x_0` = initial noise before any forward); H3 changed from k* = 0 to
   k* = 1; `x_0` bit-equality across configurations added as INVALID anchor 0.
3. Run matrix corrected: per core axis 2 full + 4 resume per trajectory (6 full + 12 resume);
   totals 35 full + 51 resume + batch; wall-time model made explicit (~6.4 GPU-h without PARALLEL).
4. `STATE_DIFFERENT` / `OUTPUT_DIFFERENT` / `STRICT_COMPATIBLE` vocabulary; "different sample"
   reserved for both flags; gate on strict compatibility.
5. Fresh-engine (new process) control per configuration for p0, descriptive, with
   `CROSS_ENGINE_*` classes and hypothesis H5.
6. "each changes exactly ONE field" replaced by the core-axis / composite-BATCH wording;
   OFFLOAD bit-invariance described as a valid matrix cell.
7. Section 1 softened: systems "implicitly require" compatible interpretation; no claim that
   they promise bit-exact portability.
8. Fingerprint fields made runtime evidence (`inductor_active`, `attention_backend_resolved`,
   `offload_backend_class`, `batch_slot`, flashinfer version).
9. `t_portable` (earliest safe switch point) replaces the monotonicity framing of H2.
10. (rev 0.2b) AX_BATCH declared NOT_MEASURED from code inspection: Wan2.2 pipeline lacks
    `SupportsStepExecution`; batch phase removed; cost ~5.5 GPU-h without PARALLEL. Fingerprint
    readback identified as an additive change to the trusted pipeline probe metadata (Section 12.3).
11. (rev 0.2b) AX_PARALLEL declared NOT_MEASURED: GPU 1 is occupied by another user; only
    GPU 0 available. Remaining open items are approvals only (Section 12.3, freeze).
12. (rev 0.2c, implementation) `HYBRID_COMPATIBLE_BOTH` added with precedence when H is compatible with
    both F_A and F_X (rev 0.2b left REPLAY_LIKE / FRESH_LIKE ambiguous in that case); environment-field
    consistency across runs added as INVALID condition 7; trusted-source set extended to the axis
    mechanisms; axis names without `AX_` prefix in code (`EAGER`, `OFFLOAD`, `ATTN_FLASHINFER`, `ATTN_CUDNN`).
13. (rev 0.2c, implementation) `runtime_fingerprint` added to the Wan2.2 trajectory-probe metadata
    (additive; records attention backend classes actually instantiated, compiled-block count, offload hook
    names, library/device environment). A run whose recorded fingerprint mismatches its declared
    configuration is saved, then stops the phase (fail closed) and makes the axis INVALID.
14. (rev 0.2d, final hostile audit) BLOCKING fix: the final-latent-only invariance class replaced by
    `FULL_TRAJECTORY_BIT_INVARIANT` (all 41 boundaries + video, 3/3); D exclusion uses it, so a final-only
    coincidence with differing intermediates enters D. `|D| >= 2 and no HYBRID_INCOMPATIBLE` moved from WEAK
    to the new `NO_GO_STATE_PORTABLE` outcome (numerics change, state portability holds = negative result).
    `model_revision` made mandatory in the recorded fingerprint (GateError if unresolved, so the environment
    check can never degenerate to None == None); `model` and declared revision also recorded by the pipeline.
    "hybrids then trivially bit-exact" wording removed.
15. (rev 0.2e, instrument fix before any comparison) The rev0_2 canonical run 1 recorded
    `attention_backend_resolved = FLASH_ATTN` against the declared SDPA and stopped fail-closed as designed.
    Cause: the host has an FA3 provider (`fa3_fwd_interface` / `flash_attn_interface`) although the FA2
    package is absent; the earlier host check only probed `flash_attn`. Canonical/EAGER/OFFLOAD expectation
    changed to FLASH_ATTN; `flash_attn_provider` added as an environment fingerprint field; namespace moved
    to rev0_3 with rev0_2 pinned in `prior_runs`. The saved rev0_2 run is bit-exact with the rev 4.1
    reference final for p0 (cross-session anchor holds). No threshold, axis, or decision rule changed.
16. (rev 0.2f, user-approved before any comparison) `ATTN_SDPA` core axis added: FLASH_ATTN -> TORCH_SDPA is the
    availability-driven fallback edge that rev0_2 revealed. Attention runtime reasons generalised to
    "platform/package-dependent backend resolution"; claim caution: Round 0 says only that backend
    availability/platform differences can resolve the same serving stack to different attention
    implementations, nothing about how often a runtime switches. `flash_attn_provider` stays an environment
    consistency field with no hard-coded expected value. Matrix 42 full + 63 resume = 105 runs, ~6.6 GPU-h,
    declared final. The rev0_2 cross-session bit-exact observation is descriptive and pre-experimental; the new
    sealed canonical must pass independently.
