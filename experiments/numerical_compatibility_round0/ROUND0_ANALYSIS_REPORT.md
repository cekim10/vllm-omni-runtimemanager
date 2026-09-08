# Numerical Compatibility Round 0 — Analysis Report

Namespace `results/numerical_compatibility_round0/rev0_3` · design rev 0.2f (`numerical-compatibility-round0-v0.2f`) ·
preregistration sha256 `c05b61e9f0140a5e654fc45fca9faf3c1e644912c85544d0a29467f361f30455` · provenance
`0c3d45c3cd7152461af830ad731226267cc4fd014d4eba51f0bb5cfecf08c3d1` · source commit `06cfb426` · GPU runs 2026-09-07/08
(single NVIDIA L40S, 46 GB) · analysis executed once by `--phase analyze`; every class and the decision below were
independently recomputed from the saved pair values and match `decision.json`.

**Decision (frozen rule): `GO_STATE_NOT_PORTABLE`.** Four core axes measured; all four in the decision set D; all four
`HYBRID_INCOMPATIBLE` at t = 12, 20 and 28. The strongest preregistered tier (`GO_STRONG_REGIME_CROSSING`, a
sample-equivalent regime whose crossing is nevertheless incompatible) did **not** occur.

## 1. Question and design (as sealed)

Does an intermediate Wan2.2-T2V-A14B denoising state created under execution configuration A remain the *same
sample* when denoising is continued under a different single-GPU configuration X?

- Observable: `H_{A->X,t} = F_X^{t:40}(x_t^A)` compared with the full runs `F_A` and `F_X`; t in {12, 20, 28} of 40 Euler
  steps; `x_k` = latent entering step k, `x_0` = seed noise before any forward.
- Vocabulary: `STATE_DIFFERENT` iff final-latent rel-L2 > 0.02; `OUTPUT_DIFFERENT` iff mean frame SSIM < 0.95;
  `STRICT_COMPATIBLE` iff neither; "different sample" is used only when both hold. `FULL_TRAJECTORY_BIT_INVARIANT`
  requires all 41 latent boundaries and the video to be bit-exact.
- Canonical phi_0: torch.compile on, CPU offload on, platform-default attention (resolved at runtime to `FLASH_ATTN`
  via an FA3 provider), bf16, TP=SP=CFG=1. Axes (one intended change each): `EAGER` (compile off), `ATTN_SDPA`
  (`TORCH_SDPA`), `ATTN_FLASHINFER`, `ATTN_CUDNN`, `OFFLOAD` (offload off). Not measurable and declared before the run:
  `BATCH` (pipeline lacks step execution), `PARALLEL` (GPU 1 belongs to another user).
- Trajectories: prompts p0 (static), p2 (articulated motion), p3 (complex city) x seed 9101; every class requires 3/3.
- Decision set D = axes that are within-configuration reproducible and not trajectory-bit-invariant. GO tiers require
  |D| >= 2 and at least one `HYBRID_INCOMPATIBLE` axis; `GO_STRONG` additionally requires that axis to be
  `FULL_SAMPLE_EQUIVALENT`. `|D| >= 1` with no hybrid incompatibility would have been `NO_GO_STATE_PORTABLE`.

## 2. What was executed

| block | planned | executed | note |
|---|---|---|---|
| canonical F_A x2 + exact-resume anchor H_{A->A,12} | 9 | 9 | |
| EAGER, ATTN_SDPA, ATTN_FLASHINFER, ATTN_CUDNN (F_X x2, H_{A->X,0}, H_{A->X,12/20/28} x 3 trajectories) | 72 | 72 | |
| OFFLOAD | 18 (+1 fresh) | 0 | engine construction failed at model load with CUDA OOM (44.35 GiB of 44.39 GiB in use; two 14B bf16 experts do not fit without sequential offload). **NOT MEASURED due to resource infeasibility**; not classified. |
| fresh-engine controls (new process, p0) | 6 | 5 | OFFLOAD fresh not run |
| total | 105 | 86 | |

Two earlier namespace events are pinned in the sealed preregistration and are not part of the result: rev0_2 stopped
fail-closed at its first run because the declared attention backend (SDPA) did not match the resolved one (FLASH_ATTN);
no comparison was made. Its single saved run was bit-exact with the rev 4.1 reference final (descriptive only).

## 3. Controls (all passed; recomputed from hashes)

- Within-configuration reproducibility: for phi_0 and every measured axis, run 2 equals run 1 bit-exactly at all 41
  latent boundaries and in the decoded video, 3/3 trajectories (15 pairs).
- Exact-resume anchor: `H_{A->A,12}` equals `F_A` run 1 bit-exactly at boundaries 12..40 and in the video, 3/3.
- Resume-plumbing anchor: `H_{A->X,0}` equals `F_X` run 1 bit-exactly (all boundaries, video), 3/3 for all four axes.
- Seed anchor: `x_0` identical across phi_0 and all four axes, 3/3.
- Fresh-engine control: a new process/engine reproduces run 1 bit-exactly for phi_0 and all four axes (5/5).
- Cross-session anchor (descriptive): `F_A` finals equal the rev 4.1 reference finals (different commit, different day)
  bit-exactly, 3/3.
- Fingerprints: every run's runtime-observed fingerprint matched its declared configuration (attention backend class,
  compiled-block count 80 vs 0, offload hook present); all non-axis environment fields identical across the 86 runs
  (torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, L40S, model revision 5be7df96, TF32 matmul off / cuDNN on, fa3-fwd 0.0.3,
  flashinfer 0.6.12).

## 4. Results

### 4.1 Compatibility matrix

| axis | within | full relation F_X vs F_A | hybrid t = 12 / 20 / 28 | t_portable | in D | cross-engine |
|---|---|---|---|---|---|---|
| EAGER | reproducible | SAMPLE_DIFFERENT | incompatible / incompatible / incompatible | none | yes | reproducible |
| ATTN_SDPA | reproducible | SAMPLE_DIFFERENT | incompatible x3 | none | yes | reproducible |
| ATTN_FLASHINFER | reproducible | SAMPLE_DIFFERENT | incompatible x3 | none | yes | reproducible |
| ATTN_CUDNN | reproducible | SAMPLE_DIFFERENT | incompatible x3 | none | yes | reproducible |
| OFFLOAD | not measured (OOM) | — | — | — | — | — |

### 4.2 Distances (final-latent rel-L2 / mean frame SSIM)

| axis | traj | F_X vs F_A | H12 vs A | H12 vs X | H20 vs A | H20 vs X | H28 vs A | H28 vs X | k_material | H1 ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| EAGER | p0 | 0.158 / 0.923 | 0.129 / 0.946 | 0.152 / 0.926 | 0.082 / 0.972 | 0.163 / 0.921 | 0.049 / 0.985 | 0.157 / 0.923 | 28 | 101x |
| EAGER | p2 | 0.302 / 0.846 | 0.281 / 0.853 | 0.183 / 0.914 | 0.050 / 0.984 | 0.305 / 0.845 | 0.045 / 0.986 | 0.303 / 0.846 | 19 | 203x |
| EAGER | p3 | 0.282 / 0.726 | 0.259 / 0.747 | 0.289 / 0.718 | 0.210 / 0.800 | 0.283 / 0.717 | 0.147 / 0.872 | 0.284 / 0.724 | 27 | 181x |
| ATTN_SDPA | p0 | 0.136 / 0.935 | 0.121 / 0.949 | 0.160 / 0.919 | 0.071 / 0.975 | 0.130 / 0.938 | 0.048 / 0.985 | 0.135 / 0.935 | 28 | 85x |
| ATTN_SDPA | p2 | 0.135 / 0.944 | 0.097 / 0.964 | 0.114 / 0.953 | 0.057 / 0.982 | 0.134 / 0.945 | 0.051 / 0.985 | 0.133 / 0.945 | 25 | 89x |
| ATTN_SDPA | p3 | 0.366 / 0.635 | 0.287 / 0.718 | 0.351 / 0.653 | 0.247 / 0.764 | 0.365 / 0.644 | 0.168 / 0.847 | 0.366 / 0.636 | 27 | 226x |
| ATTN_FLASHINFER | p0 | 0.106 / 0.958 | 0.086 / 0.969 | 0.109 / 0.956 | 0.092 / 0.970 | 0.074 / 0.974 | 0.038 / 0.991 | 0.107 / 0.957 | 31 | 92x |
| ATTN_FLASHINFER | p2 | 0.291 / 0.849 | 0.079 / 0.974 | 0.299 / 0.846 | 0.049 / 0.986 | 0.293 / 0.848 | 0.042 / 0.989 | 0.292 / 0.849 | 20 | 262x |
| ATTN_FLASHINFER | p3 | 0.256 / 0.752 | 0.241 / 0.769 | 0.241 / 0.777 | 0.238 / 0.780 | 0.253 / 0.757 | 0.150 / 0.871 | 0.251 / 0.760 | 30 | 220x |
| ATTN_CUDNN | p0 | 0.136 / 0.935 | 0.088 / 0.964 | 0.151 / 0.923 | 0.069 / 0.977 | 0.126 / 0.941 | 0.052 / 0.984 | 0.137 / 0.934 | 28 | 85x |
| ATTN_CUDNN | p2 | 0.128 / 0.949 | 0.095 / 0.965 | 0.103 / 0.961 | 0.057 / 0.982 | 0.126 / 0.951 | 0.045 / 0.986 | 0.127 / 0.950 | 26 | 83x |
| ATTN_CUDNN | p3 | 0.360 / 0.643 | 0.258 / 0.750 | 0.342 / 0.668 | 0.208 / 0.807 | 0.357 / 0.655 | 0.156 / 0.859 | 0.358 / 0.648 | 27 | 222x |

`k_material` = first boundary at which more than 2% of latent positions exceed the channel-normalised error threshold
0.02 (F_X vs F_A). `H1 ratio` = final rel-L2 divided by rel-L2 at boundary k*+4.

Every strict-incompatibility verdict above is driven by the state criterion (rel-L2 > 0.02); 25 of the 60 cells with
rel-L2 > 0.02 have SSIM >= 0.95 and are therefore "state-only" differences, not "different sample" (notably the
H20/H28 vs A cells for p0 and p2, and the full-run cell ATTN_FLASHINFER p0 at SSIM 0.958). All 12 full-run cells and
all 24 hybrid-vs-both cells at t = 12 are `STRICT_INCOMPATIBLE`; 11 of the 12 full-run cells are "different sample".

### 4.3 Divergence structure

- First bit-level divergence k* = 1 for every axis and trajectory (F_X vs F_A): the first model forward already differs;
  `x_0` is identical (hypothesis H3 holds 12/12).
- The difference then stays sub-material for most of the trajectory and becomes material late: e.g. ATTN_SDPA p0
  rel-L2 = 0.0006 at k = 1, 0.0025 at k = 10, 0.0045 at k = 20, 0.012 at k = 30, 0.035 at k = 35, 0.136 at k = 40;
  material fraction 0.0001 at k = 20, 0.02 at k = 28, 0.41 at k = 35, ~1.0 at k = 40. k_material ranges 19-31 across the
  12 cells. Final rel-L2 is 83-262x the rel-L2 at k*+4 (H1 holds 12/12).
- Hybrids diverge from `F_A` at boundary t+1 and from `F_X` at boundary t, as expected; their final distance to `F_X`
  is essentially the full-run distance d(F_A, F_X) at every t (the hybrid does not converge to the fresh run), while
  their distance to `F_A` shrinks with t (fewer steps executed under X) but never reaches the compatibility threshold:
  at t = 28 rel-L2 vs A is still 0.038-0.168. No portability window exists in the measured range (H2 fails, as
  predicted from the rev 4.1 synthetic data).
- Prompt ordering p3 > p2 > p0 (H4) holds for ATTN_SDPA and ATTN_CUDNN but not for EAGER and ATTN_FLASHINFER, where p2
  exceeds p3. The two SDPA-family axes are not identical to each other (k* = 1 between ATTN_SDPA and ATTN_CUDNN).

## 5. Frozen claim sentences (the only claims this round supports)

1. On this model and host, each measured execution regime is internally deterministic: repeated runs and
   fresh-engine runs reproduce all 41 latent boundaries and the decoded video bit-exactly, and exact resume reproduces
   the remaining trajectory bit-exactly. The canonical final output also reproduces bit-exactly across a different
   commit and session.
2. Changing exactly one regime field (compile on/off, or the attention implementation among FLASH_ATTN, SDPA,
   FLASHINFER and cuDNN) changes the first model forward at the bit level and yields a strictly incompatible final
   result under the preregistered state/output contract in 3/3 trajectories for each of the four measured regimes
   (final rel-L2 0.11-0.37, SSIM 0.64-0.96). Eleven of the twelve full-run cells satisfy the stricter "different
   sample" definition.
3. An intermediate state produced under phi_0 and continued under any of the four other regimes is strictly
   incompatible with both the replay (`F_A`) and the fresh run (`F_X`) at t = 12, 20 and 28, in 3/3 trajectories; no
   checkpoint age up to t = 28 makes the crossing compatible.
4. Latent numerical differences remain sub-material for many steps (material fraction exceeds 2% only at boundaries
   19-31) before amplifying into a material final-state divergence (83-262x growth from boundary k*+4 to the final).
5. Read together: the numerical execution regime is part of the deterministic execution semantics of a request; the
   observed behaviour is "stable within regime, different across regime", not "diffusion serving is nondeterministic".

## 6. What this round does NOT show

- It does not show the regime-crossing-only failure (`F_A` ~ `F_X` but `H` incompatible with both). Every measured
  regime change also changed the full-run sample, so a reviewer may read the hybrid result as a consequence of generic
  configuration sensitivity. The defensible systems implication is narrower: resuming or reusing state across regimes
  satisfies neither the replay nor the fresh contract, and deferring the crossing to a late checkpoint does not help.
- No claim about how often regimes change in production, about throughput or SLO cost, about multi-GPU layouts or
  batching (not measured), about offload (infeasible on 46 GB), about other models, or about any mechanism.
- No perceptual-quality claim: SSIM is used only as the preregistered output-difference criterion.
- Scope: three trajectories, one seed, one host, 40-step Euler, bf16; kill scope is "four successfully measured
  single-GPU regime changes on this host".

## 7. Round 1 question (for the next preregistration)

The result establishes the constraint; it does not yet establish that the constraint has any structure or any cost.
Round 1 is therefore staged, and each stage can end the branch:

- **Round 1A — Is numerical compatibility a non-trivial schedulable relation, rather than exact fingerprint
  equality?** All four measured regimes are incompatible with phi_0, and the stored per-boundary hashes already show
  that every pair among the five regimes diverges at boundary 1 in 3/3 trajectories (10 pairs, bit level). If the
  full pairwise 5 x 5 relation is also strictly incompatible everywhere, the compatibility matrix is the identity and
  the realistic policy is exact fingerprint pinning; the abstraction survives only if some off-diagonal pair is
  sample-equivalent (or the relation depends on t or on the request), confirmed by a minimal cross-resume probe.
- **Round 1B — Does compatibility constrain useful elasticity?** Unsafe Elastic vs Exact Fingerprint Pinning vs Oracle
  Compatibility under swept synthetic workloads; the headroom that matters is Oracle minus Pinning.
- **Round 1C — only if 1B shows headroom:** a compatibility-aware scheduler / mechanism.

No kernel or canonicalisation work precedes 1A/1B.

## 8. Retained data and figure candidates

Raw arrays (~12 GB: all-boundary latents for run 1 of every configuration and every hybrid, videos, per-boundary
hashes) are retained on the host until the figure list is fixed. Candidates: per-step rel-L2 growth curves (4 axes x
3 prompts); F_A / F_X / H triangle distances at t = 12, 20, 28; k* = 1 vs k_material = 19-31; representative frames.

## 9. Instrumentation lessons

- Probe host packages the way the selector does (FA3 modules), not by the FA2 package name; the SDPA-vs-FLASH_ATTN
  expectation error cost one 5-minute fail-closed run and one re-seal.
- Any Wan2.2-A14B configuration without CPU offload cannot run on a 46 GB device; do not plan it as an axis there.
- Runtime-observed fingerprints (backend class, compiled-block count, hook registry) caught the mismatch that a declared
  configuration would have hidden; keep them in every future round.
