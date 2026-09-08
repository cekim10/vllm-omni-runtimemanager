# Numerical Compatibility Round 1 — Preregistration DRAFT (rev 1.0a, 2026-09-08)

Status: DRAFT. Round 0 (`rev0_3`, `GO_STATE_NOT_PORTABLE`) established the constraint; Round 1 asks whether the
constraint has structure (1A) and cost (1B) before anything is built (1C). Each stage can end the branch. No GPU work
is approved by this document; 1A step 1 is CPU-only analysis of frozen Round 0 arrays.

## Round 1A — Is numerical compatibility a non-trivial schedulable relation, rather than exact fingerprint equality?

### Motivation

Round 0 measured only phi_0 -> X. If every pair of regimes is mutually incompatible, the compatibility matrix is the
identity, the realistic safe policy is exact fingerprint pinning, and an "oracle compatibility" scheduler has no room
above pinning. Bit-level evidence already points that way: from the stored per-boundary hashes, all 10 pairs among
{CANONICAL, EAGER, ATTN_SDPA, ATTN_FLASHINFER, ATTN_CUDNN} diverge at boundary k* = 1 in 3/3 trajectories. Bit-level
divergence does not by itself decide strict compatibility (rel-L2 <= 0.02 and SSIM >= 0.95), so 1A measures it.

### Step 1 (CPU only, frozen Round 0 arrays; no new runs)

- Inputs: `F_i` run 1 for i in the five regimes, 3 trajectories (`x_40.npy`, `video.npy` under `rev0_3`).
- Compute for all 10 unordered pairs x 3 trajectories: final rel-L2, mean frame SSIM, `STRICT_COMPATIBLE`,
  `different_sample`, plus k* from hashes. Vocabulary and thresholds are the Round 0 ones, unchanged.
- Classification per pair (3/3 rule, Round 0 `full_relation`): `FULL_TRAJECTORY_BIT_INVARIANT` /
  `FULL_SAMPLE_EQUIVALENT` / `FULL_SAMPLE_DIFFERENT` / `FULL_MIXED`.
- Outcome:
  - `1A_TRIVIAL_CANDIDATE`: all 10 pairs `FULL_SAMPLE_DIFFERENT`. No cross-resume probe is warranted; go to Step 3.
  - `1A_STRUCTURE_CANDIDATE`: at least one pair is `FULL_SAMPLE_EQUIVALENT` or `FULL_TRAJECTORY_BIT_INVARIANT` (or
    `FULL_MIXED` with at least one strictly compatible trajectory). Those pairs go to Step 2.

### Step 2 (minimal GPU probe, only for STRUCTURE candidates; requires separate approval)

- For each candidate pair (i, j): `H_{i->j,20}` and `H_{j->i,20}` for the 3 trajectories (resume at t = 20 from the
  stored run-1 latent `x_20` of the producer under the consumer's configuration), compared with `F_i` and `F_j`.
  Optional t = 12 and 28 only if t = 20 is compatible in at least one direction (to locate the boundary).
- Anchors as in Round 0: `x_20` supplied equals the state entering the resumed step; producer-side exact resume
  already validated in Round 0 for phi_0 only, so `H_{i->i,20}` is added as a control for each producer i.
- Pair relation: `COMPATIBLE_BOTH_WAYS`, `COMPATIBLE_ONE_WAY` (directional), `INCOMPATIBLE`.
- Cost: per candidate pair 3 trajectories x (2 hybrids + 2 self-controls) x ~145 s + 2 engine constructions
  (~0.6 GPU-h per pair).

### Step 3 — 1A decision (frozen)

- `1A_NONTRIVIAL`: at least one off-diagonal pair is `COMPATIBLE_BOTH_WAYS` or `COMPATIBLE_ONE_WAY` at t = 20 in 3/3,
  or the relation demonstrably depends on t (compatible at 28 but not at 12 for some pair, 3/3). The compatibility
  relation is richer than fingerprint equality; 1B compares Oracle Compatibility against Pinning using the measured
  relation.
- `1A_TRIVIAL`: no off-diagonal compatibility (or no candidates after Step 1). The relation is fingerprint equality
  on this host. 1B is still run, but its question narrows to "does exact pinning cost useful elasticity?" and the
  solution novelty is reduced to fingerprint-aware placement; this is recorded before 1B starts.

## Round 1B — Does compatibility constrain useful elasticity? (sketch; frozen after 1A)

Three policies, defined now so they cannot drift:

- **Unsafe Elastic**: migration / restart / reuse allowed whenever a worker is available; no semantic safety;
  performance reference (upper bound on elasticity).
- **Exact Fingerprint Pinning**: intermediate state moves only between producer and consumer with identical complete
  fingerprints; simplest strong safe baseline.
- **Oracle Compatibility**: the measured relation C(phi_i, phi_j, t, q) is known; crossings allowed iff they satisfy
  the contract. Under `1A_TRIVIAL` this collapses to Pinning by construction, and 1B reports the gap as zero.

Headroom of interest: **Oracle Compatibility minus Exact Fingerprint Pinning**. The Unsafe-vs-Pinning gap is reported
but never decides. Workload: synthetic, swept rather than a single trace, with separately controlled parameters:
arrival load rho, worker-domain composition, service-time distribution, migration / preemption opportunity rate, state
age t at which migration becomes useful, worker imbalance / skew, SLO slack. The primary metric (SLO attainment vs p99
JCT) and the kill threshold are chosen in the 1B preregistration from the simulator's noise floor and the improvement
scale prior systems papers treat as meaningful; no number is fixed here.

## Round 1C — only if 1B shows headroom

Compatibility-aware scheduler / mechanism design. Not specified before 1B.

## Open items before sealing 1A

1. Confirm Step 1 outcome from the host arrays (read-only script; no repository files change).
2. If STRUCTURE candidates exist: approve Step 2 GPU probe and its namespace (`results/numerical_compatibility_round1/1a`).
3. Decide whether 1A Step 1 and Step 3 are sealed as one preregistration (recommended: seal before Step 2 runs).
