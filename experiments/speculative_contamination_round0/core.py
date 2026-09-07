"""Frozen scientific definitions for the speculative-contamination Round-0 experiment, revision 2 (numpy only, CPU-testable).

Primary question (frozen):
    When region-selective diffusion execution introduces a localized approximation error, does correctness require a
    global dense refresh, or can an actually recomputable dependency-bounded subset recover the trajectory?

Latent geometry (trusted Wan2.2 T2V path, 480x832x33, 40 Euler steps):
    latent  x : [1, 16, F=9, H=60, W=104]
    tokens    : patch_size (1, 2, 2) -> token grid (9, 30, 52) = 14040 tokens; token (f,h,w) <-> latent [:, :, f, 2h:2h+2, 2w:2w+2]
    token index = f * 1560 + h * 52 + w (patch_embedding -> flatten(2) -> transpose(1, 2)); full self-attention with 3D RoPE.

Trusted scheduler (WanEulerScheduler.step):  x_{k+1} = x_k + (sigma_{k+1} - sigma_k) * v_k,  sigma_k = timesteps[k] / 1000.
STALE_VELOCITY (primary error model): the region R skipped its last L target updates and kept the velocity of its last active
step (the RAS / HSA "cached Euler update"), so at the checkpoint t
    x'_t[R] = x_{t-L}[R] + (sigma_t - sigma_{t-L}) * v_{t-L-1}[R],   v_{t-L-1} = (x_{t-L} - x_{t-L-1}) / (sigma_{t-L} - sigma_{t-L-1}),
everything else equal to the clean x_t. PRIMARY L = 4 (frozen on real-model evidence: L = 1 is below the bf16 resolution floor at
injection; existing region-selective systems skip regional updates across several steps). L = 1 is a descriptive magnitude arm.
Primary repair window d = 4 (aligned with L): after four stale regional updates, can correctness be restored by recomputing
substantially less than the full state?
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

LATENT_SHAPE = (1, 16, 9, 60, 104)
PATCH = (1, 2, 2)
TOKEN_GRID = (LATENT_SHAPE[2] // PATCH[0], LATENT_SHAPE[3] // PATCH[1], LATENT_SHAPE[4] // PATCH[2])
N_TOKENS = int(np.prod(TOKEN_GRID))
N_POSITIONS = int(np.prod(LATENT_SHAPE[2:]))
TOTAL_STEPS = 40
CHECKPOINTS = (12, 20, 28)
PROPAGATION_OFFSETS = (1, 2, 4)
REPAIR_OFFSET = 4  # d: normal steps executed after t before verification / repair (aligned with the primary stale window)
STALE_WINDOW_PRIMARY = 4
STALE_WINDOW_SECONDARY = 1  # descriptive magnitude / control arm only
ERROR_MODELS = ("STALE_VELOCITY", "STALE_VELOCITY_W1", "GAUSS_HIGH")
DECISION_ERROR = "STALE_VELOCITY"
GAUSS_HIGH_ALPHA = 0.30  # stress / calibration control only: fraction of per-channel clean latent std
MATERIAL_THRESHOLD = 0.02  # channel-normalised per-position RMS error > 2% of clean latent scale (~5x bf16 floor)
TRACE_THRESHOLD = 0.01  # descriptive only
RECOVERY_REL_L2 = 0.02
RECOVERY_SSIM = 0.95
RECOVERY_SENSITIVITY = (0.01, 0.02, 0.05, 0.10)
MATERIAL_CELL_FRACTION_MIN = 0.5  # STOP_NO_RECOVERY_PROBLEM if fewer decision cells have a material no-repair error
RCOMPUTE_STRONG_GO = 0.20
RCOMPUTE_GO = 0.40
RCOMPUTE_WEAK = 0.70
SPATIAL_DISTANCE_BINS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 64))
DECISION_REGIONS = ("SPATIAL_SMALL", "TEMPORAL_MEDIUM")


# --------------------------------------------------------------------------------------
# regions / recovery domains
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenBox:
    f: tuple[int, int]
    h: tuple[int, int]
    w: tuple[int, int]

    def token_mask(self) -> np.ndarray:
        mask = np.zeros(TOKEN_GRID, dtype=bool)
        mask[self.f[0]:self.f[1], self.h[0]:self.h[1], self.w[0]:self.w[1]] = True
        return mask

    @property
    def n_tokens(self) -> int:
        return (self.f[1] - self.f[0]) * (self.h[1] - self.h[0]) * (self.w[1] - self.w[0])

    @property
    def fraction(self) -> float:
        return self.n_tokens / N_TOKENS

    def to_json(self) -> dict[str, Any]:
        return {"f": list(self.f), "h": list(self.h), "w": list(self.w), "n_tokens": self.n_tokens, "fraction": self.fraction}


REGIONS: dict[str, TokenBox] = {
    "SPATIAL_SMALL": TokenBox((0, 9), (12, 18), (21, 31)),  # 6x10 tokens x 9 frames = 540 = 3.85%
    "TEMPORAL_MEDIUM": TokenBox((4, 5), (0, 30), (0, 52)),  # frame 4, whole frame = 1560 = 11.11%
}
REGION_KIND = {"SPATIAL_SMALL": "spatial", "TEMPORAL_MEDIUM": "temporal"}
FULL_DOMAIN = TokenBox((0, 9), (0, 30), (0, 52))
RECOVERY_DOMAINS: dict[str, dict[str, TokenBox]] = {
    "SPATIAL_SMALL": {  # centred on token (15, 26); h rows x w cols, all frames
        "R": REGIONS["SPATIAL_SMALL"],
        "20": TokenBox((0, 9), (7, 23), (16, 36)),  # 16x20 = 320/frame = 20.5%
        "40": TokenBox((0, 9), (3, 27), (13, 39)),  # 24x26 = 624 = 40.0%
        "70": TokenBox((0, 9), (1, 29), (7, 46)),  # 28x39 = 1092 = 70.0%
        "100": FULL_DOMAIN,
    },
    "TEMPORAL_MEDIUM": {  # frame 4 outward, whole frames
        "R": REGIONS["TEMPORAL_MEDIUM"],
        "22": TokenBox((4, 6), (0, 30), (0, 52)),
        "44": TokenBox((3, 7), (0, 30), (0, 52)),
        "67": TokenBox((2, 8), (0, 30), (0, 52)),
        "100": FULL_DOMAIN,
    },
}
# Frozen WEAK-band dependency-cone characterization (descriptive only; recompute arm; cannot upgrade WEAK to GO).
# Allowed only when 0.40 < worst-region median R_compute* <= 0.70. Domains sit strictly between the 40 and 70 grid points.
WEAK_BAND_DOMAINS: dict[str, dict[str, TokenBox]] = {
    "SPATIAL_SMALL": {"50": TokenBox((0, 9), (2, 28), (11, 41)), "60": TokenBox((0, 9), (2, 28), (8, 44))},  # 26x30 = 780 (50.0%), 26x36 = 936 (60.0%)
    "TEMPORAL_MEDIUM": {"56": TokenBox((2, 7), (0, 30), (0, 52))},  # frames 2-6 = 55.6%
}


def token_to_latent_mask(token_mask: np.ndarray) -> np.ndarray:
    if token_mask.shape != TOKEN_GRID:
        raise ValueError(f"token mask shape {token_mask.shape} != {TOKEN_GRID}")
    lat = np.repeat(np.repeat(np.repeat(token_mask, PATCH[0], axis=0), PATCH[1], axis=1), PATCH[2], axis=2)
    return np.broadcast_to(lat[None, None], LATENT_SHAPE).copy()


def position_mask(token_mask: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(np.repeat(token_mask, PATCH[0], axis=0), PATCH[1], axis=1), PATCH[2], axis=2)


def validate_regions() -> dict[str, Any]:
    doc = {}
    for name, box in REGIONS.items():
        assert box.token_mask().sum() == box.n_tokens
        doc[name] = {**box.to_json(), "kind": REGION_KIND[name]}
    for region, domains in RECOVERY_DOMAINS.items():
        base = REGIONS[region].token_mask()
        prev = 0
        for key, box in domains.items():
            m = box.token_mask()
            assert (m | base).sum() == m.sum(), f"{region}/{key} does not contain R"
            assert m.sum() >= prev, f"{region}/{key} not monotone"
            prev = m.sum()
        assert domains["100"].fraction == 1.0
    for region, extra in WEAK_BAND_DOMAINS.items():
        lo, hi = RECOVERY_DOMAINS[region][{"SPATIAL_SMALL": "40", "TEMPORAL_MEDIUM": "44"}[region]], RECOVERY_DOMAINS[region][{"SPATIAL_SMALL": "70", "TEMPORAL_MEDIUM": "67"}[region]]
        for key, box in extra.items():
            m, lm, hm = box.token_mask(), lo.token_mask(), hi.token_mask()
            assert (m | lm).sum() == m.sum() and (hm | m).sum() == hm.sum(), f"WEAK domain {region}/{key} not nested between the 40 and 70 grid points"
            assert lo.fraction < box.fraction < hi.fraction
    return doc


def geometry_document() -> dict[str, Any]:
    return {
        "latent_shape": list(LATENT_SHAPE), "patch_size": list(PATCH), "token_grid_fhw": list(TOKEN_GRID), "n_tokens": N_TOKENS, "n_latent_positions": N_POSITIONS,
        "token_index_formula": "f * (30 * 52) + h * 52 + w  (patch_embedding -> flatten(2) -> transpose(1, 2))",
        "token_to_latent": "token (f, h, w) <-> latent [:, :, f, 2h:2h+2, 2w:2w+2] (all 16 channels)",
        "attention": "full self-attention over all 14040 tokens in every block; 3D RoPE; the per-layer dependency cone is full-width by construction",
        "regions": validate_regions(),
        "recovery_domains": {r: {k: b.to_json() for k, b in d.items()} for r, d in RECOVERY_DOMAINS.items()},
        "weak_band_characterization_domains": {r: {k: b.to_json() for k, b in d.items()} for r, d in WEAK_BAND_DOMAINS.items()},
    }


# --------------------------------------------------------------------------------------
# scheduler algebra (mirrors WanEulerScheduler.step; validated on-host against probe records)
# --------------------------------------------------------------------------------------
def sigmas_from_timesteps(timesteps: list[float], num_train_timesteps: int = 1000) -> np.ndarray:
    """sigma_k for k = 0..len(timesteps)-1; the scheduler's public timesteps are sigmas[:-1] * num_train_timesteps."""
    return np.asarray(timesteps, dtype=np.float64) / float(num_train_timesteps)


def delta_sigma(sigmas: np.ndarray, k: int) -> float:
    """sigma_{k+1} - sigma_k, exactly the factor multiplying the model output in WanEulerScheduler.step at step k."""
    if k + 1 >= len(sigmas):
        raise ValueError(f"delta_sigma needs sigma_{k + 1}; only {len(sigmas)} sigmas known")
    return float(sigmas[k + 1] - sigmas[k])


def implied_velocity(x_k: np.ndarray, x_k1: np.ndarray, sigmas: np.ndarray, k: int) -> np.ndarray:
    """Effective (CFG-combined, bf16-realised) velocity v_k = (x_{k+1} - x_k) / (sigma_{k+1} - sigma_k)."""
    return (x_k1.astype(np.float64) - x_k.astype(np.float64)) / delta_sigma(sigmas, k)


def euler_step(x_k: np.ndarray, v_k: np.ndarray, sigmas: np.ndarray, k: int) -> np.ndarray:
    return (x_k.astype(np.float64) + delta_sigma(sigmas, k) * v_k.astype(np.float64)).astype(np.float32)


# --------------------------------------------------------------------------------------
# perturbation / error models
# --------------------------------------------------------------------------------------
def channel_std(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[1], -1).astype(np.float64).std(axis=1).astype(np.float32)


def bf16_round(x: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float32)
    bits = a.view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    out = rounded.astype(np.uint32).view(np.float32)
    return np.where(np.isfinite(a), out, a).astype(np.float32)


def _perturbation_stats(x_clean: np.ndarray, x_pert: np.ndarray, region: str, one_step_update: np.ndarray | None) -> dict[str, Any]:
    mask = token_to_latent_mask(REGIONS[region].token_mask())
    realized = (x_pert.astype(np.float32) - x_clean.astype(np.float32))
    inside = realized[mask].astype(np.float64)
    sigma = channel_std(x_clean).astype(np.float64)
    e = normalised_error_map(x_pert, x_clean)
    r = position_mask(REGIONS[region].token_mask())
    stats = {
        "region": region, "channel_std": sigma.tolist(),
        "delta_l2": float(np.linalg.norm(realized.astype(np.float64))),
        "relative_l2_in_region": float(np.linalg.norm(inside) / max(np.linalg.norm(x_clean[mask].astype(np.float64)), 1e-12)),
        "relative_l2_global": float(np.linalg.norm(realized.astype(np.float64)) / max(np.linalg.norm(x_clean.astype(np.float64)), 1e-12)),
        "max_abs": float(np.abs(realized).max()),
        "rms_in_region": float(np.sqrt(np.mean(inside ** 2))),
        "rms_in_region_over_mean_channel_std": float(np.sqrt(np.mean(inside ** 2)) / max(float(sigma.mean()), 1e-12)),
        "material_fraction_inside_region_at_injection": float((e > MATERIAL_THRESHOLD)[r].mean()),
        "affected_fraction_positions": float(REGIONS[region].fraction),
        "outside_region_changed": bool(np.any(realized[~mask] != 0)),
    }
    if one_step_update is not None:
        upd = one_step_update[mask].astype(np.float64)
        stats["one_step_update_rms_in_region"] = float(np.sqrt(np.mean(upd ** 2)))
        stats["rms_over_one_step_update_rms"] = float(stats["rms_in_region"] / max(stats["one_step_update_rms_in_region"], 1e-12))
    return stats


def stale_velocity_state(*, x_last_active_prev: np.ndarray, x_last_active: np.ndarray, x_t: np.ndarray, sigmas: np.ndarray, t: int, window: int, region: str) -> tuple[np.ndarray, dict[str, Any]]:
    """x'_t[R] = x_{t-L}[R] + (sigma_t - sigma_{t-L}) * v_{t-L-1}[R] with v from the last active step; bf16-realised; outside R clean.

    x_last_active_prev = x_{t-L-1}, x_last_active = x_{t-L}. With window L = 1 this is the previous-step velocity applied at step t-1.
    """
    if window < 1 or t - window - 1 < 0:
        raise ValueError("stale window out of range")
    v_stale = implied_velocity(x_last_active_prev, x_last_active, sigmas, t - window - 1)
    total_dsigma = float(sigmas[t] - sigmas[t - window])
    mask = token_to_latent_mask(REGIONS[region].token_mask())
    x_stale = (x_last_active.astype(np.float64) + total_dsigma * v_stale).astype(np.float32)
    x_pert = np.where(mask, bf16_round(x_stale), x_t.astype(np.float32)).astype(np.float32)
    stats = _perturbation_stats(x_t, x_pert, region, None)
    stats.update({"error_model": "STALE_VELOCITY" if window == STALE_WINDOW_PRIMARY else f"STALE_VELOCITY_W{window}", "window": window,
                  "last_active_step": t - window - 1, "delta_sigma_stale_step": delta_sigma(sigmas, t - window - 1), "total_delta_sigma_window": total_dsigma,
                  "formula": "x'_t[R] = x_{t-L}[R] + (sigma_t - sigma_{t-L}) * (x_{t-L} - x_{t-L-1})[R] / (sigma_{t-L} - sigma_{t-L-1})"})
    return x_pert, stats


def gaussian_state(x_clean: np.ndarray, *, trajectory_id: str, t: int, region: str, alpha: float = GAUSS_HIGH_ALPHA) -> tuple[np.ndarray, dict[str, Any]]:
    seed = int(hashlib.sha256(f"{trajectory_id}|{t}|{region}|GAUSS|{alpha}".encode()).hexdigest()[:16], 16)
    sigma = channel_std(x_clean)
    z = np.random.default_rng(seed).standard_normal(LATENT_SHAPE).astype(np.float32)
    mask = token_to_latent_mask(REGIONS[region].token_mask())
    delta = (alpha * sigma[None, :, None, None, None] * z).astype(np.float32)
    x_pert = np.where(mask, bf16_round(x_clean.astype(np.float32) + delta), x_clean.astype(np.float32)).astype(np.float32)
    stats = _perturbation_stats(x_clean, x_pert, region, None)
    stats.update({"error_model": "GAUSS_HIGH", "alpha": alpha, "seed": seed, "role": "stress / calibration control only; never decides"})
    return x_pert, stats


# --------------------------------------------------------------------------------------
# error maps and contamination
# --------------------------------------------------------------------------------------
def normalised_error_map(x: np.ndarray, x_clean: np.ndarray) -> np.ndarray:
    sigma = channel_std(x_clean).astype(np.float64)
    d = (x.astype(np.float64) - x_clean.astype(np.float64))[0] / sigma[:, None, None, None]
    return np.sqrt(np.mean(d * d, axis=0))


def _distance_maps(region: str) -> tuple[np.ndarray, np.ndarray]:
    box = REGIONS[region]
    f = np.arange(TOKEN_GRID[0])[:, None, None]
    h = np.arange(TOKEN_GRID[1])[None, :, None]
    w = np.arange(TOKEN_GRID[2])[None, None, :]
    dh = np.maximum(np.maximum(box.h[0] - h, h - (box.h[1] - 1)), 0)
    dw = np.maximum(np.maximum(box.w[0] - w, w - (box.w[1] - 1)), 0)
    df = np.maximum(np.maximum(box.f[0] - f, f - (box.f[1] - 1)), 0)
    return position_mask(np.broadcast_to(np.maximum(dh, dw), TOKEN_GRID)), position_mask(np.broadcast_to(df, TOKEN_GRID))


def contamination_metrics(x: np.ndarray, x_clean: np.ndarray, region: str, *, material=MATERIAL_THRESHOLD, trace=TRACE_THRESHOLD) -> dict[str, Any]:
    e = normalised_error_map(x, x_clean)
    r = position_mask(REGIONS[region].token_mask())
    energy = e * e
    total_energy = float(energy.sum())
    mat, tr = e > material, e > trace
    spatial_d, temporal_d = _distance_maps(region)
    frac_r = float(r.mean())
    out = {
        "total_relative_l2": float(np.linalg.norm((x.astype(np.float64) - x_clean.astype(np.float64)).ravel()) / max(np.linalg.norm(x_clean.astype(np.float64).ravel()), 1e-12)),
        "energy_fraction_inside_region": float(energy[r].sum() / total_energy) if total_energy > 0 else 1.0,
        "energy_fraction_outside_region": float(energy[~r].sum() / total_energy) if total_energy > 0 else 0.0,
        "region_fraction": frac_r, "material_fraction": float(mat.mean()), "material_fraction_inside_region": float(mat[r].mean()), "material_fraction_outside_region": float(mat[~r].mean()),
        "material_positions_outside_region": int(mat[~r].sum()), "trace_fraction": float(tr.mean()), "trace_fraction_outside_region": float(tr[~r].mean()),
        "expansion_C": float(mat.mean() / frac_r) if frac_r > 0 else None,
        "max_normalised_error_outside_region": float(e[~r].max()), "median_normalised_error_outside_region": float(np.median(e[~r])),
        "spatial_distance_histogram_material_outside": {}, "temporal_distance_histogram_material_outside": {},
    }
    outside_mat = mat & ~r
    for lo, hi in SPATIAL_DISTANCE_BINS:
        sel = (spatial_d >= lo) & (spatial_d <= hi) & ~r
        out["spatial_distance_histogram_material_outside"][f"{lo}-{hi}"] = {"positions": int(sel.sum()), "material": int((outside_mat & sel).sum()), "fraction": float((outside_mat & sel).sum() / sel.sum()) if sel.any() else None}
    for td in range(TOKEN_GRID[0]):
        sel = (temporal_d == td) & ~r
        if sel.any():
            out["temporal_distance_histogram_material_outside"][str(td)] = {"positions": int(sel.sum()), "material": int((outside_mat & sel).sum()), "fraction": float((outside_mat & sel).sum() / sel.sum())}
    return out


# --------------------------------------------------------------------------------------
# repair operators
# --------------------------------------------------------------------------------------
def oracle_repair(perturbed: np.ndarray, clean: np.ndarray, token_mask: np.ndarray) -> np.ndarray:
    """ORACLE state replacement: copy clean latent values inside the domain. Structural probe only."""
    mask = token_to_latent_mask(token_mask)
    return np.where(mask, clean, perturbed).astype(np.float32)


def compose_recompute_input(domain_mask: np.ndarray, domain_state: np.ndarray, cached_context: np.ndarray) -> np.ndarray:
    """Input of one recompute step: the domain carries its regenerated state, everything else the cached (perturbed-trajectory) context."""
    mask = token_to_latent_mask(domain_mask)
    return np.where(mask, domain_state, cached_context).astype(np.float32)


def compose_recompute_output(domain_mask: np.ndarray, step_output: np.ndarray, previous_domain_state: np.ndarray) -> np.ndarray:
    """Keep only the domain's outputs from a full target step; outside the domain nothing is consumed (its cost is the saving)."""
    mask = token_to_latent_mask(domain_mask)
    return np.where(mask, step_output, previous_domain_state).astype(np.float32)


# --------------------------------------------------------------------------------------
# convergence, R*, decision
# --------------------------------------------------------------------------------------
def final_convergence(x_final: np.ndarray, x_clean_final: np.ndarray, *, frame_ssim_mean: float | None) -> dict[str, Any]:
    rel = float(np.linalg.norm((x_final.astype(np.float64) - x_clean_final.astype(np.float64)).ravel()) / max(np.linalg.norm(x_clean_final.astype(np.float64).ravel()), 1e-12))
    e = normalised_error_map(x_final, x_clean_final)
    return {"final_relative_l2": rel, "final_material_fraction": float((e > MATERIAL_THRESHOLD).mean()), "frame_ssim_mean": frame_ssim_mean,
            "recovered": bool(rel <= RECOVERY_REL_L2 and (frame_ssim_mean is None or frame_ssim_mean >= RECOVERY_SSIM)),
            "recovered_by_rel_l2_threshold": {str(th): bool(rel <= th) for th in RECOVERY_SENSITIVITY}}


def r_star(sweep: dict[str, dict[str, Any]], fractions: dict[str, float], *, threshold_key: str | None = None) -> dict[str, Any]:
    """Minimum tested fraction such that recovery holds there and at every larger tested fraction; 1.0 if none."""
    keys = sorted(sweep, key=lambda k: fractions[k])
    ok = {k: (sweep[k]["recovered"] if threshold_key is None else sweep[k]["recovered_by_rel_l2_threshold"][threshold_key]) for k in keys}
    r = None
    for i, k in enumerate(keys):
        if all(ok[j] for j in keys[i:]):
            r = fractions[k]
            break
    violations = sum(1 for i, k in enumerate(keys) if ok[k] and any(not ok[j] for j in keys[i + 1:]))
    return {"r_star": r if r is not None else 1.0, "found": r is not None, "recovered_by_fraction": {k: ok[k] for k in keys}, "monotonicity_violations": violations}


def rcompute_band(median: float) -> str:
    if median <= RCOMPUTE_STRONG_GO:
        return "STRONG_GO"
    if median <= RCOMPUTE_GO:
        return "GO"
    if median <= RCOMPUTE_WEAK:
        return "WEAK"
    return "NO_GO"


def decide(*, valid: bool, invalid_reason: str | None, material_cell_fraction: float | None, rcompute_median_by_region: dict[str, float], rstate_median_by_region: dict[str, float]) -> dict[str, Any]:
    """Frozen gate. Only STALE_VELOCITY on the decision regions enters; Gaussian HIGH and R_state* never decide."""
    if not valid:
        return {"decision": "INVALID_EXPERIMENT", "rationale": invalid_reason or "control failure"}
    if material_cell_fraction is None or material_cell_fraction < MATERIAL_CELL_FRACTION_MIN:
        return {"decision": "STOP_NO_RECOVERY_PROBLEM", "rationale": f"only {material_cell_fraction} of decision cells have a material no-repair STALE_VELOCITY error; dense continuation already converges, so there is nothing to recover", "material_cell_fraction": material_cell_fraction}
    worst_c = max(rcompute_median_by_region.values())
    worst_s = max(rstate_median_by_region.values()) if rstate_median_by_region else None
    band = rcompute_band(worst_c)
    facts = {"material_cell_fraction": material_cell_fraction, "rcompute_median_by_region": rcompute_median_by_region, "rcompute_worst_region_median": worst_c, "rcompute_band": band,
             "rstate_median_by_region": rstate_median_by_region, "rstate_worst_region_median": worst_s,
             "oracle_only_headroom": bool(worst_s is not None and worst_s <= RCOMPUTE_GO < worst_c)}
    decision = {"STRONG_GO": "STRONG_GO_DEPENDENCY_BOUNDED_RECOMPUTE", "GO": "GO_DEPENDENCY_BOUNDED_RECOMPUTE", "WEAK": "WEAK_ONE_DEPENDENCY_CONE_CHARACTERIZATION_ALLOWED", "NO_GO": "NO_GO_GLOBAL_DENSE_REFRESH_REQUIRED"}[band]
    rationale = f"median R_compute* (worst region) = {worst_c:.3f} -> {band}"
    if facts["oracle_only_headroom"]:
        rationale += f"; R_state* median {worst_s:.3f} is small but not actionable (oracle-only headroom, item 8)"
    return {"decision": decision, "rationale": rationale, **facts}


def idealized_window_saving(domain_fraction: float, d: int = REPAIR_OFFSET) -> dict[str, float]:
    """Theoretical saved window work of SELECTIVE_STATE_RECOMPUTE(D) vs FULL_ROLLBACK, in full-forward units; the experiment executes d FULL forwards."""
    return {"full_rollback_forwards": float(d), "idealized_selective_forwards": float(d * domain_fraction), "idealized_saved_forwards": float(d * (1.0 - domain_fraction)), "experimental_forwards_executed": float(d)}


def frozen_thresholds() -> dict[str, Any]:
    return {
        "material_threshold": MATERIAL_THRESHOLD, "trace_threshold_descriptive": TRACE_THRESHOLD,
        "recovery_rel_l2": RECOVERY_REL_L2, "recovery_ssim": RECOVERY_SSIM, "recovery_sensitivity_descriptive": list(RECOVERY_SENSITIVITY),
        "material_cell_fraction_min": MATERIAL_CELL_FRACTION_MIN,
        "rcompute_strong_go": RCOMPUTE_STRONG_GO, "rcompute_go": RCOMPUTE_GO, "rcompute_weak": RCOMPUTE_WEAK,
        "gauss_high_alpha": GAUSS_HIGH_ALPHA, "stale_window_primary": STALE_WINDOW_PRIMARY, "stale_window_secondary": STALE_WINDOW_SECONDARY,
        "decision_error_model": DECISION_ERROR, "decision_regions": list(DECISION_REGIONS), "repair_offset_d": REPAIR_OFFSET,
        "weak_band_domains": {r: list(d) for r, d in WEAK_BAND_DOMAINS.items()},
    }


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
