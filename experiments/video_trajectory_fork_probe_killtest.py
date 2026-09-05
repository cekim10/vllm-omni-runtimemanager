#!/usr/bin/env python3
"""Round 4A: cheap fatal screening for a trajectory-conditioned forkability probe.

For every frozen OLD checkpoint x_k of the confirmatory fork experiment (3 seeds x k in
{10,15,20,25}) the trusted Wan2.2 Euler path is executed for exactly ONE step twice, once under
the OLD conditioning and once under the NEW conditioning, from the identical persisted state.
The within-step probe (audited in Phase 2) records the CFG-combined noise prediction that the
Euler scheduler consumes (``guidance_combined_output``) and the post-update latent
(``scheduler_output``). No trajectory is continued, no video is generated.

Primary preregistered signal:

    relative_response_l2 = ||p_new - p_old||_2 / (||p_old||_2 + eps)

Direction (frozen): larger => more responsive to the conditioning change => higher frozen
qualitative ordinal. The fatal gate is same-timestep discrimination at k=15:

    probe(2345, k15) < probe(3456, k15)  AND  probe(2345, k15) < probe(4567, k15)

Everything downstream of the frozen labels (which were blinded and finalised before this
experiment existed) is descriptive. No predictor, threshold, or mechanism is built.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_trajectory_fork_confirmatory as conf  # noqa: E402
from experiments import video_trajectory_fork_killtest as smoke  # noqa: E402

EXPERIMENT_VERSION = "video-trajectory-fork-probe-killtest-v1"
NAMESPACE = "video_trajectory_fork_probe_killtest"
DEFAULT_CONFIG = REPO_ROOT / "experiments/video_trajectory_fork_probe_killtest_config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / NAMESPACE
SOURCE_ROOT = REPO_ROOT / "results" / "video_trajectory_fork_confirmatory"
MODES = ("cpu", "probe", "analyze")
SEEDS = conf.SEEDS
KS = conf.SWITCHES
CONDITIONS = ("old", "new")
OLD_PROMPT = conf.OLD_PROMPT
NEW_PROMPT = conf.NEW_PROMPT
ORDINAL = conf.ORDINAL
EPS = 1e-12
AUDIT_POINT = (3456, 15)
PRIMARY_K = 15
SUPPORT_K = 20
LOW_SEED = 2345
HIGH_SEEDS = (3456, 4567)
CONFOUNDED_K = 25
RESPONSE_BOUNDARY = "guidance_combined_output"
UPDATE_BOUNDARY = "scheduler_output"
INPUT_BOUNDARY = "latent_entering_step"
TRUSTED_SOURCE_FILES = (
    "experiments/video_trajectory_fork_probe_killtest.py",
    "experiments/video_trajectory_fork_probe_killtest_config.yaml",
    "experiments/run_video_trajectory_fork_probe_killtest_gpu0.sh",
    "tests/diffusion/test_video_trajectory_fork_probe_killtest.py",
    "experiments/video_trajectory_fork_confirmatory.py",
    "experiments/video_trajectory_fork_killtest.py",
    "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
    "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
)
FORBIDDEN_OUTPUT_PARTS = conf.FORBIDDEN_OUTPUT_PARTS + ("video_trajectory_fork_confirmatory",)
GateError = smoke.GateError
canonical_json = smoke.canonical_json
sha256_bytes = smoke.sha256_bytes
sha256_file = smoke.sha256_file
array_sha256 = smoke.array_sha256
atomic_json = smoke.atomic_json
read_csv = smoke.read_csv
CSV_FIELDS = (
    "seed", "k", "condition", "role", "frozen_label", "ordinal", "prompt", "prompt_sha256", "resume_index", "timestep",
    "checkpoint_hash", "input_boundary_hash", "response_hash", "update_hash", "response_path", "update_path",
    "wall_time_s", "model_forwards", "execution_step_limit", "executed_local_steps", "vae_decode_skipped", "scheduler_class",
)


# --------------------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("experiment_version changed")
    if tuple(config["seeds"]) != SEEDS or tuple(config["k_values"]) != KS or tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("seeds / k values / conditions are frozen")
    if config["prompts"]["old"] != OLD_PROMPT or config["prompts"]["new"] != NEW_PROMPT:
        raise ValueError("prompt pair changed")
    if {k: int(v) for k, v in config["ordinal"].items()} != ORDINAL:
        raise ValueError("ordinal mapping changed")
    probe = config["probe"]
    if probe["primary_signal"] != "relative_response_l2" or float(probe["eps"]) != EPS or probe["direction"] != "higher_probe_means_more_forkable":
        raise ValueError("primary probe definition/direction is frozen")
    if probe["response_boundary"] != RESPONSE_BOUNDARY or probe["update_boundary"] != UPDATE_BOUNDARY or int(probe["execution_step_limit"]) != 1:
        raise ValueError("probe execution boundary is frozen")
    gate = config["gate"]
    if gate["primary_k"] != PRIMARY_K or gate["support_k"] != SUPPORT_K or gate["low_forkability_seed"] != LOW_SEED or tuple(gate["high_forkability_seeds"]) != HIGH_SEEDS or gate["confounded_k"] != CONFOUNDED_K:
        raise ValueError("same-k gate definition changed")
    if tuple(config["determinism_audit"]["point"]) != AUDIT_POINT or config["determinism_audit"]["tolerance"] != 0.0:
        raise ValueError("determinism audit point/tolerance changed")
    if float(gate["k15_gap_over_repeat_noise_min_ratio"]) != 10.0 or float(gate["k15_gap_abs_floor"]) != 1e-6:
        raise ValueError("materiality rule changed")
    if config["source"]["namespace"] != "video_trajectory_fork_confirmatory":
        raise ValueError("source namespace changed")


def validate_output_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT / "results")
    except ValueError as error:
        raise ValueError("Output must be under results/") from error
    if not relative.parts or relative.parts[0] != NAMESPACE:
        raise ValueError(f"Output must use the isolated {NAMESPACE} namespace")
    if any(part in str(relative) for part in FORBIDDEN_OUTPUT_PARTS):
        raise ValueError("Trusted prior-result namespace cannot be used")


# --------------------------------------------------------------------------------------
# frozen source artifacts (read-only)
# --------------------------------------------------------------------------------------
def load_source(config: dict[str, Any]) -> dict[str, Any]:
    root = SOURCE_ROOT
    prereg_path, sha_path = root / "preregistration.json", root / "preregistration.sha256"
    summary_path, preflight_path = root / "summary.json", root / "preflight.json"
    for path in (prereg_path, sha_path, summary_path, preflight_path):
        if not path.exists():
            raise GateError(f"Frozen confirmatory artifact missing: {path}")
    recorded = sha_path.read_text().split()[0]
    if sha256_file(prereg_path) != recorded:
        raise GateError("Confirmatory preregistration hash mismatch")
    source_pins = config["source"]
    if source_pins.get("preregistration_sha256") not in (None, recorded):
        raise GateError("Config pins a different confirmatory preregistration")
    summary = json.loads(summary_path.read_text())
    summary_sha = sha256_file(summary_path)
    if source_pins.get("summary_sha256") not in (None, summary_sha):
        raise GateError("Config pins a different confirmatory summary")
    if summary.get("decision") != "GO" or summary.get("preregistration_sha256") != recorded:
        raise GateError("Confirmatory summary is not the frozen GO result bound to its preregistration")
    prereg = json.loads(prereg_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS" or preflight.get("provenance_hash") != summary.get("provenance_hash"):
        raise GateError("Confirmatory controls not PASS or unbound")
    if prereg["prompts"]["old"] != OLD_PROMPT or prereg["prompts"]["new"] != NEW_PROMPT or tuple(prereg["seeds"]) != SEEDS or tuple(prereg["switch_steps"]) != KS:
        raise GateError("Confirmatory prompts/seeds/switch points differ from this experiment's frozen values")
    labels: dict[str, dict[str, str]] = {}
    for seed in SEEDS:
        entry = summary["per_seed"][str(seed)]
        if entry.get("status") != "LABELLED":
            raise GateError(f"Confirmatory seed {seed} is not labelled")
        labels[str(seed)] = {str(k): entry["labels"][str(k)] for k in KS}
        for k in KS:
            if labels[str(seed)][str(k)] not in ORDINAL:
                raise GateError("Confirmatory label outside the frozen scale")
    checkpoint_hashes = {str(seed): {str(k): preflight["seeds"][str(seed)]["old_checkpoint_hashes"][str(k)] for k in KS} for seed in SEEDS}
    schedule = [float(v) for v in prereg["scheduler_plan"]["timesteps"]]
    return {
        "root": str(root.relative_to(REPO_ROOT)),
        "preregistration_sha256": recorded,
        "summary_sha256": summary_sha,
        "preflight_sha256": sha256_file(preflight_path),
        "provenance_hash": summary["provenance_hash"],
        "decision": summary["decision"],
        "labels": labels,
        "checkpoint_hashes": checkpoint_hashes,
        "schedule": schedule,
        "scheduler_class": prereg["scheduler_plan"]["scheduler_class"],
        "generation": prereg["generation"],
        "scheduler": prereg["scheduler"],
        "model": prereg["model"],
        "expert_by_switch": prereg["expert_by_switch"],
    }


def checkpoint_path(seed: int, k: int) -> Path:
    metadata = json.loads((SOURCE_ROOT / "rows" / f"red_to_blue_seed{seed}" / "old_baseline_metadata.json").read_text())["worker_trajectory_probe"]
    records = {int(r["step_index"]): r for r in metadata["records"]}
    if k not in records:
        raise GateError(f"Confirmatory OLD trajectory did not capture step {k} for seed {seed}")
    return Path(records[k]["latent_path"])


# --------------------------------------------------------------------------------------
# preregistration / provenance
# --------------------------------------------------------------------------------------
def build_provenance(config_path: Path) -> dict[str, Any]:
    paths = [REPO_ROOT / value for value in TRUSTED_SOURCE_FILES]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise GateError(f"Provenance inputs missing: {missing}")
    hashes = {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).splitlines()
    except Exception:
        commit, status = "UNKNOWN", ["git status unavailable"]
    relevant = [line for line in status if any(name in line for name in TRUSTED_SOURCE_FILES)]
    document = {"git_commit": commit, "git_dirty": bool(status), "git_status": status, "relevant_git_status": relevant, "source_sha256": hashes, "config_sha256": sha256_file(config_path)}
    document["provenance_hash"] = sha256_bytes(canonical_json({k: document[k] for k in ("git_commit", "relevant_git_status", "source_sha256", "config_sha256")}))
    return document


def expected_keys() -> set[tuple[int, int, str, str]]:
    keys = {(seed, k, cond, "primary") for seed in SEEDS for k in KS for cond in CONDITIONS}
    keys |= {(AUDIT_POINT[0], AUDIT_POINT[1], cond, "audit") for cond in CONDITIONS}
    return keys


def row_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return int(row["seed"]), int(row["k"]), str(row["condition"]), str(row["role"])


def validate_key_set(rows: list[dict[str, Any]], expected: set[tuple[int, int, str, str]]) -> None:
    keys = [row_key(r) for r in rows]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates or set(keys) != expected:
        raise GateError(f"Result key-set mismatch: missing={sorted(expected - set(keys))}, unexpected={sorted(set(keys) - expected)}, duplicates={duplicates}")


def build_preregistration(config: dict[str, Any], provenance: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "title": "Round 4A - Trajectory-Conditioned Forkability Probe Kill Test",
        "broken_assumption_hypothesis": "Existing request-level reuse policies cannot distinguish realized trajectories that share the same request pair and timestep but have different downstream forkability.",
        "research_question": "At the same prompt pair and same denoising timestep, can a cheap trajectory-state probe distinguish trajectories with different downstream forkability?",
        "source": {k: source[k] for k in ("root", "preregistration_sha256", "summary_sha256", "preflight_sha256", "provenance_hash", "decision")},
        "source_checkpoint_hashes": source["checkpoint_hashes"],
        "frozen_labels": source["labels"],
        "frozen_labels_note": "blinded five-level labels finalised in the confirmatory experiment before this probe was designed",
        "ordinal": ORDINAL,
        "prompts": {"old": OLD_PROMPT, "new": NEW_PROMPT},
        "seeds": list(SEEDS),
        "k_values": list(KS),
        "schedule_timesteps_by_k": {str(k): source["schedule"][k] for k in KS},
        "expert_by_k": source["expert_by_switch"],
        "model": source["model"],
        "scheduler": source["scheduler"],
        "generation": source["generation"],
        "probe": config["probe"],
        "secondary_signals": config["secondary_signals"],
        "gate": config["gate"],
        "determinism_audit": config["determinism_audit"],
        "decision_rule": config["decision_rule"],
        "expected_keys": sorted([list(k) for k in expected_keys()]),
        "expected_local_responses": len(expected_keys()),
        "source_commit": provenance["git_commit"],
        "provenance_hash": provenance["provenance_hash"],
        "config_sha256": provenance["config_sha256"],
        "claim_boundary": config["claim_boundary"],
    }


def require_preregistration(output_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    prereg_path, sha_path = output_dir / "preregistration.json", output_dir / "preregistration.sha256"
    if not prereg_path.exists() or not sha_path.exists():
        raise GateError("Run the cpu phase first; preregistration is missing")
    if sha256_file(prereg_path) != sha_path.read_text().split()[0]:
        raise GateError("preregistration.json was modified after it was hashed")
    document = json.loads(prereg_path.read_text())
    frozen = json.loads((output_dir / "provenance.json").read_text())
    if document.get("provenance_hash") != provenance["provenance_hash"] or frozen.get("provenance_hash") != provenance["provenance_hash"]:
        raise GateError("Code/config provenance changed after preregistration; use a fresh output namespace")
    return document


def run_cpu(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(config_path)
    source = load_source(config)
    prereg_path, sha_path = output_dir / "preregistration.json", output_dir / "preregistration.sha256"
    if prereg_path.exists():
        existing = json.loads(prereg_path.read_text())
        if existing.get("provenance_hash") != provenance["provenance_hash"]:
            raise GateError("Output namespace already holds a preregistration from different code/config; use a fresh namespace")
        if (output_dir / "raw_probe_results.csv").exists():
            return {"mode": "cpu", "status": "ALREADY_FROZEN", "preregistration_sha256": sha_path.read_text().split()[0]}
    document = build_preregistration(config, provenance, source)
    atomic_json(output_dir / "provenance.json", provenance)
    atomic_json(prereg_path, document)
    digest = sha256_file(prereg_path)
    sha_path.write_text(f"{digest}  preregistration.json\n")
    return {"mode": "cpu", "status": "FROZEN", "preregistration_sha256": digest, "expected_local_responses": document["expected_local_responses"], "frozen_labels": source["labels"]}


# --------------------------------------------------------------------------------------
# GPU probe (never executed by the auditor)
# --------------------------------------------------------------------------------------
def probe_sampling_params(source: dict[str, Any], *, seed: int, k: int, label: str, artifact_dir: Path, latents: Any) -> Any:
    import torch

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    generation = source["generation"]
    sampling = OmniDiffusionSamplingParams(
        height=int(generation["height"]), width=int(generation["width"]), num_frames=int(generation["num_frames"]),
        num_inference_steps=int(generation["num_inference_steps"]), guidance_scale=float(generation["guidance_scale"]), fps=float(generation["fps"]),
        seed=seed, generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    sampling.latents = latents.detach().cpu().clone()
    sampling.step_index = int(k)
    sampling.extra_args = {
        "flow_shift": float(source["scheduler"]["flow_shift"]),
        "sample_solver": "euler",
        "execution_step_limit": 1,
        "skip_vae_decode": True,
        "within_step_probe": {"artifact_dir": str(artifact_dir), "request_label": label, "selected_local_step": 0, "selected_absolute_step": int(k)},
        "trajectory_probe": {"artifact_dir": str(artifact_dir / "trajectory_probe"), "request_label": label, "capture_steps": [0, 1], "fps": float(generation["fps"]), "save_decoded": False, "save_latents": True, "save_mp4": False},
    }
    return sampling


def _load_pt(path: str | Path) -> np.ndarray:
    import torch

    return torch.load(str(path), map_location="cpu", weights_only=True).detach().cpu().float().contiguous().numpy()


def run_probe(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from experiments.video_runtime_state_discovery import build_omni
    from vllm_omni.outputs import OmniRequestOutput

    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    source = load_source(config)
    if source["checkpoint_hashes"] != prereg["source_checkpoint_hashes"] or source["labels"] != prereg["frozen_labels"]:
        raise GateError("Frozen confirmatory artifacts changed since preregistration")
    schedule = source["schedule"]
    engine_config = {"model": source["model"], "scheduler": source["scheduler"], "generation": source["generation"]}
    omni = build_omni(engine_config, args)
    rows: list[dict[str, Any]] = []
    try:
        plan = [(seed, k, cond, "primary") for seed in SEEDS for k in KS for cond in CONDITIONS] + [(AUDIT_POINT[0], AUDIT_POINT[1], cond, "audit") for cond in CONDITIONS]
        for seed, k, cond, role in plan:
            expected_hash = source["checkpoint_hashes"][str(seed)][str(k)]
            ckpt_path = checkpoint_path(seed, k)
            x_k = _load_pt(ckpt_path)
            if array_sha256(x_k) != expected_hash:
                raise GateError(f"OLD checkpoint hash mismatch for seed {seed} k={k}")
            prompt = OLD_PROMPT if cond == "old" else NEW_PROMPT
            label = f"seed{seed}_k{k:02d}_{cond}_{role}"
            artifact_dir = output_dir / "probes" / label
            started = time.perf_counter()
            outputs = omni.generate({"prompt": prompt}, probe_sampling_params(source, seed=seed, k=k, label=label, artifact_dir=artifact_dir, latents=torch.from_numpy(np.ascontiguousarray(x_k))))
            wall = time.perf_counter() - started
            output = OmniRequestOutput.unwrap_result(outputs)
            custom = output.custom_output or {}
            control = custom.get("execution_control") or {}
            within = custom.get("within_step_probe") or {}
            records = {r["boundary"]: r for r in within.get("records", [])}
            for boundary in (INPUT_BOUNDARY, "transformer_input", RESPONSE_BOUNDARY, UPDATE_BOUNDARY):
                if boundary not in records:
                    raise GateError(f"Within-step probe missing boundary {boundary} for {label}")
            if control.get("execution_step_limit") != 1 or control.get("executed_local_steps") != 1 or control.get("resume_step_index") != k or control.get("vae_decode_skipped") is not True:
                raise GateError(f"Bounded execution semantics violated for {label}: {control}")
            entering = _load_pt(records[INPUT_BOUNDARY]["latent_path"])
            if array_sha256(entering) != expected_hash:
                raise GateError(f"State entering the step differs from the persisted checkpoint for {label} (mutation or cast)")
            timestep = float(records["transformer_input"]["timestep"])
            if abs(timestep - schedule[k]) > 1e-3:
                raise GateError(f"Execution boundary timestep {timestep} != schedule[{k}]={schedule[k]} for {label}")
            if records[RESPONSE_BOUNDARY]["runtime_dtype"] != "torch.bfloat16":
                raise GateError(f"Response runtime dtype is not bfloat16 for {label}")
            response = _load_pt(records[RESPONSE_BOUNDARY]["latent_path"])
            update = _load_pt(records[UPDATE_BOUNDARY]["latent_path"])
            trajectory_meta = json.loads(Path(custom["trajectory_probe_metadata_path"]).read_text())
            traj = {int(r["step_index"]): r for r in trajectory_meta["records"]}
            if set(traj) != {0, 1} or not np.array_equal(_load_pt(traj[1]["latent_path"]), update):
                raise GateError(f"Trajectory probe does not confirm a single executed update for {label}")
            if not str(trajectory_meta.get("scheduler_class", "")).endswith(smoke.EXPECTED_SCHEDULER):
                raise GateError("Worker did not execute the trusted Euler scheduler")
            rel_dir = Path("responses") / label
            (output_dir / rel_dir).mkdir(parents=True, exist_ok=True)
            response_path = output_dir / rel_dir / "response.npy"
            update_path = output_dir / rel_dir / "update.npy"
            np.save(response_path, response, allow_pickle=False)
            np.save(update_path, update, allow_pickle=False)
            rows.append({
                "seed": seed, "k": k, "condition": cond, "role": role,
                "frozen_label": source["labels"][str(seed)][str(k)], "ordinal": ORDINAL[source["labels"][str(seed)][str(k)]],
                "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode()), "resume_index": k, "timestep": timestep,
                "checkpoint_hash": expected_hash, "input_boundary_hash": array_sha256(entering), "response_hash": array_sha256(response), "update_hash": array_sha256(update),
                "response_path": str(rel_dir / "response.npy"), "update_path": str(rel_dir / "update.npy"),
                "wall_time_s": wall, "model_forwards": 2, "execution_step_limit": 1, "executed_local_steps": 1, "vae_decode_skipped": True,
                "scheduler_class": trajectory_meta.get("scheduler_class"),
            })
            _write_rows(output_dir, rows)
    finally:
        omni.shutdown()
    validate_key_set(rows, expected_keys())
    _write_rows(output_dir, rows)
    return {"mode": "probe", "rows": len(rows), "next": "run analyze"}


def _write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    path = output_dir / "raw_probe_results.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=row_key))
    tmp.replace(path)


# --------------------------------------------------------------------------------------
# signals (frozen)
# --------------------------------------------------------------------------------------
def signals(x_k: np.ndarray | None, p_old: np.ndarray, p_new: np.ndarray, x_next_old: np.ndarray | None, x_next_new: np.ndarray | None) -> dict[str, float]:
    po = p_old.astype(np.float64).ravel()
    pn = p_new.astype(np.float64).ravel()
    delta = pn - po
    out = {
        "relative_response_l2": float(np.linalg.norm(delta) / (np.linalg.norm(po) + EPS)),
        "response_delta_rms": float(np.sqrt(np.mean(delta * delta))),
        "response_cosine": float(np.dot(pn, po) / ((np.linalg.norm(pn) * np.linalg.norm(po)) + EPS)),
        "differing_element_fraction": float(np.count_nonzero(p_new != p_old) / p_old.size),
        "delta_over_response_rms": float(np.sqrt(np.mean(delta * delta)) / (np.sqrt(np.mean(po * po)) + EPS)),
    }
    if x_k is not None and x_next_old is not None and x_next_new is not None:
        d = x_next_new.astype(np.float64).ravel() - x_next_old.astype(np.float64).ravel()
        out["relative_step_effect"] = float(np.linalg.norm(d) / (np.linalg.norm(x_k.astype(np.float64).ravel()) + EPS))
    return out


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for idx in order[i:j + 1]:
            ranks[idx] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(list(map(float, x))), _ranks(list(map(float, y)))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else num / den


def same_k_pass(probe_at_k: dict[int, float], low: int = LOW_SEED, highs: tuple[int, ...] = HIGH_SEEDS) -> dict[str, Any]:
    comparisons = {str(high): {"low": probe_at_k[low], "high": probe_at_k[high], "pass": probe_at_k[low] < probe_at_k[high], "gap": probe_at_k[high] - probe_at_k[low]} for high in highs}
    return {"pass": all(c["pass"] for c in comparisons.values()), "comparisons": comparisons, "min_gap": min(c["gap"] for c in comparisons.values())}


def within_k_concordance(probe: dict[int, float], ordinal: dict[int, int]) -> dict[str, Any]:
    seeds = sorted(probe)
    concordant = discordant = tied = 0
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = seeds[i], seeds[j]
            do, dp = ordinal[b] - ordinal[a], probe[b] - probe[a]
            if do == 0:
                tied += 1
            elif (do > 0) == (dp > 0) and dp != 0:
                concordant += 1
            else:
                discordant += 1
    return {"probe_order_low_to_high": [s for s in sorted(seeds, key=lambda s: probe[s])], "label_order_low_to_high": [s for s in sorted(seeds, key=lambda s: ordinal[s])], "concordant": concordant, "discordant": discordant, "tied_label_pairs": tied, "contradicts_labels": discordant > 0}


def decide(*, valid: bool, invalid_reason: str | None, same_k15: dict[str, Any], same_k20: dict[str, Any], determinism_pass: bool, repeat_abs_diff: float, rho_all: float | None, rho_without_k25: float | None) -> dict[str, Any]:
    if not valid:
        return {"decision": "INVALID", "rationale": invalid_reason or "hard control failed", "ROUND4B_ELIGIBLE": False}
    if not same_k15["pass"]:
        return {"decision": "NO-GO", "rationale": "primary probe does not rank seed 2345 below both 3456 and 4567 at k=15", "ROUND4B_ELIGIBLE": False}
    material = same_k15["min_gap"] > max(10.0 * repeat_abs_diff, 1e-6)
    promising = same_k20["pass"] and determinism_pass and material and rho_all is not None and rho_all > 0 and rho_without_k25 is not None and rho_without_k25 > 0
    if promising:
        return {"decision": "PROMISING", "rationale": "k15 primary pass, k20 support pass, determinism pass, k15 gaps material vs repeat noise, rho_all > 0 and rho_without_k25 > 0", "ROUND4B_ELIGIBLE": True}
    reasons = []
    if not same_k20["pass"]:
        reasons.append("k20 supporting contrast fails")
    if not material:
        reasons.append("k15 separation not material relative to repeat noise / absolute floor")
    if rho_all is None or rho_all <= 0:
        reasons.append("rho_all not positive")
    if rho_without_k25 is None or rho_without_k25 <= 0:
        reasons.append("rho_without_k25 not positive")
    if not determinism_pass:
        reasons.append("determinism audit failed")
    return {"decision": "WEAK-PASS", "rationale": "k15 primary pass but: " + "; ".join(reasons), "ROUND4B_ELIGIBLE": False}


def run_analyze(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_preregistration(output_dir, provenance)
    if (output_dir / "summary.json").exists():
        raise GateError("summary.json already exists; the frozen analysis runs exactly once")
    source = load_source(config)
    if source["checkpoint_hashes"] != prereg["source_checkpoint_hashes"] or source["labels"] != prereg["frozen_labels"]:
        raise GateError("Frozen confirmatory artifacts changed since preregistration")
    rows = read_csv(output_dir / "raw_probe_results.csv")
    validate_key_set(rows, expected_keys())
    invalid: list[str] = []
    for r in rows:
        if r["checkpoint_hash"] != source["checkpoint_hashes"][r["seed"]][r["k"]] or r["input_boundary_hash"] != r["checkpoint_hash"]:
            invalid.append(f"checkpoint/input binding {row_key(r)}")
        if r["prompt"] != (OLD_PROMPT if r["condition"] == "old" else NEW_PROMPT):
            invalid.append(f"prompt {row_key(r)}")
        if int(r["execution_step_limit"]) != 1 or int(r["executed_local_steps"]) != 1 or r["vae_decode_skipped"] != "True" or not str(r["scheduler_class"]).endswith(smoke.EXPECTED_SCHEDULER):
            invalid.append(f"execution semantics {row_key(r)}")
        if abs(float(r["timestep"]) - source["schedule"][int(r["k"])]) > 1e-3:
            invalid.append(f"timestep {row_key(r)}")
        for field in ("response_path", "update_path"):
            arr = np.load(output_dir / r[field], allow_pickle=False)
            if array_sha256(arr) != r[field.replace("_path", "_hash")]:
                invalid.append(f"artifact hash {row_key(r)} {field}")
            if not np.all(np.isfinite(arr)):
                invalid.append(f"non-finite {row_key(r)} {field}")
    by = {row_key(r): r for r in rows}
    ckpt_cache: dict[tuple[int, int], np.ndarray] = {}

    def x_k(seed: int, k: int) -> np.ndarray | None:
        try:
            if (seed, k) not in ckpt_cache:
                ckpt_cache[(seed, k)] = _load_pt(checkpoint_path(seed, k))
            return ckpt_cache[(seed, k)]
        except Exception:
            return None

    table = []
    probe: dict[int, dict[int, float]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for k in KS:
            ro, rn = by[(seed, k, "old", "primary")], by[(seed, k, "new", "primary")]
            p_old = np.load(output_dir / ro["response_path"], allow_pickle=False); p_new = np.load(output_dir / rn["response_path"], allow_pickle=False)
            u_old = np.load(output_dir / ro["update_path"], allow_pickle=False); u_new = np.load(output_dir / rn["update_path"], allow_pickle=False)
            sig = signals(x_k(seed, k), p_old, p_new, u_old, u_new)
            probe[seed][k] = sig["relative_response_l2"]
            table.append({"seed": seed, "k": k, "frozen_label": source["labels"][str(seed)][str(k)], "ordinal": ORDINAL[source["labels"][str(seed)][str(k)]], "timestep": float(ro["timestep"]), "expert": source["expert_by_switch"][str(k)]["current_expert"], **sig, "wall_old_s": float(ro["wall_time_s"]), "wall_new_s": float(rn["wall_time_s"])})
    # determinism audit (bit-exact expected on the trusted path)
    audit_seed, audit_k = AUDIT_POINT
    a_old, a_new = by[(audit_seed, audit_k, "old", "audit")], by[(audit_seed, audit_k, "new", "audit")]
    p_old_primary = np.load(output_dir / by[(audit_seed, audit_k, "old", "primary")]["response_path"], allow_pickle=False)
    p_new_primary = np.load(output_dir / by[(audit_seed, audit_k, "new", "primary")]["response_path"], allow_pickle=False)
    p_old_audit = np.load(output_dir / a_old["response_path"], allow_pickle=False); p_new_audit = np.load(output_dir / a_new["response_path"], allow_pickle=False)
    audit_signal = signals(None, p_old_audit, p_new_audit, None, None)["relative_response_l2"]
    repeat_abs_diff = abs(audit_signal - probe[audit_seed][audit_k])
    determinism = {"point": list(AUDIT_POINT), "responses_bit_exact": bool(np.array_equal(p_old_primary, p_old_audit) and np.array_equal(p_new_primary, p_new_audit)), "primary_signal_first": probe[audit_seed][audit_k], "primary_signal_repeat": audit_signal, "abs_diff": repeat_abs_diff, "tolerance": 0.0}
    determinism["pass"] = determinism["responses_bit_exact"] and repeat_abs_diff <= 0.0
    if not determinism["pass"]:
        invalid.append("determinism audit failed")
    same_k15 = same_k_pass({s: probe[s][PRIMARY_K] for s in SEEDS})
    same_k20 = same_k_pass({s: probe[s][SUPPORT_K] for s in SEEDS})
    concordance = {str(k): within_k_concordance({s: probe[s][k] for s in SEEDS}, {s: ORDINAL[source["labels"][str(s)][str(k)]] for s in SEEDS}) for k in KS}
    pts = [(s, k) for s in SEEDS for k in KS]
    ordv = [ORDINAL[source["labels"][str(s)][str(k)]] for s, k in pts]
    prv = [probe[s][k] for s, k in pts]
    rho_all = spearman(prv, ordv)
    sub = [(s, k) for s, k in pts if k != CONFOUNDED_K]
    rho_without_k25 = spearman([probe[s][k] for s, k in sub], [ORDINAL[source["labels"][str(s)][str(k)]] for s, k in sub])
    timestep_only_rho = spearman([-k for _, k in pts], ordv)
    probe_vs_timestep_rho = spearman([k for _, k in pts], prv)
    walls = [float(r["wall_time_s"]) for r in rows]
    cost = {"responses": len(rows), "model_forwards_total": 2 * len(rows), "mean_wall_s_old": float(np.mean([float(r["wall_time_s"]) for r in rows if r["condition"] == "old"])), "mean_wall_s_new": float(np.mean([float(r["wall_time_s"]) for r in rows if r["condition"] == "new"])), "total_wall_s": float(np.sum(walls)), "note": "client wall per one-step request (includes text encoding and engine overhead); a full 40-step generation on this host measured ~237-278 s in the confirmatory run; peak GPU memory not measured"}
    verdict = decide(valid=not invalid, invalid_reason="; ".join(invalid) if invalid else None, same_k15=same_k15, same_k20=same_k20, determinism_pass=determinism["pass"], repeat_abs_diff=repeat_abs_diff, rho_all=rho_all, rho_without_k25=rho_without_k25)
    summary = {
        **verdict,
        "SAME_K_PRIMARY_PASS": same_k15["pass"], "SAME_K20_SUPPORT": same_k20["pass"],
        "same_k15": same_k15, "same_k20": same_k20, "within_k_concordance": concordance,
        "rho_all": rho_all, "rho_without_k25": rho_without_k25, "timestep_only_rho": timestep_only_rho, "probe_vs_timestep_rho": probe_vs_timestep_rho,
        "determinism": determinism, "invalid_reasons": invalid, "cost": cost,
        "probe_matrix": {str(s): {str(k): probe[s][k] for k in KS} for s in SEEDS}, "label_matrix": source["labels"], "table": table,
        "preregistration_sha256": (output_dir / "preregistration.sha256").read_text().split()[0], "provenance_hash": provenance["provenance_hash"], "source": prereg["source"], "claim_boundary": prereg["claim_boundary"],
    }
    atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "probe_matrix.csv").open("w", newline="") as handle:
        w = csv.writer(handle); w.writerow(["seed", *[f"k{k}" for k in KS]])
        for s in SEEDS:
            w.writerow([s, *[f"{probe[s][k]:.8f}" for k in KS]])
        w.writerow(["label", *["" for _ in KS]])
        for s in SEEDS:
            w.writerow([s, *[source["labels"][str(s)][str(k)] for k in KS]])
    write_report(output_dir, summary)
    print_console(summary, provenance)
    return {k: v for k, v in summary.items() if k not in ("table", "within_k_concordance")}


def write_report(output_dir: Path, s: dict[str, Any]) -> None:
    L = ["# Round 4A - Trajectory-Conditioned Forkability Probe Kill Test", "", "## Decision", "", s["decision"], "", "## Broken-assumption hypothesis", "",
         "Existing request-level reuse policies cannot distinguish realized trajectories that share the same request pair and timestep but have different downstream forkability.", "",
         "## Frozen prior evidence", "", "Qualitative labels (blinded, finalised in the confirmatory experiment BEFORE this probe was designed or run):", "", "| seed | k10 | k15 | k20 | k25 |", "|---|---|---|---|---|"]
    for seed in SEEDS:
        L.append(f"| {seed} | " + " | ".join(s["label_matrix"][str(seed)][str(k)] for k in KS) + " |")
    L += ["", "## Probe", "", "One trusted Euler step from the persisted OLD checkpoint x_k under OLD and under NEW conditioning; the CFG-combined noise prediction consumed by the scheduler (`guidance_combined_output`) is the response. Primary signal: relative_response_l2 = ||p_new - p_old|| / (||p_old|| + 1e-12); frozen direction: higher = more forkable.", "",
          "## Primary table", "", "| seed | k | frozen label | ordinal | relative response L2 | delta RMS | cosine | diff fraction | step effect |", "|---|---|---|---|---|---|---|---|---|"]
    for t in s["table"]:
        L.append(f"| {t['seed']} | {t['k']} | {t['frozen_label']} | {t['ordinal']:+d} | {t['relative_response_l2']:.6f} | {t['response_delta_rms']:.6f} | {t['response_cosine']:.6f} | {t['differing_element_fraction']:.4f} | {t.get('relative_step_effect', float('nan')):.6f} |")
    L += ["", "## Primary fatal test (k=15)", "", "| k | lower-forkability seed | comparison seed | expected | observed | pass |", "|---|---|---|---|---|---|"]
    for k, block in ((15, s["same_k15"]), (20, s["same_k20"])):
        for high, c in block["comparisons"].items():
            L.append(f"| {k} | {LOW_SEED} | {high} | probe({LOW_SEED}) < probe({high}) | {c['low']:.6f} vs {c['high']:.6f} | {c['pass']} |")
    L += ["", f"SAME_K_PRIMARY_PASS: {s['SAME_K_PRIMARY_PASS']}", f"SAME_K20_SUPPORT: {s['SAME_K20_SUPPORT']}", "", "## Within-k concordance", ""]
    for k, c in s["within_k_concordance"].items():
        L.append(f"- k{k}: probe order {c['probe_order_low_to_high']}, label order {c['label_order_low_to_high']}, concordant={c['concordant']}, discordant={c['discordant']}, tied label pairs={c['tied_label_pairs']}, contradicts labels={c['contradicts_labels']}")
    L += ["", "## Timestep confound (secondary)", "", f"- rho_all (probe vs ordinal, 12 points): {s['rho_all']}", f"- rho_without_k25: {s['rho_without_k25']}", f"- timestep-only baseline Spearman(-k, ordinal): {s['timestep_only_rho']}", f"- Spearman(k, probe): {s['probe_vs_timestep_rho']}", "",
          "## Determinism audit", "", f"- point {s['determinism']['point']}: responses bit-exact={s['determinism']['responses_bit_exact']}, primary signal first={s['determinism']['primary_signal_first']:.8f}, repeat={s['determinism']['primary_signal_repeat']:.8f}, |diff|={s['determinism']['abs_diff']}", "",
          "## Cost", "", f"- {s['cost']['responses']} local responses, {s['cost']['model_forwards_total']} model forwards; mean wall OLD {s['cost']['mean_wall_s_old']:.1f} s, NEW {s['cost']['mean_wall_s_new']:.1f} s; total {s['cost']['total_wall_s']:.0f} s. {s['cost']['note']}", "",
          "## Decision rationale", "", s["rationale"], "", "## What this result does NOT establish", "",
          "- Not a general predictor of safe rollback depth; no threshold or model was fit.", "- No claim across prompts, edit types, or models; one prompt pair, three seeds.", "- No serving-performance or NIRVANA-comparison claim; prefix fractions are not realized speedups.", "- CLIP-free here, but the frozen labels are human judgments on one edit type.", "- k=25 coincides with the Wan2.2 high-noise expert boundary and is reported with and without.", f"- ROUND4B_ELIGIBLE = {str(s['ROUND4B_ELIGIBLE']).lower()}", ""]
    (output_dir / "video_trajectory_fork_probe_killtest.md").write_text("\n".join(L))


def print_console(s: dict[str, Any], provenance: dict[str, Any]) -> None:
    print(f"git_commit={provenance['git_commit']} relevant_dirty={provenance['relevant_git_status']}")
    print(f"preregistration_sha256={s['preregistration_sha256']}")
    print(f"SAME_K_PRIMARY_PASS={s['SAME_K_PRIMARY_PASS']} SAME_K20_SUPPORT={s['SAME_K20_SUPPORT']}")
    print(f"rho_all={s['rho_all']} rho_without_k25={s['rho_without_k25']} timestep_only_rho={s['timestep_only_rho']} probe_vs_timestep_rho={s['probe_vs_timestep_rho']}")
    print(f"determinism={s['determinism']['pass']} mean_probe_cost_s(old,new)=({s['cost']['mean_wall_s_old']:.1f},{s['cost']['mean_wall_s_new']:.1f})")
    print(f"decision={s['decision']} ROUND4B_ELIGIBLE={s['ROUND4B_ELIGIBLE']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    validate_output_path(output_dir)
    if args.mode == "cpu":
        result = run_cpu(config, args.config, output_dir)
    elif args.mode == "probe":
        result = run_probe(config, args.config, output_dir, args)
    else:
        result = run_analyze(config, args.config, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
