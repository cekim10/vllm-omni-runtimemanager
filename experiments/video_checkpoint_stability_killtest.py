#!/usr/bin/env python3
"""Fixed-progress trajectory-conditioned recovery-tail kill test.

The experiment reads hash-pinned corrected-v3 and error-shape inputs. It
writes only to its isolated namespace and contains no runtime mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_runtime_error_shape_killtest as base  # noqa: E402
from experiments import video_runtime_state_discovery as v3  # noqa: E402


EXPECTED_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
EXPECTED_SCHEDULER = "WanEulerScheduler"
EXPECTED_RUNTIME_DTYPE = "torch.bfloat16"
EXPECTED_EXPERT = "high_noise_transformer"
PRIMARY_STEP = 20
PERTURBATION_FAMILY = "dense_additive_gaussian"
PRIMARY_OPERATOR = base.PRIMARY_OPERATOR
ALLOWED_MODES = ("cpu", "preflight", "smoke", "analyze-smoke")
DECISION_INPUT_FIELDS = frozenset(
    {
        "trajectory_id",
        "replicate_id",
        "frame_ssim_mean",
        "realized_runtime_bf16_mse",
        "relative_mse_mismatch",
        "realized_runtime_active_fraction",
    }
)
RAW_FIELDS = (
    "status",
    "experiment_version",
    "config_hash",
    "provenance_hash",
    "source_raw_sha256",
    "model",
    "scheduler",
    "trajectory_id",
    "prompt_id",
    "prompt_text",
    "generation_seed",
    "checkpoint_step",
    "resume_index",
    "resume_timestep",
    "expert_regime",
    "remaining_high_noise_steps",
    "remaining_low_noise_steps",
    "clean_checkpoint_hash",
    "clean_reference_video_hash",
    "clean_reference_final_latent_hash",
    "replicate_id",
    "replicate_seed",
    "perturbation_family",
    "target_mse",
    "intended_support_fraction",
    "realized_runtime_active_fraction",
    "active_elements",
    "realized_nonzero_elements",
    "total_elements",
    "realized_probe_mse",
    "realized_runtime_bf16_mse",
    "relative_mse_mismatch",
    "solved_scale",
    "runtime_input_hash",
    "final_latent_mse",
    "video_mse",
    "video_psnr",
    "frame_ssim_mean",
    "temporal_delta_mse",
    "temporal_delta_agreement",
    "prompt_clip_score",
    "resume_ms",
    "runtime_candidate_sha256",
    "recovered_final_latent_sha256",
    "recovered_video_sha256",
    "runtime_candidate_path",
    "recovered_final_latent_path",
    "recovered_video_path",
    "result_path",
)


class GlobalStopError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config))


def atomic_json(path: Path, value: Any) -> None:
    v3.atomic_json(path, value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    v3.write_csv(path, rows, fields)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _base_config(config: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(config)
    adapted["source_v3"] = config["trusted_v3"]
    return adapted


def trajectory_id(trajectory: dict[str, Any]) -> str:
    return (
        f"{trajectory['prompt_id']}_{int(trajectory['generation_seed'])}"
        f"_step{int(trajectory['checkpoint_step']):02d}"
    )


def canonical_replicate_seed(namespace: str, replicate_id: int) -> int:
    digest = hashlib.sha256(f"{namespace}|{replicate_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def derive_replicate_seeds(config: dict[str, Any]) -> list[int]:
    perturbation = config["perturbation"]
    return [
        canonical_replicate_seed(perturbation["seed_namespace"], replicate_id)
        for replicate_id in range(int(perturbation["replicate_count"]))
    ]


def scheduler_timesteps_numpy(config: dict[str, Any]) -> list[float]:
    """Reproduce the pinned WanEulerScheduler schedule without importing torch."""
    num_steps = int(config["generation"]["num_inference_steps"])
    train_steps = int(config["scheduler"]["num_train_timesteps"])
    shift = float(config["scheduler"]["flow_shift"])
    original = np.linspace(train_steps, 0, num_steps + 1, dtype=np.float32)
    sigma = original / np.float32(train_steps)
    shifted = np.float32(shift) * sigma / (np.float32(1.0) + np.float32(shift - 1.0) * sigma)
    return [float(value) for value in (shifted[:-1] * np.float32(train_steps))]


def derive_expert_metadata(config: dict[str, Any], checkpoint_step: int) -> dict[str, Any]:
    timesteps = scheduler_timesteps_numpy(config)
    if not 0 <= checkpoint_step < len(timesteps):
        raise ValueError(f"Checkpoint step outside scheduler: {checkpoint_step}")
    boundary = float(config["generation"]["boundary_ratio"]) * float(
        config["scheduler"]["num_train_timesteps"]
    )
    remaining = timesteps[checkpoint_step:]
    high = sum(value >= boundary for value in remaining)
    low = sum(value < boundary for value in remaining)
    expert = EXPECTED_EXPERT if remaining[0] >= boundary else "low_noise_transformer_2"
    return {
        "checkpoint_step": checkpoint_step,
        "resume_index": checkpoint_step,
        "resume_timestep": remaining[0],
        "expert_regime": expert,
        "remaining_high_noise_steps": high,
        "remaining_low_noise_steps": low,
        "expert_boundary_timestep": boundary,
        "crosses_expert_boundary_after_resume": expert == EXPECTED_EXPERT and low > 0,
        "expert_rule": "transformer_2 iff scheduler timestep < boundary_timestep",
    }


def validate_source_hashes(config: dict[str, Any]) -> dict[str, Any]:
    trusted = config["trusted_v3"]
    error = config["validated_error_shape_source"]
    paths = {
        "raw_results": _resolve(trusted["root"]) / "raw_results.csv",
        "trusted_config": _resolve(trusted["config_file"]),
        "trusted_provenance": _resolve(trusted["provenance_file"]),
        "error_shape_config": _resolve(error["config_file"]),
        "error_shape_matcher": _resolve(error["matcher_script"]),
    }
    expected = {
        "raw_results": trusted["raw_results_sha256"],
        "trusted_config": trusted["config_file_sha256"],
        "trusted_provenance": trusted["provenance_file_sha256"],
        "error_shape_config": error["config_file_sha256"],
        "error_shape_matcher": error["matcher_script_sha256"],
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != expected:
        raise GlobalStopError(f"GLOBAL STOP: source content hash mismatch: {actual}")
    provenance = json.loads(paths["trusted_provenance"].read_text())
    if provenance["provenance_hash"] != trusted["provenance_hash"]:
        raise GlobalStopError("GLOBAL STOP: trusted provenance identity mismatch")
    return {name: {"path": str(paths[name]), "sha256": actual[name]} for name in paths}


def validate_trusted_namespace(config: dict[str, Any]) -> None:
    root = str(config["trusted_v3"]["root"]).lower()
    if "v2" in root or "unipc" in root:
        raise GlobalStopError("GLOBAL STOP: old invalid v2/UniPC namespace rejected")


def source_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    validate_trusted_namespace(config)
    validate_source_hashes(config)
    rows = base.source_rows(_base_config(config))
    if len(rows) != int(config["trusted_v3"]["expected_raw_rows"]):
        raise GlobalStopError("GLOBAL STOP: trusted source row count changed")
    if any(row["scheduler"] != EXPECTED_SCHEDULER for row in rows):
        raise GlobalStopError("GLOBAL STOP: non-Euler trusted row")
    return rows


def derive_target(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = config["validated_error_shape_source"]
    path = _resolve(source["config_file"])
    if sha256_file(path) != source["config_file_sha256"]:
        raise GlobalStopError("GLOBAL STOP: validated error-shape config hash mismatch")
    source_config = json.loads(path.read_text())
    configured = next(
        float(row["mse"])
        for row in source_config["concentration"]["targets"]
        if row["name"] == "small"
    )
    values = [
        float(row["initial_mse_runtime_dtype"])
        for row in rows
        if row["corruption_name"] == source["target_source_condition"]
    ]
    if len(values) != 36:
        raise GlobalStopError("GLOBAL STOP: target derivation requires exactly 36 INT8 rows")
    derived = statistics.fmean(values)
    frozen = float(source["target_mse"])
    if configured != frozen or derived != frozen:
        raise GlobalStopError("GLOBAL STOP: frozen perturbation target differs from source")
    return {
        "target_mse": frozen,
        "source_file": str(path),
        "source_file_sha256": sha256_file(path),
        "source_config_key": source["target_config_key"],
        "source_condition": source["target_source_condition"],
        "source_row_count": len(values),
        "derivation": source["target_derivation"],
    }


def _trajectory_source_document(
    config: dict[str, Any], row: dict[str, Any], expert: dict[str, Any]
) -> dict[str, Any]:
    prompt_id = row["prompt_id"]
    seed = int(row["generation_seed"])
    path, manifest = base._trajectory_manifest(_base_config(config), prompt_id, seed)
    states = {int(state["step"]): state for state in manifest["states"]}
    if PRIMARY_STEP not in states or 40 not in states:
        raise GlobalStopError("GLOBAL STOP: source trajectory lacks step20/final state")
    state = states[PRIMARY_STEP]
    final = states[40]
    if state["tensor_sha256"] != row["clean_latent_hash"]:
        raise GlobalStopError("GLOBAL STOP: step20 source hash differs from trusted row")
    expected_bytes = int(state["runtime_numel"]) * 2
    if (
        state["runtime_dtype"] != EXPECTED_RUNTIME_DTYPE
        or int(state["runtime_element_size_bytes"]) != 2
        or int(state["runtime_payload_bytes"]) != expected_bytes
    ):
        raise GlobalStopError("GLOBAL STOP: BF16 runtime-state accounting mismatch")
    if not math.isclose(
        float(row["checkpoint_scheduler_timestep"]),
        float(expert["resume_timestep"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise GlobalStopError("GLOBAL STOP: source scheduler timestep differs from derived schedule")
    if (
        row["current_expert"] != expert["expert_regime"]
        or int(row["remaining_high_noise_steps"]) != expert["remaining_high_noise_steps"]
        or int(row["remaining_low_noise_steps"]) != expert["remaining_low_noise_steps"]
    ):
        raise GlobalStopError("GLOBAL STOP: source expert metadata differs from derived logic")
    return {
        "trajectory_id": f"{prompt_id}_{seed}_step20",
        "prompt_id": prompt_id,
        "prompt_text": row["prompt_text"],
        "motion_category": row["motion_category"],
        "generation_seed": seed,
        "checkpoint_step": PRIMARY_STEP,
        "clean_checkpoint_hash": state["tensor_sha256"],
        "clean_checkpoint_file_sha256": state["file_sha256"],
        "clean_reference_video_hash": manifest["baseline_video_tensor_sha256"],
        "clean_reference_video_file_sha256": manifest["baseline_video_file_sha256"],
        "clean_reference_final_latent_hash": final["tensor_sha256"],
        "clean_reference_final_latent_file_sha256": final["file_sha256"],
        "source_manifest_path": str(path),
        "source_manifest_sha256": sha256_file(path),
        "runtime_dtype": state["runtime_dtype"],
        "runtime_element_size_bytes": int(state["runtime_element_size_bytes"]),
        "runtime_numel": int(state["runtime_numel"]),
        "runtime_full_bytes": expected_bytes,
        "shape": [int(value) for value in state["shape"]],
        **expert,
    }


def derive_primary_trajectories(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    all_prompts = sorted({row["prompt_id"] for row in rows})
    expected_count = int(config["trusted_v3"]["expected_prompt_count"])
    if len(all_prompts) != expected_count:
        raise GlobalStopError("GLOBAL STOP: trusted prompt population is not exactly 12")
    expert = derive_expert_metadata(config, PRIMARY_STEP)
    if expert["expert_regime"] != config["generation"]["expected_expert_regime"]:
        raise GlobalStopError("GLOBAL STOP: primary step is not in expected expert regime")
    selected = [
        row
        for row in rows
        if row["corruption_name"] == config["selection"]["source_condition"]
        and int(row["checkpoint_step"]) == PRIMARY_STEP
    ]
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_prompt[row["prompt_id"]].append(row)
    if set(by_prompt) != set(all_prompts) or any(len(group) != 1 for group in by_prompt.values()):
        raise GlobalStopError("GLOBAL STOP: expected one step20 full-direct row per prompt")
    trajectories = []
    for prompt_id in all_prompts:
        seeds = {int(row["generation_seed"]) for row in rows if row["prompt_id"] == prompt_id}
        if len(seeds) != 1:
            raise GlobalStopError("GLOBAL STOP: prompt has more than one trusted seed")
        trajectories.append(_trajectory_source_document(config, by_prompt[prompt_id][0], expert))
    trajectories.sort(key=lambda row: (row["prompt_id"], row["generation_seed"]))
    identities = [row["trajectory_id"] for row in trajectories]
    if len(trajectories) != 12 or len(set(identities)) != 12:
        raise GlobalStopError("GLOBAL STOP: primary trajectory manifest is incomplete or duplicated")
    if any(int(row["checkpoint_step"]) != PRIMARY_STEP for row in trajectories):
        raise GlobalStopError("GLOBAL STOP: non-step20 trajectory entered primary manifest")
    return trajectories


def source_falsifiability_audit(
    rows: list[dict[str, Any]], trajectories: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    identities = {(row["prompt_id"], int(row["generation_seed"])) for row in trajectories}
    comparable = [
        row
        for row in rows
        if row["corruption_name"] == "gaussian_matched_int8"
        and int(row["checkpoint_step"]) == PRIMARY_STEP
        and (row["prompt_id"], int(row["generation_seed"])) in identities
    ]
    counts = Counter((row["prompt_id"], int(row["generation_seed"])) for row in comparable)
    expected_replicates = int(config["perturbation"]["replicate_count"])
    cannot_compute = (
        len(comparable) == len(trajectories) == 12
        and set(counts) == identities
        and all(count == 1 for count in counts.values())
        and max(counts.values()) < expected_replicates
    )
    absolute_ssim = [float(row["frame_ssim_mean"]) for row in comparable]
    return {
        "source_condition": "gaussian_matched_int8",
        "checkpoint_step": PRIMARY_STEP,
        "trajectory_count": len(trajectories),
        "source_samples_per_trajectory": {
            f"{prompt}_{seed}": counts[(prompt, seed)] for prompt, seed in sorted(identities)
        },
        "required_samples_per_trajectory": expected_replicates,
        "within_trajectory_tail_depth_computable": not cannot_compute,
        "preregistered_question_not_already_answered_by_source_data": cannot_compute,
        "descriptive_cross_prompt_absolute_ssim_range": max(absolute_ssim) - min(absolute_ssim),
        "absolute_ssim_range_used_for_decision": False,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config["experiment_version"] != "video-checkpoint-stability-killtest-v2-fixed-step":
        raise ValueError("Obsolete six-checkpoint experiment version")
    if config["model"] != EXPECTED_MODEL:
        raise ValueError("Model changed")
    if config["scheduler"]["name"] != EXPECTED_SCHEDULER or config["scheduler"]["sample_solver"] != "euler":
        raise ValueError("Explicit Euler scheduler is required")
    generation = config["generation"]
    if generation["checkpoint_steps"] != [PRIMARY_STEP] or int(generation["primary_checkpoint_step"]) != PRIMARY_STEP:
        raise ValueError("Primary experiment must use step20 only")
    if generation["expected_expert_regime"] != EXPECTED_EXPERT:
        raise ValueError("Expected expert regime changed")
    perturbation = config["perturbation"]
    frozen_perturbation = {
        "family": PERTURBATION_FAMILY,
        "intended_support_fraction": 1.0,
        "runtime_mse_relative_tolerance": 0.01,
        "trajectory_mean_mse_range_relative_to_target_limit": 0.001,
        "trajectory_mean_support_range_limit": 0.005,
        "replicate_count": 16,
    }
    for key, expected in frozen_perturbation.items():
        if perturbation[key] != expected:
            raise ValueError(f"Frozen perturbation value changed: {key}")
    seeds = derive_replicate_seeds(config)
    if seeds != [int(value) for value in perturbation["replicate_seeds"]] or len(set(seeds)) != 16:
        raise ValueError("Canonical replicate seed schedule changed or collided")
    analysis = config["analysis"]
    frozen_analysis = {
        "primary_metric": "frame_ssim_mean",
        "lower_tail_count": 4,
        "loo_lower_tail_count": 4,
        "large_drop_threshold": 0.10,
        "go_tail_depth_range": 0.10,
        "go_minimum_trajectory_tail_depth": 0.10,
        "no_go_tail_depth_range": 0.05,
        "no_go_maximum_trajectory_tail_depth": 0.05,
        "loo_tail_depth_range": 0.075,
        "loo_required_count": 12,
        "single_replicate_collapse_floor": 0.05,
        "supporting_large_drop_rate": 0.25,
    }
    for key, expected in frozen_analysis.items():
        if analysis[key] != expected:
            raise ValueError(f"Frozen analysis value changed: {key}")
    if set(analysis["decision_input_fields"]) != DECISION_INPUT_FIELDS:
        raise ValueError("Configured decision inputs differ from preregistered fields")
    excluded = set(analysis["descriptive_only_metrics"] + analysis["auxiliary_only_metrics"])
    if excluded & DECISION_INPUT_FIELDS:
        raise ValueError("Descriptive/auxiliary metric entered decision inputs")
    expected_matrix = {
        "trajectory_count": 12,
        "replicate_count": 16,
        "primary_checkpoint_step": 20,
        "primary_row_count": 192,
    }
    if config["matrix"] != expected_matrix:
        raise ValueError("Primary matrix changed")
    if config["allowed_modes"] != list(ALLOWED_MODES):
        raise ValueError("Automatic or broader mode detected")
    return config


def validate_output_namespace(config: dict[str, Any], output_dir: Path) -> None:
    output = output_dir.resolve()
    protected = [
        _resolve(config["trusted_v3"]["root"]).resolve(),
        _resolve(config["validated_error_shape_source"]["root"]).resolve(),
    ]
    if any(output == root or root in output.parents for root in protected):
        raise GlobalStopError("GLOBAL STOP: output namespace overlaps a trusted source")


def _relevant_diff(paths: list[Path]) -> str:
    relative = [str(path.resolve().relative_to(REPO_ROOT)) for path in paths]
    return _git_value("diff", "--", *relative) or ""


def build_provenance(config_path: Path) -> dict[str, Any]:
    script = Path(__file__).resolve()
    runner = REPO_ROOT / "experiments/run_video_checkpoint_stability_killtest_gpu0.sh"
    pipeline = REPO_ROOT / "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"
    scheduler = REPO_ROOT / "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py"
    error_script = REPO_ROOT / "experiments/video_runtime_error_shape_killtest.py"
    status = _git_value("status", "--short") or ""
    files = [script, config_path, runner, pipeline, scheduler, error_script]
    document = {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status.splitlines(),
        "relevant_diff_sha256": sha256_bytes(_relevant_diff(files).encode()),
        "experiment_script_sha256": sha256_file(script),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(runner),
        "pipeline_wan2_2_sha256": sha256_file(pipeline),
        "scheduler_sha256": sha256_file(scheduler),
        "error_shape_matcher_sha256": sha256_file(error_script),
        "trusted_v3_raw_sha256": sha256_file(
            _resolve("results/video_runtime_state_discovery_v3_corrected/raw_results.csv")
        ),
        "trusted_v3_config_sha256": sha256_file(
            _resolve("results/video_runtime_state_discovery_v3_corrected/preregistered_config.yaml")
        ),
        "trusted_v3_provenance_file_sha256": sha256_file(
            _resolve("results/video_runtime_state_discovery_v3_corrected/run_provenance.json")
        ),
    }
    document["provenance_hash"] = sha256_bytes(canonical_json(document))
    return document


def environment_document(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in ("torch", "diffusers", "skimage", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            versions[name] = f"unavailable: {error}"
    gpu = None
    cuda = None
    try:
        import torch

        cuda = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "cuda_version": cuda,
        "gpu_model": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": config["model"],
        "model_revision": config.get("model_revision"),
        "runtime_dtype": EXPECTED_RUNTIME_DTYPE,
        "provenance": provenance,
    }


def primary_manifest(
    config: dict[str, Any], rows: list[dict[str, Any]], target: dict[str, Any]
) -> dict[str, Any]:
    trajectories = derive_primary_trajectories(rows, config)
    document = {
        "selection_rule": config["selection"],
        "trajectories": trajectories,
        "replicate_seeds": [
            {"replicate_id": index, "seed": seed}
            for index, seed in enumerate(derive_replicate_seeds(config))
        ],
        "target": target,
        "mse_tolerances": {
            key: config["perturbation"][key]
            for key in (
                "runtime_mse_relative_tolerance",
                "trajectory_mean_mse_range_relative_to_target_limit",
                "trajectory_mean_support_range_limit",
            )
        },
        "decision_thresholds": {
            key: value
            for key, value in config["analysis"].items()
            if key not in {"secondary_metrics", "descriptive_only_metrics", "auxiliary_only_metrics"}
        },
        "source_raw_sha256": config["trusted_v3"]["raw_results_sha256"],
    }
    document["manifest_sha256"] = sha256_bytes(canonical_json(document))
    return document


def _source_trajectory(config: dict[str, Any], trajectory: dict[str, Any]) -> base.SourceTrajectory:
    source = base.load_source_trajectory(_base_config(config), trajectory, PRIMARY_STEP)
    if source.clean_hash != trajectory["clean_checkpoint_hash"]:
        raise GlobalStopError("GLOBAL STOP: loaded checkpoint differs from primary manifest")
    return source


def construct_dense_perturbation(
    clean: np.ndarray,
    *,
    target_mse: float,
    replicate_seed: int,
    relative_tolerance: float,
    max_iterations: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate, details = base.construct_fixed_mse_error(
        clean,
        target_mse=target_mse,
        active_fraction=1.0,
        operator_family=PRIMARY_OPERATOR,
        support_seed=replicate_seed,
        perturbation_value_seed=replicate_seed,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    if int(details["active_elements"]) != int(details["total_elements"]):
        raise GlobalStopError("GLOBAL STOP: dense perturbation did not intend full support")
    if details["operator_family"] != PRIMARY_OPERATOR:
        raise GlobalStopError("GLOBAL STOP: non-additive perturbation entered primary matrix")
    details["perturbation_family"] = PERTURBATION_FAMILY
    details["replicate_seed"] = replicate_seed
    return candidate, details


def expected_primary_keys(config: dict[str, Any], trajectories: list[dict[str, Any]] | None = None) -> set[tuple[str, int]]:
    if trajectories is None:
        rows = source_rows(config)
        trajectories = derive_primary_trajectories(rows, config)
    return {
        (row["trajectory_id"], replicate_id)
        for row in trajectories
        for replicate_id in range(int(config["perturbation"]["replicate_count"]))
    }


def validate_expected_keys(
    rows: list[dict[str, Any]], config: dict[str, Any], trajectories: list[dict[str, Any]] | None = None
) -> None:
    expected = expected_primary_keys(config, trajectories)
    keys = [(row["trajectory_id"], int(row["replicate_id"])) for row in rows]
    counts = Counter(keys)
    actual = set(keys)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if missing or unexpected or duplicates or len(rows) != len(expected):
        raise GlobalStopError(
            "GLOBAL STOP: primary matrix key mismatch: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )


def validate_seed_schedule(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    expected = {index: seed for index, seed in enumerate(derive_replicate_seeds(config))}
    by_trajectory: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        by_trajectory[row["trajectory_id"]][int(row["replicate_id"])].add(
            int(row["replicate_seed"])
        )
    for schedule in by_trajectory.values():
        if set(schedule) != set(expected):
            raise GlobalStopError("GLOBAL STOP: replicate IDs differ across trajectories")
        if any(schedule[index] != {expected[index]} for index in expected):
            raise GlobalStopError("GLOBAL STOP: replicate seed is trajectory-dependent")


def lower_tail_mean(values: Iterable[float], count: int = 4) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) < count:
        raise ValueError("Too few values for lower-tail statistic")
    return statistics.fmean(ordered[:count])


def tail_depth(values: Iterable[float], count: int = 4) -> float:
    samples = [float(value) for value in values]
    return statistics.fmean(samples) - lower_tail_mean(samples, count)


def construction_statistics(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trajectory_id"]].append(row)
    target = float(config["validated_error_shape_source"]["target_mse"])
    summaries = []
    for identity, cells in sorted(grouped.items()):
        mses = [float(row["realized_runtime_bf16_mse"]) for row in cells]
        supports = [float(row["realized_runtime_active_fraction"]) for row in cells]
        mean_mse = statistics.fmean(mses)
        summaries.append(
            {
                "trajectory_id": identity,
                "mean_realized_runtime_bf16_mse": mean_mse,
                "mean_signed_relative_mse_residual": (mean_mse - target) / target,
                "max_absolute_relative_mse_mismatch": max(
                    abs(float(row["relative_mse_mismatch"])) for row in cells
                ),
                "mean_realized_runtime_active_fraction": statistics.fmean(supports),
                "min_realized_runtime_active_fraction": min(supports),
                "max_realized_runtime_active_fraction": max(supports),
            }
        )
    mean_mses = [row["mean_realized_runtime_bf16_mse"] for row in summaries]
    mean_supports = [row["mean_realized_runtime_active_fraction"] for row in summaries]
    mse_range = (max(mean_mses) - min(mean_mses)) / target
    support_range = max(mean_supports) - min(mean_supports)
    all_cells_match = all(
        abs(float(row["relative_mse_mismatch"]))
        <= float(config["perturbation"]["runtime_mse_relative_tolerance"])
        for row in rows
    )
    return {
        "trajectory_summaries": summaries,
        "max_per_cell_relative_mse_mismatch": max(
            abs(float(row["relative_mse_mismatch"])) for row in rows
        ),
        "trajectory_mean_mse_absolute_range": max(mean_mses) - min(mean_mses),
        "trajectory_mean_mse_range_relative_to_target": mse_range,
        "trajectory_mean_mse_gate_passed": all_cells_match
        and mse_range
        <= float(config["perturbation"]["trajectory_mean_mse_range_relative_to_target_limit"]),
        "trajectory_mean_support_range": support_range,
        "trajectory_mean_support_gate_passed": support_range
        <= float(config["perturbation"]["trajectory_mean_support_range_limit"]),
        "global_min_realized_runtime_active_fraction": min(
            row["min_realized_runtime_active_fraction"] for row in summaries
        ),
        "global_max_realized_runtime_active_fraction": max(
            row["max_realized_runtime_active_fraction"] for row in summaries
        ),
        "all_cells_mse_matched": all_cells_match,
    }


def classify_decision(
    config: dict[str, Any],
    *,
    tail_depth_range: float,
    maximum_trajectory_tail_depth: float,
    loo_tail_depth_ranges: list[float],
    mse_confound_passed: bool,
    support_confound_passed: bool,
    correctness_passed: bool,
) -> dict[str, Any]:
    analysis = config["analysis"]
    loo_count = sum(
        value >= float(analysis["loo_tail_depth_range"]) for value in loo_tail_depth_ranges
    )
    robust = (
        loo_count >= int(analysis["loo_required_count"])
        and min(loo_tail_depth_ranges)
        >= float(analysis["single_replicate_collapse_floor"])
    )
    go = (
        tail_depth_range >= float(analysis["go_tail_depth_range"])
        and maximum_trajectory_tail_depth
        >= float(analysis["go_minimum_trajectory_tail_depth"])
        and robust
        and mse_confound_passed
        and support_confound_passed
        and correctness_passed
    )
    no_go = (
        tail_depth_range < float(analysis["no_go_tail_depth_range"])
        and maximum_trajectory_tail_depth
        < float(analysis["no_go_maximum_trajectory_tail_depth"])
    )
    decision = (
        "GO_TO_BROADER_STABILITY_MAP"
        if go
        else "NO_GO"
        if no_go
        else "WEAK_INCONCLUSIVE"
    )
    return {
        "decision": decision,
        "loo_tail_depth_threshold_count": loo_count,
        "minimum_loo_tail_depth_range": min(loo_tail_depth_ranges),
        "single_replicate_robust": robust,
    }


def analyze_rows(
    rows: list[dict[str, Any]], config: dict[str, Any], *, correctness_passed: bool
) -> dict[str, Any]:
    trajectories = derive_primary_trajectories(source_rows(config), config)
    validate_expected_keys(rows, config, trajectories)
    validate_seed_schedule(rows, config)
    if any(int(row["checkpoint_step"]) != PRIMARY_STEP for row in rows):
        raise GlobalStopError("GLOBAL STOP: non-step20 result entered primary analysis")
    if any(row["expert_regime"] != EXPECTED_EXPERT for row in rows):
        raise GlobalStopError("GLOBAL STOP: mixed expert regime entered primary analysis")
    if any(row["perturbation_family"] != PERTURBATION_FAMILY for row in rows):
        raise GlobalStopError("GLOBAL STOP: FP16 or non-primary perturbation entered analysis")
    if any(not math.isfinite(float(row["frame_ssim_mean"])) for row in rows):
        raise GlobalStopError("GLOBAL STOP: non-finite primary metric")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trajectory_id"]].append(row)
    tail_count = int(config["analysis"]["lower_tail_count"])
    large_drop_threshold = float(config["analysis"]["large_drop_threshold"])
    summaries = []
    for identity, cells in sorted(grouped.items()):
        values = [float(row["frame_ssim_mean"]) for row in cells]
        mean = statistics.fmean(values)
        median = statistics.median(values)
        bottom = lower_tail_mean(values, tail_count)
        summaries.append(
            {
                "trajectory_id": identity,
                "prompt_id": cells[0]["prompt_id"],
                "generation_seed": int(cells[0]["generation_seed"]),
                "checkpoint_step": int(cells[0]["checkpoint_step"]),
                "expert_regime": cells[0]["expert_regime"],
                "mean_ssim": mean,
                "median_ssim": median,
                "minimum_ssim": min(values),
                "bottom4_mean_ssim": bottom,
                "tail_depth": mean - bottom,
                "std_ssim": statistics.pstdev(values),
                "large_drop_count": sum(median - value >= large_drop_threshold for value in values),
                "large_drop_rate": statistics.fmean(
                    median - value >= large_drop_threshold for value in values
                ),
            }
        )
    depths = [row["tail_depth"] for row in summaries]
    depth_range = max(depths) - min(depths)
    maximum_depth = max(depths)
    max_large_drop_rate = max(row["large_drop_rate"] for row in summaries)
    max_depth_trajectory = max(summaries, key=lambda row: (row["tail_depth"], row["trajectory_id"]))
    min_depth_trajectory = min(summaries, key=lambda row: (row["tail_depth"], row["trajectory_id"]))
    loo = []
    loo_count = int(config["analysis"]["loo_lower_tail_count"])
    for replicate_id in range(int(config["perturbation"]["replicate_count"])):
        remaining_depths = []
        for cells in grouped.values():
            values = [
                float(row["frame_ssim_mean"])
                for row in cells
                if int(row["replicate_id"]) != replicate_id
            ]
            remaining_depths.append(tail_depth(values, loo_count))
        loo.append(
            {
                "removed_replicate_id": replicate_id,
                "tail_depth_range": max(remaining_depths) - min(remaining_depths),
                "lower_tail_count": loo_count,
                "remaining_samples_per_trajectory": 15,
            }
        )
    construction = construction_statistics(rows, config)
    classification = classify_decision(
        config,
        tail_depth_range=depth_range,
        maximum_trajectory_tail_depth=maximum_depth,
        loo_tail_depth_ranges=[row["tail_depth_range"] for row in loo],
        mse_confound_passed=construction["trajectory_mean_mse_gate_passed"],
        support_confound_passed=construction["trajectory_mean_support_gate_passed"],
        correctness_passed=correctness_passed,
    )
    return {
        **classification,
        "tail_depth_range": depth_range,
        "maximum_trajectory_tail_depth": maximum_depth,
        "max_tail_depth_trajectory_id": max_depth_trajectory["trajectory_id"],
        "min_tail_depth_trajectory_id": min_depth_trajectory["trajectory_id"],
        "max_large_drop_rate": max_large_drop_rate,
        "supporting_large_drop_rate_met": max_large_drop_rate
        >= float(config["analysis"]["supporting_large_drop_rate"]),
        "trajectory_summaries": summaries,
        "leave_one_replicate_out": loo,
        "construction_confounds": construction,
        "correctness_passed": correctness_passed,
        "decision_input_fields": sorted(DECISION_INPUT_FIELDS),
        "absolute_ssim_level_gap_used_for_decision": False,
        "fp16_anomaly_used_for_decision": False,
        "minimum_ssim_used_for_decision": False,
        "large_drop_rate_used_for_decision": False,
        "temporal_or_clip_used_for_decision": False,
    }


def _gate(
    name: str,
    passed: bool | None,
    evidence: Any,
    expected: str,
    artifacts: Iterable[str | Path] = (),
    *,
    required: bool = True,
) -> dict[str, Any]:
    return v3.gate_record(name, passed, evidence, artifacts, expected, required=required)


def _write_gates(path: Path, gates: list[dict[str, Any]]) -> None:
    all_passed = v3.validate_gate_records(gates)
    atomic_json(path, {"all_passed": all_passed, "gates": gates})
    if not all_passed:
        failed = [row["name"] for row in gates if row["required"] and row["status"] != "PASS"]
        raise GlobalStopError(f"GLOBAL STOP: required gates failed: {failed}")


def assert_provenance_matches(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists() or json.loads(path.read_text()) != expected:
        raise GlobalStopError("GLOBAL STOP: prior mode provenance differs from current content")


def require_mode_gate(output_dir: Path, name: str, provenance: dict[str, Any]) -> None:
    assert_provenance_matches(output_dir / "run_provenance.json", provenance)
    path = output_dir / name
    if not path.exists() or not json.loads(path.read_text()).get("all_passed"):
        raise GlobalStopError(f"GLOBAL STOP: prerequisite gate did not pass: {path}")


def _save_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "tensor_sha256": v3.array_sha256(value),
    }


def _validate_saved_array(record: dict[str, Any]) -> bool:
    path = _resolve(record["path"])
    if not path.exists() or sha256_file(path) != record["file_sha256"]:
        return False
    return v3.array_sha256(np.load(path, allow_pickle=False)) == record["tensor_sha256"]


def _result_valid(
    path: Path, provenance: dict[str, Any], expected_identity: dict[str, Any]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text())
        if row["status"] != "COMPLETE":
            raise GlobalStopError(f"GLOBAL STOP: incomplete result exists: {path}")
        if row["provenance_hash"] != provenance["provenance_hash"]:
            raise GlobalStopError(f"GLOBAL STOP: stale result provenance: {path}")
        if any(row.get(key) != value for key, value in expected_identity.items()):
            raise GlobalStopError(f"GLOBAL STOP: result identity mismatch: {path}")
        for field in (
            "runtime_candidate_artifact",
            "recovered_final_latent_artifact",
            "recovered_video_artifact",
        ):
            if not _validate_saved_array(row[field]):
                raise GlobalStopError(f"GLOBAL STOP: invalid retained artifact {field}: {path}")
        return row
    except GlobalStopError:
        raise
    except Exception as error:
        raise GlobalStopError(f"GLOBAL STOP: malformed result {path}: {error}") from error


def _shutdown(omni: Any) -> None:
    shutdown = getattr(omni, "shutdown", None)
    if callable(shutdown):
        shutdown()


def _decision_field_gate(config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    configured = set(config["analysis"]["decision_input_fields"])
    excluded = set(
        config["analysis"]["descriptive_only_metrics"]
        + config["analysis"]["auxiliary_only_metrics"]
    )
    return configured == DECISION_INPUT_FIELDS and not configured & excluded, {
        "configured_fields": sorted(configured),
        "expected_fields": sorted(DECISION_INPUT_FIELDS),
        "excluded_intersection": sorted(configured & excluded),
    }


def _cpu_gates(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    construction_rows: list[dict[str, Any]],
    construction: dict[str, Any],
    falsifiability: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    seeds = derive_replicate_seeds(config)
    max_range_ratio = max(float(row["restored_to_clean_absmax_ratio"]) for row in construction_rows)
    return [
        _gate("G1 trusted source hashes exact", True, validate_source_hashes(config), "all source hashes match"),
        _gate("G2 Euler only / invalid UniPC excluded", all(row["scheduler"] == EXPECTED_SCHEDULER for row in rows), config["trusted_v3"]["root"], "corrected-v3 Euler only"),
        _gate("G3 exactly 12 source-derived trajectories", len(trajectories) == len({row["prompt_id"] for row in trajectories}) == 12, trajectories, "all 12 prompts exactly once"),
        _gate("G4 primary checkpoint step20 only", all(row["checkpoint_step"] == PRIMARY_STEP for row in trajectories), {"steps": sorted({row["checkpoint_step"] for row in trajectories})}, "step 20 only"),
        _gate("G5 same derived expert regime", all(row["expert_regime"] == EXPECTED_EXPERT for row in trajectories), {"expert": EXPECTED_EXPERT, "metadata": derive_expert_metadata(config, PRIMARY_STEP)}, "all begin in high-noise transformer"),
        _gate("G6 clean FULL exact", None, "GPU preflight required", "all 12 clean resumes exact", required=False),
        _gate("G7 resume index/timestep", all(row["resume_index"] == PRIMARY_STEP and row["resume_timestep"] > row["expert_boundary_timestep"] for row in trajectories), trajectories, "derived step/index/timestep consistent"),
        _gate("G8 BF16 state accounting", all(row["runtime_dtype"] == EXPECTED_RUNTIME_DTYPE and row["runtime_full_bytes"] == row["runtime_numel"] * 2 for row in trajectories), trajectories, "BF16 shape and 2-byte state"),
        _gate("G9 dense-additive operator only", all(row["perturbation_family"] == PERTURBATION_FAMILY and row["active_elements"] == row["total_elements"] for row in construction_rows) and max_range_ratio <= float(config["perturbation"]["max_restored_to_clean_absmax_ratio"]), {"rows": len(construction_rows), "max_absmax_ratio": max_range_ratio}, "dense additive full intended support"),
        _gate("G10 shared 16 replicate seeds", len(seeds) == len(set(seeds)) == 16, seeds, "same collision-free schedule"),
        _gate("G11 per-cell BF16 MSE tolerance", construction["all_cells_mse_matched"], construction["max_per_cell_relative_mse_mismatch"], "all 192 within 1%"),
        _gate("G12 trajectory mean-MSE confound", construction["trajectory_mean_mse_gate_passed"], construction["trajectory_mean_mse_range_relative_to_target"], "range <=0.001 of target"),
        _gate("G13 realized-support confound", construction["trajectory_mean_support_gate_passed"], construction["trajectory_mean_support_range"], "mean support range <=0.005"),
        _gate("G14 exact 192-row key set", len(construction_rows) == 192, {"actual": len(construction_rows), "expected": 192}, "exact key equality"),
        _gate("G15 finite frame SSIM", None, "GPU smoke required", "all primary SSIM finite", required=False),
        _gate("G16 allowed decision fields only", _decision_field_gate(config)[0], _decision_field_gate(config)[1], "only preregistered primary inputs"),
        _gate("G17 temporal/CLIP excluded", _decision_field_gate(config)[0], _decision_field_gate(config)[1], "descriptive fields cannot decide"),
        _gate("G18 FP16 anomaly cannot enter decision", all(row["perturbation_family"] == PERTURBATION_FAMILY for row in construction_rows), {"family": PERTURBATION_FAMILY}, "no FP16 outcomes or roles"),
        _gate("G19 output artifacts", None, "GPU smoke required", "candidate/video/final latent retained", required=False),
        _gate("G20 deterministic source manifest", len(manifest["trajectories"]) == 12 and manifest["selection_rule"]["no_outcome_based_selection"], manifest["manifest_sha256"], "all source-derived step20 trajectories"),
        _gate("G21 source cannot answer centered tail", falsifiability["preregistered_question_not_already_answered_by_source_data"], falsifiability, "one source Gaussian sample per prompt is insufficient for 16-sample tail"),
        _gate("G22 no automatic expansion", config["allowed_modes"] == list(ALLOWED_MODES), config["allowed_modes"], "only four explicit modes"),
    ]


def run_cpu_mode(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_namespace(config, output_dir)
    rows = source_rows(config)
    target = derive_target(config, rows)
    trajectories = derive_primary_trajectories(rows, config)
    manifest = primary_manifest(config, rows, target)
    seeds = derive_replicate_seeds(config)
    provenance = build_provenance(config_path)
    construction_rows = []
    for trajectory in trajectories:
        source = _source_trajectory(config, trajectory)
        for replicate_id, seed in enumerate(seeds):
            _, details = construct_dense_perturbation(
                source.clean,
                target_mse=float(target["target_mse"]),
                replicate_seed=seed,
                relative_tolerance=float(config["perturbation"]["runtime_mse_relative_tolerance"]),
            )
            construction_rows.append(
                {
                    "trajectory_id": trajectory["trajectory_id"],
                    "prompt_id": trajectory["prompt_id"],
                    "generation_seed": trajectory["generation_seed"],
                    "checkpoint_step": PRIMARY_STEP,
                    "replicate_id": replicate_id,
                    "replicate_seed": seed,
                    **details,
                }
            )
    validate_expected_keys(construction_rows, config, trajectories)
    validate_seed_schedule(construction_rows, config)
    construction = construction_statistics(construction_rows, config)
    falsifiability = source_falsifiability_audit(rows, trajectories, config)
    gates = _cpu_gates(
        config, rows, trajectories, construction_rows, construction, falsifiability, manifest
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_manifest = output_dir / "checkpoint_selection_manifest.json"
    if stale_manifest.exists():
        stale_manifest.unlink()
    atomic_json(output_dir / "preregistered_config.json", config)
    atomic_json(output_dir / "primary_trajectory_manifest.json", manifest)
    atomic_json(output_dir / "perturbation_target_manifest.json", target)
    atomic_json(
        output_dir / "replicate_seed_manifest.json",
        {"replicate_seeds": manifest["replicate_seeds"], "seed_namespace": config["perturbation"]["seed_namespace"]},
    )
    atomic_json(
        output_dir / "expected_primary_keys.json",
        {"keys": sorted([list(key) for key in expected_primary_keys(config, trajectories)]), "count": 192},
    )
    atomic_json(output_dir / "source_falsifiability_audit.json", falsifiability)
    atomic_json(output_dir / "run_provenance.json", provenance)
    atomic_json(output_dir / "environment.json", environment_document(config, provenance))
    atomic_json(output_dir / "cpu_construction_checks.json", construction_rows)
    atomic_json(output_dir / "cpu_construction_summary.json", construction)
    _write_gates(output_dir / "cpu_gates.json", gates)
    return {
        "mode": "cpu",
        "all_passed": True,
        "trajectory_count": len(trajectories),
        "construction_rows": len(construction_rows),
        "provenance_hash": provenance["provenance_hash"],
    }


def run_preflight(
    config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    validate_output_namespace(config, output_dir)
    provenance = build_provenance(config_path)
    require_mode_gate(output_dir, "cpu_gates.json", provenance)
    rows = source_rows(config)
    target = derive_target(config, rows)
    trajectories = derive_primary_trajectories(rows, config)
    stored_manifest = json.loads((output_dir / "primary_trajectory_manifest.json").read_text())
    if primary_manifest(config, rows, target) != stored_manifest:
        raise GlobalStopError("GLOBAL STOP: primary manifest changed after CPU mode")
    actual_scheduler = v3.scheduler_document(config)
    actual_expert = v3.expert_region_metadata(config, actual_scheduler, PRIMARY_STEP)
    derived_expert = derive_expert_metadata(config, PRIMARY_STEP)
    if (
        actual_expert["current_expert"] != derived_expert["expert_regime"]
        or not math.isclose(actual_expert["checkpoint_scheduler_timestep"], derived_expert["resume_timestep"], abs_tol=1e-6)
        or actual_expert["remaining_high_noise_steps"] != derived_expert["remaining_high_noise_steps"]
        or actual_expert["remaining_low_noise_steps"] != derived_expert["remaining_low_noise_steps"]
    ):
        raise GlobalStopError("GLOBAL STOP: runtime scheduler expert metadata mismatch")
    exact_results = []
    matcher_results = []
    omni = v3.build_omni(config, args)
    try:
        for trajectory in trajectories:
            source = _source_trajectory(config, trajectory)
            _, details = construct_dense_perturbation(
                source.clean,
                target_mse=float(target["target_mse"]),
                replicate_seed=derive_replicate_seeds(config)[0],
                relative_tolerance=float(config["perturbation"]["runtime_mse_relative_tolerance"]),
            )
            matcher_results.append({"trajectory_id": trajectory["trajectory_id"], **details})
            result = base.run_resume(
                omni,
                config,
                source,
                source.clean,
                step_index=PRIMARY_STEP,
                label=f"stability_preflight_{trajectory['prompt_id']}",
                directory=output_dir / "preflight" / trajectory["trajectory_id"],
            )
            exact_results.append({"trajectory_id": trajectory["trajectory_id"], **result})
    finally:
        _shutdown(omni)
    all_exact = all(row["exact_final_latent"] and row["exact_video"] for row in exact_results)
    retained = all(
        base._validate_saved_array(record)
        for row in exact_results
        for record in (row["recovered_video_artifact"], row["recovered_final_latent_artifact"])
    )
    gates = [
        _gate("G1 trusted source hashes exact", True, validate_source_hashes(config), "all source hashes match"),
        _gate("G2 Euler only / invalid UniPC excluded", actual_scheduler["scheduler_class"].endswith(EXPECTED_SCHEDULER), actual_scheduler, "WanEulerScheduler"),
        _gate("G3 exactly 12 source-derived trajectories", len(trajectories) == 12, len(trajectories), "12"),
        _gate("G4 primary checkpoint step20 only", all(row["checkpoint_step"] == PRIMARY_STEP for row in trajectories), trajectories, "step20"),
        _gate("G5 same derived expert regime", actual_expert["current_expert"] == EXPECTED_EXPERT, actual_expert, EXPECTED_EXPERT),
        _gate("G6 clean FULL exact", all_exact, exact_results, "all 12 exact"),
        _gate("G7 resume index/timestep", all(int(row["resume_index"]) == PRIMARY_STEP for row in exact_results), exact_results, "resume index 20"),
        _gate("G8 BF16 state accounting", all(row["runtime_dtype"] == EXPECTED_RUNTIME_DTYPE and row["runtime_full_bytes"] == row["runtime_numel"] * 2 for row in trajectories), trajectories, "BF16 state"),
        _gate("G9 dense-additive operator only", all(row["perturbation_family"] == PERTURBATION_FAMILY and row["active_elements"] == row["total_elements"] for row in matcher_results), matcher_results, "dense additive"),
        _gate("G10 shared 16 replicate seeds", len(set(derive_replicate_seeds(config))) == 16, derive_replicate_seeds(config), "shared schedule"),
        _gate("G11 per-cell BF16 MSE tolerance", all(abs(float(row["relative_mse_mismatch"])) <= float(config["perturbation"]["runtime_mse_relative_tolerance"]) for row in matcher_results), matcher_results, "within 1%"),
        _gate("G12 trajectory mean-MSE confound", True, "validated across 192 CPU constructions", "CPU gate passed"),
        _gate("G13 realized-support confound", True, "validated across 192 CPU constructions", "CPU gate passed"),
        _gate("G14 exact 192-row key set", len(expected_primary_keys(config, trajectories)) == 192, 192, "192"),
        _gate("G15 finite frame SSIM", all(math.isfinite(float(row["frame_ssim_mean"])) for row in exact_results), exact_results, "finite"),
        _gate("G16 allowed decision fields only", _decision_field_gate(config)[0], _decision_field_gate(config)[1], "primary only"),
        _gate("G17 temporal/CLIP excluded", _decision_field_gate(config)[0], _decision_field_gate(config)[1], "excluded"),
        _gate("G18 FP16 anomaly cannot enter decision", all(row["perturbation_family"] == PERTURBATION_FAMILY for row in matcher_results), PERTURBATION_FAMILY, "no FP16"),
        _gate("G19 artifact existence/hash validation", retained, exact_results, "all retained"),
        _gate("G20 primary manifest deterministic", primary_manifest(config, rows, target) == stored_manifest, stored_manifest["manifest_sha256"], "unchanged"),
        _gate("G21 source cannot answer centered tail", source_falsifiability_audit(rows, trajectories, config)["preregistered_question_not_already_answered_by_source_data"], source_falsifiability_audit(rows, trajectories, config), "insufficient source replicates"),
        _gate("G22 no automatic expansion", config["allowed_modes"] == list(ALLOWED_MODES), config["allowed_modes"], "four modes"),
    ]
    atomic_json(output_dir / "preflight_results.json", {"clean_full_controls": exact_results, "matcher_controls": matcher_results, "expert_metadata": actual_expert})
    _write_gates(output_dir / "preflight_gates.json", gates)
    return {"mode": "preflight", "all_passed": True, "exact_controls": len(exact_results)}


def _scientific_row(
    config: dict[str, Any],
    provenance: dict[str, Any],
    trajectory: dict[str, Any],
    source: base.SourceTrajectory,
    replicate_id: int,
    replicate_seed: int,
    details: dict[str, Any],
    candidate_record: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    return {
        "status": "COMPLETE",
        "experiment_version": config["experiment_version"],
        "config_hash": config_hash(config),
        "provenance_hash": provenance["provenance_hash"],
        "source_raw_sha256": config["trusted_v3"]["raw_results_sha256"],
        "model": config["model"],
        "scheduler": EXPECTED_SCHEDULER,
        "trajectory_id": trajectory["trajectory_id"],
        "prompt_id": source.prompt_id,
        "prompt_text": source.prompt,
        "generation_seed": source.seed,
        "checkpoint_step": PRIMARY_STEP,
        "resume_index": result["resume_index"],
        "resume_timestep": trajectory["resume_timestep"],
        "expert_regime": trajectory["expert_regime"],
        "remaining_high_noise_steps": trajectory["remaining_high_noise_steps"],
        "remaining_low_noise_steps": trajectory["remaining_low_noise_steps"],
        "clean_checkpoint_hash": source.clean_hash,
        "clean_reference_video_hash": trajectory["clean_reference_video_hash"],
        "clean_reference_final_latent_hash": trajectory["clean_reference_final_latent_hash"],
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed,
        "perturbation_family": PERTURBATION_FAMILY,
        "target_mse": details["target_mse"],
        "intended_support_fraction": details["intended_active_fraction"],
        "realized_runtime_active_fraction": details["realized_runtime_active_fraction"],
        "active_elements": details["active_elements"],
        "realized_nonzero_elements": details["realized_nonzero_elements"],
        "total_elements": details["total_elements"],
        "realized_probe_mse": details["realized_probe_mse"],
        "realized_runtime_bf16_mse": details["realized_runtime_bf16_mse"],
        "relative_mse_mismatch": details["relative_mse_mismatch"],
        "solved_scale": details["solved_scale"],
        "runtime_input_hash": details["runtime_input_hash"],
        "final_latent_mse": result["final_latent_mse"],
        "video_mse": result["video_mse"],
        "video_psnr": result["video_psnr"],
        "frame_ssim_mean": result["frame_ssim_mean"],
        "temporal_delta_mse": result["temporal_delta_mse"],
        "temporal_delta_agreement": result["temporal_delta_agreement"],
        "prompt_clip_score": "",
        "resume_ms": result["resume_ms"],
        "runtime_candidate_sha256": candidate_record["tensor_sha256"],
        "recovered_final_latent_sha256": result["recovered_final_latent_sha256"],
        "recovered_video_sha256": result["recovered_video_sha256"],
        "runtime_candidate_path": candidate_record["path"],
        "recovered_final_latent_path": result["recovered_final_latent_artifact"]["path"],
        "recovered_video_path": result["recovered_video_artifact"]["path"],
        "runtime_candidate_artifact": candidate_record,
        "recovered_final_latent_artifact": result["recovered_final_latent_artifact"],
        "recovered_video_artifact": result["recovered_video_artifact"],
        "result_path": str(result_path),
    }


def run_smoke(
    config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    validate_output_namespace(config, output_dir)
    provenance = build_provenance(config_path)
    require_mode_gate(output_dir, "preflight_gates.json", provenance)
    rows = source_rows(config)
    target = derive_target(config, rows)
    trajectories = derive_primary_trajectories(rows, config)
    stored_manifest = json.loads((output_dir / "primary_trajectory_manifest.json").read_text())
    if primary_manifest(config, rows, target) != stored_manifest:
        raise GlobalStopError("GLOBAL STOP: primary manifest changed")
    seeds = derive_replicate_seeds(config)
    scientific_rows = []
    omni = v3.build_omni(config, args)
    try:
        for trajectory in trajectories:
            source = _source_trajectory(config, trajectory)
            for replicate_id, seed in enumerate(seeds):
                directory = output_dir / "smoke/cells" / trajectory["trajectory_id"] / f"replicate_{replicate_id:02d}"
                result_path = directory / "result.json"
                identity = {
                    "trajectory_id": trajectory["trajectory_id"],
                    "replicate_id": replicate_id,
                    "replicate_seed": seed,
                    "checkpoint_step": PRIMARY_STEP,
                    "perturbation_family": PERTURBATION_FAMILY,
                }
                cached = _result_valid(result_path, provenance, identity)
                if cached is not None:
                    scientific_rows.append(cached)
                    continue
                candidate, details = construct_dense_perturbation(
                    source.clean,
                    target_mse=float(target["target_mse"]),
                    replicate_seed=seed,
                    relative_tolerance=float(config["perturbation"]["runtime_mse_relative_tolerance"]),
                )
                candidate_record = _save_array(directory / "scientific_artifacts/runtime_candidate.npy", candidate)
                result = base.run_resume(
                    omni,
                    config,
                    source,
                    candidate,
                    step_index=PRIMARY_STEP,
                    label=f"trajectory_tail_{trajectory['prompt_id']}_replicate_{replicate_id:02d}",
                    directory=directory / "scientific_artifacts",
                )
                row = _scientific_row(
                    config, provenance, trajectory, source, replicate_id, seed,
                    details, candidate_record, result, result_path,
                )
                atomic_json(result_path, row)
                scientific_rows.append(row)
                print(
                    f"[checkpoint-stability] {trajectory['trajectory_id']} replicate={replicate_id} "
                    f"mismatch={details['relative_mse_mismatch']:.6g} ssim={result['frame_ssim_mean']:.4f}",
                    flush=True,
                )
    finally:
        _shutdown(omni)
    scientific_rows.sort(key=lambda row: (row["trajectory_id"], int(row["replicate_id"])))
    validate_expected_keys(scientific_rows, config, trajectories)
    validate_seed_schedule(scientific_rows, config)
    construction = construction_statistics(scientific_rows, config)
    all_artifacts = all(
        _validate_saved_array(row[field])
        for row in scientific_rows
        for field in (
            "runtime_candidate_artifact",
            "recovered_final_latent_artifact",
            "recovered_video_artifact",
        )
    )
    gates = [
        _gate("G1 trusted source hashes exact", True, validate_source_hashes(config), "all source hashes match"),
        _gate("G2 Euler only / invalid UniPC excluded", all(row["scheduler"] == EXPECTED_SCHEDULER for row in scientific_rows), len(scientific_rows), "Euler only"),
        _gate("G3 exactly 12 source-derived trajectories", len({row["trajectory_id"] for row in scientific_rows}) == 12, 12, "12"),
        _gate("G4 primary checkpoint step20 only", all(int(row["checkpoint_step"]) == PRIMARY_STEP for row in scientific_rows), PRIMARY_STEP, "20"),
        _gate("G5 same derived expert regime", all(row["expert_regime"] == EXPECTED_EXPERT for row in scientific_rows), EXPECTED_EXPERT, EXPECTED_EXPERT),
        _gate("G6 clean FULL exact", json.loads((output_dir / "preflight_gates.json").read_text())["all_passed"], str(output_dir / "preflight_gates.json"), "preflight passed"),
        _gate("G7 resume index/timestep", all(int(row["resume_index"]) == PRIMARY_STEP and math.isclose(float(row["resume_timestep"]), trajectories[0]["resume_timestep"], abs_tol=1e-6) for row in scientific_rows), trajectories[0]["resume_timestep"], "fixed"),
        _gate("G8 BF16 state accounting", all(row["runtime_dtype"] == EXPECTED_RUNTIME_DTYPE and row["runtime_full_bytes"] == row["runtime_numel"] * 2 for row in trajectories), trajectories, "BF16"),
        _gate("G9 dense-additive operator only", all(row["perturbation_family"] == PERTURBATION_FAMILY and int(row["active_elements"]) == int(row["total_elements"]) for row in scientific_rows), PERTURBATION_FAMILY, "dense additive"),
        _gate("G10 shared 16 replicate seeds", True, seeds, "shared schedule validated"),
        _gate("G11 per-cell BF16 MSE tolerance", construction["all_cells_mse_matched"], construction["max_per_cell_relative_mse_mismatch"], "within 1%"),
        _gate("G12 trajectory mean-MSE confound", construction["trajectory_mean_mse_gate_passed"], construction["trajectory_mean_mse_range_relative_to_target"], "<=0.001"),
        _gate("G13 realized-support confound", construction["trajectory_mean_support_gate_passed"], construction["trajectory_mean_support_range"], "<=0.005"),
        _gate("G14 exact 192-row key set", len(scientific_rows) == 192, len(scientific_rows), "192 exact keys"),
        _gate("G15 finite frame SSIM", all(math.isfinite(float(row["frame_ssim_mean"])) for row in scientific_rows), len(scientific_rows), "finite"),
        _gate("G16 allowed decision fields only", _decision_field_gate(config)[0], _decision_field_gate(config)[1], "primary only"),
        _gate("G17 temporal/CLIP excluded", _decision_field_gate(config)[0], _decision_field_gate(config)[1], "excluded"),
        _gate("G18 FP16 anomaly cannot enter decision", all(row["perturbation_family"] == PERTURBATION_FAMILY for row in scientific_rows), PERTURBATION_FAMILY, "no FP16"),
        _gate("G19 artifact existence/hash validation", all_artifacts, {"rows": 192, "artifacts_per_row": 3}, "all retained"),
        _gate("G20 primary manifest deterministic", primary_manifest(config, rows, target) == stored_manifest, stored_manifest["manifest_sha256"], "unchanged"),
        _gate("G21 source cannot answer centered tail", source_falsifiability_audit(rows, trajectories, config)["preregistered_question_not_already_answered_by_source_data"], source_falsifiability_audit(rows, trajectories, config), "insufficient source replicates"),
        _gate("G22 no automatic expansion", config["allowed_modes"] == list(ALLOWED_MODES), config["allowed_modes"], "four modes"),
    ]
    write_csv(output_dir / "smoke_raw.csv", scientific_rows, RAW_FIELDS)
    atomic_json(output_dir / "smoke_construction_summary.json", construction)
    _write_gates(output_dir / "smoke_gates.json", gates)
    atomic_json(output_dir / "smoke_summary.json", {"mode": "smoke", "row_count": 192, "all_passed": True})
    return {"mode": "smoke", "row_count": 192, "all_passed": True}


def _load_smoke_results(
    config: dict[str, Any], provenance: dict[str, Any], output_dir: Path,
    trajectories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    seeds = derive_replicate_seeds(config)
    for trajectory in trajectories:
        for replicate_id, seed in enumerate(seeds):
            path = output_dir / "smoke/cells" / trajectory["trajectory_id"] / f"replicate_{replicate_id:02d}" / "result.json"
            row = _result_valid(
                path,
                provenance,
                {
                    "trajectory_id": trajectory["trajectory_id"],
                    "replicate_id": replicate_id,
                    "replicate_seed": seed,
                    "checkpoint_step": PRIMARY_STEP,
                    "perturbation_family": PERTURBATION_FAMILY,
                },
            )
            if row is None:
                raise GlobalStopError(f"GLOBAL STOP: missing completed smoke result: {path}")
            rows.append(row)
    return rows


def run_analyze_smoke(
    config: dict[str, Any], config_path: Path, output_dir: Path
) -> dict[str, Any]:
    validate_output_namespace(config, output_dir)
    provenance = build_provenance(config_path)
    require_mode_gate(output_dir, "smoke_gates.json", provenance)
    source = source_rows(config)
    trajectories = derive_primary_trajectories(source, config)
    rows = _load_smoke_results(config, provenance, output_dir, trajectories)
    smoke_gates = json.loads((output_dir / "smoke_gates.json").read_text())
    result = analyze_rows(rows, config, correctness_passed=bool(smoke_gates["all_passed"]))
    write_csv(output_dir / "trajectory_tail_summary.csv", result["trajectory_summaries"])
    write_csv(output_dir / "leave_one_replicate_out.csv", result["leave_one_replicate_out"])
    atomic_json(output_dir / "checkpoint_stability_summary.json", result)
    report = [
        "# Fixed-Progress Trajectory-Conditioned Stability Kill Test",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"Tail-depth range: {result['tail_depth_range']:.6f}",
        f"Maximum trajectory tail depth: {result['maximum_trajectory_tail_depth']:.6f}",
        f"Maximum large-drop rate (supporting only): {result['max_large_drop_rate']:.6f}",
        f"LOO tail-depth range >= 0.075: {result['loo_tail_depth_threshold_count']}/16",
        f"Minimum LOO tail-depth range: {result['minimum_loo_tail_depth_range']:.6f}",
        "",
        "All primary trajectories are source-derived step-20 checkpoints.",
        "Absolute SSIM levels, minimum SSIM, large-drop rate, FP16, temporal metrics, and CLIP cannot rescue the decision.",
    ]
    (output_dir / "video_checkpoint_stability_killtest.md").write_text("\n".join(report) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=ALLOWED_MODES, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/video_checkpoint_stability_killtest_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/video_checkpoint_stability_killtest",
    )
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(config_path)
    if args.mode == "cpu":
        result = run_cpu_mode(config, config_path, output_dir)
    elif args.mode == "preflight":
        result = run_preflight(config, config_path, output_dir, args)
    elif args.mode == "smoke":
        result = run_smoke(config, config_path, output_dir, args)
    else:
        result = run_analyze_smoke(config, config_path, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
