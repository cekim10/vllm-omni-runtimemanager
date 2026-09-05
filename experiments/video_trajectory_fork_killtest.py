#!/usr/bin/env python3
"""Fatal screening for native Wan2.2 trajectory-prefix forkability.

This experiment changes only text conditioning after an exact Euler resume
boundary. It deliberately does not implement correction, blending, or caching.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "experiments/video_trajectory_fork_killtest_config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/video_trajectory_fork_killtest"
EXPECTED_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
EXPECTED_SCHEDULER = "WanEulerScheduler"
EXPECTED_SWITCHES = (0, 5, 10, 15, 20, 25, 30)
PRIMARY_FAMILY = "red_to_blue"
TRUSTED_SOURCE_FILES = (
    "experiments/video_trajectory_fork_killtest.py",
    "experiments/video_trajectory_fork_killtest_config.yaml",
    "experiments/run_video_trajectory_fork_killtest_gpu0.sh",
    "tests/diffusion/test_video_trajectory_fork_killtest.py",
    "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
    "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
)
FORBIDDEN_OUTPUT_PARTS = (
    "video_runtime_state_discovery",
    "video_state_protection",
    "video_bf16",
    "video_runtime_error_shape",
)
RAW_FIELDS = (
    "status",
    "experiment_version",
    "provenance_hash",
    "prompt_family",
    "severity",
    "old_prompt",
    "new_prompt",
    "active_prompt",
    "active_prompt_sha256",
    "seed",
    "trajectory_type",
    "switch_step",
    "resume_index",
    "resume_scheduler_timestep",
    "prefix_reuse_fraction",
    "scheduler",
    "scheduler_class",
    "expert_at_switch",
    "remaining_high_noise_steps",
    "remaining_low_noise_steps",
    "crosses_expert_boundary_after_resume",
    "initial_latent_hash",
    "fork_latent_hash",
    "resume_input_hash",
    "final_latent_hash",
    "video_hash",
    "score_old_condition",
    "score_new_condition",
    "new_minus_old",
    "ssim_to_old",
    "ssim_to_new",
    "video_mse_to_old",
    "video_mse_to_new",
    "final_latent_mse_to_old",
    "final_latent_mse_to_new",
    "wall_time_s",
    "control_exact",
    "final_video_npy",
    "final_video_mp4",
    "final_latent_npy",
    "condition_metadata_json",
    "notes",
)


class GateError(RuntimeError):
    """A fatal experimental-control or provenance failure."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    identity = {
        "version": "numpy-array-identity-v1",
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "bytes_sha256": sha256_bytes(value.tobytes(order="C")),
    }
    return sha256_bytes(canonical_json(identity))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    generation = config["generation"]
    scheduler = config["scheduler"]
    if config["model"] != EXPECTED_MODEL:
        raise ValueError(f"Model must remain {EXPECTED_MODEL}")
    if scheduler["name"] != EXPECTED_SCHEDULER or scheduler["sample_solver"] != "euler":
        raise ValueError("The kill test requires the trusted Wan Euler scheduler")
    expected_generation = {
        "height": 480,
        "width": 832,
        "num_frames": 33,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "fps": 16.0,
        "boundary_ratio": 0.875,
    }
    for key, expected in expected_generation.items():
        if generation[key] != expected:
            raise ValueError(f"Frozen generation field changed: {key}={generation[key]!r}")
    if tuple(generation["switch_steps"]) != EXPECTED_SWITCHES:
        raise ValueError(f"Switch points must be exactly {EXPECTED_SWITCHES}")
    families = {row["id"]: row for row in config["prompt_families"]}
    if set(families) != {"red_to_blue", "car_to_truck", "car_to_sailboat"}:
        raise ValueError("Prompt families differ from the preregistered primary and two expansions")
    primary = families[PRIMARY_FAMILY]
    expected_old = "A red sports car driving on a snowy road, cinematic video"
    expected_new = "A blue sports car driving on a snowy road, cinematic video"
    if primary["old_prompt"] != expected_old or primary["new_prompt"] != expected_new:
        raise ValueError("Primary prompt pair changed")
    expected_expansions = {
        "car_to_truck": "A red pickup truck driving on a snowy road, cinematic video",
        "car_to_sailboat": "A small sailboat moving across the ocean, cinematic video",
    }
    for family_id, expected_new_prompt in expected_expansions.items():
        if families[family_id]["old_prompt"] != expected_old or families[family_id]["new_prompt"] != expected_new_prompt:
            raise ValueError(f"Preregistered expansion prompt changed: {family_id}")
    if config["decision"]["early_fatal_steps"] != [5, 10]:
        raise ValueError("Fatal-screen steps changed")
    if float(config["decision"]["automated_margin_support_threshold"]) != 0.0:
        raise ValueError("Automated concept evidence is preregistered as a sign-only comparison")
    history = [int(value) for value in config.get("seed_history", [])]
    if not history or history[-1] != int(config["seed"]):
        raise ValueError("seed_history must end with the active seed")
    if len(history) - 1 > int(config["seed_replacement_limit"]):
        raise ValueError("The one-time baseline-only seed replacement limit was exceeded")
    if len(history) > 1 and not config.get("seed_replacement_reason"):
        raise ValueError("A replacement seed requires a recorded baseline-only reason")


def validate_output_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT / "results")
    except ValueError as error:
        raise ValueError("Output must be under this repository's results directory") from error
    if not relative.parts or relative.parts[0] != "video_trajectory_fork_killtest":
        raise ValueError("Output must use the isolated video_trajectory_fork_killtest namespace")
    if any(part in str(relative) for part in FORBIDDEN_OUTPUT_PARTS):
        raise ValueError("Trusted prior-result namespace cannot be used")


def prompt_family(config: dict[str, Any], family_id: str) -> dict[str, Any]:
    for family in config["prompt_families"]:
        if family["id"] == family_id:
            return family
    raise KeyError(f"Unknown prompt family: {family_id}")


def expected_control_keys(switches: tuple[int, ...] = EXPECTED_SWITCHES) -> set[tuple[str, int]]:
    keys = {("old_baseline", -1), ("new_baseline", -1), ("fork_new", 0)}
    keys.update(("same_condition", step) for step in switches)
    return keys


def expected_primary_keys(switches: tuple[int, ...] = EXPECTED_SWITCHES) -> set[tuple[str, int]]:
    return {("fork_new", step) for step in switches if step > 0}


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["trajectory_type"]), int(row["switch_step"])


def validate_key_set(rows: list[dict[str, Any]], expected: set[tuple[str, int]]) -> None:
    keys = [row_key(row) for row in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    actual = set(keys)
    if duplicates or actual != expected:
        raise GateError(
            f"Result key-set mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}, duplicates={duplicates}"
        )


def merge_result_rows(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in existing + incoming:
        key = (str(row["prompt_family"]), str(row["trajectory_type"]), int(row["switch_step"]))
        previous = merged.get(key)
        if previous is not None:
            identities = ("provenance_hash", "final_latent_hash", "video_hash", "resume_input_hash")
            if any(str(previous.get(field)) != str(row.get(field)) for field in identities):
                raise GateError(f"Conflicting completed rows for {key}")
            continue
        merged[key] = row
    return sorted(merged.values(), key=lambda row: (row["prompt_family"], row_key(row)))


def build_provenance(config_path: Path) -> dict[str, Any]:
    file_paths = [REPO_ROOT / value for value in TRUSTED_SOURCE_FILES]
    missing = [str(path) for path in file_paths if not path.exists()]
    if missing:
        raise GateError(f"Provenance inputs missing: {missing}")
    hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in file_paths}
    if str(config_path.resolve().relative_to(REPO_ROOT)) not in hashes:
        hashes[str(config_path.resolve().relative_to(REPO_ROOT))] = sha256_file(config_path)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except Exception:
        commit, status = "UNKNOWN", ["git status unavailable"]
    relevant = [line for line in status if any(name in line for name in TRUSTED_SOURCE_FILES)]
    document = {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status": status,
        "relevant_git_status": relevant,
        "source_sha256": hashes,
        "config_sha256": sha256_file(config_path),
    }
    hash_basis = {
        "git_commit": commit,
        "relevant_git_status": relevant,
        "source_sha256": hashes,
        "config_sha256": document["config_sha256"],
    }
    document["provenance_hash"] = sha256_bytes(canonical_json(hash_basis))
    return document


def require_provenance(output_dir: Path, current: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "provenance.json"
    if not path.exists():
        raise GateError("Run the CPU phase first; provenance.json is missing")
    frozen = json.loads(path.read_text())
    if frozen.get("provenance_hash") != current["provenance_hash"]:
        raise GateError("Code/config provenance changed; use a fresh output namespace and rerun CPU")
    return frozen


def scheduler_plan(config: dict[str, Any]) -> dict[str, Any]:
    from experiments.video_runtime_state_discovery import scheduler_document

    adapted = json.loads(json.dumps(config))
    adapted["generation"]["checkpoint_steps"] = list(EXPECTED_SWITCHES)
    plan = scheduler_document(adapted)
    if plan["num_inference_steps"] != 40 or tuple(plan["checkpoint_indices"]) != EXPECTED_SWITCHES:
        raise GateError("Scheduler plan does not match frozen switch boundaries")
    return plan


def expert_metadata(config: dict[str, Any], plan: dict[str, Any], step: int) -> dict[str, Any]:
    if step == 40:
        return {
            "checkpoint_scheduler_timestep": None,
            "current_expert": None,
            "remaining_high_noise_steps": 0,
            "remaining_low_noise_steps": 0,
            "crosses_expert_boundary_after_resume": False,
        }
    from experiments.video_runtime_state_discovery import expert_region_metadata

    return expert_region_metadata(config, plan, step)


def run_cpu_phase(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(config_path)
    frozen_path = output_dir / "provenance.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen.get("provenance_hash") != provenance["provenance_hash"]:
            raise GateError("Output namespace already contains artifacts from different code/config")
    plan = scheduler_plan(config)
    manifest = {
        "status": "PASS",
        "experiment_version": config["experiment_version"],
        "provenance_hash": provenance["provenance_hash"],
        "model": config["model"],
        "seed": int(config["seed"]),
        "seed_history": config["seed_history"],
        "seed_replacement_count": len(config["seed_history"]) - 1,
        "seed_replacement_reason": config.get("seed_replacement_reason"),
        "primary_family": prompt_family(config, PRIMARY_FAMILY),
        "switch_steps": list(EXPECTED_SWITCHES),
        "control_keys": [list(value) for value in sorted(expected_control_keys())],
        "primary_keys": [list(value) for value in sorted(expected_primary_keys())],
        "scheduler": plan,
        "expert_by_switch": {str(step): expert_metadata(config, plan, step) for step in EXPECTED_SWITCHES},
        "decision_rule": config["decision"],
        "phase3_allowed_only_after": "PROMISING",
        "scientific_scope": "native conditioning-only trajectory fork; no correction or realized speedup claim",
    }
    atomic_json(output_dir / "provenance.json", provenance)
    atomic_json(output_dir / "cpu_manifest.json", manifest)
    return manifest


def require_cpu_manifest(config: dict[str, Any], output_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "cpu_manifest.json"
    if not path.exists():
        raise GateError("CPU manifest missing")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "PASS" or manifest.get("provenance_hash") != provenance["provenance_hash"]:
        raise GateError("CPU manifest is stale or failed")
    if tuple(manifest.get("switch_steps", [])) != EXPECTED_SWITCHES:
        raise GateError("CPU manifest switch points changed")
    if {tuple(value) for value in manifest.get("control_keys", [])} != expected_control_keys():
        raise GateError("CPU manifest control matrix changed")
    if {tuple(value) for value in manifest.get("primary_keys", [])} != expected_primary_keys():
        raise GateError("CPU manifest primary matrix changed")
    if int(manifest.get("seed")) != int(config["seed"]):
        raise GateError("CPU manifest seed differs from config")
    return manifest


def _coerce_embedding_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    candidate = getattr(value, "last_hidden_state", None)
    if isinstance(candidate, torch.Tensor):
        return candidate[:, 0] if candidate.ndim >= 2 else candidate
    if isinstance(value, tuple):
        for candidate in value:
            if isinstance(candidate, torch.Tensor):
                return candidate
    raise TypeError(f"Cannot extract an embedding tensor from {type(value)!r}")


class ConceptEvaluator:
    """Descriptive, uncalibrated frame-level CLIP concept comparison."""

    def __init__(self, config: dict[str, Any], family: dict[str, Any] | None = None) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model_name = config["concept_metric"]["model"]
        self.frame_count = int(config["concept_metric"]["frame_count"])
        self.old_concept = (family or {}).get("old_concept", config["concept_metric"]["old_concept"])
        self.new_concept = (family or {}).get("new_concept", config["concept_metric"]["new_concept"])
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).eval().cpu()

    def score(self, video: np.ndarray) -> dict[str, Any]:
        from PIL import Image

        indices = np.linspace(0, len(video) - 1, self.frame_count, dtype=int)
        images = [Image.fromarray(video[index, ..., :3]) for index in indices]
        image_inputs = self.processor(images=images, return_tensors="pt")
        text_inputs = self.processor(
            text=[self.old_concept, self.new_concept], return_tensors="pt", padding=True
        )
        with self.torch.inference_mode():
            image_features = _coerce_embedding_tensor(self.model.get_image_features(**image_inputs))
            text_features = _coerce_embedding_tensor(self.model.get_text_features(**text_inputs))
            image_features = self.torch.nn.functional.normalize(image_features, dim=-1)
            text_features = self.torch.nn.functional.normalize(text_features, dim=-1)
            scores = image_features @ text_features.T
        means = scores.mean(dim=0).detach().cpu().tolist()
        return {
            "score_old_condition": float(means[0]),
            "score_new_condition": float(means[1]),
            "new_minus_old": float(means[1] - means[0]),
            "frame_indices": [int(value) for value in indices],
            "metric_model": self.model_name,
            "metric_role": "descriptive_uncalibrated_concept_similarity",
        }


def _load_probe_tensor(path: str | Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu").detach().cpu().contiguous()


def _probe_array(record: dict[str, Any]) -> np.ndarray:
    return _load_probe_tensor(record["latent_path"]).float().numpy()


def checkpoint_hashes(records: dict[int, dict[str, Any]]) -> dict[str, str]:
    return {str(step): array_sha256(_probe_array(records[step])) for step in EXPECTED_SWITCHES}


def validate_checkpoint_hashes(
    records: dict[int, dict[str, Any]], expected: dict[str, str]
) -> None:
    if set(expected) != {str(step) for step in EXPECTED_SWITCHES}:
        raise GateError("Frozen OLD checkpoint hash set is incomplete")
    actual = checkpoint_hashes(records)
    if actual != expected:
        mismatched = [step for step in expected if actual.get(step) != expected[step]]
        raise GateError(f"OLD trajectory checkpoint artifact changed at steps {mismatched}")


def _record_by_step(metadata: dict[str, Any], step: int) -> dict[str, Any]:
    records = {int(row["step_index"]): row for row in metadata["records"]}
    if step not in records:
        raise GateError(f"Trajectory probe did not capture local step {step}; found {sorted(records)}")
    return records[step]


def validate_worker_probe(
    metadata: dict[str, Any], scheduler: dict[str, Any], resume_index: int, local_steps: int
) -> None:
    if metadata.get("sample_solver") != "euler" or not str(metadata.get("scheduler_class", "")).endswith(
        EXPECTED_SCHEDULER
    ):
        raise GateError("Worker did not execute the trusted Euler path")
    if int(metadata.get("num_steps", -1)) != local_steps:
        raise GateError("Worker executed an unexpected number of resumed scheduler steps")
    input_record = _record_by_step(metadata, 0)
    final_record = _record_by_step(metadata, local_steps)
    timesteps = [float(value) for value in scheduler["timesteps"]]
    if not math.isclose(float(input_record["timestep"]), timesteps[resume_index], rel_tol=0.0, abs_tol=1e-5):
        raise GateError("Resume input is bound to the wrong scheduler timestep")
    if not math.isclose(float(final_record["timestep"]), timesteps[-1], rel_tol=0.0, abs_tol=1e-5):
        raise GateError("Resumed execution skipped or did not reach the final scheduler timestep")


def _save_artifacts(
    output_dir: Path,
    family_id: str,
    label: str,
    video: np.ndarray,
    final_latent: np.ndarray,
    fps: float,
) -> dict[str, str]:
    from experiments.temporal_dimension_killtest_preflight import _save_video
    from PIL import Image

    condition_dir = output_dir / "artifacts" / family_id / label
    condition_dir.mkdir(parents=True, exist_ok=True)
    video_path = condition_dir / "final_video.npy"
    latent_path = condition_dir / "final_latent.npy"
    mp4_path = output_dir / "qualitative" / family_id / f"{label}.mp4"
    np.save(video_path, np.ascontiguousarray(video), allow_pickle=False)
    np.save(latent_path, np.ascontiguousarray(final_latent), allow_pickle=False)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    _save_video(mp4_path, video, fps)
    selected = {"first": 0, "middle": len(video) // 2, "last": len(video) - 1}
    for name, index in selected.items():
        Image.fromarray(video[index, ..., :3]).save(mp4_path.parent / f"{label}_{name}.png")
    for path in (video_path, latent_path, mp4_path):
        if not path.exists() or path.stat().st_size == 0:
            raise GateError(f"Required scientific artifact was not persisted: {path}")
    return {
        "final_video_npy": str(video_path),
        "final_video_mp4": str(mp4_path),
        "final_latent_npy": str(latent_path),
    }


def _mse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        raise GateError(f"MSE shape mismatch: {lhs.shape} != {rhs.shape}")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    return float(np.mean(delta * delta))


def _reference_metrics(video: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    from experiments.video_runtime_state_discovery import video_metrics

    metrics = video_metrics(video, reference)
    return float(metrics["frame_ssim_mean"]), float(metrics["video_mse"])


def _load_saved_baseline(output_dir: Path, family_id: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    root = output_dir / "artifacts" / family_id / label
    video_path, latent_path = root / "final_video.npy", root / "final_latent.npy"
    if not video_path.exists() or not latent_path.exists():
        raise GateError(f"Required baseline artifacts missing for {family_id}/{label}")
    return np.load(video_path, allow_pickle=False), np.load(latent_path, allow_pickle=False)


def _result_path(output_dir: Path, family_id: str, label: str) -> Path:
    return output_dir / "rows" / family_id / f"{label}.json"


def _validate_result_artifacts(row: dict[str, Any], provenance_hash: str) -> None:
    if row.get("status") != "COMPLETE" or row.get("provenance_hash") != provenance_hash:
        raise GateError("Completed row has invalid status/provenance")
    video = np.load(row["final_video_npy"], allow_pickle=False)
    latent = np.load(row["final_latent_npy"], allow_pickle=False)
    if array_sha256(video) != row["video_hash"] or array_sha256(latent) != row["final_latent_hash"]:
        raise GateError("Persisted result artifact hash mismatch")
    if not Path(row["final_video_mp4"]).exists() or not Path(row["condition_metadata_json"]).exists():
        raise GateError("Completed row is missing qualitative or metadata artifacts")


def _save_row(output_dir: Path, row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    path = _result_path(output_dir, row["prompt_family"], metadata["label"])
    metadata_path = path.with_name(path.stem + "_metadata.json")
    atomic_json(metadata_path, metadata)
    row["condition_metadata_json"] = str(metadata_path)
    atomic_json(path, row)
    _validate_result_artifacts(row, row["provenance_hash"])
    return row


def _load_existing_row(output_dir: Path, family_id: str, label: str, provenance_hash: str) -> dict[str, Any] | None:
    path = _result_path(output_dir, family_id, label)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    _validate_result_artifacts(row, provenance_hash)
    return row


def _build_omni(config: dict[str, Any], args: argparse.Namespace) -> Any:
    from experiments.video_runtime_state_discovery import build_omni

    return build_omni(config, args)


def _run_generate(
    omni: Any,
    config: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    label: str,
    artifact_dir: Path,
    capture_steps: list[int],
    latents: Any = None,
    step_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any], float]:
    from experiments.video_runtime_state_discovery import run_generate

    return run_generate(
        omni,
        config,
        prompt=prompt,
        seed=seed,
        label=label,
        artifact_dir=artifact_dir,
        capture_steps=capture_steps,
        latents=latents,
        step_index=step_index,
    )


def _make_row(
    *,
    config: dict[str, Any],
    provenance_hash: str,
    family: dict[str, Any],
    trajectory_type: str,
    switch_step: int,
    active_prompt: str,
    input_latent: np.ndarray,
    initial_hash: str,
    video: np.ndarray,
    final_latent: np.ndarray,
    wall_ms: float,
    scheduler: dict[str, Any],
    metric: dict[str, Any],
    old_reference: tuple[np.ndarray, np.ndarray] | None,
    new_reference: tuple[np.ndarray, np.ndarray] | None,
    artifacts: dict[str, str],
    control_exact: bool | None,
    notes: str,
) -> dict[str, Any]:
    expert = expert_metadata(config, scheduler, switch_step if switch_step >= 0 else 0)
    if old_reference is None:
        ssim_old = mse_video_old = mse_latent_old = None
    else:
        ssim_old, mse_video_old = _reference_metrics(video, old_reference[0])
        mse_latent_old = _mse(final_latent, old_reference[1])
    if new_reference is None:
        ssim_new = mse_video_new = mse_latent_new = None
    else:
        ssim_new, mse_video_new = _reference_metrics(video, new_reference[0])
        mse_latent_new = _mse(final_latent, new_reference[1])
    return {
        "status": "COMPLETE",
        "experiment_version": config["experiment_version"],
        "provenance_hash": provenance_hash,
        "prompt_family": family["id"],
        "severity": family["severity"],
        "old_prompt": family["old_prompt"],
        "new_prompt": family["new_prompt"],
        "active_prompt": active_prompt,
        "active_prompt_sha256": sha256_bytes(active_prompt.encode()),
        "seed": int(config["seed"]),
        "trajectory_type": trajectory_type,
        "switch_step": switch_step,
        "resume_index": max(switch_step, 0),
        "resume_scheduler_timestep": expert["checkpoint_scheduler_timestep"],
        "prefix_reuse_fraction": 0.0 if switch_step < 0 else switch_step / 40.0,
        "scheduler": EXPECTED_SCHEDULER,
        "scheduler_class": scheduler["scheduler_class"],
        "expert_at_switch": expert["current_expert"],
        "remaining_high_noise_steps": expert["remaining_high_noise_steps"],
        "remaining_low_noise_steps": expert["remaining_low_noise_steps"],
        "crosses_expert_boundary_after_resume": expert["crosses_expert_boundary_after_resume"],
        "initial_latent_hash": initial_hash,
        "fork_latent_hash": array_sha256(input_latent),
        "resume_input_hash": array_sha256(input_latent),
        "final_latent_hash": array_sha256(final_latent),
        "video_hash": array_sha256(video),
        **{key: metric[key] for key in ("score_old_condition", "score_new_condition", "new_minus_old")},
        "ssim_to_old": ssim_old,
        "ssim_to_new": ssim_new,
        "video_mse_to_old": mse_video_old,
        "video_mse_to_new": mse_video_new,
        "final_latent_mse_to_old": mse_latent_old,
        "final_latent_mse_to_new": mse_latent_new,
        "wall_time_s": wall_ms / 1000.0,
        "control_exact": control_exact,
        **artifacts,
        "condition_metadata_json": None,
        "notes": notes,
    }


def _execute_one(
    omni: Any,
    evaluator: ConceptEvaluator,
    config: dict[str, Any],
    provenance: dict[str, Any],
    scheduler: dict[str, Any],
    output_dir: Path,
    family: dict[str, Any],
    *,
    label: str,
    trajectory_type: str,
    switch_step: int,
    prompt: str,
    input_tensor: Any,
    initial_hash: str,
    old_reference: tuple[np.ndarray, np.ndarray] | None,
    new_reference: tuple[np.ndarray, np.ndarray] | None,
    exact_reference: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    existing = _load_existing_row(output_dir, family["id"], label, provenance["provenance_hash"])
    if existing is not None:
        return existing
    local_steps = 40 if switch_step < 0 else 40 - switch_step
    capture_steps = sorted({0, local_steps})
    probe_dir = output_dir / "trajectory_probes" / family["id"] / label
    video, metadata, wall_ms = _run_generate(
        omni,
        config,
        prompt=prompt,
        seed=int(config["seed"]),
        label=label,
        artifact_dir=probe_dir,
        capture_steps=capture_steps,
        latents=input_tensor,
        step_index=max(switch_step, 0),
    )
    validate_worker_probe(metadata, scheduler, max(switch_step, 0), local_steps)
    input_record = _record_by_step(metadata, 0)
    final_record = _record_by_step(metadata, local_steps)
    realized_input = _probe_array(input_record)
    expected_input = input_tensor.detach().cpu().float().numpy()
    if array_sha256(realized_input) != array_sha256(expected_input):
        raise GateError(f"Resume input changed before execution for {label}")
    final_latent = _probe_array(final_record)
    exact = None
    if exact_reference is not None:
        exact = array_sha256(video) == array_sha256(exact_reference[0]) and array_sha256(final_latent) == array_sha256(
            exact_reference[1]
        )
    artifacts = _save_artifacts(
        output_dir, family["id"], label, video, final_latent, float(config["generation"]["fps"])
    )
    metric = evaluator.score(video)
    row = _make_row(
        config=config,
        provenance_hash=provenance["provenance_hash"],
        family=family,
        trajectory_type=trajectory_type,
        switch_step=switch_step,
        active_prompt=prompt,
        input_latent=realized_input,
        initial_hash=initial_hash,
        video=video,
        final_latent=final_latent,
        wall_ms=wall_ms,
        scheduler=scheduler,
        metric=metric,
        old_reference=old_reference,
        new_reference=new_reference,
        artifacts=artifacts,
        control_exact=exact,
        notes="Only conditioning changes after the exact resume boundary; CLIP scores are descriptive.",
    )
    return _save_row(
        output_dir,
        row,
        {
            "label": label,
            "worker_trajectory_probe": metadata,
            "concept_metric": metric,
            "scheduler_plan": scheduler,
            "generation_parameters": config["generation"],
        },
    )


def _run_baseline(
    omni: Any,
    evaluator: ConceptEvaluator,
    config: dict[str, Any],
    provenance: dict[str, Any],
    scheduler: dict[str, Any],
    output_dir: Path,
    family: dict[str, Any],
    *,
    which: str,
    capture_steps: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"{which}_baseline"
    existing = _load_existing_row(output_dir, family["id"], label, provenance["provenance_hash"])
    metadata_path = output_dir / "rows" / family["id"] / f"{label}_metadata.json"
    if existing is not None:
        metadata = json.loads(metadata_path.read_text())["worker_trajectory_probe"]
        return existing, metadata
    prompt = family[f"{which}_prompt"]
    probe_dir = output_dir / "trajectory_probes" / family["id"] / label
    video, metadata, wall_ms = _run_generate(
        omni,
        config,
        prompt=prompt,
        seed=int(config["seed"]),
        label=label,
        artifact_dir=probe_dir,
        capture_steps=capture_steps,
        latents=None,
        step_index=0,
    )
    validate_worker_probe(metadata, scheduler, 0, 40)
    input_latent = _probe_array(_record_by_step(metadata, 0))
    final_latent = _probe_array(_record_by_step(metadata, 40))
    artifacts = _save_artifacts(
        output_dir, family["id"], label, video, final_latent, float(config["generation"]["fps"])
    )
    metric = evaluator.score(video)
    row = _make_row(
        config=config,
        provenance_hash=provenance["provenance_hash"],
        family=family,
        trajectory_type=label,
        switch_step=-1,
        active_prompt=prompt,
        input_latent=input_latent,
        initial_hash=array_sha256(input_latent),
        video=video,
        final_latent=final_latent,
        wall_ms=wall_ms,
        scheduler=scheduler,
        metric=metric,
        old_reference=None,
        new_reference=None,
        artifacts=artifacts,
        control_exact=True,
        notes="Independent full baseline with the frozen seed and initial noise.",
    )
    _save_row(
        output_dir,
        row,
        {
            "label": label,
            "worker_trajectory_probe": metadata,
            "concept_metric": metric,
            "scheduler_plan": scheduler,
            "generation_parameters": config["generation"],
        },
    )
    return row, metadata


def validate_baseline_judgment(document: dict[str, Any], provenance_hash: str, family_id: str) -> None:
    if document.get("provenance_hash") != provenance_hash or document.get("prompt_family") != family_id:
        raise GateError("Baseline judgment provenance/family mismatch")
    required = ("old_matches_expected", "new_matches_expected", "clearly_different")
    if any(not isinstance(document.get(key), bool) for key in required):
        raise GateError("Baseline judgment must contain explicit booleans")
    if not all(document[key] for key in required):
        raise GateError("C3 failed: baseline pair is not informative; do not inspect/run forks")
    if document.get("fork_outcomes_examined") is not False:
        raise GateError("C3 must be frozen before fork outcomes are examined")


def _write_baseline_template(output_dir: Path, provenance_hash: str, family: dict[str, Any]) -> None:
    path = output_dir / "baseline_judgment_template.json"
    if path.exists():
        return
    atomic_json(
        path,
        {
            "provenance_hash": provenance_hash,
            "prompt_family": family["id"],
            "old_matches_expected": None,
            "new_matches_expected": None,
            "clearly_different": None,
            "fork_outcomes_examined": False,
            "notes": "Inspect only OLD/NEW baseline videos and fixed first/middle/last frames.",
        },
    )


def run_controls(config: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(args.config)
    require_provenance(output_dir, provenance)
    manifest = require_cpu_manifest(config, output_dir, provenance)
    scheduler = manifest["scheduler"]
    family = prompt_family(config, PRIMARY_FAMILY)
    evaluator = ConceptEvaluator(config, family)
    omni = _build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        old_row, old_meta = _run_baseline(
            omni, evaluator, config, provenance, scheduler, output_dir, family,
            which="old", capture_steps=list(EXPECTED_SWITCHES) + [40]
        )
        new_row, new_meta = _run_baseline(
            omni, evaluator, config, provenance, scheduler, output_dir, family,
            which="new", capture_steps=[0, 40]
        )
        old_ref = _load_saved_baseline(output_dir, family["id"], "old_baseline")
        new_ref = _load_saved_baseline(output_dir, family["id"], "new_baseline")
        old_records = {int(row["step_index"]): row for row in old_meta["records"]}
        old_initial = _probe_array(old_records[0])
        new_initial = _probe_array(_record_by_step(new_meta, 0))
        if array_sha256(old_initial) != array_sha256(new_initial):
            raise GateError("OLD and NEW baselines did not use identical initial latent")
        initial_hash = array_sha256(old_initial)
        rows.extend((old_row, new_row))
        for step in EXPECTED_SWITCHES:
            checkpoint = _load_probe_tensor(old_records[step]["latent_path"])
            rows.append(
                _execute_one(
                    omni, evaluator, config, provenance, scheduler, output_dir, family,
                    label=f"same_condition_k{step:02d}", trajectory_type="same_condition",
                    switch_step=step, prompt=family["old_prompt"], input_tensor=checkpoint,
                    initial_hash=initial_hash, old_reference=old_ref, new_reference=new_ref,
                    exact_reference=old_ref,
                )
            )
        initial_tensor = _load_probe_tensor(old_records[0]["latent_path"])
        rows.append(
            _execute_one(
                omni, evaluator, config, provenance, scheduler, output_dir, family,
                label="fork_new_k00", trajectory_type="fork_new", switch_step=0,
                prompt=family["new_prompt"], input_tensor=initial_tensor, initial_hash=initial_hash,
                old_reference=old_ref, new_reference=new_ref, exact_reference=new_ref,
            )
        )
    finally:
        omni.shutdown()
    validate_key_set(rows, expected_control_keys())
    c1 = all(row["control_exact"] is True for row in rows if row["trajectory_type"] == "same_condition")
    c2_rows = [row for row in rows if row_key(row) == ("fork_new", 0)]
    c2 = len(c2_rows) == 1 and c2_rows[0]["control_exact"] is True
    preflight = {
        "status": "PASS" if c1 and c2 else "FAIL",
        "provenance_hash": provenance["provenance_hash"],
        "scheduler_euler": True,
        "same_condition_exact_all_switches": c1,
        "k0_new_baseline_exact": c2,
        "identical_initial_latent": True,
        "old_checkpoint_hashes": checkpoint_hashes(old_records),
        "control_row_count": len(rows),
        "expected_control_row_count": len(expected_control_keys()),
        "baseline_semantic_separation": "AWAITING_MANUAL_C3",
        "expert_boundary_timestep": float(config["generation"]["boundary_ratio"]) * 1000.0,
    }
    atomic_json(output_dir / "preflight.json", preflight)
    write_csv(output_dir / "raw_results.csv", merge_result_rows(read_csv(output_dir / "raw_results.csv"), rows))
    _write_baseline_template(output_dir, provenance["provenance_hash"], family)
    if preflight["status"] != "PASS":
        raise GateError("C1/C2 failed; INVALID / STOP")
    return preflight


def require_controls(output_dir: Path, provenance_hash: str) -> dict[str, Any]:
    path = output_dir / "preflight.json"
    if not path.exists():
        raise GateError("GPU controls have not completed")
    result = json.loads(path.read_text())
    required_true = (
        "scheduler_euler",
        "same_condition_exact_all_switches",
        "k0_new_baseline_exact",
        "identical_initial_latent",
    )
    if result.get("status") != "PASS" or result.get("provenance_hash") != provenance_hash:
        raise GateError("GPU preflight failed or has stale provenance")
    if not all(result.get(key) is True for key in required_true):
        raise GateError("GPU control gate is incomplete/failed")
    if int(result.get("control_row_count", -1)) != len(expected_control_keys()):
        raise GateError("GPU control result count mismatch")
    hashes = result.get("old_checkpoint_hashes", {})
    if set(hashes) != {str(step) for step in EXPECTED_SWITCHES} or not all(
        isinstance(value, str) and len(value) == 64 for value in hashes.values()
    ):
        raise GateError("GPU control gate lacks the complete frozen OLD checkpoint hash set")
    return result


def run_primary(config: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(args.config)
    require_provenance(output_dir, provenance)
    manifest = require_cpu_manifest(config, output_dir, provenance)
    controls = require_controls(output_dir, provenance["provenance_hash"])
    judgment_path = output_dir / "baseline_judgment.json"
    if not judgment_path.exists():
        raise GateError(
            "Fill baseline_judgment.json from baseline_judgment_template.json before running primary forks"
        )
    family = prompt_family(config, PRIMARY_FAMILY)
    validate_baseline_judgment(json.loads(judgment_path.read_text()), provenance["provenance_hash"], family["id"])
    old_ref = _load_saved_baseline(output_dir, family["id"], "old_baseline")
    new_ref = _load_saved_baseline(output_dir, family["id"], "new_baseline")
    old_meta_path = output_dir / "rows" / family["id"] / "old_baseline_metadata.json"
    old_meta = json.loads(old_meta_path.read_text())["worker_trajectory_probe"]
    old_records = {int(row["step_index"]): row for row in old_meta["records"]}
    validate_checkpoint_hashes(old_records, controls["old_checkpoint_hashes"])
    initial_hash = array_sha256(_probe_array(old_records[0]))
    evaluator = ConceptEvaluator(config, family)
    omni = _build_omni(config, args)
    rows: list[dict[str, Any]] = []
    try:
        for step in EXPECTED_SWITCHES[1:]:
            rows.append(
                _execute_one(
                    omni, evaluator, config, provenance, manifest["scheduler"], output_dir, family,
                    label=f"fork_new_k{step:02d}", trajectory_type="fork_new", switch_step=step,
                    prompt=family["new_prompt"], input_tensor=_load_probe_tensor(old_records[step]["latent_path"]),
                    initial_hash=initial_hash, old_reference=old_ref, new_reference=new_ref,
                    exact_reference=None,
                )
            )
    finally:
        omni.shutdown()
    validate_key_set(rows, expected_primary_keys())
    prior_rows = read_csv(output_dir / "raw_results.csv")
    for row in prior_rows:
        _validate_result_artifacts(row, provenance["provenance_hash"])
    all_rows = merge_result_rows(prior_rows, rows)
    write_csv(output_dir / "raw_results.csv", all_rows)
    _write_qualitative_template(output_dir, provenance["provenance_hash"], family, rows)
    provisional = analyze_primary(config, args.config, output_dir, require_manual=False)
    return provisional


def _write_qualitative_template(
    output_dir: Path, provenance_hash: str, family: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    path = output_dir / "qualitative_judgment_template.json"
    if path.exists():
        return
    atomic_json(
        path,
        {
            "provenance_hash": provenance_hash,
            "prompt_family": family["id"],
            "allowed_outcomes": ["new", "old", "mixed", "corrupted", "ambiguous"],
            "outcomes": {
                str(int(row["switch_step"])): {"outcome": None, "notes": ""}
                for row in sorted(rows, key=lambda value: int(value["switch_step"]))
            },
            "instructions": "Inspect each fixed first/middle/last frame and complete video; do not change thresholds.",
        },
    )


def validate_qualitative_judgment(
    document: dict[str, Any], provenance_hash: str, expected_steps: set[int], allowed: set[str]
) -> dict[int, str]:
    if document.get("provenance_hash") != provenance_hash or document.get("prompt_family") != PRIMARY_FAMILY:
        raise GateError("Qualitative judgment provenance/family mismatch")
    outcomes = document.get("outcomes", {})
    if {int(step) for step in outcomes} != expected_steps:
        raise GateError("Qualitative judgment switch-step set mismatch")
    result: dict[int, str] = {}
    for step, entry in outcomes.items():
        outcome = entry.get("outcome") if isinstance(entry, dict) else None
        if outcome not in allowed:
            raise GateError(f"Invalid/missing qualitative outcome at k={step}: {outcome!r}")
        result[int(step)] = outcome
    return result


def classify_primary(
    rows: list[dict[str, Any]],
    qualitative: dict[int, str] | None,
    controls_pass: bool,
) -> tuple[str, int | None, str]:
    if not controls_pass:
        return "INVALID", None, "Required Euler/fork controls failed."
    expected = expected_primary_keys()
    validate_key_set(rows, expected)
    if qualitative is None:
        return "INCONCLUSIVE", None, "Qualitative fork judgments are not yet frozen."
    by_step = {int(row["switch_step"]): row for row in rows}
    automated = {step: float(row["new_minus_old"]) for step, row in by_step.items()}
    disagreements = [
        step for step, outcome in qualitative.items()
        if (outcome == "new" and automated[step] <= 0.0) or (outcome == "old" and automated[step] >= 0.0)
    ]
    if disagreements:
        return "INCONCLUSIVE", None, f"Automated and qualitative evidence disagree at k={disagreements}."
    unresolved = [step for step, outcome in qualitative.items() if outcome in {"mixed", "corrupted", "ambiguous"}]
    if unresolved:
        return "INCONCLUSIVE", None, f"Qualitative result is mixed, corrupted, or ambiguous at k={unresolved}."
    successful = [
        step for step, outcome in qualitative.items()
        if step >= 10 and outcome == "new" and automated[step] > 0.0
    ]
    if successful:
        largest = max(successful)
        return "PROMISING", largest, f"NEW conditioning is clear with {largest / 40:.1%} prefix reuse."
    if all(qualitative[step] == "old" and automated[step] < 0.0 for step in (5, 10)):
        return "CLEAR NO-GO", None, "Both preregistered earliest meaningful forks remain OLD-conditioned."
    return "INCONCLUSIVE", None, "Evidence is mixed, ambiguous, corrupted, or only the k=5 fork works."


def _report(config: dict[str, Any], output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    primary_rows = sorted(
        [row for row in rows if row["prompt_family"] == PRIMARY_FAMILY and row["trajectory_type"] == "fork_new" and int(row["switch_step"]) > 0],
        key=lambda row: int(row["switch_step"]),
    )
    lines = [
        "# Video Trajectory Fork Kill Test",
        "",
        "## Decision",
        summary["decision"],
        "",
        "## Primary question",
        "Can a trajectory computed under P reuse a meaningful prefix and then respond to new conditioning P'?",
        "",
        "## Controls",
        f"- Same-conditioning exact resume: {summary['controls']['same_condition_exact_all_switches']}",
        f"- k=0 NEW-baseline equivalence: {summary['controls']['k0_new_baseline_exact']}",
        f"- Baseline semantic separation: {summary['controls']['baseline_semantic_separation']}",
        f"- Scheduler: {EXPECTED_SCHEDULER}; resume index is exactly k.",
        "",
        "## Primary result",
        "| switch k | prefix reused | old score | new score | new-old | SSIM OLD | SSIM NEW | qualitative |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    qualitative = summary.get("qualitative_outcomes", {})
    for row in primary_rows:
        step = int(row["switch_step"])
        lines.append(
            f"| {step} | {step / 40:.1%} | {float(row['score_old_condition']):.4f} | "
            f"{float(row['score_new_condition']):.4f} | {float(row['new_minus_old']):+.4f} | "
            f"{float(row['ssim_to_old']):.4f} | {float(row['ssim_to_new']):.4f} | "
            f"{qualitative.get(str(step), 'NOT_REVIEWED')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "The concept margin describes responsiveness to P'; SSIM to OLD and NEW are separate pixel-fidelity descriptors. "
            "Prefix reuse is theoretical compute reuse, not measured serving speedup.",
            "",
            "Wan2.2 expert identity and remaining high/low-noise steps are recorded per row; progress and expert regime are confounded.",
            "",
            "## Decision rationale",
            summary["decision_rationale"],
            "",
            "## Next action",
            summary["next_action"],
            "",
            "Automated CLIP-style concept scores are uncalibrated descriptive evidence and cannot independently determine the decision.",
        ]
    )
    (output_dir / "video_trajectory_fork_killtest.md").write_text("\n".join(lines) + "\n")


def analyze_primary(
    config: dict[str, Any], config_path: Path, output_dir: Path, *, require_manual: bool = True
) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    frozen = require_provenance(output_dir, provenance)
    require_cpu_manifest(config, output_dir, provenance)
    require_controls(output_dir, frozen["provenance_hash"])
    all_rows = read_csv(output_dir / "raw_results.csv")
    for row in all_rows:
        _validate_result_artifacts(row, frozen["provenance_hash"])
    primary = [
        row for row in all_rows
        if row["prompt_family"] == PRIMARY_FAMILY and row["trajectory_type"] == "fork_new" and int(row["switch_step"]) > 0
    ]
    validate_key_set(primary, expected_primary_keys())
    qualitative_path = output_dir / "qualitative_judgment.json"
    qualitative: dict[int, str] | None = None
    if qualitative_path.exists():
        qualitative = validate_qualitative_judgment(
            json.loads(qualitative_path.read_text()),
            frozen["provenance_hash"],
            {step for _, step in expected_primary_keys()},
            set(config["decision"]["manual_outcomes"]),
        )
    elif require_manual:
        raise GateError("Fill qualitative_judgment.json from the generated template before final analysis")
    decision, largest, rationale = classify_primary(primary, qualitative, controls_pass=True)
    baseline_judgment = json.loads((output_dir / "baseline_judgment.json").read_text())
    validate_baseline_judgment(baseline_judgment, frozen["provenance_hash"], PRIMARY_FAMILY)
    summary = {
        "decision": decision,
        "decision_rationale": rationale,
        "largest_working_switch_step": largest,
        "largest_prefix_reuse_fraction": None if largest is None else largest / 40.0,
        "prompt_family": PRIMARY_FAMILY,
        "seed": int(config["seed"]),
        "controls": {
            **json.loads((output_dir / "preflight.json").read_text()),
            "baseline_semantic_separation": all(
                baseline_judgment[key]
                for key in ("old_matches_expected", "new_matches_expected", "clearly_different")
            ),
        },
        "qualitative_outcomes": {} if qualitative is None else {str(k): v for k, v in qualitative.items()},
        "phase3_run": False,
        "phase3_reason": "Not allowed unless primary decision is PROMISING.",
        "next_action": (
            "EXPAND TO TWO PREDECLARED EDIT SEVERITIES"
            if decision == "PROMISING"
            else "STOP — primitive lacks useful headroom"
            if decision == "CLEAR NO-GO"
            else "AUDIT METRIC / CONTROL ONLY"
        ),
        "claim_boundary": "No scheduler, serving mechanism, Jacobian, caching, or realized speedup claim.",
    }
    atomic_json(output_dir / "summary.json", summary)
    _report(config, output_dir, summary, all_rows)
    return summary


def run_expansion(config: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists() or json.loads(summary_path.read_text()).get("decision") != "PROMISING":
        raise GateError("Phase 3 is forbidden unless the primary frozen decision is PROMISING")
    verified_primary = analyze_primary(config, args.config, output_dir, require_manual=True)
    if verified_primary.get("decision") != "PROMISING":
        raise GateError("Phase 3 is forbidden unless the primary frozen decision is PROMISING")
    provenance = build_provenance(args.config)
    require_provenance(output_dir, provenance)
    manifest = require_cpu_manifest(config, output_dir, provenance)
    controls = require_controls(output_dir, provenance["provenance_hash"])
    old_family = prompt_family(config, PRIMARY_FAMILY)
    old_ref = _load_saved_baseline(output_dir, old_family["id"], "old_baseline")
    old_meta_path = output_dir / "rows" / old_family["id"] / "old_baseline_metadata.json"
    old_meta = json.loads(old_meta_path.read_text())["worker_trajectory_probe"]
    old_records = {int(row["step_index"]): row for row in old_meta["records"]}
    validate_checkpoint_hashes(old_records, controls["old_checkpoint_hashes"])
    initial_hash = array_sha256(_probe_array(old_records[0]))
    expansion_rows: list[dict[str, Any]] = []
    omni = _build_omni(config, args)
    try:
        for family_id in ("car_to_truck", "car_to_sailboat"):
            family = prompt_family(config, family_id)
            evaluator = ConceptEvaluator(config, family)
            new_row, new_meta = _run_baseline(
                omni, evaluator, config, provenance, manifest["scheduler"], output_dir, family,
                which="new", capture_steps=[0, 40]
            )
            new_ref = _load_saved_baseline(output_dir, family_id, "new_baseline")
            new_initial = _probe_array(_record_by_step(new_meta, 0))
            if array_sha256(new_initial) != initial_hash:
                raise GateError(f"Expansion {family_id} did not use the frozen initial latent")
            expansion_rows.append(new_row)
            for step in EXPECTED_SWITCHES:
                row = _execute_one(
                    omni, evaluator, config, provenance, manifest["scheduler"], output_dir, family,
                    label=f"fork_new_k{step:02d}", trajectory_type="fork_new", switch_step=step,
                    prompt=family["new_prompt"], input_tensor=_load_probe_tensor(old_records[step]["latent_path"]),
                    initial_hash=initial_hash, old_reference=old_ref, new_reference=new_ref,
                    exact_reference=new_ref if step == 0 else None,
                )
                if step == 0 and row["control_exact"] is not True:
                    raise GateError(f"Expansion C2 failed for {family_id}")
                expansion_rows.append(row)
    finally:
        omni.shutdown()
    expected_per_family = {("new_baseline", -1)} | {("fork_new", step) for step in EXPECTED_SWITCHES}
    for family_id in ("car_to_truck", "car_to_sailboat"):
        validate_key_set(
            [row for row in expansion_rows if row["prompt_family"] == family_id], expected_per_family
        )
    prior_rows = read_csv(output_dir / "raw_results.csv")
    for row in prior_rows:
        _validate_result_artifacts(row, provenance["provenance_hash"])
    all_rows = merge_result_rows(prior_rows, expansion_rows)
    write_csv(output_dir / "raw_results.csv", all_rows)
    _write_expansion_template(output_dir, provenance["provenance_hash"], expansion_rows)
    summary = json.loads(summary_path.read_text())
    summary["phase3_run"] = True
    summary["phase3_reason"] = "Primary Phase 2 decision was PROMISING."
    summary["phase3_families"] = ["car_to_truck", "car_to_sailboat"]
    summary["phase3_row_count"] = len(expansion_rows)
    atomic_json(summary_path, summary)
    _report(config, output_dir, summary, all_rows)
    return summary


def _write_expansion_template(
    output_dir: Path, provenance_hash: str, rows: list[dict[str, Any]]
) -> None:
    path = output_dir / "expansion_qualitative_judgment_template.json"
    if path.exists():
        return
    outcomes: dict[str, dict[str, dict[str, str | None]]] = {}
    for family_id in ("car_to_truck", "car_to_sailboat"):
        outcomes[family_id] = {
            str(int(row["switch_step"])): {"outcome": None, "notes": ""}
            for row in rows
            if row["prompt_family"] == family_id and row["trajectory_type"] == "fork_new"
        }
    atomic_json(
        path,
        {
            "provenance_hash": provenance_hash,
            "allowed_outcomes": ["new", "old", "mixed", "corrupted", "ambiguous"],
            "outcomes": outcomes,
            "instructions": "Descriptive Phase-3 review only; do not revise the primary decision.",
        },
    )


def print_console_summary(
    mode: str,
    config: dict[str, Any],
    output_dir: Path,
    provenance: dict[str, Any],
    result: dict[str, Any],
) -> None:
    raw_rows = read_csv(output_dir / "raw_results.csv")
    completed_gpu_phases: list[str] = []
    if any(row["trajectory_type"] in {"old_baseline", "same_condition"} for row in raw_rows):
        completed_gpu_phases.append("controls")
    if any(
        row["prompt_family"] == PRIMARY_FAMILY
        and row["trajectory_type"] == "fork_new"
        and int(row["switch_step"]) > 0
        for row in raw_rows
    ):
        completed_gpu_phases.append("primary")
    if any(row["prompt_family"] != PRIMARY_FAMILY for row in raw_rows):
        completed_gpu_phases.append("expansion")
    family = prompt_family(config, PRIMARY_FAMILY)
    print(f"git_commit={provenance['git_commit']}")
    print(f"git_dirty={provenance['git_dirty']} relevant_status={provenance['relevant_git_status']}")
    print(f"files_created={[str(REPO_ROOT / path) for path in TRUSTED_SOURCE_FILES[:4]]}")
    print(f"gpu_runs_performed={completed_gpu_phases or 'none'}")
    print(
        f"prompt_family={PRIMARY_FAMILY} seed={result.get('seed', 'see config')} "
        f"old_prompt={family['old_prompt']!r} new_prompt={family['new_prompt']!r}"
    )
    print(f"controls={result.get('controls', result.get('status', 'not run'))}")
    print(f"phase2_decision={result.get('decision', 'NOT RUN')}")
    print(f"phase3_run={result.get('phase3_run', False)} reason={result.get('phase3_reason', 'gated')}")
    print(f"largest_working_prefix={result.get('largest_prefix_reuse_fraction')}")
    print(f"recommendation={result.get('next_action', 'continue staged execution only')}")
    print("caveat=concept scores are descriptive; prefix fraction is not realized serving speedup")
    print(f"output_dir={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "controls", "primary", "analyze", "expansion"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config(args.config)
    validate_output_path(output_dir)
    if args.mode == "cpu":
        result = run_cpu_phase(config, args.config, output_dir)
    elif args.mode == "controls":
        result = run_controls(config, args, output_dir)
    elif args.mode == "primary":
        result = run_primary(config, args, output_dir)
    elif args.mode == "analyze":
        result = analyze_primary(config, args.config, output_dir)
    else:
        result = run_expansion(config, args, output_dir)
    provenance = json.loads((output_dir / "provenance.json").read_text())
    print_console_summary(args.mode, config, output_dir, provenance, result)


if __name__ == "__main__":
    main()
