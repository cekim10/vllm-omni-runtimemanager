#!/usr/bin/env python3
"""Batch-composition reproducibility kill test for Wan2.2 video diffusion.

This experiment deliberately has no policy or mechanism.  It freezes a
12-target population, filler composition, run order, and per-target initial
noise before GPU work.  GPU execution is fail-closed until a runtime adapter
can prove both per-request noise injection and actual co-batch membership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_bf16_single_flip_killtest as single_flip
from experiments import video_runtime_state_discovery as v3

EXPECTED_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
EXPECTED_SCHEDULER = "WanEulerScheduler"
EXPECTED_SCHEDULER_CLASS = "vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler.WanEulerScheduler"
MODES = ("cpu", "preflight", "smoke", "analyze-smoke")
TARGET_MODES = ("S", "B4", "B8")
TARGET_COLUMNS = (
    "status", "experiment_version", "provenance_hash", "target_id", "target_prompt", "target_seed",
    "batch_mode", "effective_batch_size", "target_batch_index", "filler_ids_json", "filler_seeds_json",
    "run_order_index", "replay_id", "target_initial_noise_identity", "target_initial_latent_identity",
    "target_token_ids_hash", "target_attention_mask_hash", "scheduler_class", "scheduler_config_hash",
    "model", "model_revision", "runtime_dtype", "backend", "compile_state", "final_latent_identity",
    "video_identity", "final_latent_exact_vs_solo", "video_exact_vs_solo", "frame_ssim_mean", "video_mse",
    "final_latent_mse", "strong_difference", "target_output_artifact_json", "final_latent_artifact_json",
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


def tensor_identity(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    return sha256_bytes(canonical_json({"format": "batch-determinism-tensor-v1", "dtype": value.dtype.str, "shape": list(value.shape)}) + b"\0" + value.tobytes())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(canonical_json(value))
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config))


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config["model"] != EXPECTED_MODEL:
        raise GlobalStopError("GLOBAL STOP: model differs from the trusted Wan configuration")
    if config["scheduler"]["name"] != EXPECTED_SCHEDULER or config["scheduler"]["sample_solver"] != "euler":
        raise GlobalStopError("GLOBAL STOP: only Euler is permitted")
    if tuple(config["allowed_modes"]) != MODES or tuple(config["batch_modes"]) != TARGET_MODES:
        raise GlobalStopError("GLOBAL STOP: modes or batch matrix changed")
    if config["batch_modes"] != {"S": 1, "B4": 4, "B8": 8} or int(config["target_batch_index"]) != 0:
        raise GlobalStopError("GLOBAL STOP: batch sizes or frozen target index changed")
    if config["generation"] != {"height": 480, "width": 832, "num_frames": 33, "num_inference_steps": 40, "guidance_scale": 4.0, "fps": 16.0, "boundary_ratio": 0.875}:
        raise GlobalStopError("GLOBAL STOP: generation configuration differs from the trusted setup")
    analysis = config["analysis"]
    if analysis["material_frame_ssim_below"] != 0.99 or analysis["strong_frame_ssim_below"] != 0.95:
        raise GlobalStopError("GLOBAL STOP: registered thresholds changed")
    if int(config["replay"]["total_runs_per_detected_pair"]) != 3:
        raise GlobalStopError("GLOBAL STOP: replay count changed")
    return config


def provenance(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    files = (
        "experiments/video_batch_determinism_killtest.py",
        "experiments/video_batch_determinism_killtest_config.yaml",
        str(config["trusted_v3_config"]), str(config["trusted_prompt_set"]),
        "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
        "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
    )
    hashes = {name: sha256_file(REPO_ROOT / name) for name in files}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
    record = {"config_sha256": sha256_file(config_path), "files": hashes, "git_commit": commit, "git_dirty": bool(dirty), "model": config["model"]}
    return {**record, "provenance_hash": sha256_bytes(canonical_json(record))}


def trusted_targets(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = read_json(REPO_ROOT / config["trusted_v3_config"])
    prompt_rows = read_json(REPO_ROOT / config["trusted_prompt_set"])
    by_id = {row["prompt_id"]: row for row in prompt_rows}
    ids, seeds = source["prompt_ids"], source["generation_seeds"]
    if len(ids) != 12 or len(set(ids)) != 12:
        raise GlobalStopError("GLOBAL STOP: trusted v3 population is not exactly 12 unique prompts")
    targets = []
    for prompt_id in ids:
        row = by_id.get(prompt_id)
        if row is None or prompt_id not in seeds:
            raise GlobalStopError(f"GLOBAL STOP: missing trusted prompt/seed for {prompt_id}")
        targets.append({"prompt_id": prompt_id, "prompt": row["prompt"], "motion_category": row["motion_category"], "generation_seed": int(seeds[prompt_id])})
    return targets


def stable_int(text: str) -> int:
    return int(sha256_bytes(text.encode())[:16], 16)


def filler_manifest(targets: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    ids = [row["prompt_id"] for row in targets]
    seeds = {row["prompt_id"]: row["generation_seed"] for row in targets}
    output: dict[str, Any] = {"mapping_seed": int(config["filler_mapping_seed"]), "targets": {}}
    for target in targets:
        eligible = [prompt_id for prompt_id in ids if prompt_id != target["prompt_id"]]
        rng = random.Random(int(config["filler_mapping_seed"]) ^ stable_int(target["prompt_id"]))
        rng.shuffle(eligible)
        output["targets"][target["prompt_id"]] = {
            "B4": [{"prompt_id": item, "generation_seed": seeds[item]} for item in eligible[:3]],
            "B8": [{"prompt_id": item, "generation_seed": seeds[item]} for item in eligible[:7]],
        }
    return output


def run_order(targets: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    keys = [{"target_id": item["prompt_id"], "batch_mode": mode} for item in targets for mode in TARGET_MODES]
    random.Random(int(config["run_order_seed"])).shuffle(keys)
    return {"seed": int(config["run_order_seed"]), "order": [{**key, "run_order_index": index} for index, key in enumerate(keys)]}


def expected_keys(targets: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(item["prompt_id"], mode) for item in targets for mode in TARGET_MODES]


def make_noise(seed: int, config: dict[str, Any]) -> np.ndarray:
    """Preconstruct request-local noise; this is the only accepted RNG strategy."""
    import torch
    shape = v3.expected_latent_shape({"generation": config["generation"]})
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).numpy()


def anchor_forensics(config: dict[str, Any]) -> dict[str, Any]:
    """Descriptive only: rederive the known single-flip anchor from trusted data."""
    single_cfg = single_flip.load_config(REPO_ROOT / "experiments/video_bf16_single_flip_killtest_config.yaml")
    derived = single_flip.derive_all(single_cfg)
    clean = derived["source"].clean
    flat = 516515
    bits = single_flip.float32_to_bf16_bits(clean).reshape(-1)
    clean_bits = int(bits[flat])
    down_bits = single_flip.adjacent_bf16_bits(clean_bits, "down")
    up_bits = single_flip.adjacent_bf16_bits(clean_bits, "up")
    value = lambda raw: float(single_flip.bf16_bits_to_float32(np.array([raw], dtype=np.uint16))[0])
    fp16_bits = int(single_flip.float32_to_bf16_bits(derived["candidate"]).reshape(-1)[flat])
    return {
        "role": "descriptive_only_not_a_batch_decision_input",
        "anchor": {"prompt_id": "recovery_008", "generation_seed": 9234, "checkpoint_step": 10, "flat_index": flat},
        "persisted_npy_storage_dtype": str(clean.dtype),
        "runtime_semantics_dtype": "torch.bfloat16",
        "clean": {"bits_hex": f"0x{clean_bits:04x}", "value": value(clean_bits)},
        "adjacent_down": {"bits_hex": f"0x{down_bits:04x}", "value": value(down_bits), "distance": abs(value(clean_bits) - value(down_bits))},
        "adjacent_up": {"bits_hex": f"0x{up_bits:04x}", "value": value(up_bits), "distance": abs(value(up_bits) - value(clean_bits))},
        "historical_fp16_reconstructed": {"bits_hex": f"0x{fp16_bits:04x}", "value": value(fp16_bits), "ulp_steps_from_clean": abs(int(fp16_bits) - clean_bits)},
    }


def batch_execution_capability() -> dict[str, Any]:
    """Report whether the checked-in Wan runtime can execute a true request batch.

    This is deliberately source-level: loading the model would be GPU work, and
    the result is used only to stop an invalid experiment before that work.
    """
    pipeline = (REPO_ROOT / "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py").read_text()
    runner = (REPO_ROOT / "vllm_omni/diffusion/worker/diffusion_model_runner.py").read_text()
    implements_step_contract = all(
        marker in pipeline
        for marker in ("supports_step_execution", "def prepare_encode", "def denoise_step", "def step_scheduler", "def post_decode")
    )
    request_mode_forward = "output = self.pipeline.forward(req)" in runner
    return {
        "status": "UNSUPPORTED" if not implements_step_contract and request_mode_forward else "REQUIRES_RUNTIME_AUDIT",
        "pipeline_supports_step_execution_contract": implements_step_contract,
        "runner_uses_single_request_forward": request_mode_forward,
        "reason": (
            "Wan2.2 does not implement the step-execution batch contract and the current runner invokes pipeline.forward(req). "
            "Submitting several requests would not prove B4/B8 co-batched denoising."
            if not implements_step_contract and request_mode_forward
            else "Static source audit was inconclusive; a validated adapter audit is required before GPU execution."
        ),
    }


def gate(name: str, passed: bool, evidence: Any, expected: str, *, status: str | None = None) -> dict[str, Any]:
    return {"name": name, "status": status or ("PASS" if passed else "FAIL"), "passed": bool(passed), "evidence": evidence, "expected": expected}


def write_gates(path: Path, gates: list[dict[str, Any]], prov: dict[str, Any], manifest_hash: str) -> None:
    required = [row for row in gates if row["status"] != "NOT_TESTED"]
    payload = {
        "scope": "cpu_only",
        "gates": gates,
        "all_passed": all(row["status"] == "PASS" for row in required),
        "gpu_required_gates_not_tested": [row["name"] for row in gates if row["status"] == "NOT_TESTED"],
        "provenance_hash": prov["provenance_hash"],
        "manifest_hash": manifest_hash,
    }
    payload["gates_hash"] = sha256_bytes(canonical_json(payload))
    atomic_json(path, payload)
    if not payload["all_passed"]:
        raise GlobalStopError(f"GLOBAL STOP: gate failure in {path}")


def validate_output_dir(path: Path, config: dict[str, Any]) -> None:
    trusted = ("video_runtime_state_discovery", "video_bf16_single_flip", "video_runtime_error_shape", "video_state_protection")
    if any(name in str(path) for name in trusted) or path.name != Path(config["output_namespace"]).name:
        raise GlobalStopError("GLOBAL STOP: output namespace is not the isolated batch-determinism namespace")


def run_cpu(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_dir(output_dir, config)
    prov = provenance(config_path, config)
    targets = trusted_targets(config)
    fillers = filler_manifest(targets, config)
    order = run_order(targets, config)
    keys = expected_keys(targets)
    if len(keys) != 36 or len(set(keys)) != 36:
        raise GlobalStopError("GLOBAL STOP: expected batch target key set is not exactly 36")
    target_manifest = {"targets": targets, "target_batch_index": 0, "generation": config["generation"], "scheduler": config["scheduler"], "target_noise_strategy": "preconstructed_cpu_noise_per_target_seed_reused_by_identity"}
    manifest_hash = sha256_bytes(canonical_json({"target_manifest": target_manifest, "fillers": fillers, "order": order, "keys": keys, "config_hash": config_hash(config)}))
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "target_manifest.json", target_manifest)
    atomic_json(output_dir / "filler_manifest.json", fillers)
    atomic_json(output_dir / "run_order.json", order)
    atomic_json(output_dir / "expected_keys.json", {"keys": [{"target_id": a, "batch_mode": b} for a, b in keys], "count": 36})
    atomic_json(output_dir / "anchor_bf16_forensics.json", anchor_forensics(config))
    capability = batch_execution_capability()
    atomic_json(output_dir / "runtime_batching_capability.json", capability)
    atomic_json(output_dir / "provenance.json", prov)
    gates = [
        gate("G1 trusted source hashes unchanged", True, prov["files"], "all source hashes recorded"),
        gate("G2 exact 12-prompt target manifest", len(targets) == 12, [x["prompt_id"] for x in targets], "12 trusted v3 prompts"),
        gate("G3 exact target seed identity", all(isinstance(x["generation_seed"], int) for x in targets), {x["prompt_id"]: x["generation_seed"] for x in targets}, "trusted v3 seeds"),
        gate("G4 Euler scheduler and frozen scheduler configuration", config["scheduler"]["name"] == EXPECTED_SCHEDULER and config["scheduler"]["sample_solver"] == "euler", config["scheduler"], "Euler"),
        gate("G5 model revision/runtime provenance pinned", prov["model"] == EXPECTED_MODEL, prov, EXPECTED_MODEL),
        gate("G6 target initial noise bit-identical across S/B4/B8", False, "requires persisted target initial-latent artifacts", "runtime execution", status="NOT_TESTED"),
        gate("G7 target token IDs/conditioning identity unchanged", False, "requires runtime tokenizer audit", "runtime execution", status="NOT_TESTED"),
        gate("G8 target batch index frozen", int(config["target_batch_index"]) == 0, 0, "batch index 0"),
        gate("G9 filler composition matches frozen manifest", all(len(fillers["targets"][x["prompt_id"]]["B4"]) == 3 and len(fillers["targets"][x["prompt_id"]]["B8"]) == 7 for x in targets), fillers, "B4=3, B8=7; target excluded"),
        gate("G10 filler seeds match frozen manifest", True, fillers, "seeds copied from trusted targets"),
        gate("G11 exact expected 36 target-condition key set", len(keys) == 36 and len(set(keys)) == 36, keys, "36 unique keys"),
        gate("G12 warmup completed for batch 1/4/8", False, "requires runtime execution", "runtime execution", status="NOT_TESTED"),
        gate("G13 primary target extraction correct", False, "requires target-only output extraction", "runtime execution", status="NOT_TESTED"),
        gate("G14 final target artifacts exist/hash validate", False, "requires persisted target artifacts", "runtime execution", status="NOT_TESTED"),
        gate("G15 finite SSIM/MSE", False, "requires recovered outputs", "runtime execution", status="NOT_TESTED"),
        gate("G16 replay input/filler composition identical", False, "requires replay rows", "runtime execution", status="NOT_TESTED"),
        gate("G17 replay determinism before counting differences", False, "requires replay artifacts", "runtime execution", status="NOT_TESTED"),
        gate("G18 no metric other than registered fields influences decision", set(config["analysis"]["decision_input_fields"]) == {"final_latent_identity", "video_identity", "frame_ssim_mean", "video_mse", "final_latent_mse", "replay_input_identity"}, config["analysis"]["decision_input_fields"], "registered fields only"),
        gate("G19 no automatic B16 / TP / backend expansion", tuple(config["batch_modes"]) == TARGET_MODES, config["batch_modes"], "S/B4/B8 only"),
        gate("G20 provenance/config/manifests hash-bound", bool(manifest_hash), manifest_hash, "nonempty frozen manifest hash"),
        gate("G21 run order matches frozen randomized order", len(order["order"]) == 36 and { (x["target_id"], x["batch_mode"]) for x in order["order"] } == set(keys), order, "permutation of exact keys"),
        gate("G22 no source/trusted namespace mutation", True, "new output namespace only", "no trusted output path"),
        gate("GPU execution blocked without true co-batch capability", capability["status"] != "SUPPORTED", capability, "fail closed before GPU work"),
    ]
    write_gates(output_dir / "cpu_gates.json", gates, prov, manifest_hash)
    return {"mode": "cpu", "all_passed": True, "targets": targets, "primary_rows": 36, "manifest_hash": manifest_hash}


def _load_cpu(output_dir: Path, config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    prov = provenance(config_path, config)
    required = ["target_manifest.json", "filler_manifest.json", "run_order.json", "expected_keys.json", "cpu_gates.json", "provenance.json"]
    if any(not (output_dir / name).exists() for name in required):
        raise GlobalStopError("GLOBAL STOP: CPU manifest artifacts are missing")
    if read_json(output_dir / "provenance.json")["provenance_hash"] != prov["provenance_hash"]:
        raise GlobalStopError("GLOBAL STOP: CPU provenance is stale")
    target, fillers, order = read_json(output_dir / "target_manifest.json"), read_json(output_dir / "filler_manifest.json"), read_json(output_dir / "run_order.json")
    keys = read_json(output_dir / "expected_keys.json")["keys"]
    manifest_hash = sha256_bytes(canonical_json({"target_manifest": target, "fillers": fillers, "order": order, "keys": [(x["target_id"], x["batch_mode"]) for x in keys], "config_hash": config_hash(config)}))
    gates = read_json(output_dir / "cpu_gates.json")
    if not gates.get("all_passed") or gates.get("provenance_hash") != prov["provenance_hash"] or gates.get("manifest_hash") != manifest_hash:
        raise GlobalStopError("GLOBAL STOP: CPU gates fail closed")
    return target, fillers, order, manifest_hash


def expected_filler_rows(target_id: str, mode: str, fillers: dict[str, Any]) -> list[dict[str, Any]]:
    return [] if mode == "S" else list(fillers["targets"][target_id][mode])


def validate_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    target_manifest: dict[str, Any],
    fillers: dict[str, Any],
    order: dict[str, Any],
    *,
    token_identities: dict[str, dict[str, str]],
    runtime_identity: dict[str, str],
    allow_replays: bool = True,
) -> None:
    targets = {x["prompt_id"]: x for x in target_manifest["targets"]}
    expected = {(x["target_id"], x["batch_mode"]) for x in order["order"]}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["target_id"], row["batch_mode"]); groups[key].append(row)
        if key not in expected: raise GlobalStopError(f"GLOBAL STOP: unexpected key {key}")
        target = targets.get(row["target_id"])
        if target is None or int(row["target_seed"]) != int(target["generation_seed"]): raise GlobalStopError("GLOBAL STOP: target seed identity mismatch")
        if int(row["effective_batch_size"]) != int(config["batch_modes"][row["batch_mode"]]): raise GlobalStopError("GLOBAL STOP: wrong batch size")
        if int(row["target_batch_index"]) != 0: raise GlobalStopError("GLOBAL STOP: target batch index changed")
        expected_fillers = expected_filler_rows(row["target_id"], row["batch_mode"], fillers)
        if json.loads(row["filler_ids_json"]) != [x["prompt_id"] for x in expected_fillers]: raise GlobalStopError("GLOBAL STOP: filler composition mismatch")
        if json.loads(row["filler_seeds_json"]) != [x["generation_seed"] for x in expected_fillers]: raise GlobalStopError("GLOBAL STOP: filler seed mismatch")
        expected_index = next(item["run_order_index"] for item in order["order"] if item["target_id"] == row["target_id"] and item["batch_mode"] == row["batch_mode"])
        if int(row["run_order_index"]) != int(expected_index): raise GlobalStopError("GLOBAL STOP: run order differs from frozen manifest")
        expected_noise = tensor_identity(make_noise(target["generation_seed"], config))
        if row["target_initial_noise_identity"] != expected_noise or row["target_initial_latent_identity"] != expected_noise: raise GlobalStopError("GLOBAL STOP: target realized initial noise differs")
        if row["scheduler_class"] != EXPECTED_SCHEDULER_CLASS or row["model"] != EXPECTED_MODEL: raise GlobalStopError("GLOBAL STOP: scheduler/model mismatch")
        token_identity = token_identities.get(row["target_id"])
        if token_identity is None or row["target_token_ids_hash"] != token_identity["token_ids_hash"] or row["target_attention_mask_hash"] != token_identity["attention_mask_hash"]:
            raise GlobalStopError("GLOBAL STOP: target token/conditioning identity mismatch")
        for field in ("model_revision", "runtime_dtype", "backend", "compile_state", "scheduler_config_hash"):
            if row[field] != runtime_identity[field]:
                raise GlobalStopError(f"GLOBAL STOP: runtime identity mismatch for {field}")
        for metric in ("frame_ssim_mean", "video_mse", "final_latent_mse"):
            if not math.isfinite(float(row[metric])): raise GlobalStopError("GLOBAL STOP: non-finite registered metric")
    if set(groups) != expected: raise GlobalStopError(f"GLOBAL STOP: incomplete target-condition matrix missing={sorted(expected-set(groups))}")
    for key, group in groups.items():
        ids = sorted(int(x["replay_id"]) for x in group)
        if ids[0] != 0 or ids != list(range(len(ids))): raise GlobalStopError("GLOBAL STOP: replay ids are invalid")
        if len(group) > 1 and len(group) != 3: raise GlobalStopError("GLOBAL STOP: a replay block is not exactly three executions")
        if not allow_replays and len(group) != 1: raise GlobalStopError("GLOBAL STOP: unexpected replay in primary screen")


def classify_target_pair(rows: list[dict[str, Any]], material: float) -> str:
    primary = next(row for row in rows if int(row["replay_id"]) == 0)
    exact = primary["final_latent_exact_vs_solo"] and primary["video_exact_vs_solo"]
    if exact:
        return "BIT_EXACT"
    if float(primary["frame_ssim_mean"]) >= material:
        return "NUMERICALLY_DIFFERENT"
    equality = ("target_initial_noise_identity", "filler_ids_json", "filler_seeds_json", "final_latent_identity", "video_identity", "frame_ssim_mean", "video_mse", "final_latent_mse")
    deterministic = len(rows) == 3 and all(len({str(x[field]) for x in rows}) == 1 for field in equality)
    return "MATERIALLY_DIFFERENT_DETERMINISTIC" if deterministic else "MATERIALLY_DIFFERENT_UNSTABLE"


def analyze(rows: list[dict[str, Any]], config: dict[str, Any], *, gates_passed: bool) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[(row["target_id"], row["batch_mode"])].append(row)
    material = float(config["analysis"]["material_frame_ssim_below"])
    summary = []
    affected = set(); replay_stable = set(); numerical = False
    for key in sorted(groups):
        target_id, mode = key
        if mode == "S": continue
        label = classify_target_pair(groups[key], material)
        primary = next(x for x in groups[key] if int(x["replay_id"]) == 0)
        summary.append({"target_id": target_id, "batch_mode": mode, "classification": label, "frame_ssim_mean": float(primary["frame_ssim_mean"]), "strong_difference": float(primary["frame_ssim_mean"]) < float(config["analysis"]["strong_frame_ssim_below"])})
        if label.startswith("MATERIALLY_DIFFERENT"): affected.add(target_id)
        if label == "MATERIALLY_DIFFERENT_DETERMINISTIC": replay_stable.add(target_id)
        if label == "NUMERICALLY_DIFFERENT": numerical = True
    all_exact = all(x["classification"] == "BIT_EXACT" for x in summary)
    # Breadth is defined over deterministic, materially different targets. A
    # non-reproducible first observation cannot contribute to the GO count.
    if (
        gates_passed
        and len(replay_stable) >= int(config["analysis"]["go_min_material_targets"])
        and len(replay_stable) >= int(config["analysis"]["go_min_replay_stable_material_targets"])
    ):
        decision = "GO_SYSTEMS_RELEVANCE"
    elif gates_passed and all_exact:
        decision = "NO_GO"
    elif gates_passed and numerical and not affected:
        decision = "WEAK_NUMERICAL_ONLY"
    else:
        decision = "WEAK_INCONCLUSIVE"
    return {"decision": decision, "affected_targets": sorted(affected), "replay_stable_material_targets": sorted(replay_stable), "pair_summary": summary, "decision_inputs": config["analysis"]["decision_input_fields"]}


def gpu_unavailable(mode: str) -> None:
    raise GlobalStopError(
        "GLOBAL STOP: no validated true co-batch execution path is available for Wan2.2 in this checkout. "
        f"Refusing {mode}: Wan2.2 lacks the step-execution batch contract and the runner invokes pipeline.forward(req), "
        "so several submitted requests would not establish actual B4/B8 co-batched denoising."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--config", type=Path, default=Path("experiments/video_batch_determinism_killtest_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/video_batch_determinism_killtest"))
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "cpu":
        print(json.dumps(run_cpu(config, args.config, args.output_dir), indent=2, sort_keys=True))
        return
    _load_cpu(args.output_dir, config, args.config)
    gpu_unavailable(args.mode)


if __name__ == "__main__":
    main()
