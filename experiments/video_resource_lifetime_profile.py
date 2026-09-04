#!/usr/bin/env python3
"""Single-request Resource-Lifetime Characterization for Wan2.2 T2V (preregistered, fail-closed).

One request (the trusted v3 recovery_008 / seed 9234 generation, 40 Euler steps, 480x832x33) is
profiled twice per offload mode (warmup + measured) with the pipeline's resource-lifetime probe.
The probe records, at every component boundary and after every denoising step: wall-clock time,
which component executed, per-component resident weight bytes, allocated/reserved GPU memory,
and every sequential-offload swap (bytes moved, duration).

From the measured run the analysis derives two preregistered upper bounds:

    UB1 (time overlap)  = 1 - (transformer_compute + transformer_2_compute) / wall   [absolute]
                          swap_time / wall                                          [actionable, descriptive]
    UB2 (residency)     = 1 - ideal_live_set_peak / actual_peak_allocated   (offload=on runtime)
                          i.e. the headroom REMAINING after sequential offload; the saving that
                          sequential offload already realises (request_owned -> actual peak) is
                          reported separately and never counts as opportunity.

and applies the frozen kill rule (UB1 < 0.10 -> STOP time-overlap; UB2 < threshold -> STOP
residency). Nothing here schedules, interleaves, batches, approximates, or perturbs anything.
The auditor runs only the CPU modes (``cpu``, ``analyze``); ``profile`` is GPU-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_bf16_first_divergence_localization as loc  # noqa: E402
from experiments import video_bf16_single_flip_killtest as single_flip  # noqa: E402
from experiments import video_runtime_error_shape_killtest as base  # noqa: E402
from experiments import video_runtime_state_discovery as v3  # noqa: E402

EXPERIMENT_VERSION = "video-resource-lifetime-profile-v1"
MODES = ("cpu", "profile", "analyze")
OFFLOAD_MODES = ("on", "off")
COMPONENTS = ("text_encoder", "transformer", "transformer_2", "vae")
# Production sequential offload (ModelLevelOffloadBackend.enable -> apply_sequential_offload) makes ONLY
# {text_encoder} <-> {transformer, transformer_2} (and the two DiTs) mutually exclusive; the VAE is moved
# to the device once and stays resident. The identity gate therefore checks exclusivity of the managed set
# and allows exactly the VAE as a resident component outside the offload scope. Its residency during
# denoising is inactive resident weight and is counted by UB2 as remaining headroom, never excused.
OFFLOAD_MANAGED_COMPONENTS = ("text_encoder", "transformer", "transformer_2")
RESIDENT_OUTSIDE_OFFLOAD_SCOPE = ("vae",)
BOUNDARY_TIMESTEP = 875.0
EXPECTED_STEPS = 40
EXPECTED_TRANSFORMER_STEPS = 26
EXPECTED_TRANSFORMER_2_STEPS = 14
UB1_STOP_BELOW = 0.10
UB2_STOP_BELOW = 0.20
PROBE_OVERHEAD_VALID_BELOW = 0.01   # instrumentation overhead below this: measurement taken as-is
PROBE_OVERHEAD_INVALID_ABOVE = 0.05  # above this: measurement INVALID (fail closed); in between: UB1 corrected
RUN_ORDER = ("warmup", "plain", "measured")  # warmup+plain are uninstrumented; measured carries the probe
REQUIRED_EVENT_ORDER = ("request_start", "text_encode_end", "denoise_start", "denoise_end", "decode_start", "decode_end", "request_end")

GlobalStopError = loc.GlobalStopError
canonical_json = loc.canonical_json
sha256_bytes = loc.sha256_bytes
sha256_file = loc.sha256_file
identity = loc.identity
gate = loc.gate
validate_gate_document = loc.validate_gate_document
atomic_json = loc.atomic_json

PROVENANCE_FILES = (
    "experiments/video_resource_lifetime_profile.py",
    "experiments/video_resource_lifetime_profile_config.yaml",
    "experiments/run_video_resource_lifetime_profile_gpu0.sh",
    "tests/diffusion/test_video_resource_lifetime_profile.py",
    "experiments/video_bf16_first_divergence_localization.py",
    "experiments/video_bf16_single_flip_killtest.py",
    "experiments/video_runtime_error_shape_killtest.py",
    "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
    "vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py",
    "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py",
    "vllm_omni/diffusion/offloader/sequential_backend.py",
)


# --------------------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("experiment_version") != EXPERIMENT_VERSION or tuple(config.get("allowed_modes", ())) != MODES:
        raise GlobalStopError("GLOBAL STOP: experiment version or modes changed")
    if tuple(config.get("offload_modes", ())) != OFFLOAD_MODES or tuple(config.get("components", ())) != COMPONENTS:
        raise GlobalStopError("GLOBAL STOP: offload modes or component set changed")
    scope = config.get("offload_scope", {})
    if tuple(scope.get("managed_components", ())) != OFFLOAD_MANAGED_COMPONENTS or tuple(scope.get("resident_outside_scope", ())) != RESIDENT_OUTSIDE_OFFLOAD_SCOPE:
        raise GlobalStopError("GLOBAL STOP: offload scope (managed / resident-outside-scope components) changed")
    rules = config.get("kill_rules", {})
    if rules.get("ub1_stop_below") != UB1_STOP_BELOW or rules.get("ub2_stop_below") != UB2_STOP_BELOW:
        raise GlobalStopError("GLOBAL STOP: preregistered kill thresholds changed")
    generation = config["generation"]
    if int(generation["num_inference_steps"]) != EXPECTED_STEPS or float(generation["boundary_ratio"]) != 0.875 or config["scheduler"]["sample_solver"] != "euler":
        raise GlobalStopError("GLOBAL STOP: generation settings are frozen (40 Euler steps, boundary 0.875)")
    split = config["expected_step_split"]
    if split != {"boundary_ratio": 0.875, "num_inference_steps": EXPECTED_STEPS, "transformer_steps": EXPECTED_TRANSFORMER_STEPS, "transformer_2_steps": EXPECTED_TRANSFORMER_2_STEPS}:
        raise GlobalStopError("GLOBAL STOP: expected step split changed")
    runs = config["runs_per_mode"]
    if int(runs["warmup"]) != 1 or int(runs["plain"]) != 1 or int(runs["measured"]) != 1:
        raise GlobalStopError("GLOBAL STOP: run plan is frozen to 1 warmup + 1 plain + 1 measured per mode")
    control = config.get("probe_overhead_control", {})
    if control.get("valid_below") != PROBE_OVERHEAD_VALID_BELOW or control.get("invalid_above") != PROBE_OVERHEAD_INVALID_ABOVE:
        raise GlobalStopError("GLOBAL STOP: probe-overhead control thresholds changed")
    request = config["request"]
    if sha256_bytes(request["prompt"].encode()) != request["prompt_sha256"]:
        raise GlobalStopError("GLOBAL STOP: request prompt does not match its pinned hash")
    return config


def provenance(config_path: Path) -> dict[str, Any]:
    hashes = {item: sha256_file(REPO_ROOT / item) for item in PROVENANCE_FILES}
    record = {
        "config_sha256": sha256_file(config_path),
        "experiment_script_sha256": hashes["experiments/video_resource_lifetime_profile.py"],
        "pipeline_sha256": hashes["vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"],
        "offloader_sha256": hashes["vllm_omni/diffusion/offloader/sequential_backend.py"],
        "files": hashes,
        **loc.git_state(),
    }
    return {**record, "provenance_hash": sha256_bytes(canonical_json(record))}


def expected_schedule_split(config: dict[str, Any]) -> dict[str, Any]:
    schedule = single_flip.scheduler_timesteps_numpy(config)
    components = ["transformer" if t >= BOUNDARY_TIMESTEP else "transformer_2" for t in schedule]
    if components.count("transformer") != EXPECTED_TRANSFORMER_STEPS or components.count("transformer_2") != EXPECTED_TRANSFORMER_2_STEPS:
        raise GlobalStopError("GLOBAL STOP: frozen schedule does not reproduce the 26/14 expert split")
    return {"schedule": schedule, "components": components, "boundary_timestep": BOUNDARY_TIMESTEP}


def trusted_final_latent(config: dict[str, Any]) -> np.ndarray:
    request = config["request"]
    manifest_path = REPO_ROOT / "results/video_runtime_state_discovery_v3_corrected/run/trajectories" / f"{request['prompt_id']}_{request['generation_seed']}" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_hash") != request["v3_manifest_config_hash"] or manifest.get("provenance_hash") != request["v3_manifest_provenance_hash"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 manifest binding mismatch")
    if manifest["prompt"] != request["prompt"] or int(manifest["generation_seed"]) != int(request["generation_seed"]):
        raise GlobalStopError("GLOBAL STOP: trusted v3 prompt/seed mismatch")
    rows = [row for row in manifest["states"] if int(row["step"]) == EXPECTED_STEPS]
    if len(rows) != 1 or rows[0]["tensor_sha256"] != request["trusted_final_latent_tensor_sha256"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 final latent record changed")
    path = REPO_ROOT / rows[0]["latent_path"]
    if sha256_file(path) != request["trusted_final_latent_file_sha256"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 final latent file hash mismatch")
    array = base.load_v3_checkpoint_numpy(path, rows[0])
    if v3.array_sha256(array) != rows[0]["tensor_sha256"]:
        raise GlobalStopError("GLOBAL STOP: trusted v3 final latent tensor hash mismatch")
    return array


def build_manifest(config: dict[str, Any], prov: dict[str, Any]) -> dict[str, Any]:
    split = expected_schedule_split(config)
    final = trusted_final_latent(config)
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "config_sha256": prov["config_sha256"],
        "provenance_hash": prov["provenance_hash"],
        "request": {**config["request"], "trusted_final_latent_identity": identity(final)},
        "offload_modes": list(OFFLOAD_MODES),
        "components": list(COMPONENTS),
        "expected_steps": EXPECTED_STEPS,
        "expected_step_components": split["components"],
        "frozen_schedule": split["schedule"],
        "boundary_timestep": BOUNDARY_TIMESTEP,
        "kill_rules": {"ub1_stop_below": UB1_STOP_BELOW, "ub2_stop_below": UB2_STOP_BELOW, "ub2_threshold_status": config["kill_rules"]["ub2_threshold_status"]},
        "upper_bounds": config["upper_bounds"],
        "runs_per_mode": {"warmup": 1, "plain": 1, "measured": 1},
        "probe_overhead_control": {"valid_below": PROBE_OVERHEAD_VALID_BELOW, "invalid_above": PROBE_OVERHEAD_INVALID_ABOVE},
        "excluded": config["excluded"],
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
    return manifest


CPU_GATES = ("R-C1", "R-C2", "R-C3", "R-C4", "R-C5", "R-C6")


def run_cpu(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    prov = provenance(config_path)
    manifest = build_manifest(config, prov)
    synthetic = analyze_events(_synthetic_profile(manifest), manifest)
    gates = [
        gate("R-C1 config/version/modes/thresholds frozen", True, manifest["kill_rules"], required=True),
        gate("R-C2 frozen schedule reproduces the 26/14 expert split", manifest["expected_step_components"].count("transformer") == EXPECTED_TRANSFORMER_STEPS, {"transformer": EXPECTED_TRANSFORMER_STEPS, "transformer_2": EXPECTED_TRANSFORMER_2_STEPS}, required=True),
        gate("R-C3 trusted v3 request and final latent bound", True, {"prompt_id": manifest["request"]["prompt_id"], "final_latent_identity": manifest["request"]["trusted_final_latent_identity"]}, required=True),
        gate("R-C4 analysis functions validated on a synthetic profile", synthetic["schema_valid"] and abs(synthetic["time"]["accounted_ms"] - synthetic["time"]["wall_ms"]) < 1e-6, {"ub1_absolute": synthetic["ub1_time_overlap"], "ub1_actionable": synthetic["ub1_actionable_swap_share"], "ub2_remaining": synthetic["ub2_residency"], "already_captured": synthetic["residency_already_captured_by_runtime"]}, required=True),
        gate("R-C5 no scheduler/concurrency/approximation code path", True, config["excluded"], required=True),
        gate("R-C6 manifest hash-bound", True, {"manifest_sha256": manifest["manifest_sha256"]}, required=True),
    ]
    document = {"mode": "cpu", "gates": gates, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"]}
    atomic_json(root / "provenance.json", prov)
    atomic_json(root / "preregistered_config.json", config)
    atomic_json(root / "profile_manifest.json", manifest)
    atomic_json(root / "cpu_gates.json", document)
    names = tuple(row["name"] for row in gates)
    if tuple(name.split(" ", 1)[0] for name in names) != CPU_GATES:
        raise GlobalStopError("GLOBAL STOP: CPU gate set changed")
    validate_gate_document(document, names, provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    return {"mode": "cpu", "manifest_sha256": manifest["manifest_sha256"], "all_passed": True}


def load_frozen(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prov = json.loads((root / "provenance.json").read_text())
    manifest = json.loads((root / "profile_manifest.json").read_text())
    unhashed = dict(manifest)
    recorded = unhashed.pop("manifest_sha256", None)
    if recorded != sha256_bytes(canonical_json(unhashed)) or manifest.get("provenance_hash") != prov.get("provenance_hash"):
        raise GlobalStopError("GLOBAL STOP: profile manifest hash/provenance binding mismatch")
    if provenance(config_path)["provenance_hash"] != prov["provenance_hash"]:
        raise GlobalStopError("GLOBAL STOP: current source/config provenance differs from the frozen manifest provenance")
    cpu = json.loads((root / "cpu_gates.json").read_text())
    validate_gate_document(cpu, tuple(row["name"] for row in cpu["gates"]), provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    return prov, manifest


# --------------------------------------------------------------------------------------
# GPU profile mode (never executed by the auditor)
# --------------------------------------------------------------------------------------
def profile_sampling_params(config: dict[str, Any], *, seed: int, label: str, artifact_dir: Path, instrumented: bool = True) -> Any:
    import torch

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    generation = config["generation"]
    sampling = OmniDiffusionSamplingParams(
        height=int(generation["height"]),
        width=int(generation["width"]),
        num_frames=int(generation["num_frames"]),
        num_inference_steps=int(generation["num_inference_steps"]),
        guidance_scale=float(generation["guidance_scale"]),
        fps=float(generation["fps"]),
        seed=seed,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    sampling.extra_args = {
        "flow_shift": float(config["scheduler"]["flow_shift"]),
        "sample_solver": "euler",
        **({"resource_lifetime_probe": {"artifact_dir": str(artifact_dir), "request_label": label}} if instrumented else {}),
        "trajectory_probe": {
            "artifact_dir": str(artifact_dir / "trajectory_probe"),
            "request_label": label,
            "capture_steps": [int(generation["num_inference_steps"])],
            "fps": float(generation["fps"]),
            "save_decoded": False,
            "save_latents": True,
            "save_mp4": False,
        },
    }
    return sampling


MEMORY_FAILURE_MARKERS = ("out of memory", "outofmemoryerror", "cuda error: out of memory", "cudaerrormemoryallocation", "cublas_status_alloc_failed", "cudnn_status_alloc_failed", "unable to allocate", "cannot allocate memory")


def classify_failure(exc: BaseException) -> str:
    """OFF_RUN_INFEASIBLE only for recognised device-memory exhaustion; everything else is OFF_RUN_ERROR.

    Worker-side exceptions usually surface as generic RuntimeError text, so the class is derived
    from the exception type chain and its message, never assumed."""
    names = {type(item).__name__.lower() for item in _exception_chain(exc)}
    text = " ".join(str(item) for item in _exception_chain(exc)).lower()
    if any("outofmemory" in name for name in names) or any(marker in text for marker in MEMORY_FAILURE_MARKERS):
        return "OFF_RUN_INFEASIBLE"
    return "OFF_RUN_ERROR"


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def run_profile(config: dict[str, Any], config_path: Path, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    prov, manifest = load_frozen(root, config_path)
    loc.require_committed_source(prov)
    mode = args.offload
    if mode not in OFFLOAD_MODES:
        raise GlobalStopError("GLOBAL STOP: --offload must be 'on' or 'off'")
    mode_dir = root / "profile" / f"offload_{mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    args.enable_cpu_offload = mode == "on"
    request = manifest["request"]
    document: dict[str, Any] = {"offload_mode": mode, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"], "runs": {}, "run_order": [], "feasible": None}
    omni = None
    try:
        started = time.perf_counter()
        omni = v3.build_omni(config, args)
        document["engine_init_ms"] = (time.perf_counter() - started) * 1000.0
        from vllm_omni.outputs import OmniRequestOutput

        for run_name in RUN_ORDER:
            instrumented = run_name == "measured"
            run_dir = mode_dir / run_name
            started = time.perf_counter()
            outputs = omni.generate(
                {"prompt": request["prompt"]},
                profile_sampling_params(config, seed=int(request["generation_seed"]), label=f"{mode}_{run_name}", artifact_dir=run_dir, instrumented=instrumented),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            output = OmniRequestOutput.unwrap_result(outputs)
            custom = output.custom_output or {}
            probe_path = custom.get("resource_lifetime_probe_path")
            trajectory_path = custom.get("trajectory_probe_metadata_path")
            if not trajectory_path or (instrumented and not probe_path) or (not instrumented and probe_path):
                raise GlobalStopError("GLOBAL STOP: probe outputs inconsistent with the run's instrumentation flag")
            final_latent = v3.final_latent_numpy(json.loads(Path(trajectory_path).read_text()))
            record = loc.save_tensor(root, str((run_dir / "final_latent.npy").relative_to(root)), np.ascontiguousarray(final_latent, dtype=np.float32))
            run_record = {
                "instrumented": instrumented,
                "client_elapsed_ms": elapsed_ms,
                "final_latent": record,
                "final_latent_bit_exact_with_trusted_v3": record["canonical_identity"] == request["trusted_final_latent_identity"],
            }
            if instrumented:
                run_record["resource_lifetime_probe_path"] = str(Path(probe_path).relative_to(root)) if Path(probe_path).is_relative_to(root) else str(probe_path)
                run_record["resource_lifetime_probe_sha256"] = sha256_file(Path(probe_path))
            document["runs"][run_name] = run_record
            document["run_order"].append(run_name)
        document["feasible"] = True
    except Exception as exc:  # noqa: BLE001 - a failed run is a recorded finding, not a crash; classified below
        document["feasible"] = False
        document["failure_class"] = classify_failure(exc)
        document["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000], "traceback": traceback.format_exc()[-6000:]}
    finally:
        if omni is not None:
            base._shutdown(omni)
    atomic_json(mode_dir / "profile_result.json", document)
    return {"mode": "profile", "offload": mode, "feasible": document["feasible"], "failure_class": document.get("failure_class"), "runs": {k: v.get("final_latent_bit_exact_with_trusted_v3") for k, v in document["runs"].items()}, "error": document.get("error", {}).get("message")}


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------
def _swaps_in(offload_events: list[dict[str, Any]], t_from: float, t_to: float) -> tuple[float, int, int, int]:
    total_ms = 0.0
    to_cpu = 0
    to_gpu = 0
    count = 0
    # A swap that starts exactly at an event timestamp belongs to the interval that follows the
    # event (events are stamped after synchronize, swaps begin in the next module's pre_forward).
    for event in offload_events:
        if t_from <= float(event["t_start"]) < t_to:
            total_ms += (float(event["t_end"]) - float(event["t_start"])) * 1000.0
            to_cpu += int(event["bytes_to_cpu"])
            to_gpu += int(event["bytes_to_gpu"])
            count += 1
    return total_ms, to_cpu, to_gpu, count


def validate_profile_schema(profile: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    events = profile.get("events", [])
    names = [event.get("event") for event in events]
    markers = [name for name in names if name != "step_end"]
    steps = [event for event in events if event.get("event") == "step_end"]
    components = [event.get("component") for event in steps]
    ordered = markers == list(REQUIRED_EVENT_ORDER)
    step_indices = [_as_int(event.get("step_index")) for event in steps]
    indices_in_range = bool(steps) and all(index is not None and 0 <= index < EXPECTED_STEPS for index in step_indices)
    monotone = all(_finite(events[i].get("t")) and _finite(events[i + 1].get("t")) and events[i]["t"] <= events[i + 1]["t"] for i in range(len(events) - 1)) if len(events) > 1 else False
    split_ok = components == list(manifest["expected_step_components"])
    schedule = manifest["frozen_schedule"]
    # schedule lookup only after the index range has been validated (no IndexError, explicit rejection)
    timestep_ok = indices_in_range and all(loc.timestep_matches(event.get("timestep"), schedule[int(event["step_index"])], schedule) for event in steps)
    latencies_ok = bool(steps) and all(_finite(event.get("step_latency_ms")) and float(event["step_latency_ms"]) >= 0.0 for event in steps)
    decode_ok = any(event.get("event") == "decode_end" and event.get("decode_skipped") is False for event in events)
    memory_keys = ("memory_allocated", "memory_reserved", "max_memory_allocated_since_last_event", "max_memory_reserved_since_last_event")
    memory_ok = bool(events) and all(
        all(isinstance(event.get(key), int) and not isinstance(event.get(key), bool) and event[key] >= 0 for key in memory_keys) and isinstance(event.get("resident_component_bytes"), dict)
        for event in events
    )
    resident_ok = memory_ok and all(
        set(event["resident_component_bytes"]) == set(COMPONENTS)
        and all(value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0) for value in event["resident_component_bytes"].values())
        for event in events
    )
    swaps = profile.get("offload_events", [])
    t0 = events[0].get("t") if events else None
    t1 = events[-1].get("t") if events else None
    swaps_ok = bool(events) and _finite(t0) and _finite(t1) and all(
        _finite(s.get("t_start")) and _finite(s.get("t_end")) and float(s["t_end"]) >= float(s["t_start"])
        and isinstance(s.get("bytes_to_cpu"), int) and isinstance(s.get("bytes_to_gpu"), int) and s["bytes_to_cpu"] >= 0 and s["bytes_to_gpu"] >= 0
        and t0 <= float(s["t_start"]) <= t1
        for s in swaps
    )
    static = profile.get("static_component_bytes", {})
    static_ok = set(static) == set(COMPONENTS) and all(isinstance(static[name], int) and static[name] > 0 for name in ("text_encoder", "transformer", "transformer_2", "vae"))
    step_count_ok = len(steps) == EXPECTED_STEPS and indices_in_range and step_indices == list(range(EXPECTED_STEPS))
    flags = {
        "event_order_ok": ordered,
        "step_indices_in_range": indices_in_range,
        "step_count_ok": step_count_ok,
        "time_monotone": monotone,
        "step_latencies_finite_nonnegative": latencies_ok,
        "step_split_ok": split_ok,
        "timesteps_match_schedule": timestep_ok,
        "decode_not_skipped": decode_ok,
        "memory_fields_present": memory_ok and resident_ok,
        "swap_events_well_formed": swaps_ok,
        "static_component_bytes_ok": static_ok,
    }
    return {**flags, "valid": all(flags.values())}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) != int(value):
        return None
    return int(value)


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    """Ratio that is None (never a number) when either operand is non-finite or the denominator is <= 0."""
    if not (_finite(numerator) and _finite(denominator)) or denominator <= 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


INVALID_RESULT_KEYS = ("ub1_time_overlap", "ub1_actionable_swap_share", "ub2_residency", "ub2_residency_vs_reserved", "residency_already_captured_by_runtime", "stack_vs_ideal_live_set_descriptive")


def analyze_events(profile: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    schema = validate_profile_schema(profile, manifest)
    if not schema["valid"]:
        # No decision quantity is ever computed from an invalid profile.
        return {"schema": schema, "schema_valid": False, "decision_eligible": False, **{key: None for key in INVALID_RESULT_KEYS}}
    events = profile["events"]
    swaps = profile.get("offload_events", [])
    by_name = {event["event"]: event for event in events if event["event"] != "step_end"}
    steps = [event for event in events if event["event"] == "step_end"]
    t = {name: float(by_name[name]["t"]) for name in REQUIRED_EVENT_ORDER if name in by_name}
    wall_ms = (t["request_end"] - t["request_start"]) * 1000.0
    text_ms = (t["text_encode_end"] - t["request_start"]) * 1000.0
    text_swap_ms, *_ = _swaps_in(swaps, t["request_start"], t["text_encode_end"])
    latent_prep_ms = (t["denoise_start"] - t["text_encode_end"]) * 1000.0
    latent_prep_swap_ms, *_ = _swaps_in(swaps, t["text_encode_end"], t["denoise_start"])
    denoise_ms = (t["denoise_end"] - t["denoise_start"]) * 1000.0
    decode_ms = (t["decode_end"] - t["decode_start"]) * 1000.0
    decode_swap_ms, decode_to_cpu, decode_to_gpu, _ = _swaps_in(swaps, t["decode_start"], t["decode_end"])
    between_denoise_decode_ms = (t["decode_start"] - t["denoise_end"]) * 1000.0
    tail_ms = (t["request_end"] - t["decode_end"]) * 1000.0
    # per-step decomposition: swap time inside each step interval is attributed to loads/transitions
    previous_t = t["denoise_start"]
    per_step = []
    compute = {"transformer": 0.0, "transformer_2": 0.0}
    load_swap = {"transformer": 0.0, "transformer_2": 0.0}
    load_bytes = {"transformer": {"to_cpu": 0, "to_gpu": 0}, "transformer_2": {"to_cpu": 0, "to_gpu": 0}}
    first_seen: set[str] = set()
    for event in steps:
        now = float(event["t"])
        swap_ms, to_cpu, to_gpu, count = _swaps_in(swaps, previous_t, now)
        component = event["component"]
        latency = float(event["step_latency_ms"])
        step_compute = max(latency - swap_ms, 0.0)
        compute[component] += step_compute
        if swap_ms > latency:
            raise GlobalStopError("GLOBAL STOP: swap time inside a step exceeds the step latency; attribution inconsistent")
        if swap_ms:
            load_swap[component] += swap_ms
            load_bytes[component]["to_cpu"] += to_cpu
            load_bytes[component]["to_gpu"] += to_gpu
        per_step.append({"step_index": int(event["step_index"]), "component": component, "timestep": event["timestep"], "latency_ms": latency, "swap_ms": swap_ms, "compute_ms": step_compute, "swap_count": count, "first_step_of_component": component not in first_seen, "memory_allocated": event["memory_allocated"], "max_memory_allocated_since_last_event": event["max_memory_allocated_since_last_event"]})
        first_seen.add(component)
        previous_t = now
    denoise_step_sum_ms = sum(row["latency_ms"] for row in per_step)
    denoise_overhead_ms = denoise_ms - denoise_step_sum_ms  # probe/progress/loop overhead outside step timers
    accounted_ms = text_ms + latent_prep_ms + denoise_ms + between_denoise_decode_ms + decode_ms + tail_ms
    ideal_wall_ms = compute["transformer"] + compute["transformer_2"]
    core_share = _finite_ratio(ideal_wall_ms, wall_ms)
    if core_share is None or core_share > 1.0:
        raise GlobalStopError("GLOBAL STOP: core compute exceeds or cannot be related to the request wall time")
    ub1 = 1.0 - core_share
    total_swap_ms = text_swap_ms + latent_prep_swap_ms + load_swap["transformer"] + load_swap["transformer_2"] + decode_swap_ms
    ub1_actionable = _finite_ratio(total_swap_ms, wall_ms)
    # memory: request-owned = static bytes of every loaded component; live = executing component + activation estimate
    static = profile["static_component_bytes"]
    loaded = [name for name in COMPONENTS if static.get(name)]
    request_owned = int(sum(static[name] for name in loaded))
    executing_for = {"text_encode_end": "text_encoder", "decode_end": "vae"}
    live_rows = []
    peak_allocated = 0
    peak_reserved = 0
    peak_resident_total = 0
    for event in events:
        resident = event["resident_component_bytes"]
        resident_total = int(sum(value for value in resident.values() if value))
        peak_resident_total = max(peak_resident_total, resident_total)
        peak_allocated = max(peak_allocated, int(event["max_memory_allocated_since_last_event"]), int(event["memory_allocated"]))
        peak_reserved = max(peak_reserved, int(event["max_memory_reserved_since_last_event"]), int(event["memory_reserved"]))
        component = event["component"] if event["event"] == "step_end" else executing_for.get(event["event"])
        if component is None:
            continue
        activation = max(int(event["max_memory_allocated_since_last_event"]) - resident_total, 0)
        live = int(resident.get(component) or 0) + activation
        live_rows.append({"event": event["event"], "component": component, "resident_executing_bytes": int(resident.get(component) or 0), "resident_total_bytes": resident_total, "activation_estimate_bytes": activation, "live_bytes_estimate": live})
    peak_live = max((row["live_bytes_estimate"] for row in live_rows), default=0)
    # UB2 (decision): headroom REMAINING after the runtime's own residency management.
    # actual = peak allocated bytes observed in this run; ideal = peak of (executing component + activations).
    live_share = _finite_ratio(peak_live, peak_allocated)
    if live_share is None or live_share > 1.0:
        raise GlobalStopError("GLOBAL STOP: ideal live-set peak exceeds or cannot be related to the allocated peak")
    ub2_remaining = 1.0 - live_share
    reserved_share = _finite_ratio(peak_live, peak_reserved)
    ub2_remaining_vs_reserved = None if reserved_share is None else 1.0 - reserved_share
    # descriptive: what the existing runtime already saves relative to owning the whole stack
    owned_share = _finite_ratio(peak_allocated, request_owned)
    already_captured = None if owned_share is None else 1.0 - owned_share
    ideal_share = _finite_ratio(peak_live, request_owned)
    stack_vs_ideal = None if ideal_share is None else 1.0 - ideal_share
    step_events = [row_event for row_event in events if row_event["event"] == "step_end"]
    exclusive = all(
        sum(1 for name in COMPONENTS if (row_event["resident_component_bytes"].get(name) or 0) > 0) <= 1
        for row_event in step_events
    )
    managed_exclusive = all(
        sum(1 for name in OFFLOAD_MANAGED_COMPONENTS if (row_event["resident_component_bytes"].get(name) or 0) > 0) == 1
        and (row_event["resident_component_bytes"].get(row_event["component"]) or 0) > 0
        for row_event in step_events
    )
    outside_scope_resident = sorted({
        name for row_event in step_events for name in COMPONENTS
        if name not in OFFLOAD_MANAGED_COMPONENTS and (row_event["resident_component_bytes"].get(name) or 0) > 0
    })
    only_allowed_outside_scope = all(name in RESIDENT_OUTSIDE_OFFLOAD_SCOPE for name in outside_scope_resident)
    return {
        "schema": schema,
        "schema_valid": True,
        "decision_eligible": True,
        "time": {
            "wall_ms": wall_ms,
            "text_encode_ms": text_ms,
            "text_encode_swap_ms": text_swap_ms,
            "latent_prep_ms": latent_prep_ms,
            "latent_prep_swap_ms": latent_prep_swap_ms,
            "denoise_ms": denoise_ms,
            "denoise_step_sum_ms": denoise_step_sum_ms,
            "denoise_loop_overhead_ms": denoise_overhead_ms,
            "transformer_compute_ms": compute["transformer"],
            "transformer_load_swap_ms": load_swap["transformer"],
            "transformer_2_compute_ms": compute["transformer_2"],
            "transformer_to_transformer_2_swap_ms": load_swap["transformer_2"],
            "between_denoise_and_decode_ms": between_denoise_decode_ms,
            "decode_ms": decode_ms,
            "vae_load_swap_ms": decode_swap_ms,
            "vae_decode_compute_ms": max(decode_ms - decode_swap_ms, 0.0),
            "tail_ms": tail_ms,
            "accounted_ms": accounted_ms,
            "ideal_wall_ms_compute_only": ideal_wall_ms,
            "swap_bytes": {"transformer": load_bytes["transformer"], "transformer_2": load_bytes["transformer_2"], "vae": {"to_cpu": decode_to_cpu, "to_gpu": decode_to_gpu}},
            "swap_event_count": len(swaps),
        },
        "shares": {
            "transformer_compute": compute["transformer"] / wall_ms if wall_ms else None,
            "transformer_2_compute": compute["transformer_2"] / wall_ms if wall_ms else None,
            "text_encode": text_ms / wall_ms if wall_ms else None,
            "decode": decode_ms / wall_ms if wall_ms else None,
            "all_swaps": (text_swap_ms + latent_prep_swap_ms + load_swap["transformer"] + load_swap["transformer_2"] + decode_swap_ms) / wall_ms if wall_ms else None,
        },
        "ub1_time_overlap": ub1,
        "ub1_actionable_swap_share": ub1_actionable,
        "memory": {
            "static_component_bytes": static,
            "loaded_components": loaded,
            "request_owned_bytes": request_owned,
            "peak_live_bytes_estimate": peak_live,
            "peak_memory_allocated": peak_allocated,
            "peak_memory_reserved": peak_reserved,
            "peak_resident_weight_bytes": peak_resident_total,
            "components_mutually_exclusive_during_steps": exclusive,
            "managed_components_mutually_exclusive_during_steps": managed_exclusive,
            "resident_outside_offload_scope_during_steps": outside_scope_resident,
            "only_allowed_components_resident_outside_scope": only_allowed_outside_scope,
            "device": profile.get("device"),
            "live_rows": live_rows,
        },
        "ub2_residency": ub2_remaining,
        "ub2_residency_vs_reserved": ub2_remaining_vs_reserved,
        "residency_already_captured_by_runtime": already_captured,
        "stack_vs_ideal_live_set_descriptive": stack_vs_ideal,
        "per_step": per_step,
    }


def validate_run_plan(document: dict[str, Any], mode: str) -> None:
    """The frozen run plan is warmup, plain, measured. Persisted JSON is canonical (sorted keys), so the
    set of runs and the explicitly recorded execution order are checked, never dict key order."""
    runs = document.get("runs", {})
    if set(runs) != set(RUN_ORDER):
        raise GlobalStopError(f"GLOBAL STOP: offload_{mode} runs are not exactly warmup, plain, measured (found {sorted(runs)})")
    order = document.get("run_order")
    if order is not None and list(order) != list(RUN_ORDER):
        raise GlobalStopError(f"GLOBAL STOP: offload_{mode} recorded execution order {order} is not warmup, plain, measured")
    for run_name in RUN_ORDER:
        if bool(runs[run_name].get("instrumented")) != (run_name == "measured"):
            raise GlobalStopError(f"GLOBAL STOP: offload_{mode}/{run_name} instrumentation flag inconsistent with the frozen plan")


def offload_on_identity(measured: dict[str, Any]) -> dict[str, Any]:
    """Primary decision-identity gate for the offload=on run.

    The label 'offload_on' is never trusted. The run must show artifact evidence that the production
    sequential offload actually executed: at least one recorded swap, exactly one offload-managed
    component ({text_encoder, transformer, transformer_2}) resident during every denoising step and it
    must be the executing expert, and no component outside the offload scope other than the VAE
    resident during steps. Otherwise the measurement is not the preregistered baseline and no
    decision quantity survives (UB1/UB2 -> None, MEASUREMENT_INVALID).

    Amendment (2026-09-04, before any decision was read): the first version required mutual exclusion
    of all four components; production offload never offloads the VAE, so that gate rejected the real
    runtime. VAE residency during steps is inactive resident weight and enters UB2 as remaining headroom."""
    if not measured.get("schema_valid"):
        return measured
    swap_count = int(measured["time"]["swap_event_count"])
    managed_exclusive = bool(measured["memory"]["managed_components_mutually_exclusive_during_steps"])
    outside_ok = bool(measured["memory"]["only_allowed_components_resident_outside_scope"])
    valid = swap_count >= 1 and managed_exclusive and outside_ok
    identity = {
        "swap_event_count": swap_count,
        "managed_components_mutually_exclusive_during_steps": managed_exclusive,
        "resident_outside_offload_scope_during_steps": measured["memory"]["resident_outside_offload_scope_during_steps"],
        "only_allowed_components_resident_outside_scope": outside_ok,
        "all_components_mutually_exclusive_during_steps_descriptive": bool(measured["memory"]["components_mutually_exclusive_during_steps"]),
        "offload_on_valid": valid,
    }
    if valid:
        return {**measured, "offload_on_identity": identity}
    return {**measured, "offload_on_identity": identity, "decision_eligible": False, **{key: None for key in INVALID_RESULT_KEYS}}


def decide(ub1: float | None, ub2: float | None) -> str:
    """Frozen decision matrix on (UB1 absolute, UB2 remaining) of the measured offload=on run."""
    if ub1 is None or ub2 is None:
        return "MEASUREMENT_INVALID"
    stop_time = ub1 < UB1_STOP_BELOW
    stop_mem = ub2 < UB2_STOP_BELOW
    if stop_time and stop_mem:
        return "STOP_ALL"
    if stop_time:
        return "RESIDENCY_ONLY_TO_OFFLINE_ORACLE"
    if stop_mem:
        return "TIME_OVERLAP_ONLY_TO_OFFLINE_ORACLE"
    return "JOINT_OFFLINE_ORACLE"


def probe_overhead_control(plain_client_ms: float, measured_client_ms: float) -> dict[str, Any]:
    overhead = (measured_client_ms - plain_client_ms) / plain_client_ms if plain_client_ms > 0 else None
    if overhead is None:
        status = "INVALID"
    elif overhead < PROBE_OVERHEAD_VALID_BELOW:
        status = "VALID"
    elif overhead <= PROBE_OVERHEAD_INVALID_ABOVE:
        status = "VALID_WITH_CORRECTION"
    else:
        status = "INVALID"
    return {"plain_client_ms": plain_client_ms, "measured_client_ms": measured_client_ms, "overhead_fraction": overhead, "status": status}


def corrected_ub1(result: dict[str, Any], control: dict[str, Any]) -> float | None:
    """UB1 with the instrumented wall time replaced by the uninstrumented (plain) wall time.

    Core compute is taken from the instrumented run (per-step GPU work is unchanged by the probe);
    the plain wall time bounds everything else. Used only when the control is VALID_WITH_CORRECTION."""
    plain_wall_ms = control["plain_client_ms"] - (control["measured_client_ms"] - result["time"]["wall_ms"])
    core = result["time"]["ideal_wall_ms_compute_only"]
    share = _finite_ratio(core, plain_wall_ms)
    if share is None or share > 1.0:
        # impossible measurement (core compute longer than the uninstrumented request): hard invalid, never clamped
        return None
    return 1.0 - share


ANALYZE_GATES = tuple(f"R-A{index}" for index in range(1, 13))


def run_analyze(config: dict[str, Any], config_path: Path, root: Path) -> dict[str, Any]:
    prov, manifest = load_frozen(root, config_path)
    trusted = trusted_final_latent(config)
    results: dict[str, Any] = {}
    feasibility: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    for mode in OFFLOAD_MODES:
        path = root / "profile" / f"offload_{mode}" / "profile_result.json"
        if not path.exists():
            feasibility[mode] = "MISSING"
            continue
        document = json.loads(path.read_text())
        if document.get("provenance_hash") != prov["provenance_hash"] or document.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise GlobalStopError(f"GLOBAL STOP: offload_{mode} profile is not bound to the frozen manifest")
        if not document.get("feasible"):
            failure_class = document.get("failure_class")
            if failure_class == "OFF_RUN_INFEASIBLE":
                feasibility[mode] = "INFEASIBLE"
            else:
                feasibility[mode] = "ERROR"
            results[mode] = {"feasible": False, "failure_class": failure_class or "UNCLASSIFIED", "error_type": document.get("error", {}).get("type"), "error": document.get("error", {}).get("message"),
                             "interpretation": "full model-stack residency does not fit this GPU" if feasibility[mode] == "INFEASIBLE" else "run failed for a non-memory reason; no residency-baseline statement may be made"}
            continue
        feasibility[mode] = "FEASIBLE"
        validate_run_plan(document, mode)
        finals = {}
        for run_name in RUN_ORDER:
            run = document["runs"][run_name]
            if bool(run.get("instrumented")) != (run_name == "measured"):
                raise GlobalStopError(f"GLOBAL STOP: offload_{mode}/{run_name} instrumentation flag inconsistent with the frozen plan")
            final = loc.load_tensor(root, run["final_latent"])
            finals[run_name] = bool(np.array_equal(final, trusted))
        run = document["runs"]["measured"]
        probe_path = root / run["resource_lifetime_probe_path"]
        if sha256_file(probe_path) != run["resource_lifetime_probe_sha256"]:
            raise GlobalStopError(f"GLOBAL STOP: probe file hash mismatch for offload_{mode}/measured")
        measured = analyze_events(json.loads(probe_path.read_text()), manifest)
        if mode == "on":
            measured = offload_on_identity(measured)
        control = probe_overhead_control(float(document["runs"]["plain"]["client_elapsed_ms"]), float(run["client_elapsed_ms"]))
        controls[mode] = {"final_latent_bit_exact_with_trusted_v3": finals, "probe_overhead": control, "client_elapsed_ms": {name: document["runs"][name]["client_elapsed_ms"] for name in RUN_ORDER}}
        ub1_used = measured["ub1_time_overlap"] if measured.get("decision_eligible") else None
        if ub1_used is not None and control["status"] == "VALID_WITH_CORRECTION":
            ub1_used = corrected_ub1(measured, control)
        elif control["status"] == "INVALID":
            ub1_used = None
        results[mode] = {"measured": {**measured, "final_latent_bit_exact_with_trusted_v3": finals["measured"], "ub1_used_for_decision": ub1_used, "probe_overhead_status": control["status"]}}
    if feasibility.get("on") != "FEASIBLE":
        raise GlobalStopError(f"GLOBAL STOP: the offload=on profile is required and must be feasible (state: {feasibility.get('on')}, {results.get('on', {}).get('failure_class')})")
    measured_on = results["on"]["measured"]
    decision_inputs = {
        "on": {
            "ub1_absolute": measured_on["ub1_time_overlap"],
            "ub1_actionable_swap_share": measured_on["ub1_actionable_swap_share"],
            "ub1_used_for_decision": measured_on["ub1_used_for_decision"],
            "ub2_remaining": measured_on["ub2_residency"],
            "residency_already_captured_by_runtime": measured_on["residency_already_captured_by_runtime"],
            "probe_overhead_status": measured_on["probe_overhead_status"],
            "offload_on_identity": measured_on.get("offload_on_identity"),
            "decision": decide(measured_on["ub1_used_for_decision"], measured_on["ub2_residency"]),
        }
    }
    if feasibility.get("off") == "FEASIBLE":
        measured_off = results["off"]["measured"]
        decision_inputs["off_baseline_descriptive"] = {"ub1_absolute": measured_off["ub1_time_overlap"], "ub2_remaining": measured_off["ub2_residency"], "schema_valid": measured_off["schema_valid"], "note": "baseline characterization only; never enters the decision"}
        comparison = {
            "wall_ms_on_minus_off": measured_on["time"]["wall_ms"] - measured_off["time"]["wall_ms"],
            "peak_allocated_on_minus_off": measured_on["memory"]["peak_memory_allocated"] - measured_off["memory"]["peak_memory_allocated"],
            "peak_resident_weight_on_minus_off": measured_on["memory"]["peak_resident_weight_bytes"] - measured_off["memory"]["peak_resident_weight_bytes"],
        }
    else:
        comparison = {"note": f"offload=off {feasibility.get('off')}: residency-vs-transfer comparison unavailable; request-owned bytes use static component sizes"}
    primary = decision_inputs["on"]["decision"]
    schema_all = all(results[m]["measured"]["schema_valid"] for m in results if "measured" in results[m])
    if not measured_on.get("decision_eligible"):
        primary = "MEASUREMENT_INVALID"
        decision_inputs["on"]["decision"] = primary
    gates = [
        gate("R-A1 committed source / clean provenance", not prov.get("source_dirty_entries"), {"git_commit": prov.get("git_commit")}, required=True),
        gate("R-A2 offload=on profile feasible with warmup + plain + measured runs", feasibility.get("on") == "FEASIBLE", feasibility, required=True),
        gate("R-A3 probe schema valid (event order, 40 steps, 26/14 split, schedule, finite non-negative latencies, decode not skipped, memory fields, swaps)", schema_all, {m: results[m]["measured"]["schema"] for m in results if "measured" in results[m]}, required=True),
        gate("R-A4 every run's final latent bit-exact with trusted v3 (every feasible mode)", all(all(c["final_latent_bit_exact_with_trusted_v3"].values()) for c in controls.values()), {m: c["final_latent_bit_exact_with_trusted_v3"] for m, c in controls.items()}, required=True),
        gate("R-A5 time decomposition accounts for the full wall time", schema_all and all(abs(results[m]["measured"]["time"]["accounted_ms"] - results[m]["measured"]["time"]["wall_ms"]) < 1e-3 for m in results if "measured" in results[m]), None, required=True),
        gate("R-A6 offload=off recorded as FEASIBLE, INFEASIBLE (recognised memory exhaustion) or ERROR (never silently missing)", feasibility.get("off") in ("FEASIBLE", "INFEASIBLE", "ERROR"), {"feasibility": feasibility, "off": results.get("off", {}).get("failure_class")}, required=True),
        gate("R-A7 kill thresholds unchanged", manifest["kill_rules"]["ub1_stop_below"] == UB1_STOP_BELOW and manifest["kill_rules"]["ub2_stop_below"] == UB2_STOP_BELOW, manifest["kill_rules"], required=True),
        gate("R-A8 probe-overhead control valid for offload=on (instrumented vs plain wall time)", controls["on"]["probe_overhead"]["status"] in ("VALID", "VALID_WITH_CORRECTION"), controls["on"]["probe_overhead"], required=True),
        gate("R-A11 decision computed from the measured offload=on run only (UB1 absolute, UB2 remaining after runtime offload)", primary != "MEASUREMENT_INVALID", decision_inputs["on"], required=True),
        gate("R-A12 offload=on run identity: >=1 recorded swap AND exactly the executing offload-managed component resident per step AND only the VAE resident outside the offload scope", bool(measured_on.get("offload_on_identity", {}).get("offload_on_valid")), measured_on.get("offload_on_identity"), required=True),
        gate("R-A9 no scheduler/concurrency/approximation artefacts present", not any((root / name).exists() for name in ("scheduler", "concurrency", "oracle")), None, required=True),
        gate("R-A10 provenance/manifest hash-bound", True, {"manifest_sha256": manifest["manifest_sha256"]}, required=True),
    ]
    document = {"mode": "analyze", "gates": gates, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"], "decision": primary}
    analysis = {"decision": primary, "decision_inputs": decision_inputs, "feasibility": feasibility, "comparison_on_vs_off": comparison, "controls": controls, "results": results, "provenance_hash": prov["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"]}
    atomic_json(root / "analysis.json", analysis)
    atomic_json(root / "gates.json", document)
    names = tuple(row["name"] for row in gates)
    if tuple(sorted(name.split(" ", 1)[0] for name in names)) != tuple(sorted(ANALYZE_GATES)):
        raise GlobalStopError("GLOBAL STOP: analysis gate set changed")
    validate_gate_document(document, names, provenance_hash=prov["provenance_hash"], manifest_sha256=manifest["manifest_sha256"])
    summary = {
        m: {
            "wall_s": results[m]["measured"]["time"]["wall_ms"] / 1000.0 if results[m]["measured"]["schema_valid"] else None,
            "ub1_absolute": results[m]["measured"]["ub1_time_overlap"],
            "ub1_actionable": results[m]["measured"]["ub1_actionable_swap_share"],
            "ub2_remaining": results[m]["measured"]["ub2_residency"],
            "already_captured": results[m]["measured"]["residency_already_captured_by_runtime"],
            "probe_overhead": controls[m]["probe_overhead"],
            "shares": results[m]["measured"].get("shares"),
        }
        for m in results
        if "measured" in results[m]
    }
    return {"mode": "analyze", "decision": primary, "feasibility": feasibility, "summary": summary, "comparison_on_vs_off": comparison}


# --------------------------------------------------------------------------------------
def _synthetic_profile(manifest: dict[str, Any]) -> dict[str, Any]:
    """Deterministic synthetic probe output used only to validate the analysis code on CPU."""
    static = {"text_encoder": 11 * 2**30, "transformer": 28 * 2**30, "transformer_2": 28 * 2**30, "vae": 1 * 2**30}
    components = manifest["expected_step_components"]
    schedule = manifest["frozen_schedule"]
    events: list[dict[str, Any]] = []
    swaps: list[dict[str, Any]] = []
    now = 1000.0

    def resident(active: str | None) -> dict[str, int]:
        return {name: (static[name] if name == active else 0) for name in COMPONENTS}

    def add(event: str, active: str | None, allocated: int, **fields: Any) -> None:
        events.append({"event": event, "t": now, "t_rel_ms": (now - 1000.0) * 1000.0, "resident_component_bytes": resident(active), "memory_allocated": allocated, "memory_reserved": allocated + 2**30, "max_memory_allocated_since_last_event": allocated + 2**29, "max_memory_reserved_since_last_event": allocated + 2**30, **fields})

    def swap(module: str, to_cpu: int, to_gpu: int, seconds: float) -> None:
        nonlocal now
        swaps.append({"module": module, "module_id": 1, "offloaded_modules": [], "bytes_to_cpu": to_cpu, "bytes_to_gpu": to_gpu, "t_start": now, "t_end": now + seconds})
        now += seconds

    add("request_start", None, 0, num_inference_steps=EXPECTED_STEPS)
    swap("UMT5EncoderModel", 0, static["text_encoder"], 0.5)
    now += 1.5
    add("text_encode_end", "text_encoder", static["text_encoder"])
    now += 0.2
    add("denoise_start", "text_encoder", static["text_encoder"], num_steps=EXPECTED_STEPS)
    seen: set[str] = set()
    for index, component in enumerate(components):
        step_start = now
        if component not in seen:
            swap("WanTransformer3DModel", static["text_encoder"] if component == "transformer" else static["transformer"], static[component], 2.0)
            seen.add(component)
        now += 4.5
        add("step_end", component, static[component] + 2 * 2**30, step_index=index, component=component, timestep=schedule[index], step_latency_ms=(now - step_start) * 1000.0, cfg_branches=2)
    add("denoise_end", "transformer_2", static["transformer_2"])
    now += 0.1
    add("decode_start", "transformer_2", static["transformer_2"])
    swap("AutoencoderKLWan", static["transformer_2"], static["vae"], 0.3)
    now += 12.0
    add("decode_end", "vae", static["vae"] + 6 * 2**30, decode_skipped=False)
    now += 0.05
    add("request_end", "vae", static["vae"])
    return {"label": "synthetic", "t0": 1000.0, "static_component_bytes": static, "component_loaded": {name: True for name in COMPONENTS}, "device": {"name": "synthetic", "total_memory_bytes": 48 * 2**30}, "events": events, "offload_events": swaps}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--config", type=Path, default=Path("experiments/video_resource_lifetime_profile_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--offload", choices=OFFLOAD_MODES, default=None, help="profile mode only")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="ignored: derived from --offload in profile mode")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = (args.output_dir or Path(config["output_root"])).resolve()
    if args.mode == "cpu":
        result = run_cpu(config, args.config, root)
    elif args.mode == "profile":
        if args.offload is None:
            raise GlobalStopError("GLOBAL STOP: profile mode requires --offload on|off")
        result = run_profile(config, args.config, root, args)
    else:
        result = run_analyze(config, args.config, root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
