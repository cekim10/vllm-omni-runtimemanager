#!/usr/bin/env python3
"""Numerical Compatibility Round 0 - phased, preregistered, fail-closed orchestration (rev 0.2d).

Phases: cpu | preregister | canonical | axis --axis {EAGER,OFFLOAD,ATTN_FLASHINFER,ATTN_CUDNN} | freshcheck --axis {CANONICAL,...} | analyze

Every GPU run is one request into the trusted Wan2.2 pipeline (no denoising is re-implemented).  The runner only produces
arrays and hashes; every class and the decision are computed by core.py from saved artifacts in `analyze`.
Boundary indexing: x_k = latent entering step k; x_40 = final.  H_{A->X,t} = resume under configuration X from the canonical
run-1 latent x_t.  The worker records a runtime-observed fingerprint into the trajectory-probe metadata; a run whose fingerprint
does not match its declared configuration stops the phase (fail closed) and is INVALID for the axis.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import video_trajectory_fork_killtest as smoke  # noqa: E402
from experiments.numerical_compatibility_round0 import core  # noqa: E402

PKG = "experiments/numerical_compatibility_round0"
EXPERIMENT_VERSION = "numerical-compatibility-round0-v0.2d"
NAMESPACE = "numerical_compatibility_round0"
ATTEMPT_SUBDIR = "rev0_2"
DEFAULT_CONFIG = REPO_ROOT / PKG / "config.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / NAMESPACE / ATTEMPT_SUBDIR
PHASES = ("cpu", "preregister", "canonical", "axis", "freshcheck", "analyze")
MOTIVATION_ROOT = REPO_ROOT / "results" / "regional_recompute_round0" / "rev4_1"
TRUSTED_SOURCE_FILES = (
    f"{PKG}/__init__.py", f"{PKG}/core.py", f"{PKG}/run.py", f"{PKG}/config.json", f"{PKG}/run_gpu0.sh",
    "tests/diffusion/test_numerical_compatibility_round0.py",
    "experiments/video_trajectory_fork_killtest.py", "experiments/video_runtime_state_discovery.py",
    "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py", "vllm_omni/diffusion/models/wan2_2/scheduling_wan_euler.py", "vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py",
    # mechanisms that define the axes
    "vllm_omni/diffusion/attention/selector.py", "vllm_omni/diffusion/attention/layer.py", "vllm_omni/platforms/cuda/platform.py",
    "vllm_omni/diffusion/compile.py", "vllm_omni/diffusion/offloader/sequential_backend.py", "vllm_omni/diffusion/worker/diffusion_model_runner.py",
)
FORBIDDEN_OUTPUT_PARTS = ("video_trajectory_fork", "video_runtime_state_discovery", "video_bf16", "video_execution_ordering", "video_resource_lifetime", "speculative_contamination_round0", "regional_recompute_round0")
GateError = smoke.GateError
sha256_file, sha256_bytes, array_sha256, atomic_json = smoke.sha256_file, smoke.sha256_bytes, smoke.array_sha256, smoke.atomic_json
PHASE_DIR = {"canonical": "canonical", "axis": "axis", "freshcheck": "freshcheck"}
ATTENTION_ENV = "DIFFUSION_ATTENTION_BACKEND"


# --------------------------------------------------------------------------------------
# config / plan
# --------------------------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["version"] != EXPERIMENT_VERSION:
        raise ValueError(f"config version {config['version']} != {EXPERIMENT_VERSION}")
    if config["scheduler"]["name"] != smoke.EXPECTED_SCHEDULER or config["scheduler"]["sample_solver"] != "euler":
        raise ValueError("Only the trusted WanEulerScheduler / euler path is allowed")
    if int(config["generation"]["num_inference_steps"]) != core.TOTAL_STEPS:
        raise ValueError("40 inference steps are frozen")
    th = config["thresholds"]
    if th["same_sample_rel_l2_max"] != core.STATE_REL_L2_MAX or th["same_sample_frame_ssim_min"] != core.OUTPUT_SSIM_MIN or th["material_position_threshold"] != core.MATERIAL_POSITION_THRESHOLD:
        raise ValueError("thresholds frozen")
    if tuple(config["checkpoints_t"]) != core.CHECKPOINTS_T:
        raise ValueError("checkpoints frozen")
    if set(config["axes"]) != set(core.CORE_AXES) | set(core.NOT_MEASURED_AXES):
        raise ValueError("axis set frozen")
    for ax in core.CORE_AXES:
        if config["axes"][ax]["class"] != "core":
            raise ValueError(f"{ax} must be a core axis")
    for ax in core.NOT_MEASURED_AXES:
        if not config["axes"][ax]["class"].startswith("NOT_MEASURED"):
            raise ValueError(f"{ax} must be NOT_MEASURED")
    ids = [t["id"] for t in config["trajectories"]]
    if len(ids) != core.N_TRAJECTORIES_REQUIRED or len(set(ids)) != len(ids):
        raise ValueError("exactly three distinct trajectories")
    if config["batch_partner"]["id"] in ids:
        raise ValueError("batch partner must not be a decision trajectory")


def validate_output_path(path: Path) -> None:
    try:
        rel = path.resolve().relative_to((REPO_ROOT / "results").resolve())
    except ValueError as error:
        raise GateError(f"Output must live under results/: {path}") from error
    if len(rel.parts) < 2 or rel.parts[0] != NAMESPACE or rel.parts[1] != ATTEMPT_SUBDIR:
        raise GateError(f"Output must be results/{NAMESPACE}/{ATTEMPT_SUBDIR}[/...]: {path}")
    if any(part in str(rel) for part in FORBIDDEN_OUTPUT_PARTS):
        raise GateError(f"Refusing to write into a frozen namespace: {path}")


def trajectories(config: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = {p["id"]: p for p in config["prompts"]}
    return [{"id": t["id"], "prompt_id": t["prompt_id"], "prompt": prompts[t["prompt_id"]]["prompt"], "category": prompts[t["prompt_id"]]["category"], "seed": int(t["seed"])} for t in config["trajectories"]]


def fresh_trajectory(config: dict[str, Any]) -> str:
    return str(config["run_matrix"]["fresh_engine_control"]["trajectory"])


def plan_for(config: dict[str, Any]) -> list[dict[str, Any]]:
    return core.run_plan(trajectories(config), fresh_trajectory(config))


def build_provenance(config_path: Path) -> dict[str, Any]:
    paths = [REPO_ROOT / v for v in TRUSTED_SOURCE_FILES]
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
    doc = {"git_commit": commit, "git_dirty": bool(status), "git_status": status, "relevant_git_status": relevant, "source_sha256": hashes, "config_sha256": sha256_file(config_path)}
    doc["provenance_hash"] = sha256_bytes(core.canonical({k: doc[k] for k in ("git_commit", "relevant_git_status", "source_sha256", "config_sha256")}))
    return doc


def motivation_reference_hashes() -> dict[str, Any]:
    """Frozen rev 4.1 reference finals (descriptive cross-session anchor).  Missing files are recorded, never fatal."""
    out: dict[str, Any] = {}
    for tr in ("p0_s9101", "p2_s9101", "p3_s9101"):
        p = MOTIVATION_ROOT / "reference" / f"{tr}_reference" / "run.json"
        out[tr] = json.loads(p.read_text())["artifact_sha256"].get("final.npy") if p.exists() else None
    pre = MOTIVATION_ROOT / "preregistration.sha256"
    return {"root": str(MOTIVATION_ROOT.relative_to(REPO_ROOT)), "preregistration_sha256": pre.read_text().split()[0] if pre.exists() else None, "final_sha256_by_traj": out}


def build_preregistration(config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    plan_cfg = json.loads(json.dumps(config))
    plan_cfg["seed"] = int(config["trajectories"][0]["seed"])
    plan_cfg["generation"]["switch_steps"] = list(smoke.EXPECTED_SWITCHES)
    sched = smoke.scheduler_plan(plan_cfg)
    runs = plan_for(config)
    return {
        "title": "Numerical Compatibility Round 0: is intermediate Wan2.2 denoising state portable across single-GPU execution configurations?",
        "experiment_version": EXPERIMENT_VERSION, "revision": "0.2d", "primary_question": config["primary_question"], "broken_assumption": config["broken_assumption"],
        "motivation_evidence": {**config["motivation_evidence"], "frozen_reference_finals": motivation_reference_hashes(),
                                 "status": "synthetic perturbation evidence; motivation only, never evidence for this round"},
        "model": config["model"], "generation": config["generation"], "scheduler": config["scheduler"], "scheduler_plan": sched,
        "trajectories": trajectories(config), "fresh_engine_trajectory": fresh_trajectory(config), "batch_partner_unused": config["batch_partner"],
        "design": core.frozen_design(), "canonical_configuration": config["canonical_configuration"], "axes": config["axes"], "excluded_axes": config["excluded_axes"],
        "host_facts": config["host_facts_2026_09_07"], "vocabulary": config["vocabulary"], "boundary_indexing": config["boundary_indexing"],
        "run_plan_labels": [p["label"] for p in runs], "n_runs": len(runs), "cost_estimate": core.estimate_cost(runs),
        "invalid_conditions": config["invalid_conditions"], "descriptive_controls": config["descriptive_controls"], "hypotheses_non_gating": config["hypotheses_non_gating"],
        "decision": config["decision"], "result_ranking": list(core.RESULT_RANKING), "kill_scope": config["kill_scope"], "claim_boundary": config["claim_boundary"],
        "novelty_boundary": config["novelty_boundary"], "changes_from_rev0_1": config["changes_from_rev0_1"],
        "gpu_execution": "requires separate explicit user approval after CPU tests and the final hostile audit",
        "verification": "analyze recomputes every class and the decision from saved arrays and hashes; declared booleans are never trusted",
        "provenance_hash": provenance["provenance_hash"], "source_commit": provenance["git_commit"], "config_sha256": provenance["config_sha256"],
    }


def require_sealed(output_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    pp, sp = output_dir / "preregistration.json", output_dir / "preregistration.sha256"
    if not pp.exists() or not sp.exists():
        raise GateError("Preregistration is not sealed; run --phase preregister first")
    if sha256_file(pp) != sp.read_text().split()[0]:
        raise GateError("preregistration.json was modified after sealing")
    doc = json.loads(pp.read_text())
    frozen = json.loads((output_dir / "provenance.json").read_text())
    if doc.get("provenance_hash") != provenance["provenance_hash"] or frozen.get("provenance_hash") != provenance["provenance_hash"]:
        raise GateError("Code/config provenance changed after preregistration; use a fresh namespace")
    if provenance["relevant_git_status"]:
        raise GateError(f"Scientific source files are not committed: {provenance['relevant_git_status']}")
    return doc


def _write_manifest(output_dir: Path, provenance: dict[str, Any], extra: dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"provenance_hash": provenance["provenance_hash"], "immutable_sha256": {}, "phases": []}
    for name in ("preregistration.json", "provenance.json", "prompts.json"):
        if (output_dir / name).exists():
            manifest["immutable_sha256"][name] = sha256_file(output_dir / name)
    manifest["phases"].append({**extra, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    atomic_json(path, manifest)


def _require_no_invalid(output_dir: Path) -> None:
    p = output_dir / "decision.json"
    if p.exists() and json.loads(p.read_text()).get("decision") == core.INVALID:
        raise GateError("Experiment already marked INVALID_EXPERIMENT; no further phases")


# --------------------------------------------------------------------------------------
# CPU phases
# --------------------------------------------------------------------------------------
def phase_cpu(config: dict[str, Any]) -> dict[str, Any]:
    runs = plan_for(config)
    cost = core.estimate_cost(runs)
    # self-check of the decision truth table on the frozen ranking
    ok = core.decide(invalid_reasons=[], axes={"EAGER": {"measured": True, "within": core.WITHIN_REPRODUCIBLE, "full": core.FULL_SAMPLE_EQUIVALENT, "hybrid_t12": core.HYBRID_INCOMPATIBLE},
                                               "ATTN_CUDNN": {"measured": True, "within": core.WITHIN_REPRODUCIBLE, "full": core.FULL_SAMPLE_DIFFERENT, "hybrid_t12": core.HYBRID_MIXED}})
    if ok["decision"] != core.GO_STRONG:
        raise GateError("decision self-check failed")
    return {"phase": "cpu", "version": EXPERIMENT_VERSION, "runs": len(runs), "cost_estimate": cost, "design": core.frozen_design(), "not_measured": core.NOT_MEASURED_AXES}


def phase_preregister(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = build_provenance(config_path)
    pp, sp = output_dir / "preregistration.json", output_dir / "preregistration.sha256"
    if pp.exists():
        if json.loads(pp.read_text()).get("provenance_hash") != provenance["provenance_hash"]:
            raise GateError("Namespace already holds a preregistration from different code/config; use a fresh namespace")
        return {"phase": "preregister", "status": "ALREADY_SEALED", "preregistration_sha256": sp.read_text().split()[0]}
    doc = build_preregistration(config, provenance)
    atomic_json(output_dir / "provenance.json", provenance)
    atomic_json(output_dir / "prompts.json", {"trajectories": doc["trajectories"], "prompts": config["prompts"], "fresh_engine_trajectory": doc["fresh_engine_trajectory"]})
    atomic_json(pp, doc)
    digest = sha256_file(pp)
    sp.write_text(f"{digest}  preregistration.json\n")
    _write_manifest(output_dir, provenance, {"phase": "preregister"})
    return {"phase": "preregister", "status": "SEALED", "preregistration_sha256": digest, "n_runs": doc["n_runs"], "cost_estimate": doc["cost_estimate"], "relevant_git_status": provenance["relevant_git_status"]}


# --------------------------------------------------------------------------------------
# GPU primitive (trusted pipeline only)
# --------------------------------------------------------------------------------------
class Engine:
    """One engine == one execution configuration.  The attention backend is selected through the platform env var BEFORE
    the Omni engine is constructed; the recorded fingerprint (from the worker) is checked against the declared configuration."""

    def __init__(self, config: dict[str, Any], configuration: str, output_dir: Path, prereg: dict[str, Any], provenance: dict[str, Any]):
        from experiments.video_runtime_state_discovery import build_omni

        if configuration not in core.CONFIGURATIONS:
            raise GateError(f"unknown configuration {configuration}")
        self.config, self.configuration, self.output_dir, self.prereg, self.provenance = config, configuration, output_dir, prereg, provenance
        spec = core.CONFIGURATIONS[configuration]
        if spec["attention_backend_env"] is None:
            os.environ.pop(ATTENTION_ENV, None)
        else:
            os.environ[ATTENTION_ENV] = spec["attention_backend_env"]
        self.schedule = [float(v) for v in prereg["scheduler_plan"]["timesteps"]]
        args = argparse.Namespace(enforce_eager=bool(spec["enforce_eager"]), enable_cpu_offload=bool(spec["enable_cpu_offload"]))
        self.omni = build_omni(config, args)
        self.model_revision = None
        try:
            from experiments.video_runtime_state_discovery import environment_document

            env = environment_document(config, prereg["scheduler_plan"], provenance)
            self.model_revision = env.get("resolved_model_revision")
            atomic_json(output_dir / f"environment_{configuration}.json", env)
        except Exception as error:
            atomic_json(output_dir / f"environment_{configuration}.json", {"status": f"deferred: {error}", "provenance_hash": provenance["provenance_hash"]})

    def shutdown(self) -> None:
        self.omni.shutdown()

    def _sampling(self, *, seed: int, label: str, artifact_dir: Path, capture_steps: list[int], latents: np.ndarray | None, step_index: int):
        import torch

        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        g = self.config["generation"]
        s = OmniDiffusionSamplingParams(height=int(g["height"]), width=int(g["width"]), num_frames=int(g["num_frames"]), num_inference_steps=int(g["num_inference_steps"]),
                                        guidance_scale=float(g["guidance_scale"]), fps=float(g["fps"]), seed=seed, generator=torch.Generator(device="cpu").manual_seed(seed))
        s.latents = None if latents is None else torch.from_numpy(np.ascontiguousarray(latents.astype(np.float32)))
        s.step_index = int(step_index)
        s.extra_args = {"flow_shift": float(self.config["scheduler"]["flow_shift"]), "sample_solver": "euler",
                        "trajectory_probe": {"artifact_dir": str(artifact_dir / "trajectory"), "request_label": label, "capture_steps": sorted(set(capture_steps)), "fps": float(g["fps"]),
                                             "save_decoded": False, "save_latents": True, "save_mp4": False}}
        return s

    def generate(self, *, prompt: str, seed: int, label: str, latents: np.ndarray | None, step_index: int) -> dict[str, Any]:
        """Run from step_index to the end, capturing the latent at every boundary; returns absolute-boundary arrays, video, fingerprint."""
        import torch

        from experiments.video_runtime_state_discovery import normalize_video
        from vllm_omni.outputs import OmniRequestOutput

        local_steps = core.TOTAL_STEPS - step_index
        capture = list(range(local_steps + 1))
        probe_dir = self.output_dir / "_probe_tmp" / label
        if probe_dir.exists():
            shutil.rmtree(probe_dir)
        started = time.perf_counter()
        outputs = self.omni.generate({"prompt": prompt}, self._sampling(seed=seed, label=label, artifact_dir=probe_dir, capture_steps=capture, latents=latents, step_index=step_index))
        wall = time.perf_counter() - started
        output = OmniRequestOutput.unwrap_result(outputs)
        custom = output.custom_output or {}
        meta = json.loads(Path(custom["trajectory_probe_metadata_path"]).read_text())
        if meta.get("sample_solver") != "euler" or not str(meta.get("scheduler_class", "")).endswith(smoke.EXPECTED_SCHEDULER):
            raise GateError(f"Worker did not execute the trusted Euler path for {label}")
        if int(meta.get("num_steps", -1)) != local_steps:
            raise GateError(f"Executed {meta.get('num_steps')} local steps, expected {local_steps} for {label}")
        records = {int(r["step_index"]): r for r in meta["records"]}
        if set(records) != set(capture):
            raise GateError(f"Trajectory probe captured {sorted(records)} but {capture} was requested for {label}")
        if abs(float(records[0]["timestep"]) - self.schedule[step_index]) > 1e-3:
            raise GateError(f"Resume timestep {records[0]['timestep']} != schedule[{step_index}] for {label}")

        def load(p: str) -> np.ndarray:
            return torch.load(p, map_location="cpu").detach().float().contiguous().numpy()

        arrays = {step_index + k: load(r["latent_path"]) for k, r in records.items()}
        if latents is not None and array_sha256(arrays[step_index]) != array_sha256(np.ascontiguousarray(latents.astype(np.float32))):
            raise GateError(f"State entering the resumed step differs from the supplied latents for {label} (mutation or cast)")
        fingerprint = dict(meta.get("runtime_fingerprint") or {})
        if "error" in fingerprint or not fingerprint:
            raise GateError(f"runtime_fingerprint missing or errored for {label}: {fingerprint}")
        if not self.model_revision:
            raise GateError(f"Resolved model revision unavailable for {label}; environment consistency cannot be checked")
        fingerprint["model_revision"] = self.model_revision
        fingerprint["batch_slot"] = 0
        mismatches = core.fingerprint_check(self.configuration, fingerprint)
        video, _ = normalize_video(outputs)
        if video.shape[0] != int(self.config["generation"]["num_frames"]):
            raise GateError(f"Decoded video has {video.shape[0]} frames for {label}")
        shutil.rmtree(probe_dir, ignore_errors=True)
        return {"arrays": arrays, "video": video, "wall_s": wall, "fingerprint": fingerprint, "fingerprint_mismatches": mismatches, "runtime_dtype": records[0].get("runtime_dtype")}


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------
def _run_dir(output_dir: Path, phase_dir: str, label: str) -> Path:
    return output_dir / phase_dir / label


def _load_run(output_dir: Path, phase_dir: str, label: str, provenance_hash: str) -> dict[str, Any] | None:
    rd = _run_dir(output_dir, phase_dir, label)
    if not (rd / "run.json").exists():
        return None
    doc = json.loads((rd / "run.json").read_text())
    if doc.get("provenance_hash") != provenance_hash or doc.get("status") != "COMPLETE":
        return None
    for rel, h in doc["artifact_sha256"].items():
        p = rd / rel
        if not p.exists():
            return None
        if rel.endswith(".npy") and array_sha256(np.load(p, allow_pickle=False)) != h:
            raise GateError(f"Persisted artifact hash mismatch: {p}")
    return doc


def _save_run(output_dir: Path, plan_item: dict[str, Any], provenance_hash: str, result: dict[str, Any], *, keep_video_npy: bool, fps: float) -> dict[str, Any]:
    from experiments.temporal_dimension_killtest_preflight import _save_video

    rd = _run_dir(output_dir, PHASE_DIR[plan_item["phase"]], plan_item["label"])
    rd.mkdir(parents=True, exist_ok=True)
    arrays: dict[int, np.ndarray] = result["arrays"]
    boundary_sha = {str(k): array_sha256(np.ascontiguousarray(arrays[k].astype(np.float32))) for k in sorted(arrays)}
    hashes: dict[str, str] = {}
    keep = sorted(arrays) if plan_item["keep_all_latents"] else [plan_item["t"], core.TOTAL_STEPS]
    for k in keep:
        a = np.ascontiguousarray(arrays[k].astype(np.float32))
        np.save(rd / f"x_{k:02d}.npy", a, allow_pickle=False)
        hashes[f"x_{k:02d}.npy"] = boundary_sha[str(k)]
    video = result["video"]
    _save_video(rd / "video.mp4", video, float(fps))
    hashes["video.mp4"] = sha256_file(rd / "video.mp4")
    video_hash = array_sha256(video)
    if keep_video_npy:
        np.save(rd / "video.npy", video, allow_pickle=False)
        hashes["video.npy"] = video_hash
    doc = {"label": plan_item["label"], "status": "COMPLETE", "provenance_hash": provenance_hash, "config": plan_item["config"], "kind": plan_item["kind"], "traj": plan_item["traj"], "rep": plan_item["rep"],
           "t": plan_item["t"], "role": plan_item.get("role", "primary"), "boundary_sha256": boundary_sha, "final_sha256": boundary_sha[str(core.TOTAL_STEPS)], "video_sha256": video_hash,
           "artifact_sha256": hashes, "fingerprint": result["fingerprint"], "fingerprint_mismatches": result["fingerprint_mismatches"], "runtime_dtype": result["runtime_dtype"], "wall_s": result["wall_s"]}
    atomic_json(rd / "run.json", doc)
    return doc


def _canonical_latent(output_dir: Path, traj: str, t: int) -> np.ndarray:
    p = _run_dir(output_dir, "canonical", f"{traj}_CANONICAL_F1") / f"x_{t:02d}.npy"
    if not p.exists():
        raise GateError(f"Canonical run-1 latent missing: {p}; run --phase canonical first")
    return np.load(p, allow_pickle=False)


def _execute(engine: Engine, output_dir: Path, item: dict[str, Any], tr: dict[str, Any], provenance_hash: str) -> dict[str, Any]:
    existing = _load_run(output_dir, PHASE_DIR[item["phase"]], item["label"], provenance_hash)
    if existing is not None:
        return existing
    latents = None if item["kind"] == "FULL" else _canonical_latent(output_dir, item["traj"], item["t"])
    result = engine.generate(prompt=tr["prompt"], seed=tr["seed"], label=item["label"], latents=latents, step_index=item["t"])
    keep_video = item["rep"] == 1 and item["phase"] != "freshcheck"
    doc = _save_run(output_dir, item, provenance_hash, result, keep_video_npy=keep_video, fps=float(engine.config["generation"]["fps"]))
    if result["fingerprint_mismatches"]:
        raise GateError(f"Recorded fingerprint does not match declared configuration {item['config']} for {item['label']}: {result['fingerprint_mismatches']} (run saved; axis INVALID)")
    return doc


def _traj(config: dict[str, Any], traj_id: str) -> dict[str, Any]:
    return next(x for x in trajectories(config) if x["id"] == traj_id)


def _run_items(config: dict[str, Any], config_path: Path, output_dir: Path, items: list[dict[str, Any]], configuration: str, phase: str) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_sealed(output_dir, provenance)
    _require_no_invalid(output_dir)
    todo = [p for p in items if _load_run(output_dir, PHASE_DIR[p["phase"]], p["label"], provenance["provenance_hash"]) is None]
    if todo:
        engine = Engine(config, configuration, output_dir, prereg, provenance)
        try:
            for item in todo:
                _execute(engine, output_dir, item, _traj(config, item["traj"]), provenance["provenance_hash"])
        finally:
            engine.shutdown()
    _write_manifest(output_dir, provenance, {"phase": phase, "configuration": configuration, "runs": len(items)})
    return {"phase": phase, "configuration": configuration, "runs": len(items), "executed_now": len(todo)}


def phase_canonical(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    items = [p for p in plan_for(config) if p["phase"] == "canonical"]
    # order: all F1 first (hybrids need x_t of F1), then F2, then the exact-resume anchors
    ordered = [p for p in items if p["kind"] == "FULL" and p["rep"] == 1] + [p for p in items if p["kind"] == "FULL" and p["rep"] == 2] + [p for p in items if p["kind"] == "HYBRID"]
    return _run_items(config, config_path, output_dir, ordered, core.CANONICAL, "canonical")


def phase_axis(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.axis not in core.CORE_AXES:
        raise GateError(f"--axis must be one of {core.CORE_AXES}; {args.axis!r} is not measurable in this round ({core.NOT_MEASURED_AXES.get(args.axis, 'unknown')})")
    provenance = build_provenance(config_path)
    for tr in trajectories(config):
        if _load_run(output_dir, "canonical", f"{tr['id']}_CANONICAL_F1", provenance["provenance_hash"]) is None:
            raise GateError("Canonical phase incomplete; run --phase canonical first")
    items = [p for p in plan_for(config) if p["phase"] == "axis" and p["config"] == args.axis]
    ordered = [p for p in items if p["kind"] == "FULL" and p["rep"] == 1] + [p for p in items if p["kind"] == "FULL" and p["rep"] == 2] + [p for p in items if p["kind"] == "HYBRID"]
    return _run_items(config, config_path, output_dir, ordered, args.axis, "axis")


def phase_freshcheck(config: dict[str, Any], config_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.axis not in core.CONFIGURATIONS:
        raise GateError(f"--axis must be one of {tuple(core.CONFIGURATIONS)}")
    items = [p for p in plan_for(config) if p["phase"] == "freshcheck" and p["config"] == args.axis]
    return _run_items(config, config_path, output_dir, items, args.axis, "freshcheck")


# --------------------------------------------------------------------------------------
# analyze (everything recomputed from artifacts)
# --------------------------------------------------------------------------------------
def _load_all_boundaries(rd: Path, doc: dict[str, Any]) -> dict[int, np.ndarray] | None:
    ks = sorted(int(k) for k in doc["boundary_sha256"])
    if any(not (rd / f"x_{k:02d}.npy").exists() for k in ks):
        return None
    return {k: np.load(rd / f"x_{k:02d}.npy", allow_pickle=False) for k in ks}


def _final(rd: Path) -> np.ndarray:
    return np.load(rd / f"x_{core.TOTAL_STEPS:02d}.npy", allow_pickle=False)


def _video(rd: Path) -> np.ndarray | None:
    p = rd / "video.npy"
    return np.load(p, allow_pickle=False) if p.exists() else None


def _pair(doc_x: dict[str, Any], rd_x: Path, doc_ref: dict[str, Any], rd_ref: Path) -> dict[str, Any]:
    """Final-state comparison of run x against reference run (both from artifacts)."""
    from experiments.video_runtime_state_discovery import video_metrics

    bit = doc_x["final_sha256"] == doc_ref["final_sha256"]
    rel = 0.0 if bit else core.rel_l2(_final(rd_x), _final(rd_ref))
    vx, vr = _video(rd_x), _video(rd_ref)
    video_bit = doc_x["video_sha256"] == doc_ref["video_sha256"]
    ssim = 1.0 if video_bit else (video_metrics(vx, vr)["frame_ssim_mean"] if vx is not None and vr is not None else None)
    ks = sorted(set(doc_x["boundary_sha256"]) & set(doc_ref["boundary_sha256"]), key=int)
    k_star = next((int(k) for k in ks if doc_x["boundary_sha256"][k] != doc_ref["boundary_sha256"][k]), None)
    all_boundaries = len(ks) == core.N_BOUNDARIES
    traj_bit = (k_star is None and video_bit) if all_boundaries else None
    out = core.compare_final(rel=rel, ssim=ssim, bit_exact=bit and video_bit, trajectory_bit_exact=traj_bit)
    out["latent_bit_exact"], out["video_bit_exact"], out["k_star"], out["boundaries_compared"] = bit, video_bit, k_star, len(ks)
    return out


def _curve(doc_x: dict[str, Any], rd_x: Path, doc_ref: dict[str, Any], rd_ref: Path) -> dict[str, Any] | None:
    ax, ar = _load_all_boundaries(rd_x, doc_x), _load_all_boundaries(rd_ref, doc_ref)
    if ax is None or ar is None:
        return None
    ks = sorted(set(ax) & set(ar))
    rel = {k: core.rel_l2(ax[k], ar[k]) for k in ks}
    mat = {k: core.material_fraction(ax[k], ar[k]) for k in ks}
    k_star = next((k for k in ks if doc_x["boundary_sha256"][str(k)] != doc_ref["boundary_sha256"][str(k)]), None)
    k_mat = next((k for k in ks if mat[k] > core.MATERIAL_FRACTION_STEP_THRESHOLD), None)
    return {"boundaries": ks, "rel_l2_by_k": rel, "material_fraction_by_k": mat, "k_star": k_star, "k_material": k_mat}


def phase_analyze(config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    provenance = build_provenance(config_path)
    prereg = require_sealed(output_dir, provenance)
    if (output_dir / "summary.json").exists():
        raise GateError("summary.json already exists; the frozen analysis runs exactly once")
    H = provenance["provenance_hash"]
    plan = plan_for(config)
    trajs = [t["id"] for t in trajectories(config)]
    docs: dict[str, dict[str, Any]] = {}
    for p in plan:
        d = _load_run(output_dir, PHASE_DIR[p["phase"]], p["label"], H)
        if d is not None:
            docs[p["label"]] = d
    rd = {lab: _run_dir(output_dir, PHASE_DIR[next(p for p in plan if p["label"] == lab)["phase"]], lab) for lab in docs}
    gate_errors: list[str] = []
    fp_fail = [f"{lab}: {d['fingerprint_mismatches']}" for lab, d in docs.items() if d.get("fingerprint_mismatches")]
    env_fail = core.environment_consistency({lab: d["fingerprint"] for lab, d in docs.items()})

    def have(lab: str) -> bool:
        return lab in docs

    # canonical
    missing_canon = [f"{tr}_CANONICAL_F{r}" for tr in trajs for r in (1, 2) if not have(f"{tr}_CANONICAL_F{r}")] + [f"{tr}_CANONICAL_H12" for tr in trajs if not have(f"{tr}_CANONICAL_H12")]
    if missing_canon:
        raise GateError(f"Canonical runs missing: {missing_canon}")
    canon_within = core.within_class({tr: _pair(docs[f"{tr}_CANONICAL_F2"], rd[f"{tr}_CANONICAL_F2"], docs[f"{tr}_CANONICAL_F1"], rd[f"{tr}_CANONICAL_F1"]) for tr in trajs})
    exact_resume = {tr: docs[f"{tr}_CANONICAL_H12"]["final_sha256"] == docs[f"{tr}_CANONICAL_F1"]["final_sha256"] and docs[f"{tr}_CANONICAL_H12"]["video_sha256"] == docs[f"{tr}_CANONICAL_F1"]["video_sha256"] for tr in trajs}

    axes_out: dict[str, Any] = {}
    x0_by_axis: dict[str, dict[str, bool]] = {}
    plumbing_by_axis: dict[str, dict[str, bool]] = {}
    for ax in core.CORE_AXES:
        labels = [f"{tr}_{ax}_{s}" for tr in trajs for s in ("F1", "F2", "H0", "H12", "H20", "H28")]
        if not all(have(l) for l in labels):
            axes_out[ax] = {"measured": False, "missing": [l for l in labels if not have(l)]}
            continue
        x0_by_axis[ax] = {tr: docs[f"{tr}_{ax}_F1"]["boundary_sha256"]["0"] == docs[f"{tr}_CANONICAL_F1"]["boundary_sha256"]["0"] for tr in trajs}
        plumbing_by_axis[ax] = {tr: docs[f"{tr}_{ax}_H0"]["final_sha256"] == docs[f"{tr}_{ax}_F1"]["final_sha256"] and docs[f"{tr}_{ax}_H0"]["video_sha256"] == docs[f"{tr}_{ax}_F1"]["video_sha256"] for tr in trajs}
        within_pairs = {tr: _pair(docs[f"{tr}_{ax}_F2"], rd[f"{tr}_{ax}_F2"], docs[f"{tr}_{ax}_F1"], rd[f"{tr}_{ax}_F1"]) for tr in trajs}
        full_pairs = {tr: _pair(docs[f"{tr}_{ax}_F1"], rd[f"{tr}_{ax}_F1"], docs[f"{tr}_CANONICAL_F1"], rd[f"{tr}_CANONICAL_F1"]) for tr in trajs}
        hyb_vs_a = {t: {tr: _pair(docs[f"{tr}_{ax}_H{t}"], rd[f"{tr}_{ax}_H{t}"], docs[f"{tr}_CANONICAL_F1"], rd[f"{tr}_CANONICAL_F1"]) for tr in trajs} for t in core.CHECKPOINTS_T}
        hyb_vs_x = {t: {tr: _pair(docs[f"{tr}_{ax}_H{t}"], rd[f"{tr}_{ax}_H{t}"], docs[f"{tr}_{ax}_F1"], rd[f"{tr}_{ax}_F1"]) for tr in trajs} for t in core.CHECKPOINTS_T}
        curves = {tr: _curve(docs[f"{tr}_{ax}_F1"], rd[f"{tr}_{ax}_F1"], docs[f"{tr}_CANONICAL_F1"], rd[f"{tr}_CANONICAL_F1"]) for tr in trajs}
        h1 = {tr: (core.amplification_h1([c["rel_l2_by_k"][k] for k in range(core.N_BOUNDARIES)], c["k_star"]) if c and len(c["boundaries"]) == core.N_BOUNDARIES else None) for tr, c in curves.items()}
        fresh_lab = f"{fresh_trajectory(config)}_{ax}_FRESH"
        fresh = _pair(docs[fresh_lab], rd[fresh_lab], docs[f"{fresh_trajectory(config)}_{ax}_F1"], rd[f"{fresh_trajectory(config)}_{ax}_F1"]) if have(fresh_lab) else None
        within, full = core.within_class(within_pairs), core.full_relation(full_pairs)
        axes_out[ax] = {"measured": True, "within": within, "full": full, "hybrid_t12": core.hybrid_class(hyb_vs_a[core.HYBRID_DECISION_T], hyb_vs_x[core.HYBRID_DECISION_T]),
                        "hybrid_by_t": {str(t): core.hybrid_class(hyb_vs_a[t], hyb_vs_x[t]) for t in core.CHECKPOINTS_T}, "t_portable": core.t_portable(hyb_vs_a),
                        "in_decision_set": core.axis_in_decision_set(within, full), "cross_engine": core.cross_engine_class(fresh),
                        "within_pairs": within_pairs, "full_pairs": full_pairs, "hybrid_vs_canonical": {str(t): v for t, v in hyb_vs_a.items()}, "hybrid_vs_axis": {str(t): v for t, v in hyb_vs_x.items()},
                        "curves_vs_canonical": {tr: (None if c is None else {"k_star": c["k_star"], "k_material": c["k_material"], "rel_l2_by_k": {str(k): v for k, v in c["rel_l2_by_k"].items()}, "material_fraction_by_k": {str(k): v for k, v in c["material_fraction_by_k"].items()}}) for tr, c in curves.items()},
                        "h1_amplification": h1, "h3_k_star_is_1": {tr: (c["k_star"] == 1 if c else None) for tr, c in curves.items()}, "fresh_engine_pair": fresh,
                        "fingerprints": {tr: docs[f"{tr}_{ax}_F1"]["fingerprint"] for tr in trajs}}
    n_core = sum(1 for a in axes_out.values() if a.get("measured"))
    invalid = core.anchor_failures(x0_bit_exact_by_axis=x0_by_axis, canonical_within=canon_within, exact_resume_bit_exact=exact_resume, resume_plumbing_bit_exact_by_axis=plumbing_by_axis,
                                   fingerprint_failures=fp_fail + [f"environment: {e}" for e in env_fail], n_core_measured=n_core, gate_errors=gate_errors)
    decision = core.decide(invalid_reasons=invalid, axes={ax: {k: a.get(k) for k in ("measured", "within", "full", "hybrid_t12")} for ax, a in axes_out.items() if a.get("measured")} if not invalid else {})
    canon_fresh_lab = f"{fresh_trajectory(config)}_CANONICAL_FRESH"
    canon_fresh = _pair(docs[canon_fresh_lab], rd[canon_fresh_lab], docs[f"{fresh_trajectory(config)}_CANONICAL_F1"], rd[f"{fresh_trajectory(config)}_CANONICAL_F1"]) if have(canon_fresh_lab) else None
    motivation = motivation_reference_hashes()
    cross_session = {tr: (None if motivation["final_sha256_by_traj"].get(tr) is None else docs[f"{tr}_CANONICAL_F1"]["final_sha256"] == motivation["final_sha256_by_traj"][tr]) for tr in trajs}
    h4 = {ax: {tr: a["full_pairs"][tr]["rel_l2"] for tr in trajs} for ax, a in axes_out.items() if a.get("measured")}
    summary = {"experiment_version": EXPERIMENT_VERSION, "provenance_hash": H, "preregistration_sha256": (output_dir / "preregistration.sha256").read_text().split()[0],
               "decision": decision, "invalid_reasons": invalid, "canonical": {"within": canon_within, "exact_resume_anchor_bit_exact": exact_resume, "cross_engine": core.cross_engine_class(canon_fresh),
                                                                                "fresh_engine_pair": canon_fresh, "cross_session_vs_rev4_1": cross_session, "fingerprint": docs[f"{trajs[0]}_CANONICAL_F1"]["fingerprint"]},
               "axes": axes_out, "not_measured_axes": core.NOT_MEASURED_AXES, "x0_bit_exact_by_axis": x0_by_axis, "resume_plumbing_by_axis": plumbing_by_axis,
               "hypotheses": {"H1": {ax: a.get("h1_amplification") for ax, a in axes_out.items() if a.get("measured")}, "H2_t_portable": {ax: a.get("t_portable") for ax, a in axes_out.items() if a.get("measured")},
                              "H3": {ax: a.get("h3_k_star_is_1") for ax, a in axes_out.items() if a.get("measured")}, "H4_final_rel_l2_by_traj": h4,
                              "H5_cross_engine": {**{ax: a.get("cross_engine") for ax, a in axes_out.items() if a.get("measured")}, "CANONICAL": core.cross_engine_class(canon_fresh)}},
               "compatibility_matrix": {ax: {k: a.get(k) for k in ("within", "full", "hybrid_by_t", "t_portable", "in_decision_set", "cross_engine")} for ax, a in axes_out.items() if a.get("measured")},
               "runs_present": sorted(docs), "runs_missing": sorted(set(p["label"] for p in plan) - set(docs)), "kill_scope": config["kill_scope"], "claim_boundary": config["claim_boundary"]}
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "decision.json", {"decision": decision["decision"], "rationale": decision["rationale"], "provenance_hash": H, "preregistration_sha256": summary["preregistration_sha256"]})
    write_report(output_dir, summary)
    _write_manifest(output_dir, provenance, {"phase": "analyze", "decision": decision["decision"]})
    return {"phase": "analyze", "decision": decision["decision"], "rationale": decision["rationale"], "D": decision["decision_set_D"], "invalid_reasons": invalid}


def write_report(output_dir: Path, s: dict[str, Any]) -> None:
    lines = [f"# Numerical Compatibility Round 0 ({s['experiment_version']})", "", f"Decision: **{s['decision']['decision']}**", "", s["decision"]["rationale"], ""]
    if s["invalid_reasons"]:
        lines += ["INVALID reasons:"] + [f"- {r}" for r in s["invalid_reasons"]] + [""]
    lines += ["| axis | within | full relation | hybrid t=12 / 20 / 28 | t_portable | in D | cross-engine |", "|---|---|---|---|---|---|---|"]
    for ax, m in s["compatibility_matrix"].items():
        hb = m["hybrid_by_t"] or {}
        lines.append(f"| {ax} | {m['within']} | {m['full']} | {hb.get('12')} / {hb.get('20')} / {hb.get('28')} | {m['t_portable']} | {m['in_decision_set']} | {m['cross_engine']} |")
    lines += ["", f"Not measured: {s['not_measured_axes']}", "", f"Canonical: {s['canonical']['within']}, exact-resume anchor {s['canonical']['exact_resume_anchor_bit_exact']}, cross-engine {s['canonical']['cross_engine']}, cross-session vs rev4.1 {s['canonical']['cross_session_vs_rev4_1']}", "",
              f"Kill scope: {s['kill_scope']}", "", f"Claim boundary: {s['claim_boundary']}", ""]
    (output_dir / "numerical_compatibility_round0.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--axis", default=None, help="configuration for --phase axis / freshcheck")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out = args.output_dir.resolve()
    validate_output_path(out)
    if args.phase == "cpu":
        result = phase_cpu(config)
    elif args.phase == "preregister":
        result = phase_preregister(config, args.config, out)
    elif args.phase == "analyze":
        result = phase_analyze(config, args.config, out)
    else:
        if args.phase in ("axis", "freshcheck") and not args.axis:
            raise GateError(f"--phase {args.phase} requires --axis")
        result = {"canonical": phase_canonical, "axis": phase_axis, "freshcheck": phase_freshcheck}[args.phase](config, args.config, out, args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
