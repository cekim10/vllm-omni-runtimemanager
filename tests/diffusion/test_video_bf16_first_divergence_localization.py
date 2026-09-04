"""CPU contracts for the isolated BF16 first-divergence localization trace."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments import video_bf16_first_divergence_localization as killtest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/video_bf16_first_divergence_localization_config.yaml")


def _config():
    return killtest.load_config(CONFIG_PATH)


_TRUSTED_ARRAY_CACHE: dict = {}


def _trusted_arrays(config, manifest):
    """Actual trusted final arrays (CLEAN from the v3 trajectory, PLUS1/HIST from preserved artifacts)."""
    if "arrays" not in _TRUSTED_ARRAY_CACHE:
        source = killtest.derive_anchor(config)["source"]
        trusted = manifest["trusted_final_identities"]
        arrays = {"CLEAN": (source.final_latent, source.video)}
        for name in ("PLUS1", "HISTORICAL_PLUS14"):
            arrays[name] = (
                np.load(killtest.REPO_ROOT / trusted[name]["final_latent"]["artifact_relative_path"], allow_pickle=False),
                np.load(killtest.REPO_ROOT / trusted[name]["video"]["artifact_relative_path"], allow_pickle=False),
            )
        _TRUSTED_ARRAY_CACHE["arrays"] = arrays
    return _TRUSTED_ARRAY_CACHE["arrays"]


def _valid_preflight(root, manifest, provenance, *, config=None, with_artifacts=True):
    """A preflight document whose controls point at persisted repeat artifacts equal to the trusted finals."""
    controls = {}
    if with_artifacts:
        arrays = _trusted_arrays(config or _config(), manifest)
        for name in killtest.TRAJECTORIES:
            latent, video = arrays[name]
            latent_record = killtest.save_tensor(root, f"preflight/{name}/repeat_0/recovered_final_latent.npy", latent)
            video_record = killtest.save_tensor(root, f"preflight/{name}/repeat_0/recovered_video.npy", video, runtime_semantics="uint8 decoded video")
            controls[name] = [
                {"repeat_id": repeat, "final_latent_identity": latent_record["canonical_identity"], "video_identity": video_record["canonical_identity"],
                 "final_latent_artifact": latent_record, "video_artifact": video_record, "exact_vs_clean": name == "CLEAN"}
                for repeat in range(3)
            ]
    gates = [killtest.gate(name, True, "synthetic validated evidence", required=True) for name in killtest.PREFLIGHT_REQUIRED_GATES]
    document = {
        "gpu_all_passed": True,
        "gates": gates,
        "controls": controls,
        "provenance_hash": provenance["provenance_hash"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    killtest.atomic_json(root / "preflight/preflight_gates.json", document)
    return document


def _trace_root(tmp_path, *, transient=False, final_only=False, never=False):
    root = tmp_path / "trace"
    boundaries = ["input", "after_step_001", "after_step_002"]
    arrays = {
        "CLEAN": [np.zeros((1, 2), dtype=np.float32)] * 3,
        "PLUS1": [np.array([[1, 0]], dtype=np.float32), np.array([[2, 0]], dtype=np.float32), np.array([[3, 0]], dtype=np.float32)],
        "HISTORICAL_PLUS14": [np.array([[4, 0]], dtype=np.float32), np.array([[5, 0]], dtype=np.float32), np.array([[3, 0]], dtype=np.float32)],
    }
    if not transient and not final_only and not never:
        arrays["HISTORICAL_PLUS14"] = [np.array([[4, 0]], dtype=np.float32), np.array([[2, 0]], dtype=np.float32), np.array([[3, 0]], dtype=np.float32)]
    if transient:
        arrays["HISTORICAL_PLUS14"] = [np.array([[1, 0]], dtype=np.float32), np.array([[5, 0]], dtype=np.float32), np.array([[3, 0]], dtype=np.float32)]
    if final_only:
        arrays["HISTORICAL_PLUS14"] = [np.array([[4, 0]], dtype=np.float32), np.array([[5, 0]], dtype=np.float32), np.array([[3, 0]], dtype=np.float32)]
    if never:
        arrays["HISTORICAL_PLUS14"] = [np.array([[4, 0]], dtype=np.float32), np.array([[5, 0]], dtype=np.float32), np.array([[6, 0]], dtype=np.float32)]
    traces = {}
    for name, values in arrays.items():
        rows = []
        for boundary, value in zip(boundaries, values, strict=True):
            rows.append({"boundary": boundary, "artifact": killtest.save_tensor(root, f"phase1/{name}/{boundary}.npy", value)})
        traces[name] = rows
    return root, {"expected_boundaries": boundaries, "traces": traces}


def _synthetic_phase2(root: Path):
    config = _config()
    provenance = {"provenance_hash": "p2-provenance", "source_dirty_entries": []}
    specs = killtest.boundary_specifications(config, 10)
    phase1_values = {
        "CLEAN": (np.array([[0.0, 0.0]], dtype=np.float32), np.array([[0.0, 0.0]], dtype=np.float32)),
        "PLUS1": (np.array([[1.0, 0.0]], dtype=np.float32), np.array([[4.0, 0.0]], dtype=np.float32)),
        "HISTORICAL_PLUS14": (np.array([[2.0, 0.0]], dtype=np.float32), np.array([[4.0, 0.0]], dtype=np.float32)),
    }
    trusted_trajectories = {}
    for name, (entry, exit_state) in phase1_values.items():
        trusted_trajectories[name] = {
            "entry_artifact": killtest.save_tensor(root, f"phase1/artifacts/{name}/input.npy", entry),
            "entry_identity": killtest.identity(entry),
            "exit_artifact": killtest.save_tensor(root, f"phase1/artifacts/{name}/after_step_001.npy", exit_state),
            "exit_identity": killtest.identity(exit_state),
        }
    phase1_trace = {"synthetic": True}
    phase1_analysis = {"outcome": "EARLY_EXACT_MERGE"}
    killtest.atomic_json(root / "phase1/trace_manifest.json", phase1_trace)
    killtest.atomic_json(root / "phase1/phase1_analysis.json", phase1_analysis)

    final_latents = {
        "CLEAN": np.array([[0.0, 0.0]], dtype=np.float32),
        "PLUS1": np.array([[5.0, 0.0]], dtype=np.float32),
        "HISTORICAL_PLUS14": np.array([[5.0, 0.0]], dtype=np.float32),
    }
    final_videos = {
        "CLEAN": np.zeros((1, 1, 1, 3), dtype=np.uint8),
        "PLUS1": np.ones((1, 1, 1, 3), dtype=np.uint8),
        "HISTORICAL_PLUS14": np.ones((1, 1, 1, 3), dtype=np.uint8),
    }
    manifest = {
        "manifest_sha256": "p2-manifest",
        "anchor": {"checkpoint_step": 10},
        "boundary_specifications": specs,
        "phase2_freeze": killtest.phase2_freeze(config),
        "trusted_phase1": {
            "trace_manifest_sha256": killtest.sha256_file(root / "phase1/trace_manifest.json"),
            "phase1_analysis_sha256": killtest.sha256_file(root / "phase1/phase1_analysis.json"),
            "trajectories": trusted_trajectories,
        },
        "trajectories": {
            name: {"canonical_identity": killtest.identity(values[0])}
            for name, values in phase1_values.items()
        },
        "trusted_final_identities": {
            "CLEAN": {
                "final_latent_identity": killtest.identity(final_latents["CLEAN"]),
                "video_identity": killtest.identity(final_videos["CLEAN"]),
            },
            "PLUS1": {
                "final_latent": {"canonical_identity": killtest.identity(final_latents["PLUS1"])},
                "video": {"canonical_identity": killtest.identity(final_videos["PLUS1"])},
            },
            "HISTORICAL_PLUS14": {
                "final_latent": {"canonical_identity": killtest.identity(final_latents["HISTORICAL_PLUS14"])},
                "video": {"canonical_identity": killtest.identity(final_videos["HISTORICAL_PLUS14"])},
            },
        },
    }
    values = {
        "CLEAN": [phase1_values["CLEAN"][0], np.array([[0.0, 0.0]], dtype=np.float32), np.array([[0.0, 0.0]], dtype=np.float32), phase1_values["CLEAN"][0], phase1_values["CLEAN"][1]],
        "PLUS1": [phase1_values["PLUS1"][0], np.array([[1.0, 0.0]], dtype=np.float32), np.array([[3.0, 0.0]], dtype=np.float32), phase1_values["PLUS1"][0], phase1_values["PLUS1"][1]],
        "HISTORICAL_PLUS14": [phase1_values["HISTORICAL_PLUS14"][0], np.array([[2.0, 0.0]], dtype=np.float32), np.array([[3.0, 0.0]], dtype=np.float32), phase1_values["HISTORICAL_PLUS14"][0], phase1_values["HISTORICAL_PLUS14"][1]],
    }
    timestep = killtest.single_flip.scheduler_timesteps_numpy(config)[10]
    traces = {}
    for name in killtest.TRAJECTORIES:
        rows = []
        for boundary, value in zip(killtest.PHASE2_AVAILABLE_BOUNDARIES, values[name], strict=True):
            rows.append({
                "boundary": boundary,
                "absolute_step": 10,
                "timestep": timestep,
                "phase1_entry_boundary": "input",
                "runtime_dtype": killtest._phase2_runtime_dtype(boundary),
                "storage_dtype": "<f4",
                "actual_shape": list(value.shape),
                **killtest.PHASE2_BOUNDARY_SEMANTICS[boundary],
                "artifact": killtest.save_tensor(root, f"phase2/artifacts/{name}/{boundary}.npy", value),
            })
        traces[name] = rows
    mapping = killtest.phase2_selection_mapping(config, manifest, 10)
    trace = {
        "provenance_hash": provenance["provenance_hash"],
        "manifest_sha256": manifest["manifest_sha256"],
        "selected_step": 10,
        "selection_mapping": mapping,
        "expected_boundaries": list(killtest.PHASE2_AVAILABLE_BOUNDARIES),
        "boundary_semantics": killtest.PHASE2_BOUNDARY_SEMANTICS,
        "traces": traces,
        "final_latents": {},
        "final_videos": {},
        "unavailable_boundaries": list(killtest.PHASE2_UNAVAILABLE_BOUNDARIES),
    }
    for name in killtest.TRAJECTORIES:
        trace["final_latents"][name] = killtest.save_tensor(root, f"phase2/artifacts/{name}/final_latent.npy", final_latents[name])
        trace["final_videos"][name] = killtest.save_tensor(root, f"phase2/artifacts/{name}/final_video.npy", final_videos[name], runtime_semantics="uint8 decoded video")
    return config, manifest, provenance, trace


def test_cpu_manifest_reconstructs_frozen_anchor_and_scoped_gates(tmp_path):
    config = _config(); config["output_root"] = "localization"
    result = killtest.run_cpu(config, CONFIG_PATH, tmp_path / "localization")
    assert result["boundary_count"] == 31 and result["cpu_all_passed"] and not result["gpu_all_passed"]
    root = tmp_path / "localization"
    manifest = json.loads((root / "anchor_manifest.json").read_text())
    historical = manifest["trajectories"]["HISTORICAL_PLUS14"]["historical_delta"]
    critical = next(row for row in historical["changed_coordinates"] if row["coordinate_flat_index"] == 516515)
    assert historical["changed_coordinate_count"] == 6
    assert critical["adjacent_steps"] == 14 and critical["direction"] == "up"
    gates = json.loads((root / "cpu_gates.json").read_text())
    assert gates["cpu_all_passed"] and not gates["gpu_all_passed"]
    specs = manifest["boundary_specifications"]
    assert specs[0] == {
        "boundary": "input",
        "resumed_update_index": None,
        "absolute_diffusion_step_index": 10,
        "scheduler_timestep": 972.9729614257812,
        "expected_storage_dtype": "<f4",
        "expected_runtime_dtype": "torch.float32",
        "runtime_dtype_semantics": "float32 resume input containing BF16-exact values",
    }
    assert specs[1]["resumed_update_index"] == 0
    assert specs[1]["absolute_diffusion_step_index"] == 11
    assert specs[1]["expected_runtime_dtype"] == "torch.bfloat16"
    assert specs[-1]["absolute_diffusion_step_index"] == 40
    assert manifest["early_late_cutoff"]["first_late_boundary"] == "after_step_023"
    assert not (root / "trajectory_manifest.json").exists()


def test_anchor_manifest_rejects_extra_plus1_coordinate(tmp_path):
    config = _config(); data = killtest.derive_anchor(config)
    bits = killtest.single_flip.float32_to_bf16_bits(data["plus1"]).reshape(-1).copy()
    bits[1] = killtest.single_flip.adjacent_bf16_bits(int(bits[1]), "up")
    tampered = killtest.single_flip.bf16_bits_to_float32(bits).reshape(data["plus1"].shape)
    changed = np.count_nonzero(
        killtest.single_flip.float32_to_bf16_bits(tampered) != killtest.single_flip.float32_to_bf16_bits(data["clean"])
    )
    assert changed != 1


def test_plus1_construction_is_bound_to_the_frozen_adjacent_value():
    data = killtest.derive_anchor(_config())
    flat = 516515
    clean_bits = killtest.single_flip.float32_to_bf16_bits(data["clean"]).reshape(-1)
    plus_bits = killtest.single_flip.float32_to_bf16_bits(data["plus1"]).reshape(-1)
    assert int(plus_bits[flat]) == killtest.single_flip.adjacent_bf16_bits(int(clean_bits[flat]), "up")
    assert killtest.single_flip.adjacent_bf16_bits(int(clean_bits[flat]), "down") != int(plus_bits[flat])


@pytest.mark.parametrize("mutate", [
    lambda m: m["anchor"].__setitem__("resume_timestep", -1.0),
    lambda m: m["anchor"].__setitem__("scheduler_class", "UniPC"),
    lambda m: m["critical_coordinate"].__setitem__("flat_index", 7),
])
def test_frozen_anchor_identity_mutations_are_detectable(tmp_path, mutate):
    config = _config(); config["output_root"] = "localization"
    killtest.run_cpu(config, CONFIG_PATH, tmp_path / "localization")
    manifest_path = tmp_path / "localization/anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text()); mutate(manifest); manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(killtest.GlobalStopError):
        killtest.require_cpu(tmp_path / "localization", CONFIG_PATH, config)


def test_self_consistent_manifest_edit_is_rejected_against_rederived_anchor(tmp_path):
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"
    killtest.run_cpu(config, CONFIG_PATH, root)
    manifest_path = root / "anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["anchor"]["prompt_id"] = "forged"
    unhashed = dict(manifest); unhashed.pop("manifest_sha256")
    manifest["manifest_sha256"] = killtest.sha256_bytes(killtest.canonical_json(unhashed))
    manifest_path.write_text(json.dumps(manifest))
    gates_path = root / "cpu_gates.json"
    gates = json.loads(gates_path.read_text()); gates["manifest_sha256"] = manifest["manifest_sha256"]
    gates_path.write_text(json.dumps(gates))
    with pytest.raises(killtest.GlobalStopError, match="independently derived"):
        killtest.require_cpu(root, CONFIG_PATH, config)


def test_boundary_mapping_and_cutoff_are_rederived_not_trusted(tmp_path):
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"; killtest.run_cpu(config, CONFIG_PATH, root)
    manifest_path = root / "anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["boundary_specifications"][1]["absolute_diffusion_step_index"] = 12
    manifest["early_late_cutoff"]["first_late_boundary_index"] = 22
    unhashed = dict(manifest); unhashed.pop("manifest_sha256")
    manifest["manifest_sha256"] = killtest.sha256_bytes(killtest.canonical_json(unhashed))
    manifest_path.write_text(json.dumps(manifest))
    gates_path = root / "cpu_gates.json"
    gates = json.loads(gates_path.read_text()); gates["manifest_sha256"] = manifest["manifest_sha256"]
    gates_path.write_text(json.dumps(gates))
    with pytest.raises(killtest.GlobalStopError, match="boundary mapping"):
        killtest.require_cpu(root, CONFIG_PATH, config)


@pytest.mark.parametrize("kind,expected,event", [
    ("persistent", "EARLY_EXACT_MERGE", "PERSISTENT_EXACT_MERGE"),
    ("transient", "TRACE_INVALID", "TRANSIENT_EXACT_MATCH"),
    ("final_only", "FINAL_ONLY_MATCH", "FINAL_ONLY_MATCH"),
    ("never", "NO_EXACT_MERGE", "NO_EXACT_MERGE"),
])
def test_trace_outcomes_are_threshold_free_and_row_order_invariant(tmp_path, kind, expected, event):
    root, trace = _trace_root(tmp_path, transient=kind == "transient", final_only=kind == "final_only", never=kind == "never")
    result = killtest.analyze_trace(root, trace)
    assert result["outcome"] == expected and result["plus1_historical_event"]["classification"] == event
    assert {row["pair"] for row in result["pairwise_rows"]} == {"CLEAN_VS_PLUS1", "CLEAN_VS_HISTORICAL_PLUS14", "PLUS1_VS_HISTORICAL_PLUS14"}
    assert all("mse" in row and "relative_l2" in row for row in result["pairwise_rows"])
    shuffled = copy.deepcopy(trace)
    for rows in shuffled["traces"].values():
        rows.reverse()
    with pytest.raises(killtest.GlobalStopError):
        killtest.analyze_trace(root, shuffled)


def test_missing_duplicate_and_stale_artifacts_fail_closed(tmp_path):
    root, trace = _trace_root(tmp_path)
    missing = copy.deepcopy(trace); missing["traces"]["CLEAN"].pop()
    with pytest.raises(killtest.GlobalStopError): killtest.analyze_trace(root, missing)
    duplicate = copy.deepcopy(trace); duplicate["traces"]["CLEAN"].append(copy.deepcopy(duplicate["traces"]["CLEAN"][-1]))
    with pytest.raises(killtest.GlobalStopError): killtest.analyze_trace(root, duplicate)
    path = root / trace["traces"]["CLEAN"][0]["artifact"]["relative_path"]
    path.write_bytes(b"stale")
    with pytest.raises(killtest.GlobalStopError): killtest.analyze_trace(root, trace)


def test_artifact_identity_covers_shape_and_metrics_ignore_row_claims(tmp_path):
    root, trace = _trace_root(tmp_path)
    artifact = trace["traces"]["PLUS1"][0]["artifact"]
    tensor = killtest.load_tensor(root, artifact)
    reshaped = tensor.reshape(2, 1)
    assert killtest.identity(tensor) != killtest.identity(reshaped)
    # Analysis has no row metric input: an attacker cannot influence a
    # result by adding a claimed bit-exact/MSE field.
    trace["traces"]["PLUS1"][0]["bit_exact"] = True
    trace["traces"]["PLUS1"][0]["mse"] = 0.0
    result = killtest.analyze_trace(root, trace)
    row = next(item for item in result["pairwise_rows"] if item["boundary"] == "input" and item["pair"] == "CLEAN_VS_PLUS1")
    assert row["bit_exact"] is False and row["mse"] > 0


def test_trace_boundary_mapping_mutation_is_rejected(tmp_path):
    root, trace = _trace_root(tmp_path)
    trace["boundary_specifications"] = [
        {"boundary": name, "resumed_update_index": index - 1 if index else None, "absolute_diffusion_step_index": 10 + index, "scheduler_timestep": float(100 - index)}
        for index, name in enumerate(trace["expected_boundaries"])
    ]
    for trajectory in killtest.TRAJECTORIES:
        for row, spec in zip(trace["traces"][trajectory], trace["boundary_specifications"], strict=True):
            row.update(spec)
    trace["traces"]["PLUS1"][1]["absolute_diffusion_step_index"] = 99
    with pytest.raises(killtest.GlobalStopError, match="boundary mapping"):
        killtest.analyze_trace(root, trace)


def test_traced_final_mismatch_is_trace_alters_execution(tmp_path):
    root = tmp_path / "trace"
    clean = np.zeros((1, 2), dtype=np.float32)
    plus = np.ones((1, 2), dtype=np.float32)
    historical = np.full((1, 2), 2, dtype=np.float32)
    videos = {"CLEAN": np.zeros((1, 1, 1, 3), dtype=np.uint8), "PLUS1": np.ones((1, 1, 1, 3), dtype=np.uint8), "HISTORICAL_PLUS14": np.full((1, 1, 1, 3), 2, dtype=np.uint8)}
    latents = {"CLEAN": clean, "PLUS1": plus, "HISTORICAL_PLUS14": historical}
    trace = {"final_latents": {}, "final_videos": {}}
    trusted = {}
    for name in killtest.TRAJECTORIES:
        trace["final_latents"][name] = killtest.save_tensor(root, f"{name}/latent.npy", latents[name])
        trace["final_videos"][name] = killtest.save_tensor(root, f"{name}/video.npy", videos[name], runtime_semantics="uint8")
        if name == "CLEAN":
            trusted[name] = {"final_latent_identity": killtest.identity(latents[name]), "video_identity": killtest.identity(videos[name])}
        else:
            trusted[name] = {"final_latent": {"canonical_identity": killtest.identity(latents[name])}, "video": {"canonical_identity": killtest.identity(videos[name])}}
    manifest = {"trusted_final_identities": trusted}
    assert all(row["matches"] for row in killtest.validate_traced_finals(root, manifest, trace).values())
    tampered = np.full((1, 2), 9, dtype=np.float32)
    trace["final_latents"]["HISTORICAL_PLUS14"] = killtest.save_tensor(root, "HISTORICAL_PLUS14/tampered.npy", tampered)
    with pytest.raises(killtest.GlobalStopError, match="TRACE_ALTERS_EXECUTION"):
        killtest.validate_traced_finals(root, manifest, trace)
    assert json.loads((root / "phase1/trace_alters_execution.json").read_text())["classification"] == "TRACE_ALTERS_EXECUTION"


def test_relocated_relative_artifacts_analyze_without_original_root(tmp_path):
    root, trace = _trace_root(tmp_path / "original")
    copied = tmp_path / "copied"; shutil.copytree(root, copied)
    assert killtest.analyze_trace(copied, trace)["outcome"] == "EARLY_EXACT_MERGE"


def test_phase2_requires_frozen_step_through_production_path(tmp_path):
    config = _config(); config.pop("phase2")
    with pytest.raises(killtest.GlobalStopError, match="configuration is absent"):
        killtest.run_phase2(config, CONFIG_PATH, tmp_path, object())
    assert config["phase3"]["enabled"] is True
    assert config["phase3"]["phase4_enabled"] is False
    assert config["phase3"]["auto_expand"] is False


def test_phase2_selection_maps_absolute_step_to_frozen_phase1_boundary(tmp_path):
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"
    killtest.run_cpu(config, CONFIG_PATH, root)
    manifest, _ = killtest.require_cpu(root, CONFIG_PATH, config)
    mapping = killtest.phase2_selection_mapping(config, manifest, 10)
    assert mapping["selected_absolute_diffusion_step_index"] == 10
    assert mapping["selected_resumed_update_index"] == 0
    assert mapping["phase1_entry_boundary"] == "input"
    assert mapping["phase1_exit_boundary"] == "after_step_001"
    assert mapping["phase1_entry_boundary_specification"]["absolute_diffusion_step_index"] == 10
    assert mapping["selected_scheduler_timestep"] == killtest.single_flip.scheduler_timesteps_numpy(config)[10]


def test_phase2_invalid_absolute_step_fails_before_gpu_build(tmp_path, monkeypatch):
    config = _config()
    config["phase2"]["selected_step"] = 40
    monkeypatch.setattr(killtest, "_build_omni", lambda *_: pytest.fail("GPU builder must not be reached"))
    with pytest.raises(killtest.GlobalStopError, match="selected step or operation-boundary freeze"):
        killtest.run_phase2(config, CONFIG_PATH, tmp_path, object())


def test_cpu_gate_file_cannot_claim_untested_gpu_gates_passed(tmp_path):
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"
    killtest.run_cpu(config, CONFIG_PATH, root)
    gates_path = root / "cpu_gates.json"
    gates = json.loads(gates_path.read_text()); gates["cpu_all_passed"] = True
    target = next(row for row in gates["gates"] if row["name"] == killtest.CPU_REQUIRED_GATES[0])
    target["status"] = "NOT_TESTED"
    gates_path.write_text(json.dumps(gates))
    with pytest.raises(killtest.GlobalStopError, match="required gate"):
        killtest.require_cpu(root, CONFIG_PATH, config)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed"])
def test_preflight_gate_set_fails_closed_despite_forged_boolean(tmp_path, mutation):
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"; killtest.run_cpu(config, CONFIG_PATH, root)
    manifest, provenance = killtest.require_cpu(root, CONFIG_PATH, config)
    document = _valid_preflight(root, manifest, provenance, config=config)
    if mutation == "missing":
        document["gates"].pop()
    elif mutation == "duplicate":
        document["gates"].append(copy.deepcopy(document["gates"][0]))
    else:
        document["gates"][0]["status"] = "FAIL"
    document["gpu_all_passed"] = True
    killtest.atomic_json(root / "preflight/preflight_gates.json", document)
    with pytest.raises(killtest.GlobalStopError):
        killtest.require_preflight(root, manifest, provenance)


def test_dirty_source_provenance_blocks_gpu_preflight():
    with pytest.raises(killtest.GlobalStopError, match="committed source revision"):
        killtest.require_committed_source({"source_dirty_entries": [" M experiments/probe.py"]})
    killtest.require_committed_source({"source_dirty_entries": [], "dirty_entries": ["?? results/video_bf16_first_divergence_localization/x.json"]})


def test_provenance_freezes_execution_and_validation_sources():
    provenance = killtest.provenance(CONFIG_PATH, _config())
    assert provenance["git_commit"]
    assert provenance["config_sha256"] == killtest.sha256_file(CONFIG_PATH)
    assert provenance["experiment_script_sha256"] == provenance["files"]["experiments/video_bf16_first_divergence_localization.py"]
    assert provenance["pipeline_sha256"] == provenance["files"]["vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"]
    assert provenance["runner_sha256"] == provenance["files"]["experiments/run_video_bf16_first_divergence_localization_gpu0.sh"]
    assert provenance["test_sha256"] == provenance["files"]["tests/diffusion/test_video_bf16_first_divergence_localization.py"]


def test_trusted_historical_identity_comes_from_v3_not_plus1():
    trusted = killtest.derive_anchor(_config())["trusted_finals"]
    assert "video_runtime_state_discovery_v3_corrected" in trusted["HISTORICAL_PLUS14"]["final_latent"]["source_relative_path"]
    assert "video_bf16_single_flip_killtest" in trusted["PLUS1"]["final_latent"]["source_relative_path"]
    assert trusted["HISTORICAL_PLUS14"]["final_latent"]["source_file_sha256"] != trusted["PLUS1"]["final_latent"]["source_file_sha256"]


def test_storage_dtype_is_derived_from_persisted_artifact(tmp_path):
    record = killtest.save_tensor(tmp_path, "value.npy", np.ones((2,), dtype=np.float64), runtime_semantics="control")
    assert record["storage_dtype"] == "<f8"
    assert killtest.load_tensor(tmp_path, record).dtype == np.float64


def test_phase2_valid_artifacts_emit_all_required_gates_and_coarse_result(tmp_path):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    result = killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)
    assert result["plus1_historical_exact_merge"]["classification"] == "MERGED_AT_GUIDANCE_OUTPUT"
    assert [row["name"] for row in result["gates"]] == list(killtest.PHASE2_REQUIRED_GATES)
    assert all(row["status"] == "PASS" for row in result["gates"])
    assert result["unavailable_boundaries"] == ["transformer_raw_output"]


@pytest.mark.parametrize("field,value", [
    ("entry_phase1_boundary", "after_step_001"),
    ("exit_phase1_boundary", "after_step_002"),
])
def test_phase2_wrong_frozen_entry_or_exit_rejected(tmp_path, field, value):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    config["phase2"][field] = value
    with pytest.raises(killtest.GlobalStopError, match="operation-boundary freeze"):
        killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)


def test_phase2_wrong_scheduler_timestep_rejected(tmp_path):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    trace["traces"]["PLUS1"][0]["timestep"] += 1.0
    with pytest.raises(killtest.GlobalStopError, match="boundary metadata"):
        killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)


@pytest.mark.parametrize("trajectory", killtest.TRAJECTORIES)
def test_phase2_wrong_input_artifact_rejected(tmp_path, trajectory):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    row = next(item for item in trace["traces"][trajectory] if item["boundary"] == "latent_entering_step")
    wrong = np.array([[99.0, 0.0]], dtype=np.float32)
    row["artifact"] = killtest.save_tensor(tmp_path, f"wrong/{trajectory}.npy", wrong)
    row["actual_shape"] = list(wrong.shape)
    with pytest.raises(killtest.GlobalStopError, match="one changed coordinate|PHASE1_PHASE2_BOUNDARY_MISMATCH"):
        killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_phase2_boundary_key_set_fails_closed(tmp_path, mutation):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    rows = trace["traces"]["CLEAN"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[-1]["boundary"] = "invented_boundary"
    with pytest.raises(killtest.GlobalStopError, match="boundary set"):
        killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)


def test_phase2_mutated_artifact_with_unchanged_record_rejected(tmp_path):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    record = trace["traces"]["PLUS1"][1]["artifact"]
    np.save(tmp_path / record["relative_path"], np.array([[9.0, 0.0]], dtype=np.float32), allow_pickle=False)
    with pytest.raises(killtest.GlobalStopError, match="artifact file validation"):
        killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)


def test_phase2_declared_pairwise_metrics_and_descriptive_fields_are_ignored(tmp_path):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    baseline = killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, copy.deepcopy(trace))
    for rows in trace["traces"].values():
        for row in rows:
            row.update({"bit_exact": True, "differing_element_count": 0, "mse": 999.0, "largest_support_increase": -1})
    attacked = killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)
    assert attacked["pairwise_rows"] == baseline["pairwise_rows"]
    assert attacked["plus1_historical_exact_merge"] == baseline["plus1_historical_exact_merge"]
    assert attacked["largest_support_increase"] == baseline["largest_support_increase"]


@pytest.mark.parametrize("kind", ["shape", "dtype"])
def test_phase2_same_values_wrong_shape_or_dtype_rejected(tmp_path, kind):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    row = trace["traces"]["PLUS1"][1]
    value = np.array([[1.0, 0.0]], dtype=np.float32)
    wrong = value.reshape(2) if kind == "shape" else value.astype(np.float64)
    row["artifact"] = killtest.save_tensor(tmp_path, f"wrong/{kind}.npy", wrong)
    row["actual_shape"] = list(wrong.shape)
    row["storage_dtype"] = np.dtype(wrong.dtype).newbyteorder("<").str
    with pytest.raises(killtest.GlobalStopError, match="metadata changed|pairwise tensors differ"):
        killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)


def test_phase2_scheduler_output_must_match_phase1_and_plus_historical(tmp_path):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    row = next(item for item in trace["traces"]["HISTORICAL_PLUS14"] if item["boundary"] == "scheduler_output")
    wrong = np.array([[7.0, 0.0]], dtype=np.float32)
    row["artifact"] = killtest.save_tensor(tmp_path, "wrong/historical_scheduler_output.npy", wrong)
    row["actual_shape"] = list(wrong.shape)
    with pytest.raises(killtest.GlobalStopError, match="PHASE1_PHASE2_BOUNDARY_MISMATCH"):
        killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)


def test_phase2_plus_historical_nonexact_scheduler_output_is_contradiction():
    rows = [
        {
            "boundary": boundary,
            "pair": "PLUS1_VS_HISTORICAL_PLUS14",
            "bit_exact": boundary != "scheduler_output",
        }
        for boundary in killtest.PHASE2_AVAILABLE_BOUNDARIES
    ]
    with pytest.raises(killtest.GlobalStopError, match="CONTRADICTS_PHASE1"):
        killtest._phase2_merge_event(rows)


@pytest.mark.parametrize("trajectory", killtest.TRAJECTORIES)
@pytest.mark.parametrize("artifact_kind", ["final_latents", "final_videos"])
def test_phase2_traced_final_mismatch_is_global_stop(tmp_path, trajectory, artifact_kind):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    wrong = (
        np.array([[8.0, 0.0]], dtype=np.float32)
        if artifact_kind == "final_latents"
        else np.full((1, 1, 1, 3), 8, dtype=np.uint8)
    )
    trace[artifact_kind][trajectory] = killtest.save_tensor(
        tmp_path,
        f"wrong/{trajectory}_{artifact_kind}.npy",
        wrong,
        runtime_semantics=(
            "uint8 decoded video"
            if artifact_kind == "final_videos"
            else killtest.EXPECTED_RUNTIME_DTYPE
        ),
    )
    with pytest.raises(killtest.GlobalStopError, match="PHASE2_TRACE_ALTERS_EXECUTION"):
        killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)
    failure = json.loads((tmp_path / "phase2/phase2_trace_alters_execution.json").read_text())
    assert failure["classification"] == "PHASE2_TRACE_ALTERS_EXECUTION"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "not_tested"])
def test_phase2_gate_document_fails_closed(tmp_path, mutation):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    result = killtest.analyze_phase2_artifacts(
        tmp_path, config, manifest, provenance, trace
    )
    document = {
        "gates": copy.deepcopy(result["gates"]),
        "provenance_hash": provenance["provenance_hash"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    if mutation == "missing":
        document["gates"].pop()
    elif mutation == "duplicate":
        document["gates"].append(copy.deepcopy(document["gates"][0]))
    elif mutation == "failed":
        document["gates"][0]["status"] = "FAIL"
    else:
        document["gates"][0]["status"] = "NOT_TESTED"
    with pytest.raises(killtest.GlobalStopError, match="gate"):
        killtest.validate_gate_document(
            document,
            killtest.PHASE2_REQUIRED_GATES,
            provenance_hash=provenance["provenance_hash"],
            manifest_sha256=manifest["manifest_sha256"],
        )


def test_phase2_row_shuffle_and_relocation_preserve_analysis(tmp_path):
    original = tmp_path / "original"
    config, manifest, provenance, trace = _synthetic_phase2(original)
    baseline = killtest.analyze_phase2_artifacts(original, config, manifest, provenance, copy.deepcopy(trace))
    for rows in trace["traces"].values():
        rows.reverse()
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    relocated = killtest.analyze_phase2_artifacts(copied, config, manifest, provenance, trace)
    assert relocated["pairwise_rows"] == baseline["pairwise_rows"]
    assert relocated["plus1_historical_exact_merge"] == baseline["plus1_historical_exact_merge"]


def test_phase2_missing_relocated_artifact_rejected(tmp_path):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)
    record = trace["traces"]["CLEAN"][0]["artifact"]
    (tmp_path / record["relative_path"]).unlink()
    with pytest.raises(killtest.GlobalStopError, match="artifact file validation"):
        killtest.analyze_phase2_artifacts(tmp_path, config, manifest, provenance, trace)


def test_phase2_unavailable_boundary_not_fabricated_and_no_expansion_modes():
    config = _config()
    assert "transformer_raw_output" not in killtest.PHASE2_AVAILABLE_BOUNDARIES
    assert killtest.PHASE2_UNAVAILABLE_BOUNDARIES == ("transformer_raw_output",)
    assert config["phase3"]["enabled"] is True
    assert config["phase3"]["phase4_enabled"] is False
    assert config["phase3"]["auto_expand"] is False
    assert all("fp32" not in mode and "search" not in mode for mode in killtest.MODES)


# ----------------------------------------------------------------------------- hostile tests added for the corrective patch

TRUSTED_PROBE = Path("results/video_bf16_single_flip_killtest/smoke/rows/c0516515_up/replay_00/trajectory_probe")


def _specs():
    config = _config()
    return config, killtest.boundary_specifications(config, 10)


def test_probe_record_validation_matches_real_resumed_production_semantics():
    """Records from a trusted resumed GPU run: input float32 (BF16-exact), scheduler outputs bfloat16."""
    metadata_files = sorted(TRUSTED_PROBE.glob("*trajectory_probe.json")) if TRUSTED_PROBE.exists() else []
    if not metadata_files:
        pytest.skip("trusted resumed probe metadata not available locally")
    records = json.loads(metadata_files[0].read_text())["records"]
    by_step = {int(row["step_index"]): row for row in records}
    assert by_step[0]["runtime_dtype"] == "torch.float32" and by_step[30]["runtime_dtype"] == "torch.bfloat16"
    config, specs = _specs()
    # Build the full 31-record list from the real endpoint records and spec-consistent interior rows.
    synthetic = []
    for spec in specs:
        step = 0 if spec["boundary"] == "input" else int(spec["boundary"].split("_")[-1])
        base = by_step.get(step, {})
        synthetic.append({"step_index": step, "runtime_dtype": base.get("runtime_dtype", spec["expected_runtime_dtype"]), "timestep": base.get("timestep", spec["scheduler_timestep"]), "latent_path": base.get("latent_path", f"/x/{step}.pt")})
    assert synthetic[0]["timestep"] == specs[0]["scheduler_timestep"] == 972.9729614257812
    assert synthetic[-1]["timestep"] == specs[-1]["scheduler_timestep"]
    killtest.validate_probe_records(synthetic, specs)  # genuine semantics accepted
    # The old (wrong) requirement: bf16 at the input boundary must now be a rejected deviation.
    mutated = copy.deepcopy(synthetic); mutated[0]["runtime_dtype"] = "torch.bfloat16"
    with pytest.raises(killtest.GlobalStopError, match="input runtime dtype torch.bfloat16"):
        killtest.validate_probe_records(mutated, specs)
    mutated = copy.deepcopy(synthetic); mutated[5]["runtime_dtype"] = "torch.float32"
    with pytest.raises(killtest.GlobalStopError, match="after_step_005 runtime dtype torch.float32"):
        killtest.validate_probe_records(mutated, specs)
    mutated = copy.deepcopy(synthetic); mutated[7]["timestep"] = specs[8]["scheduler_timestep"]
    with pytest.raises(killtest.GlobalStopError, match="after_step_007 scheduler timestep"):
        killtest.validate_probe_records(mutated, specs)
    # Last-bit float32 accelerator differences are accepted; anything beyond the frozen tolerance is not.
    ulp = float(np.spacing(np.float32(specs[5]["scheduler_timestep"])))
    accepted = copy.deepcopy(synthetic); accepted[5]["timestep"] = specs[5]["scheduler_timestep"] + ulp
    killtest.validate_probe_records(accepted, specs)
    accepted[5]["timestep"] = specs[5]["scheduler_timestep"] - 3 * ulp
    killtest.validate_probe_records(accepted, specs)
    rejected = copy.deepcopy(synthetic); rejected[5]["timestep"] = specs[5]["scheduler_timestep"] + 0.01
    with pytest.raises(killtest.GlobalStopError, match="after_step_005 scheduler timestep"):
        killtest.validate_probe_records(rejected, specs)
    rejected = copy.deepcopy(synthetic); rejected[5]["timestep"] = float("nan")
    with pytest.raises(killtest.GlobalStopError, match="after_step_005 scheduler timestep"):
        killtest.validate_probe_records(rejected, specs)
    with pytest.raises(killtest.GlobalStopError, match="every requested boundary"):
        killtest.validate_probe_records(synthetic[:-1], specs)
    mutated = copy.deepcopy(synthetic); mutated[3]["latent_path"] = ""
    with pytest.raises(killtest.GlobalStopError, match="no persisted latent"):
        killtest.validate_probe_records(mutated, specs)


def test_boundary_specifications_freeze_label_index_absolute_step_and_timestep():
    config, specs = _specs()
    timesteps = killtest.single_flip.scheduler_timesteps_numpy(config)
    assert len(specs) == 31
    for k in range(1, 31):
        spec = specs[k]
        assert spec["boundary"] == f"after_step_{k:03d}"
        assert spec["resumed_update_index"] == k - 1
        assert spec["absolute_diffusion_step_index"] == 10 + k
        assert spec["scheduler_timestep"] == timesteps[10 + k - 1]
        assert spec["expected_runtime_dtype"] == "torch.bfloat16"
    cutoff = killtest.early_late_cutoff(31)
    assert cutoff == {"rule": cutoff["rule"], "remaining_updates": 30, "first_late_boundary_index": 23, "last_early_boundary_index": 22, "first_late_boundary": "after_step_023"}
    # Phase 2 is preregistered only for absolute step 10.
    manifest = {"anchor": {"checkpoint_step": 10}, "boundary_specifications": specs}
    mapping = killtest.phase2_selection_mapping(config, manifest, 10)
    assert mapping["phase1_entry_boundary_specification"]["absolute_diffusion_step_index"] == 10
    assert mapping["phase1_exit_boundary_specification"]["absolute_diffusion_step_index"] == 11
    assert mapping["selected_scheduler_timestep"] == timesteps[10]
    for bad in (9, 11, 40):
        with pytest.raises(killtest.GlobalStopError):
            killtest.phase2_selection_mapping(config, manifest, bad)


def test_dirty_source_blocks_preflight_before_any_gpu_build(tmp_path, monkeypatch):
    dirty = {"git_commit": "deadbeef", "git_dirty": True, "dirty_entries": [" M vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"], "source_dirty_entries": [" M vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py"], "dirty_allowlist": ["results/video_bf16_first_divergence_localization/"]}
    monkeypatch.setattr(killtest, "git_state", lambda: dirty)
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"
    killtest.run_cpu(config, CONFIG_PATH, root)  # CPU work is allowed on a dirty tree
    monkeypatch.setattr(killtest, "_build_omni", lambda *_: pytest.fail("GPU builder must not be reached on a dirty source tree"))
    with pytest.raises(killtest.GlobalStopError, match="committed source revision"):
        killtest.run_preflight(config, CONFIG_PATH, root, object())
    # Only allowlisted result files dirty: the source check passes (GPU build then attempted and stopped by the sentinel).
    clean = dict(dirty, dirty_entries=["?? results/video_bf16_first_divergence_localization/x.json"], source_dirty_entries=[])
    monkeypatch.setattr(killtest, "git_state", lambda: clean)
    killtest.run_cpu(config, CONFIG_PATH, root)
    with pytest.raises(pytest.fail.Exception):
        killtest.run_preflight(config, CONFIG_PATH, root, object())


def test_provenance_hash_changes_with_git_state_and_pipeline_hash(monkeypatch):
    config = _config()
    base = killtest.provenance(CONFIG_PATH, config)
    assert base["git_commit"] and "git_dirty" in base and "source_dirty_entries" in base
    original = killtest.git_state()
    monkeypatch.setattr(killtest, "git_state", lambda: {**original, "git_commit": "0" * 40})
    assert killtest.provenance(CONFIG_PATH, config)["provenance_hash"] != base["provenance_hash"]


def test_trusted_historical_identity_equals_preserved_v3_artifact_independently():
    trusted = killtest.derive_anchor(_config())["trusted_finals"]
    cell = Path("results/video_runtime_state_discovery_v3_corrected/run/cells/recovery_008/seed_9234/step_10/fp16/scientific_artifacts")
    latent = np.load(cell / "recovered_final_latent.npy", allow_pickle=False)
    video = np.load(cell / "recovered_video.npy", allow_pickle=False)
    assert trusted["HISTORICAL_PLUS14"]["final_latent"]["canonical_identity"] == killtest.identity(latent)
    assert trusted["HISTORICAL_PLUS14"]["video"]["canonical_identity"] == killtest.identity(video)
    assert trusted["HISTORICAL_PLUS14"]["final_latent"]["artifact_relative_path"].startswith("results/video_runtime_state_discovery_v3_corrected/")
    # CLEAN identities come from the trusted v3 baseline, distinct from the anomaly identities.
    assert trusted["CLEAN"]["final_latent_identity"] != trusted["HISTORICAL_PLUS14"]["final_latent"]["canonical_identity"]


def test_require_cpu_rejects_edited_trusted_final_identities_and_rewritten_gate_statuses(tmp_path):
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"; killtest.run_cpu(config, CONFIG_PATH, root)
    manifest_path = root / "anchor_manifest.json"; gates_path = root / "cpu_gates.json"
    genuine_manifest, genuine_gates = manifest_path.read_text(), gates_path.read_text()

    def resign(manifest):
        unhashed = dict(manifest); unhashed.pop("manifest_sha256")
        manifest["manifest_sha256"] = killtest.sha256_bytes(killtest.canonical_json(unhashed))
        manifest_path.write_text(json.dumps(manifest))
        gates = json.loads(gates_path.read_text()); gates["manifest_sha256"] = manifest["manifest_sha256"]; gates_path.write_text(json.dumps(gates))

    manifest = json.loads(genuine_manifest)
    manifest["trusted_final_identities"]["HISTORICAL_PLUS14"]["final_latent"]["canonical_identity"] = "0" * 64
    resign(manifest)
    with pytest.raises(killtest.GlobalStopError, match="trusted final identities"):
        killtest.require_cpu(root, CONFIG_PATH, config)
    manifest_path.write_text(genuine_manifest); gates_path.write_text(genuine_gates)
    gates = json.loads(genuine_gates)
    for row in gates["gates"]:
        if row["required"]:
            row["status"] = "FAIL"
    gates_path.write_text(json.dumps(gates))  # cpu_all_passed left true
    with pytest.raises(killtest.GlobalStopError, match="required gate"):
        killtest.require_cpu(root, CONFIG_PATH, config)
    manifest_path.write_text(genuine_manifest); gates_path.write_text(genuine_gates)
    manifest = json.loads(genuine_manifest); manifest["expected_boundaries"].pop(); resign(manifest)
    with pytest.raises(killtest.GlobalStopError, match="boundary mapping"):
        killtest.require_cpu(root, CONFIG_PATH, config)
    manifest_path.write_text(genuine_manifest); gates_path.write_text(genuine_gates)
    killtest.require_cpu(root, CONFIG_PATH, config)


def test_analyze_phase2_rejects_unfrozen_or_mismatched_selected_step(tmp_path):
    config, manifest, provenance, trace = _synthetic_phase2(tmp_path)

    wrong_step = copy.deepcopy(trace)
    wrong_step["selected_step"] = 20
    with pytest.raises(killtest.GlobalStopError, match="selected-step"):
        killtest.analyze_phase2_artifacts(
            tmp_path, config, manifest, provenance, wrong_step
        )

    wrong_mapping = copy.deepcopy(trace)
    wrong_mapping["selection_mapping"]["phase1_exit_boundary"] = "after_step_011"
    with pytest.raises(killtest.GlobalStopError, match="selected-step binding"):
        killtest.analyze_phase2_artifacts(
            tmp_path, config, manifest, provenance, wrong_mapping
        )

    assert killtest.analyze_phase2_artifacts(
        tmp_path, config, manifest, provenance, trace
    )["selected_step"] == 10


def test_no_binary_search_or_expansion_modes_exist():
    assert not any("binary" in m or "sweep" in m or "fp32" in m for m in killtest.MODES)
    assert killtest.MODES == ("cpu", "preflight", "phase1", "analyze-phase1", "phase2", "analyze-phase2", "phase3", "analyze-phase3")
    source = Path("experiments/video_bf16_first_divergence_localization.py").read_text().lower()
    for word in ("bisect", "binary_search", "ulp_sweep", "chaos", "jacobian", "tie-to-even", "branch basin", "certificate"):
        assert word not in source, word



def test_git_state_ignores_untracked_files_outside_source_scope(monkeypatch):
    rows = [
        "?? .venv-vllm-cu12/bin/python",
        "?? scratch/notes.txt",
        "?? results/video_bf16_first_divergence_localization/x.json",
        "?? experiments/new_probe.py",
        "?? tests/diffusion/test_new.py",
        "?? vllm_omni/diffusion/models/wan2_2/extra.py",
        " M vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
        " M README.md",
    ]
    monkeypatch.setattr(killtest, "_git", lambda *args: "\n".join(rows) if args[0] == "status" else "abc123")
    state = killtest.git_state()
    assert state["git_dirty"] is True
    assert state["source_dirty_entries"] == sorted([
        " M README.md",
        " M vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
        "?? experiments/new_probe.py",
        "?? tests/diffusion/test_new.py",
        "?? vllm_omni/diffusion/models/wan2_2/extra.py",
    ])
    with pytest.raises(killtest.GlobalStopError, match="committed source revision"):
        killtest.require_committed_source(state)
    # Only venv/scratch/results untracked: clean for GPU purposes, and provenance does not carry them.
    monkeypatch.setattr(killtest, "_git", lambda *args: "\n".join(rows[:3]) if args[0] == "status" else "abc123")
    state = killtest.git_state()
    assert state["git_dirty"] is False and state["source_dirty_entries"] == []
    killtest.require_committed_source(state)
    assert not any(".venv" in json.dumps(v) for v in state.values())



def test_timestep_match_policy_is_frozen_and_unambiguous():
    config, specs = _specs()
    schedule = killtest.single_flip.scheduler_timesteps_numpy(config)
    assert killtest.TIMESTEP_MATCH_ABS_TOL == 1e-3
    gaps = [abs(schedule[i] - schedule[i + 1]) for i in range(len(schedule) - 1)]
    assert min(gaps) > 100 * killtest.TIMESTEP_MATCH_ABS_TOL  # tolerance cannot bridge two distinct steps
    for index, expected in enumerate(schedule):
        assert killtest.timestep_matches(expected, expected, schedule)
        assert killtest.timestep_matches(expected + 5e-5, expected, schedule)
        if index > 0:
            assert not killtest.timestep_matches(expected + 5e-5, schedule[index - 1], schedule)
    assert not killtest.timestep_matches("x", schedule[0], schedule)
    assert not killtest.timestep_matches(None, schedule[0], schedule)
    manifest_policy = killtest.timestep_match_policy()
    assert manifest_policy["abs_tol"] == killtest.TIMESTEP_MATCH_ABS_TOL and "nearest" in manifest_policy["rule"]


# ----------------------------------------------------------------------------- Phase-3 hostile CPU contracts

_PHASE3_TEST_SHAPES = {"latent": [1, 1, 1, 1, 2], "block_hidden": [1, 2, 2]}
_PHASE3_MODEL_CLASS = (
    "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer."
    "WanTransformer3DModel"
)


def _phase3_value(shape, scalar):
    return killtest.single_flip.base.cast_runtime_bf16(
        np.full(shape, scalar, dtype=np.float32)
    )


def _save_phase3_value(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    bits = killtest.single_flip.float32_to_bf16_bits(value)
    np.save(path, np.ascontiguousarray(bits, dtype=np.uint16), allow_pickle=False)
    return killtest.phase3_artifact_record(root, path, shape=list(value.shape))


def _find_cfg_only_merge_pair():
    """Two (positive, negative) BF16 scalar pairs that differ in both branches yet combine identically under the
    production rule; found by search so the test does not hard-code arithmetic assumptions."""
    rng = np.random.default_rng(11)
    bf = lambda x: float(killtest.single_flip.base.cast_runtime_bf16(np.array([x], dtype=np.float32))[0])
    bits = lambda x: killtest.single_flip.float32_to_bf16_bits(np.array([x], dtype=np.float32))
    for _ in range(20000):
        p1 = bf(rng.uniform(0.5, 2.0)); n1 = bf(-rng.uniform(500.0, 2000.0))
        p2 = bf(p1 + rng.choice([2 ** -7, -(2 ** -7)])); n2 = bf(n1 + rng.choice([4.0, -4.0, 8.0, -8.0]))
        if p1 == p2 or n1 == n2:
            continue
        if np.array_equal(killtest.reconstruct_cfg_combined_bits(bits(p1), bits(n1), 4.0), killtest.reconstruct_cfg_combined_bits(bits(p2), bits(n2), 4.0)):
            return {"positive": (p1, p2), "negative": (n1, n2)}
    raise AssertionError("no cfg-only merge pair found")


def _synthetic_phase3(tmp_path, monkeypatch, *, merge_at="after_block_002"):
    config = _config()
    config["output_root"] = tmp_path.name
    monkeypatch.setattr(
        killtest,
        "_phase3_expected_shapes",
        lambda _config: copy.deepcopy(_PHASE3_TEST_SHAPES),
    )
    freeze = killtest.phase3_freeze(config)
    provenance = {
        "provenance_hash": "synthetic-phase3-provenance",
        "source_dirty_entries": [],
    }
    latent_shape = _PHASE3_TEST_SHAPES["latent"]
    block_shape = _PHASE3_TEST_SHAPES["block_hidden"]
    entry = {
        "CLEAN": _phase3_value(latent_shape, 0),
        "PLUS1": _phase3_value(latent_shape, 0),
        "HISTORICAL_PLUS14": _phase3_value(latent_shape, 0),
    }
    entry["PLUS1"].reshape(-1)[0] = 1
    entry["HISTORICAL_PLUS14"].reshape(-1)[0] = 2
    cfg = {}  # filled from the raw branch outputs via the production reconstruction rule (see below)
    final_latents = {
        "CLEAN": _phase3_value(latent_shape, 0),
        "PLUS1": _phase3_value(latent_shape, 8),
        "HISTORICAL_PLUS14": _phase3_value(latent_shape, 8),
    }
    final_videos = {
        "CLEAN": np.zeros((1, 1, 1, 3), dtype=np.uint8),
        "PLUS1": np.ones((1, 1, 1, 3), dtype=np.uint8),
        "HISTORICAL_PLUS14": np.ones((1, 1, 1, 3), dtype=np.uint8),
    }
    boundary_index = {name: index for index, name in enumerate(killtest.PHASE3_FIXED_BOUNDARIES)}

    def pair_values(boundary, branch):
        shape = latent_shape if boundary in ("transformer_entry", "raw_transformer_output") else block_shape
        if boundary == "transformer_entry":
            return {name: value.copy() for name, value in entry.items()}
        offset = 0.0 if branch == "positive" else 0.5  # negative branch carries distinct values
        plus = _phase3_value(shape, 3 + offset)
        historical = _phase3_value(shape, 4 + offset)
        if merge_at == "transient":
            if boundary == "after_block_000":
                historical = plus.copy()
        elif merge_at in boundary_index and boundary_index[boundary] >= boundary_index[merge_at]:
            historical = plus.copy()
        if merge_at in (None, "transient") and boundary == "raw_transformer_output" and cfg_only_pair is not None:
            # raw outputs differ in BOTH branches while the production CFG combination is identical
            plus_val, hist_val = cfg_only_pair[branch]
            plus = _phase3_value(shape, plus_val); historical = _phase3_value(shape, hist_val)
        return {
            "CLEAN": _phase3_value(shape, 1 + offset),  # non-zero so single-bit operand attacks are not absorbed
            "PLUS1": plus,
            "HISTORICAL_PLUS14": historical,
        }

    cfg_only_pair = _find_cfg_only_merge_pair() if merge_at in (None, "transient") else None
    values_by_boundary = {
        (boundary, branch): pair_values(boundary, branch)
        for boundary in killtest.PHASE3_FIXED_BOUNDARIES for branch in killtest.PHASE3_BRANCHES
    }
    for name in killtest.TRAJECTORIES:
        pos = killtest.single_flip.float32_to_bf16_bits(values_by_boundary[("raw_transformer_output", "positive")][name])
        neg = killtest.single_flip.float32_to_bf16_bits(values_by_boundary[("raw_transformer_output", "negative")][name])
        cfg[name] = killtest.single_flip.bf16_bits_to_float32(killtest.reconstruct_cfg_combined_bits(pos, neg, 4.0)).reshape(latent_shape)
    trajectories = {}
    architecture = copy.deepcopy(freeze["expected_architecture"])
    assert architecture["model_class"] == _PHASE3_MODEL_CLASS and architecture["inner_dim"] == 5120
    for trajectory in killtest.TRAJECTORIES:
        branches = {}
        for invocation_index, branch in enumerate(killtest.PHASE3_BRANCHES):
            records = []
            for boundary in killtest.PHASE3_FIXED_BOUNDARIES:
                value = values_by_boundary[(boundary, branch)][trajectory]
                artifact = _save_phase3_value(
                    tmp_path,
                    f"phase3/artifacts/{trajectory}/{branch}/{boundary}.npy",
                    value,
                )
                records.append(
                    {
                        "boundary": boundary,
                        "block_index": (
                            int(boundary.rsplit("_", 1)[1])
                            if boundary.startswith("after_block_")
                            else None
                        ),
                        "branch": branch,
                        "invocation_index": invocation_index,
                        "runtime_dtype": killtest.EXPECTED_RUNTIME_DTYPE,
                        "shape": list(value.shape),
                        "artifact": artifact,
                    }
                )
            branches[branch] = {
                "architecture": copy.deepcopy(architecture),
                "records": records,
            }
        cfg_artifact = _save_phase3_value(
            tmp_path,
            f"phase3/artifacts/{trajectory}/cfg.npy",
            cfg[trajectory],
        )
        trajectories[trajectory] = {
            "trajectory": trajectory,
            "branches": branches,
            "cfg_combined_output": {
                "boundary": "cfg_combined_output",
                "absolute_step": 10,
                "local_step": 0,
                "timestep_bits": freeze["selected_scheduler_timestep_bits"],
                "runtime_dtype": killtest.EXPECTED_RUNTIME_DTYPE,
                "shape": latent_shape,
                "guidance_scale": 4.0,
                "guidance_scale_bits": killtest.float64_bit_pattern(4.0),
                "cfg_normalize": False,
                "artifact": cfg_artifact,
            },
            "final_latent": killtest.save_tensor(
                tmp_path,
                f"phase3/artifacts/{trajectory}/final_latent.npy",
                final_latents[trajectory],
            ),
            "final_video": killtest.save_tensor(
                tmp_path,
                f"phase3/artifacts/{trajectory}/final_video.npy",
                final_videos[trajectory],
                runtime_semantics="uint8 decoded video",
            ),
        }
    manifest = {
        "manifest_sha256": "synthetic-phase3-manifest",
        "trusted_phase1": {"exact_only": True},
        "trusted_phase2": {
            "source_root_relative_path": "synthetic",
            "phase2_manifest_sha256": "synthetic",
            "trajectories": {},
        },
        "trusted_final_identities": {
            "CLEAN": {
                "final_latent_identity": killtest.identity(final_latents["CLEAN"]),
                "video_identity": killtest.identity(final_videos["CLEAN"]),
            },
            "PLUS1": {
                "final_latent": {"canonical_identity": killtest.identity(final_latents["PLUS1"])},
                "video": {"canonical_identity": killtest.identity(final_videos["PLUS1"])},
            },
            "HISTORICAL_PLUS14": {
                "final_latent": {"canonical_identity": killtest.identity(final_latents["HISTORICAL_PLUS14"])},
                "video": {"canonical_identity": killtest.identity(final_videos["HISTORICAL_PLUS14"])},
            },
        },
    }
    trusted_phase2 = {
        name: {"entry": entry[name], "exit": cfg[name]}
        for name in killtest.TRAJECTORIES
    }
    monkeypatch.setattr(
        killtest,
        "_load_trusted_phase2_boundary",
        lambda _root, _manifest, name, which: trusted_phase2[name][which],
    )
    trace = {
        "provenance_hash": provenance["provenance_hash"],
        "manifest_sha256": manifest["manifest_sha256"],
        "selected_step": 10,
        "selected_scheduler_timestep_bits": freeze["selected_scheduler_timestep_bits"],
        "phase3_freeze": freeze,
        "trajectories": trajectories,
    }
    return config, manifest, provenance, trace


def _analyze_phase3(bundle, root):
    config, manifest, provenance, trace = bundle
    return killtest.analyze_phase3_artifacts(
        root, config, manifest, provenance, trace
    )


def _replace_phase3_artifact(root, row, value):
    suffix = killtest.sha256_bytes(np.ascontiguousarray(value).tobytes())[:10]
    artifact = _save_phase3_value(root, f"phase3/mutations/{suffix}.npy", value)
    row["shape"] = list(value.shape)
    row["artifact"] = artifact


@pytest.mark.parametrize("selected", [9, 11, None])
def test_phase3_selected_step_is_explicitly_frozen(selected):
    config = _config()
    if selected is None:
        config["phase3"].pop("selected_step")
    else:
        config["phase3"]["selected_step"] = selected
    with pytest.raises(killtest.GlobalStopError, match="Phase-3 target"):
        killtest.validate_phase3_config(config)


def test_phase3_valid_artifacts_emit_exact_gate_set_and_events(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    result = _analyze_phase3(bundle, tmp_path)
    assert [row["name"] for row in result["gates"]] == list(killtest.PHASE3_REQUIRED_GATES)
    assert all(row["required"] and row["status"] == "PASS" for row in result["gates"])
    assert result["branch_events"]["positive"]["classification"] == "MERGED_AFTER_BLOCK_002"
    assert result["branch_events"]["negative"]["classification"] == "MERGED_AFTER_BLOCK_002"
    assert result["cfg_event"]["classification"] == "MERGED_WITHIN_BOTH_TRANSFORMER_BRANCHES"


@pytest.mark.parametrize(
    "mutation",
    ["timestep", "invocation", "block_count", "block_order", "branch_label", "branch_swap", "shape", "dtype"],
)
def test_phase3_execution_identity_mutations_fail_closed(tmp_path, monkeypatch, mutation):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    config, manifest, provenance, trace = bundle
    row = trace["trajectories"]["CLEAN"]["branches"]["positive"]["records"][0]
    if mutation == "timestep":
        trace["selected_scheduler_timestep_bits"] = "00" * 8
    elif mutation == "invocation":
        row["invocation_index"] = 1
    elif mutation == "block_count":
        trace["trajectories"]["CLEAN"]["branches"]["positive"]["architecture"]["configured_num_layers"] = 39
    elif mutation == "block_order":
        order = trace["trajectories"]["CLEAN"]["branches"]["positive"]["architecture"]["executed_block_order"]
        order[0], order[1] = order[1], order[0]
    elif mutation == "branch_label":
        row["branch"] = "unconditional"
    elif mutation == "branch_swap":
        branches = trace["trajectories"]["CLEAN"]["branches"]
        branches["positive"], branches["negative"] = branches["negative"], branches["positive"]
    elif mutation == "shape":
        wrong = _phase3_value([1, 1, 1, 2, 1], 0)
        _replace_phase3_artifact(tmp_path, row, wrong)
    else:
        row["runtime_dtype"] = "torch.float16"
    with pytest.raises(killtest.GlobalStopError):
        killtest.analyze_phase3_artifacts(tmp_path, config, manifest, provenance, trace)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_phase3_boundary_key_set_fails_closed(tmp_path, monkeypatch, mutation):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    rows = bundle[3]["trajectories"]["CLEAN"]["branches"]["positive"]["records"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[-1]["boundary"] = "attention_internal"
    with pytest.raises(killtest.GlobalStopError, match="block boundary set"):
        _analyze_phase3(bundle, tmp_path)


def test_phase3_boundary_set_audit_reports_concrete_labels():
    expected = ["a", "b", "c"]
    clean = killtest.phase3_boundary_set_audit(["a", "b", "c"], expected)
    assert clean == {"recorded_count": 3, "expected_count": 3, "missing": [], "duplicate": [], "unexpected": []}
    assert killtest.phase3_boundary_set_audit(["a", "c"], expected)["missing"] == ["b"]
    assert killtest.phase3_boundary_set_audit(["a", "b", "c", "a"], expected)["duplicate"] == ["a"]
    assert killtest.phase3_boundary_set_audit(["a", "b", "c", "zzz"], expected)["unexpected"] == ["zzz"]
    assert killtest.phase3_boundary_set_audit(["a", "b", None], expected)["unexpected"] == ["None"]
    audit = killtest.phase3_boundary_set_audit(["x", "x"], expected)
    assert audit["missing"] == expected and audit["duplicate"] == ["x"] and audit["unexpected"] == ["x"]


def test_phase3_boundary_gates_carry_computed_audit_not_expected_list(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    result = _analyze_phase3(bundle, tmp_path)
    gates = {row["name"]: row for row in result["gates"]}
    branches = sorted(f"{name}/{branch}" for name in killtest.TRAJECTORIES for branch in killtest.PHASE3_BRANCHES)
    for number, key in ((12, "missing"), (13, "duplicate"), (14, "unexpected")):
        gate = gates[f"P3-G{number}"]
        assert gate["status"] == "PASS"
        evidence = gate["evidence"]
        assert evidence != list(killtest.PHASE3_FIXED_BOUNDARIES)
        assert evidence["audited_branches"] == branches
        assert evidence["expected_boundary_count"] == len(killtest.PHASE3_FIXED_BOUNDARIES)
        assert evidence[key] == {branch: [] for branch in branches}
        assert evidence["recorded_counts"] == {branch: len(killtest.PHASE3_FIXED_BOUNDARIES) for branch in branches}


@pytest.mark.parametrize("trajectory", killtest.TRAJECTORIES)
def test_phase3_entry_must_equal_phase2_transformer_input(tmp_path, monkeypatch, trajectory):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    row = bundle[3]["trajectories"][trajectory]["branches"]["positive"]["records"][0]
    _replace_phase3_artifact(tmp_path, row, _phase3_value(_PHASE3_TEST_SHAPES["latent"], 9))
    with pytest.raises(killtest.GlobalStopError, match="PHASE2_PHASE3_BOUNDARY_MISMATCH"):
        _analyze_phase3(bundle, tmp_path)


@pytest.mark.parametrize("trajectory", killtest.TRAJECTORIES)
def test_phase3_exit_must_equal_phase2_guidance_output(tmp_path, monkeypatch, trajectory):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    row = bundle[3]["trajectories"][trajectory]["cfg_combined_output"]
    _replace_phase3_artifact(tmp_path, row, _phase3_value(_PHASE3_TEST_SHAPES["latent"], 9))
    with pytest.raises(killtest.GlobalStopError, match="PHASE2_PHASE3_BOUNDARY_MISMATCH"):
        _analyze_phase3(bundle, tmp_path)


@pytest.mark.parametrize("trajectory", killtest.TRAJECTORIES)
def test_phase3_traced_final_mutation_is_global_stop(tmp_path, monkeypatch, trajectory):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    bundle[3]["trajectories"][trajectory]["final_latent"] = killtest.save_tensor(
        tmp_path,
        f"phase3/mutations/{trajectory}_final.npy",
        _phase3_value(_PHASE3_TEST_SHAPES["latent"], 99),
    )
    with pytest.raises(killtest.GlobalStopError, match="PHASE3_TRACE_ALTERS_EXECUTION"):
        _analyze_phase3(bundle, tmp_path)


def test_phase3_declared_metrics_are_ignored_and_recomputed(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    baseline = _analyze_phase3(copy.deepcopy(bundle), tmp_path)
    for trajectory in bundle[3]["trajectories"].values():
        for branch in trajectory["branches"].values():
            for row in branch["records"]:
                row.update({"bit_exact": True, "differing_element_count": 0, "mse": 1e99, "relative_l2": 1e99})
    attacked = _analyze_phase3(bundle, tmp_path)
    assert attacked["pairwise_rows"] == baseline["pairwise_rows"]
    assert attacked["branch_events"] == baseline["branch_events"]


def test_phase3_float_descriptions_are_not_in_provenance_binding():
    exact = {
        "canonical_identity": "abc",
        "differing_element_count": 1,
        "bit_exact": False,
    }
    changed_description = {"relative_l2": np.nextafter(1.0, 2.0)}
    assert killtest.canonical_json(exact) == killtest.canonical_json(copy.deepcopy(exact))
    assert not killtest._binding_contains_float_reduction(exact)
    assert killtest._binding_contains_float_reduction(changed_description)
    exact["differing_element_count"] = 2
    assert killtest.canonical_json(exact) != killtest.canonical_json({
        "canonical_identity": "abc", "differing_element_count": 1, "bit_exact": False,
    })


def test_phase3_artifact_identity_mutation_is_rejected(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    row = bundle[3]["trajectories"]["PLUS1"]["branches"]["positive"]["records"][3]
    row["artifact"]["runtime_canonical_identity"] = "forged"
    with pytest.raises(killtest.GlobalStopError, match="canonical identity"):
        _analyze_phase3(bundle, tmp_path)


@pytest.mark.parametrize(
    "merge_at,expected",
    [
        ("transient", "TRANSIENT_EXACT_EQUALITY"),
        ("after_block_017", "MERGED_AFTER_BLOCK_017"),
        ("raw_transformer_output", "MERGED_AT_RAW_TRANSFORMER_OUTPUT"),
        (None, "NO_INTERNAL_EXACT_MERGE"),
    ],
)
def test_phase3_persistent_merge_classification(tmp_path, monkeypatch, merge_at, expected):
    bundle = _synthetic_phase3(tmp_path, monkeypatch, merge_at=merge_at)
    result = _analyze_phase3(bundle, tmp_path)
    assert result["branch_events"]["positive"]["classification"] == expected
    if merge_at is None:
        assert result["cfg_event"]["classification"] == "MERGED_AT_CFG_COMBINATION"


def test_phase3_row_shuffle_and_relocation_preserve_analysis(tmp_path, monkeypatch):
    original = tmp_path / "original"
    bundle = _synthetic_phase3(original, monkeypatch)
    baseline = _analyze_phase3(copy.deepcopy(bundle), original)
    for trajectory in bundle[3]["trajectories"].values():
        for branch in trajectory["branches"].values():
            branch["records"].reverse()
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    result = _analyze_phase3(bundle, relocated)
    assert result["branch_events"] == baseline["branch_events"]
    assert result["pairwise_rows"] == baseline["pairwise_rows"]


def test_phase3_missing_relocated_artifact_is_rejected(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    record = bundle[3]["trajectories"]["CLEAN"]["branches"]["negative"]["records"][5]["artifact"]
    (tmp_path / record["relative_path"]).unlink()
    with pytest.raises(killtest.GlobalStopError, match="artifact file validation"):
        _analyze_phase3(bundle, tmp_path)


@pytest.mark.parametrize("mutation", ["phase4", "internal", "expansion"])
def test_phase3_forbidden_followup_modes_are_rejected(mutation):
    config = _config()
    if mutation == "phase4":
        config["phase3"]["phase4_enabled"] = True
    elif mutation == "internal":
        config["phase3"]["trace_attention"] = True
    else:
        config["phase3"]["auto_expand"] = True
    with pytest.raises(killtest.GlobalStopError, match="Phase-3 target"):
        killtest.validate_phase3_config(config)



# ----------------------------------------------------------------------------- Phase-3 additions (Claude)


def test_phase3_cfg_classification_distinguishes_branches(tmp_path, monkeypatch):
    assert killtest._phase3_cfg_classification({"positive": True, "negative": True}) == "MERGED_WITHIN_BOTH_TRANSFORMER_BRANCHES"
    assert killtest._phase3_cfg_classification({"positive": True, "negative": False}) == "MERGED_WITHIN_POSITIVE_BRANCH_ONLY_CFG_COMBINATION_EXACT"
    assert killtest._phase3_cfg_classification({"positive": False, "negative": True}) == "MERGED_WITHIN_NEGATIVE_BRANCH_ONLY_CFG_COMBINATION_EXACT"
    assert killtest._phase3_cfg_classification({"positive": False, "negative": False}) == "MERGED_AT_CFG_COMBINATION"
    # Production path: negative branch never merges (raw outputs differ) while positive merges and CFG output is exact.
    bundle = _synthetic_phase3(tmp_path, monkeypatch, merge_at="after_block_005")
    config, manifest, provenance, trace = bundle
    latent_shape = _PHASE3_TEST_SHAPES["latent"]
    bits = lambda arr: killtest.single_flip.float32_to_bf16_bits(arr)
    plus_pos = killtest.load_phase3_artifact(tmp_path, _raw_row(trace, "PLUS1", "positive")["artifact"])
    plus_neg = killtest.load_phase3_artifact(tmp_path, _raw_row(trace, "PLUS1", "negative")["artifact"])
    rng = np.random.default_rng(2); hist_neg = None
    for _ in range(20000):  # search a distinct negative raw output whose combine with the same positive is identical
        cand = _phase3_value(latent_shape, float(killtest.single_flip.base.cast_runtime_bf16(np.array([plus_neg.reshape(-1)[0] + rng.choice([-8.0, -4.0, 4.0, 8.0]) * rng.integers(1, 4)], np.float32))[0]))
        if not np.array_equal(cand, plus_neg) and np.array_equal(killtest.reconstruct_cfg_combined_bits(bits(plus_pos), bits(cand), 4.0), killtest.reconstruct_cfg_combined_bits(bits(plus_pos), bits(plus_neg), 4.0)):
            hist_neg = cand; break
    if hist_neg is None:
        big = _phase3_value(latent_shape, -1000.0); big2 = _phase3_value(latent_shape, -1004.0); one = _phase3_value(latent_shape, 1.0)
        for trajectory in ("PLUS1", "HISTORICAL_PLUS14"):
            _replace_phase3_artifact(tmp_path, _raw_row(trace, trajectory, "positive"), one)
        _replace_phase3_artifact(tmp_path, _raw_row(trace, "PLUS1", "negative"), big); plus_pos, plus_neg, hist_neg = one, big, big2
        assert np.array_equal(killtest.reconstruct_cfg_combined_bits(bits(one), bits(big), 4.0), killtest.reconstruct_cfg_combined_bits(bits(one), bits(big2), 4.0))
    for row in trace["trajectories"]["HISTORICAL_PLUS14"]["branches"]["negative"]["records"]:
        if row["boundary"] == "raw_transformer_output":
            _replace_phase3_artifact(tmp_path, row, hist_neg)
        elif row["boundary"] != "transformer_entry":
            _replace_phase3_artifact(tmp_path, row, _phase3_value(row["shape"], 11))
    combined = killtest.single_flip.bf16_bits_to_float32(killtest.reconstruct_cfg_combined_bits(bits(plus_pos), bits(plus_neg), 4.0)).reshape(latent_shape)
    for trajectory in ("PLUS1", "HISTORICAL_PLUS14"):
        _replace_phase3_artifact(tmp_path, trace["trajectories"][trajectory]["cfg_combined_output"], combined)
    trusted = killtest._load_trusted_phase2_boundary
    monkeypatch.setattr(killtest, "_load_trusted_phase2_boundary", lambda r, m, name, which: combined if (which == "exit" and name != "CLEAN") else trusted(r, m, name, which))
    result = killtest.analyze_phase3_artifacts(tmp_path, config, manifest, provenance, trace)
    assert result["branch_events"]["positive"]["classification"] == "MERGED_AFTER_BLOCK_005"
    assert result["branch_events"]["negative"]["classification"] == "NO_INTERNAL_EXACT_MERGE"
    assert result["cfg_event"]["classification"] == "MERGED_WITHIN_POSITIVE_BRANCH_ONLY_CFG_COMBINATION_EXACT"


def test_phase3_architecture_heads_or_patch_mismatch_fails_closed(tmp_path, monkeypatch):
    for key, value in (("num_attention_heads", 32), ("attention_head_dim", 64), ("inner_dim", 4096), ("patch_size", [1, 1, 1]), ("model_class", "other.Model")):
        bundle = _synthetic_phase3(tmp_path / key, monkeypatch)
        config, manifest, provenance, trace = bundle
        trace["trajectories"]["PLUS1"]["branches"]["negative"]["architecture"][key] = value
        with pytest.raises(killtest.GlobalStopError, match="required gate"):
            killtest.analyze_phase3_artifacts(tmp_path / key, config, manifest, provenance, trace)
        gates = json.loads((tmp_path / key / "phase3/phase3_gates.json").read_text())["gates"]
        assert {g["name"] for g in gates if g["status"] == "FAIL"} == {"P3-G3", "P3-G11"}


@pytest.mark.parametrize("trajectory", killtest.TRAJECTORIES)
def test_phase3_traced_final_video_mutation_is_global_stop(tmp_path, monkeypatch, trajectory):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    bundle[3]["trajectories"][trajectory]["final_video"] = killtest.save_tensor(
        tmp_path, f"phase3/mutations/{trajectory}_video.npy", np.full((1, 1, 1, 3), 5, dtype=np.uint8), runtime_semantics="uint8 decoded video"
    )
    with pytest.raises(killtest.GlobalStopError, match="PHASE3_TRACE_ALTERS_EXECUTION"):
        _analyze_phase3(bundle, tmp_path)
    assert not (tmp_path / "phase3/phase3_analysis.json").exists()


def test_phase3_bf16_bit_artifacts_match_torch_runtime_encoding(tmp_path):
    torch = pytest.importorskip("torch")
    runtime = torch.randn(2, 3, 4).to(torch.bfloat16)
    bits = runtime.view(torch.uint16).numpy()  # exactly what the production probe persists
    path = tmp_path / "x.npy"; np.save(path, np.ascontiguousarray(bits), allow_pickle=False)
    record = killtest.phase3_artifact_record(tmp_path, path, shape=list(runtime.shape))
    widened = killtest.load_phase3_artifact(tmp_path, record)
    assert np.array_equal(widened, runtime.float().numpy()) and widened.dtype == np.float32
    assert record["runtime_dtype"] == "torch.bfloat16" and record["artifact_encoding"] == "bf16_bits_v1" and record["nbytes"] == bits.nbytes
    # Same bytes reinterpreted with another shape must produce another identity and be rejected against the record.
    other = np.save(tmp_path / "y.npy", np.ascontiguousarray(bits.reshape(4, 3, 2)), allow_pickle=False)
    other_record = killtest.phase3_artifact_record(tmp_path, tmp_path / "y.npy", shape=[4, 3, 2])
    assert other_record["runtime_canonical_identity"] != record["runtime_canonical_identity"]
    forged = dict(record, shape=[4, 3, 2])
    with pytest.raises(killtest.GlobalStopError):
        killtest.load_phase3_artifact(tmp_path, forged)


needs_trusted_phases = pytest.mark.skipif(
    not Path("results/video_bf16_first_divergence_localization/phase2/phase2_manifest.json").exists(),
    reason="trusted Phase-1/Phase-2 artifacts absent",
)


@needs_trusted_phases
def test_provenance_binding_is_invariant_to_last_bit_float_metric_differences(monkeypatch):
    """Cross-host reproducibility: only exact invariants enter the binding."""
    config = _config()
    baseline1 = killtest.derive_trusted_phase1(config)
    baseline2 = killtest.derive_trusted_phase2(config)
    assert not killtest._binding_contains_float_reduction(baseline1) and not killtest._binding_contains_float_reduction(baseline2)
    original_metrics = killtest.metrics

    def perturbed_metrics(lhs, rhs):
        row = original_metrics(lhs, rhs)
        for key in ("relative_l2", "mse", "mean_abs_diff", "max_abs_diff", "l2"):
            if row[key] != 0.0:
                row[key] = float(np.nextafter(row[key], np.inf))  # emulate another host's reduction order
        return row

    monkeypatch.setattr(killtest, "metrics", perturbed_metrics)
    assert killtest.derive_trusted_phase1(config) == baseline1
    assert killtest.derive_trusted_phase2(config) == baseline2
    # The same perturbation still changes the descriptive analysis rows, proving the monkeypatch is live.
    a = original_metrics(np.ones(4, np.float32), np.zeros(4, np.float32)); b = perturbed_metrics(np.ones(4, np.float32), np.zeros(4, np.float32))
    assert a != b and a["differing_element_count"] == b["differing_element_count"]


@needs_trusted_phases
def test_require_cpu_rejects_trusted_phase2_identity_or_integer_mutation(tmp_path):
    config = _config(); config["output_root"] = "video_bf16_first_divergence_localization_phase3"
    root = tmp_path / "video_bf16_first_divergence_localization_phase3"
    killtest.run_cpu(config, CONFIG_PATH, root)
    manifest_path, gates_path = root / "anchor_manifest.json", root / "cpu_gates.json"
    genuine_manifest, genuine_gates = manifest_path.read_text(), gates_path.read_text()

    def resign(mutate):
        manifest = json.loads(genuine_manifest); mutate(manifest)
        unhashed = dict(manifest); unhashed.pop("manifest_sha256")
        manifest["manifest_sha256"] = killtest.sha256_bytes(killtest.canonical_json(unhashed))
        manifest_path.write_text(json.dumps(manifest))
        gates = json.loads(genuine_gates); gates["manifest_sha256"] = manifest["manifest_sha256"]; gates_path.write_text(json.dumps(gates))

    resign(lambda m: m["trusted_phase2"]["trajectories"]["CLEAN"].__setitem__("exit_identity", "0" * 64))
    with pytest.raises(killtest.GlobalStopError, match="trusted Phase-2 binding"):
        killtest.require_cpu(root, CONFIG_PATH, config)
    resign(lambda m: m["trusted_phase1"]["trusted_exit_clean_vs_plus1"].__setitem__("differing_element_count", 41640))
    with pytest.raises(killtest.GlobalStopError, match="trusted Phase-1 binding"):
        killtest.require_cpu(root, CONFIG_PATH, config)
    resign(lambda m: m["phase3_freeze"]["expected_architecture"].__setitem__("inner_dim", 4096))
    with pytest.raises(killtest.GlobalStopError, match="Phase-3 freeze"):
        killtest.require_cpu(root, CONFIG_PATH, config)
    manifest_path.write_text(genuine_manifest); gates_path.write_text(genuine_gates)
    manifest, _ = killtest.require_cpu(root, CONFIG_PATH, config)
    assert not killtest._binding_contains_float_reduction(manifest["trusted_phase1"])
    assert not killtest._binding_contains_float_reduction(manifest["trusted_phase2"])
    assert manifest["phase3_freeze"]["expected_artifact_budget"]["bytes_total_three_trajectories"] > 30e9



# ----------------------------------------------------------------------------- M1: CFG operand reconstruction (production path)


def _raw_row(trace, trajectory, branch):
    return next(r for r in trace["trajectories"][trajectory]["branches"][branch]["records"] if r["boundary"] == "raw_transformer_output")


def _flip_one_bf16_bit(root, row, flat=0, mask=0x0080):
    """Flip one BF16 bit of one element. Default flips the top mantissa bit (a finite, non-absorbable change):
    an LSB flip of a tiny value can be absorbed by the (non-injective) BF16 combine and an exponent flip can
    produce Inf, so neither is a meaningful reconstruction attack."""
    path = root / row["artifact"]["relative_path"]
    bits = np.load(path, allow_pickle=False).copy(); bits.reshape(-1)[flat] ^= np.uint16(mask)
    np.save(path, bits, allow_pickle=False)
    row["artifact"] = killtest.phase3_artifact_record(root, path, shape=row["shape"])


def test_cfg_reconstruction_rule_matches_torch_eager_bf16_semantics():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(3)
    for scale in (4.0, 3.5, 7.25):
        p = (torch.randn(3, 16, 9, 20, 24, generator=g) * 3).to(torch.bfloat16)
        n = (torch.randn(3, 16, 9, 20, 24, generator=g) * 3).to(torch.bfloat16)
        torch_comb = (n + scale * (p - n)).view(torch.uint16).numpy()  # exact production expression, eager bf16 ops
        ours = killtest.reconstruct_cfg_combined_bits(p.view(torch.uint16).numpy(), n.view(torch.uint16).numpy(), scale)
        assert np.array_equal(ours, torch_comb), scale
        swapped = killtest.reconstruct_cfg_combined_bits(n.view(torch.uint16).numpy(), p.view(torch.uint16).numpy(), scale)
        assert not np.array_equal(swapped, torch_comb)
    assert "bf16(p - n)" in killtest.CFG_RECONSTRUCTION_RULE and "cfg_normalize=False" in killtest.CFG_RECONSTRUCTION_RULE


def test_m1_A_correct_raw_operands_pass_and_gate_31_is_required(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    result = _analyze_phase3(bundle, tmp_path)
    assert "P3-G31" in killtest.PHASE3_REQUIRED_GATES and [g["name"] for g in result["gates"]][-1] == "P3-G31"
    assert all(g["status"] == "PASS" for g in result["gates"])
    assert all(r["bit_exact"] and not r["swapped_operands_bit_exact"] for r in result["cfg_operand_reconstruction"].values())
    assert result["cfg_operand_reconstruction"]["CLEAN"]["guidance_scale_bits"] == killtest.float64_bit_pattern(4.0)


def test_m1_B_swapped_branch_artifact_contents_with_intact_labels_rejected(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    trace = bundle[3]
    pos, neg = trace["trajectories"]["PLUS1"]["branches"]["positive"]["records"], trace["trajectories"]["PLUS1"]["branches"]["negative"]["records"]
    for a, c in zip(pos, neg):
        a["artifact"], c["artifact"] = c["artifact"], a["artifact"]
    with pytest.raises(killtest.GlobalStopError, match="CFG_OPERAND_RECONSTRUCTION_MISMATCH"):
        _analyze_phase3(bundle, tmp_path)
    failure = json.loads((tmp_path / "phase3/cfg_operand_reconstruction_mismatch.json").read_text())
    assert failure["reconstruction"]["PLUS1"]["bit_exact"] is False and failure["reconstruction"]["PLUS1"]["swapped_operands_bit_exact"] is True
    assert not (tmp_path / "phase3/phase3_analysis.json").exists()


@pytest.mark.parametrize("target", ["positive", "negative", "cfg"])
@pytest.mark.parametrize("trajectory", killtest.TRAJECTORIES)
def test_m1_CDE_single_bf16_bit_mutation_rejected(tmp_path, monkeypatch, target, trajectory):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    trace = bundle[3]
    row = trace["trajectories"][trajectory]["cfg_combined_output"] if target == "cfg" else _raw_row(trace, trajectory, target)
    _flip_one_bf16_bit(tmp_path, row)
    with pytest.raises(killtest.GlobalStopError, match="CFG_OPERAND_RECONSTRUCTION_MISMATCH|PHASE2_PHASE3_BOUNDARY_MISMATCH"):
        _analyze_phase3(bundle, tmp_path)


def test_m1_F_wrong_guidance_scale_rejected(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    config, manifest, provenance, trace = bundle
    row = trace["trajectories"]["CLEAN"]["cfg_combined_output"]
    row["guidance_scale"] = 3.5; row["guidance_scale_bits"] = killtest.float64_bit_pattern(3.5)
    with pytest.raises(killtest.GlobalStopError, match="CFG_OPERAND_RECONSTRUCTION_MISMATCH"):
        _analyze_phase3(bundle, tmp_path)
    # config-level scale change is rejected before any analysis (frozen generation configuration)
    row["guidance_scale"] = 4.0; row["guidance_scale_bits"] = killtest.float64_bit_pattern(4.0)
    bad = copy.deepcopy(config); bad["generation"]["guidance_scale"] = 3.5
    with pytest.raises(killtest.GlobalStopError, match="binding mismatch"):
        killtest.analyze_phase3_artifacts(tmp_path, bad, manifest, provenance, trace)


def test_m1_G_arbitrary_cfg_tensor_with_correct_shape_dtype_rejected(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    trace = bundle[3]
    for trajectory in killtest.TRAJECTORIES:
        row = trace["trajectories"][trajectory]["cfg_combined_output"]
        _replace_phase3_artifact(tmp_path, row, _phase3_value(_PHASE3_TEST_SHAPES["latent"], 7.0))
    # PLUS1/HIST cfg are still equal (both 7), so the only guard that can fire is reconstruction or the Phase-2 exit binding.
    with pytest.raises(killtest.GlobalStopError, match="PHASE2_PHASE3_BOUNDARY_MISMATCH|CFG_OPERAND_RECONSTRUCTION_MISMATCH"):
        _analyze_phase3(bundle, tmp_path)


def test_m1_H_forged_reconstruction_pass_in_rows_is_ignored(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    trace = bundle[3]
    _flip_one_bf16_bit(tmp_path, _raw_row(trace, "HISTORICAL_PLUS14", "negative"))
    for trajectory in trace["trajectories"].values():
        trajectory["cfg_combined_output"].update({"reconstruction_bit_exact": True, "P3-G31": "PASS"})
        trajectory["cfg_operand_reconstruction"] = {"bit_exact": True}
    with pytest.raises(killtest.GlobalStopError, match="CFG_OPERAND_RECONSTRUCTION_MISMATCH"):
        _analyze_phase3(bundle, tmp_path)


def test_m1_I_wrong_branch_labels_rejected_structurally(tmp_path, monkeypatch):
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    trace = bundle[3]
    for row in trace["trajectories"]["CLEAN"]["branches"]["positive"]["records"]:
        row["branch"] = "negative"
    with pytest.raises(killtest.GlobalStopError, match="required gate"):
        _analyze_phase3(bundle, tmp_path)


def test_m1_J_float32_reconstruction_is_not_the_production_path(tmp_path, monkeypatch):
    """A cfg tensor produced by single-rounding float32 arithmetic must be rejected where it differs from the
    three-step BF16 emulation, proving the validator uses the canonical path."""
    bf = killtest.single_flip.base.cast_runtime_bf16
    rng = np.random.default_rng(5)
    p = bf(rng.normal(scale=3, size=(1, 1, 1, 1, 2)).astype(np.float32)); n = bf(rng.normal(scale=3, size=(1, 1, 1, 1, 2)).astype(np.float32))
    for _ in range(5000):
        approx = bf((n + np.float32(4.0) * (p - n)).astype(np.float32))
        exact = killtest.single_flip.bf16_bits_to_float32(killtest.reconstruct_cfg_combined_bits(killtest.single_flip.float32_to_bf16_bits(p), killtest.single_flip.float32_to_bf16_bits(n), 4.0)).reshape(p.shape)
        if not np.array_equal(approx, exact):
            break
        p = bf(rng.normal(scale=3, size=p.shape).astype(np.float32)); n = bf(rng.normal(scale=3, size=p.shape).astype(np.float32))
    else:
        pytest.skip("no float32-vs-BF16-path divergence found")
    bundle = _synthetic_phase3(tmp_path, monkeypatch)
    trace = bundle[3]
    for trajectory in killtest.TRAJECTORIES:
        _replace_phase3_artifact(tmp_path, _raw_row(trace, trajectory, "positive"), p)
        _replace_phase3_artifact(tmp_path, _raw_row(trace, trajectory, "negative"), n)
        _replace_phase3_artifact(tmp_path, trace["trajectories"][trajectory]["cfg_combined_output"], approx)
    with pytest.raises(killtest.GlobalStopError, match="CFG_OPERAND_RECONSTRUCTION_MISMATCH|PHASE2_PHASE3_BOUNDARY_MISMATCH"):
        _analyze_phase3(bundle, tmp_path)
    for trajectory in killtest.TRAJECTORIES:
        _replace_phase3_artifact(tmp_path, trace["trajectories"][trajectory]["cfg_combined_output"], exact)
    # with the exact BF16 path the only remaining guard is the Phase-2 exit identity (synthetic exit differs)
    with pytest.raises(killtest.GlobalStopError, match="PHASE2_PHASE3_BOUNDARY_MISMATCH"):
        _analyze_phase3(bundle, tmp_path)


# ----------------------------------------------------------------------------- M2: preflight authentication (production require_preflight)


@needs_trusted_phases
def test_m2_fabricated_all_pass_preflight_without_artifacts_rejected(tmp_path):
    config = _config(); config["output_root"] = "video_bf16_first_divergence_localization_phase3"
    root = tmp_path / "video_bf16_first_divergence_localization_phase3"
    killtest.run_cpu(config, CONFIG_PATH, root)
    manifest, provenance = killtest.require_cpu(root, CONFIG_PATH, config)
    _valid_preflight(root, manifest, provenance, config=config, with_artifacts=False)
    with pytest.raises(killtest.GlobalStopError, match="three trajectories"):
        killtest.require_preflight(root, manifest, provenance)
    # names/PASS/hashes all correct, controls present but artifacts deleted
    _valid_preflight(root, manifest, provenance, config=config)
    killtest.require_preflight(root, manifest, provenance)  # genuine passes
    shutil.rmtree(root / "preflight/PLUS1")
    with pytest.raises(killtest.GlobalStopError, match="artifact file validation"):
        killtest.require_preflight(root, manifest, provenance)


@needs_trusted_phases
def test_m2_mutated_repeat_artifact_with_resigned_document_rejected(tmp_path):
    config = _config(); config["output_root"] = "video_bf16_first_divergence_localization_phase3"
    root = tmp_path / "video_bf16_first_divergence_localization_phase3"
    killtest.run_cpu(config, CONFIG_PATH, root)
    manifest, provenance = killtest.require_cpu(root, CONFIG_PATH, config)
    document = _valid_preflight(root, manifest, provenance, config=config)
    latent = killtest.load_tensor(root, document["controls"]["HISTORICAL_PLUS14"][0]["final_latent_artifact"]).copy()
    latent.reshape(-1)[0] += np.float32(1e-3)
    record = killtest.save_tensor(root, "preflight/HISTORICAL_PLUS14/repeat_0/recovered_final_latent.npy", latent)
    for row in document["controls"]["HISTORICAL_PLUS14"]:
        row["final_latent_artifact"] = record; row["final_latent_identity"] = record["canonical_identity"]
    killtest.atomic_json(root / "preflight/preflight_gates.json", document)  # all gates still declared PASS
    with pytest.raises(killtest.GlobalStopError, match="re-derived from artifacts fail"):
        killtest.require_preflight(root, manifest, provenance)
    # repeat key set violations
    document = _valid_preflight(root, manifest, provenance, config=config)
    document["controls"]["CLEAN"].pop(); killtest.atomic_json(root / "preflight/preflight_gates.json", document)
    with pytest.raises(killtest.GlobalStopError, match="repeat key set"):
        killtest.require_preflight(root, manifest, provenance)
    document = _valid_preflight(root, manifest, provenance, config=config)
    document["controls"]["CLEAN"][2]["repeat_id"] = 0; killtest.atomic_json(root / "preflight/preflight_gates.json", document)
    with pytest.raises(killtest.GlobalStopError, match="repeat key set"):
        killtest.require_preflight(root, manifest, provenance)
    # a declared FAIL contradicting artifacts, or a gate document from the trusted Phase-1/2 root, are rejected
    document = _valid_preflight(root, manifest, provenance, config=config)
    document["gates"][0]["status"] = "FAIL"; killtest.atomic_json(root / "preflight/preflight_gates.json", document)
    with pytest.raises(killtest.GlobalStopError):
        killtest.require_preflight(root, manifest, provenance)
    old = json.loads(Path("results/video_bf16_first_divergence_localization/preflight/preflight_gates.json").read_text())
    killtest.atomic_json(root / "preflight/preflight_gates.json", old)
    with pytest.raises(killtest.GlobalStopError, match="binding mismatch"):
        killtest.require_preflight(root, manifest, provenance)


# ----------------------------------------------------------------------------- M3: require_cpu validates the Phase-3 config


@needs_trusted_phases
@pytest.mark.parametrize("mutate", [
    lambda c: c["phase3"].__setitem__("selected_step", 9),
    lambda c: c["phase3"].__setitem__("selected_step", 11),
    lambda c: c["phase3"].pop("selected_step"),
    lambda c: c["phase3"].__setitem__("trace_attention", True),
    lambda c: c["phase3"].__setitem__("phase4_enabled", True),
    lambda c: c["phase3"].__setitem__("auto_expand", True),
])
def test_m3_require_cpu_rejects_mutated_loaded_config(tmp_path, mutate):
    config = _config(); config["output_root"] = "video_bf16_first_divergence_localization_phase3"
    root = tmp_path / "video_bf16_first_divergence_localization_phase3"
    killtest.run_cpu(config, CONFIG_PATH, root)
    mutated = copy.deepcopy(config); mutate(mutated)
    with pytest.raises(killtest.GlobalStopError, match="Phase-3 target"):
        killtest.require_cpu(root, CONFIG_PATH, mutated)


# ----------------------------------------------------------------------------- portability: exact invariants only


@needs_trusted_phases
def test_portability_float_last_bit_in_manifest_does_not_affect_authorization(tmp_path):
    config = _config(); config["output_root"] = "video_bf16_first_divergence_localization_phase3"
    root = tmp_path / "video_bf16_first_divergence_localization_phase3"
    killtest.run_cpu(config, CONFIG_PATH, root)
    manifest_path, gates_path = root / "anchor_manifest.json", root / "cpu_gates.json"
    genuine_manifest, genuine_gates = manifest_path.read_text(), gates_path.read_text()

    def resign(mutate):
        manifest = json.loads(genuine_manifest); mutate(manifest)
        unhashed = dict(manifest); unhashed.pop("manifest_sha256")
        manifest["manifest_sha256"] = killtest.sha256_bytes(killtest.canonical_json(unhashed))
        manifest_path.write_text(json.dumps(manifest))
        gates = json.loads(genuine_gates); gates["manifest_sha256"] = manifest["manifest_sha256"]; gates_path.write_text(json.dumps(gates))

    hist = "trajectories"; up = lambda x: float(np.nextafter(x, np.inf))
    resign(lambda m: m[hist]["HISTORICAL_PLUS14"]["historical_delta"].__setitem__("delta_l2", up(m[hist]["HISTORICAL_PLUS14"]["historical_delta"]["delta_l2"])))
    killtest.require_cpu(root, CONFIG_PATH, config)  # descriptive float: authorization unchanged
    resign(lambda m: m[hist]["PLUS1"]["construction"].__setitem__("realized_mse", up(m[hist]["PLUS1"]["construction"]["realized_mse"])))
    killtest.require_cpu(root, CONFIG_PATH, config)
    resign(lambda m: m[hist]["HISTORICAL_PLUS14"]["historical_delta"]["changed_coordinates"][2].__setitem__("perturbed_bf16_bits_hex", "0xb4a1"))
    with pytest.raises(killtest.GlobalStopError, match="historical support"):
        killtest.require_cpu(root, CONFIG_PATH, config)
    resign(lambda m: m[hist]["PLUS1"]["construction"].__setitem__("expected_perturbed_bf16_bits", 0xb4ac))
    with pytest.raises(killtest.GlobalStopError, match="PLUS1 construction"):
        killtest.require_cpu(root, CONFIG_PATH, config)
    resign(lambda m: m[hist]["HISTORICAL_PLUS14"]["historical_delta"].__setitem__("changed_coordinate_count", 5))
    with pytest.raises(killtest.GlobalStopError, match="historical support"):
        killtest.require_cpu(root, CONFIG_PATH, config)
    manifest_path.write_text(genuine_manifest); gates_path.write_text(genuine_gates)
    killtest.require_cpu(root, CONFIG_PATH, config)
