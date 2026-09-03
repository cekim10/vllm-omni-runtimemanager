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


def _valid_preflight(root, manifest, provenance):
    gates = [killtest.gate(name, True, "synthetic validated evidence", required=True) for name in killtest.PREFLIGHT_REQUIRED_GATES]
    document = {
        "gpu_all_passed": True,
        "gates": gates,
        "controls": {},
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
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"
    killtest.run_cpu(config, CONFIG_PATH, root)
    manifest, provenance = killtest.require_cpu(root, CONFIG_PATH, config)
    _valid_preflight(root, manifest, provenance)
    with pytest.raises(killtest.GlobalStopError, match="selected_step"):
        killtest.run_phase2(config, CONFIG_PATH, root, object())
    assert config["phase3"] == {"enabled": False, "auto_expand": False}


def test_phase2_selection_maps_absolute_step_to_frozen_phase1_boundary(tmp_path):
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"
    killtest.run_cpu(config, CONFIG_PATH, root)
    manifest, _ = killtest.require_cpu(root, CONFIG_PATH, config)
    mapping = killtest.phase2_selection_mapping(config, manifest, 20)
    assert mapping["selected_absolute_diffusion_step_index"] == 20
    assert mapping["selected_resumed_update_index"] == 10
    assert mapping["phase1_entry_boundary"] == "after_step_010"
    assert mapping["phase1_entry_boundary_specification"]["absolute_diffusion_step_index"] == 20
    assert mapping["selected_scheduler_timestep"] == killtest.single_flip.scheduler_timesteps_numpy(config)[20]


def test_phase2_invalid_absolute_step_fails_before_gpu_build(tmp_path, monkeypatch):
    config = _config(); config["output_root"] = "localization"
    config["phase2"]["selected_step"] = 40
    root = tmp_path / "localization"
    killtest.run_cpu(config, CONFIG_PATH, root)
    manifest, provenance = killtest.require_cpu(root, CONFIG_PATH, config)
    _valid_preflight(root, manifest, provenance)
    monkeypatch.setattr(killtest, "_build_omni", lambda *_: pytest.fail("GPU builder must not be reached"))
    with pytest.raises(killtest.GlobalStopError, match="outside the resumed denoising range"):
        killtest.run_phase2(config, CONFIG_PATH, root, object())


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
    document = _valid_preflight(root, manifest, provenance)
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
    # Phase-2 mapping is the inverse of the phase-1 spec for every legal absolute step.
    manifest = {"anchor": {"checkpoint_step": 10}, "boundary_specifications": specs}
    for step in range(10, 40):
        mapping = killtest.phase2_selection_mapping(config, manifest, step)
        assert mapping["phase1_entry_boundary_specification"]["absolute_diffusion_step_index"] == step
        assert mapping["selected_scheduler_timestep"] == timesteps[step]
    for bad in (9, 40):
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
    config = _config(); config["output_root"] = "localization"
    root = tmp_path / "localization"; killtest.run_cpu(config, CONFIG_PATH, root)
    manifest, provenance = killtest.require_cpu(root, CONFIG_PATH, config)
    _valid_preflight(root, manifest, provenance)
    boundaries = ["latent_entering_step", "transformer_input", "guidance_combined_output", "scheduler_input", "scheduler_output"]
    traces = {name: [{"boundary": b, "artifact": killtest.save_tensor(root, f"phase2/artifacts/{name}/{b}.npy", np.full((1, 2), i, dtype=np.float32))} for b in boundaries] for i, name in enumerate(killtest.TRAJECTORIES)}
    document = {"provenance_hash": provenance["provenance_hash"], "manifest_sha256": manifest["manifest_sha256"], "selected_step": 20, "selection_mapping": killtest.phase2_selection_mapping(config, manifest, 20), "traces": traces, "unavailable_boundaries": ["transformer_raw_output"]}
    killtest.atomic_json(root / "phase2/phase2_manifest.json", document)
    with pytest.raises(killtest.GlobalStopError, match="explicitly frozen config step"):
        killtest.run_analyze_phase2(config, CONFIG_PATH, root)  # config step still null
    config["phase2"]["selected_step"] = 21
    with pytest.raises(killtest.GlobalStopError, match="explicitly frozen config step"):
        killtest.run_analyze_phase2(config, CONFIG_PATH, root)
    config["phase2"]["selected_step"] = 20
    forged = copy.deepcopy(document); forged["selection_mapping"]["phase1_entry_boundary"] = "after_step_011"
    killtest.atomic_json(root / "phase2/phase2_manifest.json", forged)
    with pytest.raises(killtest.GlobalStopError, match="selection mapping"):
        killtest.run_analyze_phase2(config, CONFIG_PATH, root)
    killtest.atomic_json(root / "phase2/phase2_manifest.json", document)
    assert killtest.run_analyze_phase2(config, CONFIG_PATH, root)["selected_step"] == 20


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
