"""CPU contracts for the single-request resource-lifetime characterization."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments import video_resource_lifetime_profile as rl

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/video_resource_lifetime_profile_config.yaml")


def _config():
    return rl.load_config(CONFIG_PATH)


def _manifest():
    config = _config()
    split = rl.expected_schedule_split(config)
    return {"expected_step_components": split["components"], "frozen_schedule": split["schedule"], "kill_rules": {"ub1_stop_below": 0.10, "ub2_stop_below": 0.20}}


def test_config_frozen_and_split_is_26_14():
    config = _config()
    split = rl.expected_schedule_split(config)
    assert split["components"].count("transformer") == 26 and split["components"].count("transformer_2") == 14
    assert split["components"][:26] == ["transformer"] * 26  # high-noise expert first, then low-noise
    assert config["kill_rules"]["ub1_stop_below"] == 0.10 and config["kill_rules"]["ub2_stop_below"] == 0.20


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["kill_rules"].__setitem__("ub1_stop_below", 0.05),
        lambda c: c["kill_rules"].__setitem__("ub2_stop_below", 0.10),
        lambda c: c["generation"].__setitem__("num_inference_steps", 30),
        lambda c: c["runs_per_mode"].__setitem__("measured", 3),
        lambda c: c["runs_per_mode"].__setitem__("plain", 0),
        lambda c: c["probe_overhead_control"].__setitem__("invalid_above", 0.5),
        lambda c: c.__setitem__("offload_modes", ["on"]),
        lambda c: c["request"].__setitem__("prompt", "changed"),
        lambda c: c["expected_step_split"].__setitem__("transformer_steps", 25),
    ],
)
def test_config_mutations_fail_closed(tmp_path, mutate):
    config = json.loads(CONFIG_PATH.read_text())
    mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(rl.GlobalStopError):
        rl.load_config(path)


def test_synthetic_profile_decomposition_and_upper_bounds():
    manifest = _manifest()
    profile = rl._synthetic_profile(manifest)
    result = rl.analyze_events(profile, manifest)
    assert result["schema_valid"], result["schema"]
    t = result["time"]
    assert abs(t["accounted_ms"] - t["wall_ms"]) < 1e-6
    assert t["transformer_compute_ms"] == pytest.approx(26 * 4500.0)
    assert t["transformer_2_compute_ms"] == pytest.approx(14 * 4500.0)
    assert t["transformer_load_swap_ms"] == pytest.approx(2000.0) and t["transformer_to_transformer_2_swap_ms"] == pytest.approx(2000.0)
    assert t["text_encode_swap_ms"] == pytest.approx(500.0) and t["vae_load_swap_ms"] == pytest.approx(300.0)
    assert t["vae_decode_compute_ms"] == pytest.approx(12000.0)
    assert t["swap_event_count"] == 4
    expected_wall = 180000.0 + 4800.0 + 1500.0 + 200.0 + 100.0 + 12000.0 + 50.0
    assert t["wall_ms"] == pytest.approx(expected_wall)
    assert result["ub1_time_overlap"] == pytest.approx(1 - 180000.0 / expected_wall)
    assert result["ub1_actionable_swap_share"] == pytest.approx(4800.0 / expected_wall)
    m = result["memory"]
    assert m["request_owned_bytes"] == 68 * 2**30
    assert m["peak_live_bytes_estimate"] == 28 * 2**30 + int(2.5 * 2**30)
    assert m["peak_memory_allocated"] == 28 * 2**30 + int(2.5 * 2**30)
    # sequential offload already keeps only the executing component resident: no remaining headroom
    assert result["ub2_residency"] == pytest.approx(0.0)
    assert result["residency_already_captured_by_runtime"] == pytest.approx(1 - 30.5 / 68)
    assert result["stack_vs_ideal_live_set_descriptive"] == pytest.approx(1 - 30.5 / 68)
    assert m["components_mutually_exclusive_during_steps"]
    assert rl.decide(result["ub1_time_overlap"], result["ub2_residency"]) == "STOP_ALL"


def test_decision_thresholds():
    assert rl.decide(0.05, 0.10) == "STOP_ALL"
    assert rl.decide(0.05, 0.50) == "RESIDENCY_ONLY_TO_OFFLINE_ORACLE"
    assert rl.decide(0.30, 0.10) == "TIME_OVERLAP_ONLY_TO_OFFLINE_ORACLE"
    assert rl.decide(0.10, 0.20) == "JOINT_OFFLINE_ORACLE"  # thresholds are strict lower bounds
    assert rl.decide(None, 0.5) == "MEASUREMENT_INVALID" and rl.decide(0.5, None) == "MEASUREMENT_INVALID"


def test_probe_overhead_control_and_correction():
    assert rl.probe_overhead_control(100.0, 100.5)["status"] == "VALID"
    mid = rl.probe_overhead_control(100.0, 103.0)
    assert mid["status"] == "VALID_WITH_CORRECTION" and mid["overhead_fraction"] == pytest.approx(0.03)
    assert rl.probe_overhead_control(100.0, 106.0)["status"] == "INVALID"
    assert rl.probe_overhead_control(0.0, 1.0)["status"] == "INVALID"
    # correction: plain wall = plain client - (measured client - measured pipeline wall)
    result = {"time": {"wall_ms": 200.0, "ideal_wall_ms_compute_only": 170.0}}
    control = {"plain_client_ms": 200.0, "measured_client_ms": 210.0}  # pipeline wall 200 of client 210 -> plain pipeline wall 190
    assert rl.corrected_ub1(result, control) == pytest.approx(1 - 170.0 / 190.0)
    assert rl.corrected_ub1({"time": {"wall_ms": 5.0, "ideal_wall_ms_compute_only": 1.0}}, {"plain_client_ms": 1.0, "measured_client_ms": 10.0}) is None


@pytest.mark.parametrize(
    "mutate,flag",
    [
        (lambda p: p["events"].pop(1), "event_order_ok"),
        (lambda p: [e for e in p["events"] if e["event"] == "step_end"][0].__setitem__("component", "transformer_2"), "step_split_ok"),
        (lambda p: [e for e in p["events"] if e["event"] == "decode_end"][0].__setitem__("decode_skipped", True), "decode_not_skipped"),
        (lambda p: [e for e in p["events"] if e["event"] == "step_end"][3].__setitem__("timestep", 1.0), "timesteps_match_schedule"),
        (lambda p: p["events"].remove([e for e in p["events"] if e["event"] == "step_end"][5]), "step_count_ok"),
        (lambda p: p["static_component_bytes"].__setitem__("vae", 0), "static_component_bytes_ok"),
        (lambda p: p["offload_events"][0].__setitem__("bytes_to_gpu", -1), "swap_events_well_formed"),
        (lambda p: p["events"][2].pop("memory_allocated"), "memory_fields_present"),
    ],
)
def test_schema_validation_fails_closed(mutate, flag):
    manifest = _manifest()
    profile = rl._synthetic_profile(manifest)
    mutate(profile)
    schema = rl.validate_profile_schema(profile, manifest)
    assert schema[flag] is False and schema["valid"] is False


def test_swaps_in_uses_half_open_interval():
    swaps = [{"t_start": 1.0, "t_end": 1.5, "bytes_to_cpu": 1, "bytes_to_gpu": 2}, {"t_start": 2.0, "t_end": 2.25, "bytes_to_cpu": 3, "bytes_to_gpu": 4}]
    # a swap starting exactly at an event timestamp belongs to the following interval
    assert rl._swaps_in(swaps, 0.0, 1.0) == (0.0, 0, 0, 0)
    assert rl._swaps_in(swaps, 1.0, 2.0) == (500.0, 1, 2, 1)
    assert rl._swaps_in(swaps, 2.0, 3.0) == (250.0, 3, 4, 1)


def test_trusted_final_latent_binding():
    config = _config()
    path = rl.REPO_ROOT / config["request"]["trusted_final_latent_path"]
    if not path.exists():
        pytest.skip("trusted v3 root not present")
    array = rl.trusted_final_latent(config)
    assert array.shape == (1, 16, 9, 60, 104)
    bad = copy.deepcopy(config)
    bad["request"]["trusted_final_latent_tensor_sha256"] = "00" * 32
    with pytest.raises(rl.GlobalStopError):
        rl.trusted_final_latent(bad)


def test_profile_sampling_params_request_probes():
    pytest.importorskip("torch")
    try:
        sampling = rl.profile_sampling_params(_config(), seed=1, label="x", artifact_dir=Path("/tmp/x"))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"vllm_omni sampling params unavailable: {exc}")
    assert sampling.extra_args["resource_lifetime_probe"]["request_label"] == "x"
    assert sampling.extra_args["trajectory_probe"]["capture_steps"] == [40]
    assert "execution_step_limit" not in sampling.extra_args and "skip_vae_decode" not in sampling.extra_args
    plain = rl.profile_sampling_params(_config(), seed=1, label="x", artifact_dir=Path("/tmp/x"), instrumented=False)
    assert "resource_lifetime_probe" not in plain.extra_args and plain.extra_args["trajectory_probe"]["capture_steps"] == [40]


# ---------------------------------------------------------------- audit acceptance checks (frozen)
def _valid_profile():
    manifest = _manifest()
    return manifest, rl._synthetic_profile(manifest)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5000.0, "4500"])
def test_nonfinite_or_negative_latency_is_measurement_invalid(bad):
    manifest, profile = _valid_profile()
    [e for e in profile["events"] if e["event"] == "step_end"][5]["step_latency_ms"] = bad
    result = rl.analyze_events(profile, manifest)
    assert result["schema_valid"] is False and result["schema"]["step_latencies_finite_nonnegative"] is False
    assert result["ub1_time_overlap"] is None and result["ub2_residency"] is None
    assert rl.decide(result["ub1_time_overlap"], result["ub2_residency"]) == "MEASUREMENT_INVALID"


def test_nonfinite_timestamp_or_swap_is_rejected():
    manifest, profile = _valid_profile()
    profile["events"][4]["t"] = float("nan")
    assert rl.validate_profile_schema(profile, manifest)["time_monotone"] is False
    manifest, profile = _valid_profile()
    profile["offload_events"][1]["t_end"] = float("inf")
    assert rl.validate_profile_schema(profile, manifest)["swap_events_well_formed"] is False


def test_core_compute_exceeding_plain_wall_is_hard_invalid():
    result = {"time": {"wall_ms": 100.0, "ideal_wall_ms_compute_only": 95.0}}
    assert rl.corrected_ub1(result, {"plain_client_ms": 90.0, "measured_client_ms": 100.0}) is None
    assert rl.corrected_ub1(result, {"plain_client_ms": 105.0, "measured_client_ms": 100.0}) == pytest.approx(1 - 95.0 / 105.0)
    assert rl.corrected_ub1(result, {"plain_client_ms": float("nan"), "measured_client_ms": 100.0}) is None
    assert rl.decide(None, 0.5) == "MEASUREMENT_INVALID"


def test_schema_invalid_profile_emits_no_decision_quantities():
    manifest, profile = _valid_profile()
    [e for e in profile["events"] if e["event"] == "step_end"][0].pop("memory_allocated")
    result = rl.analyze_events(profile, manifest)
    assert result["schema_valid"] is False and result["decision_eligible"] is False
    assert "time" not in result and "memory" not in result
    assert all(result[key] is None for key in rl.INVALID_RESULT_KEYS)
    assert rl.decide(result["ub1_time_overlap"], result["ub2_residency"]) == "MEASUREMENT_INVALID"


def test_out_of_range_step_index_is_rejected_cleanly():
    manifest, profile = _valid_profile()
    steps = [e for e in profile["events"] if e["event"] == "step_end"]
    extra = dict(steps[-1], step_index=40, t=steps[-1]["t"] + 1.0)
    profile["events"].insert(profile["events"].index(steps[-1]) + 1, extra)
    schema = rl.validate_profile_schema(profile, manifest)  # must not raise
    assert schema["step_indices_in_range"] is False and schema["step_count_ok"] is False and schema["valid"] is False
    manifest, profile = _valid_profile()
    [e for e in profile["events"] if e["event"] == "step_end"][3]["step_index"] = -1
    assert rl.validate_profile_schema(profile, manifest)["step_indices_in_range"] is False


def test_failure_classification_distinguishes_memory_exhaustion():
    assert rl.classify_failure(RuntimeError("CUDA out of memory. Tried to allocate 28.00 GiB")) == "OFF_RUN_INFEASIBLE"
    assert rl.classify_failure(RuntimeError("cudaErrorMemoryAllocation: out of memory")) == "OFF_RUN_INFEASIBLE"
    wrapped = RuntimeError("worker died")
    wrapped.__cause__ = MemoryError("unable to allocate 4 GiB")
    assert rl.classify_failure(wrapped) == "OFF_RUN_INFEASIBLE"
    assert rl.classify_failure(ImportError("No module named 'aenum'")) == "OFF_RUN_ERROR"
    assert rl.classify_failure(RuntimeError("Resume step_index must be in [0, 39]")) == "OFF_RUN_ERROR"
    assert rl.classify_failure(KeyError("trajectory_probe_metadata_path")) == "OFF_RUN_ERROR"


def test_swap_exceeding_step_latency_fails_closed():
    manifest, profile = _valid_profile()
    profile["offload_events"][1]["t_end"] = profile["offload_events"][1]["t_start"] + 10.0  # 10 s swap inside a 6.5 s step
    with pytest.raises(rl.GlobalStopError, match="swap time"):
        rl.analyze_events(profile, manifest)


def test_offloader_recording_is_off_by_default_and_skips_accounting(monkeypatch):
    try:
        from vllm_omni.diffusion.offloader import sequential_backend as sb
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"offloader import unavailable: {exc}")
    import torch

    assert sb._RECORDING_ENABLED is False
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return 0

    monkeypatch.setattr(sb, "_module_bytes", counting)
    hook = sb.SequentialOffloadHook(offload_targets=[torch.nn.Linear(2, 2)], device=torch.device("cpu"), pin_memory=False)
    hook.pre_forward(torch.nn.Linear(2, 2))
    assert calls["n"] == 0 and sb.drain_offload_events() == []
    sb.enable_offload_event_recording()
    try:
        hook.pre_forward(torch.nn.Linear(2, 2))
        assert calls["n"] > 0
    finally:
        sb.disable_offload_event_recording()
    assert sb._RECORDING_ENABLED is False


# ---------------------------------------------------------------- offload=on identity gate (frozen)
def test_offload_on_zero_swaps_is_measurement_invalid():
    manifest, profile = _valid_profile()
    profile["offload_events"] = []
    measured = rl.offload_on_identity(rl.analyze_events(profile, manifest))
    assert measured["schema_valid"] is True  # the profile itself is well-formed ...
    assert measured["offload_on_identity"] == {"swap_event_count": 0, "components_mutually_exclusive_during_steps": True, "offload_on_valid": False}
    assert measured["decision_eligible"] is False and measured["ub1_time_overlap"] is None and measured["ub2_residency"] is None
    assert rl.decide(measured["ub1_time_overlap"], measured["ub2_residency"]) == "MEASUREMENT_INVALID"


def test_offload_on_without_mutual_exclusion_is_measurement_invalid():
    manifest, profile = _valid_profile()
    for event in profile["events"]:
        if event["event"] == "step_end" and event["component"] == "transformer_2":
            event["resident_component_bytes"]["transformer"] = 28 * 2**30  # expert A left resident during expert-B steps
            for key in ("memory_allocated", "max_memory_allocated_since_last_event", "memory_reserved", "max_memory_reserved_since_last_event"):
                event[key] += 28 * 2**30
    raw = rl.analyze_events(profile, manifest)
    assert raw["memory"]["components_mutually_exclusive_during_steps"] is False and raw["ub2_residency"] > 0.20  # would have looked like residency headroom
    measured = rl.offload_on_identity(raw)
    assert measured["offload_on_identity"]["offload_on_valid"] is False
    assert measured["decision_eligible"] is False and measured["ub2_residency"] is None
    assert rl.decide(measured["ub1_time_overlap"], measured["ub2_residency"]) == "MEASUREMENT_INVALID"


def test_offload_on_valid_run_keeps_frozen_ub_values():
    manifest, profile = _valid_profile()
    raw = rl.analyze_events(profile, manifest)
    measured = rl.offload_on_identity(raw)
    assert measured["offload_on_identity"]["offload_on_valid"] is True
    assert measured["decision_eligible"] is True
    assert measured["ub1_time_overlap"] == raw["ub1_time_overlap"] == pytest.approx(0.0938837, abs=1e-6)
    assert measured["ub2_residency"] == raw["ub2_residency"] == pytest.approx(0.0)
    assert rl.decide(measured["ub1_time_overlap"], measured["ub2_residency"]) == "STOP_ALL"


def test_identity_gate_is_not_applied_to_off_and_never_enters_decide():
    manifest, profile = _valid_profile()
    profile["offload_events"] = []  # an OFF run legitimately has no swaps
    off = rl.analyze_events(profile, manifest)
    assert off["schema_valid"] and off["decision_eligible"] is True  # OFF is analysed descriptively without the identity gate
    import inspect

    assert list(inspect.signature(rl.decide).parameters) == ["ub1", "ub2"]
    source = inspect.getsource(rl.run_analyze)
    assert 'if mode == "on":' in source and "offload_on_identity(measured)" in source


def test_forward_wrapper_releases_recorder_on_exception(monkeypatch):
    try:
        from vllm_omni.diffusion.offloader import sequential_backend as sb
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"pipeline/offloader import unavailable: {exc}")

    class _Boom(Exception):
        pass

    def failing_impl(self, req, *args, **kwargs):
        sb.enable_offload_event_recording()
        sb._OFFLOAD_EVENTS.append({"module": "x"})
        raise _Boom()

    monkeypatch.setattr(Wan22Pipeline, "_forward_impl", failing_impl)
    pipeline = Wan22Pipeline.__new__(Wan22Pipeline)
    with pytest.raises(_Boom):
        Wan22Pipeline.forward(pipeline, object())
    assert sb._RECORDING_ENABLED is False and sb._OFFLOAD_EVENTS == []
