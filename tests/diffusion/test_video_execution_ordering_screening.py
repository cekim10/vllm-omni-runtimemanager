"""CPU contracts for the preregistered execution-ordering screening."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import video_execution_ordering_screening as screening

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/video_execution_ordering_screening_config.yaml")


def _config():
    return screening.load_config(CONFIG_PATH)


def _trusted_roots_present(config) -> bool:
    v3 = screening.REPO_ROOT / config["trusted_v3"]["root"] / "run" / "trajectories"
    phase1 = screening.REPO_ROOT / config["trusted_phase1"]["root"] / "phase1" / "trace_manifest.json"
    return v3.exists() and phase1.exists()


# ---------------------------------------------------------------- config freeze
def test_config_loads_and_matches_frozen_constants():
    config = _config()
    assert tuple(row["name"] for row in config["candidates"]) == screening.CANDIDATE_NAMES
    assert config["primary"]["meaningful_reversal"]["y_abs_floor"] == screening.Y_ABS_FLOOR == 1e-4
    assert config["primary"]["pair_eligibility"]["x_ratio_max"] == screening.X_RATIO_MAX == 2.0
    assert config["primary"]["continue_rule"] == {"min_distinct_prompts": 4, "min_distinct_steps": 2, "min_distinct_pair_types": 2}
    assert len(config["trusted_v3"]["trajectory_ids"]) == 12


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["primary"]["meaningful_reversal"].__setitem__("y_abs_floor", 1e-5),
        lambda c: c["primary"]["meaningful_reversal"].__setitem__("y_ratio_min", 1.5),
        lambda c: c["primary"]["pair_eligibility"].__setitem__("x_ratio_max", 3.0),
        lambda c: c["primary"]["continue_rule"].__setitem__("min_distinct_prompts", 3),
        lambda c: c["candidates"].pop(),
        lambda c: c["candidates"].append({"name": "int2_group4", "tier": "int2", "format": "grouped_symmetric_v1", "bits": 2, "group_count": 4}),
        lambda c: c.__setitem__("checkpoint_steps", [5, 10, 20]),
        lambda c: c["candidates"][2].__setitem__("group_count", 2),
        lambda c: c["generation"].__setitem__("num_inference_steps", 30),
        lambda c: c["trusted_v3"]["trajectory_ids"].pop(),
    ],
)
def test_config_mutations_fail_closed(tmp_path, mutate):
    config = json.loads(CONFIG_PATH.read_text())
    mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(screening.GlobalStopError):
        screening.load_config(path)


# ---------------------------------------------------------------- quantizer
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("count", [1, 7, 8, 9, 1000])
def test_pack_unpack_round_trip(bits, count):
    rng = np.random.default_rng(count * 31 + bits)
    values = rng.integers(0, 1 << bits, size=count).astype(np.uint8)
    packed = screening.pack_unsigned(values, bits)
    assert len(packed) == (count if bits == 8 else (count + 1) // 2)
    assert np.array_equal(screening.unpack_unsigned(packed, bits, count), values)


def test_quantize_values_scale_rounding_and_clamp():
    values = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    unsigned, scale, restored = screening.quantize_values(values, 8)
    assert scale == 1.0 / 127
    assert unsigned.dtype == np.uint8 and unsigned.min() == 0 and unsigned.max() == 254
    assert np.array_equal(restored[[0, 5]], np.array([-1.0, 1.0], dtype=np.float32))
    # exact half rounds to even: 0.5 / (1/127) = 63.5 -> 64 (even), 0.25/(1/127) = 31.75 -> 32
    assert unsigned[4] - 127 == 64 and unsigned[3] - 127 == 32
    zeros = np.zeros(5, dtype=np.float32)
    unsigned0, scale0, restored0 = screening.quantize_values(zeros, 4)
    assert scale0 == 1.0 and np.all(unsigned0 == 7) and np.all(restored0 == 0)


@pytest.mark.parametrize("bits,group_count", [(8, 1), (8, 4), (8, 16), (4, 1), (4, 8), (4, 16)])
def test_grouped_encode_decode_round_trip(bits, group_count):
    rng = np.random.default_rng(bits * 100 + group_count)
    clean = rng.standard_normal((1, 16, 2, 4, 6)).astype(np.float32)
    clean[:, 3] *= 50.0  # one loud channel makes per-tensor vs grouped scales differ
    payload, metadata = screening.encode_grouped(clean, bits, group_count)
    assert metadata["group_count"] == group_count and len(metadata["groups"]) == group_count
    assert sum(g["nbytes"] for g in metadata["groups"]) == len(payload)
    per_group_elements = [g["count"] for g in metadata["groups"]]
    assert sum(per_group_elements) == clean.size
    restored = screening.decode_grouped(payload, metadata)
    assert restored.shape == clean.shape and restored.dtype == np.float32
    # decode is a function of bytes+metadata only: re-decoding gives identical bits
    assert screening.decode_grouped(bytes(payload), json.loads(json.dumps(metadata))).tobytes() == restored.tobytes()
    # quantization error bounded by half a step per group
    for (start, end), g in zip(screening.group_ranges(16, group_count), metadata["groups"], strict=True):
        step = g["scale"]
        assert np.max(np.abs(restored[:, start:end] - clean[:, start:end])) <= step / 2 + 1e-6


def test_grouped_scales_differ_between_granularities():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((1, 16, 2, 4, 6)).astype(np.float32)
    clean[:, 0] *= 40.0
    _, m1 = screening.encode_grouped(clean, 8, 1)
    _, m4 = screening.encode_grouped(clean, 8, 4)
    assert len({g["scale"] for g in m4["groups"]}) > 1
    assert m1["groups"][0]["scale"] == max(g["scale"] for g in m4["groups"])


def test_quantizer_matches_trusted_torch_reference():
    torch = pytest.importorskip("torch")
    try:
        from experiments import video_propagation_aware_checkpoint_killtest as reference
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"reference module unavailable: {exc}")
    rng = np.random.default_rng(7)
    values = (rng.standard_normal(5000) * rng.uniform(0.01, 3.0)).astype(np.float32)
    for bits in (4, 8):
        unsigned, scale, restored = screening.quantize_values(values, bits)
        ref_unsigned, ref_scale, ref_restored = reference._quantize_values(torch.from_numpy(values.copy()), bits)
        assert scale == ref_scale
        assert np.array_equal(unsigned, ref_unsigned)
        assert np.array_equal(restored, ref_restored.numpy())


def test_decode_candidate_control_formats():
    rng = np.random.default_rng(3)
    clean = screening.base.cast_runtime_bf16(rng.standard_normal((1, 16, 1, 2, 2)).astype(np.float32) * 1e-3)
    specs = {row["name"]: row for row in _config()["candidates"]}
    payload, metadata, restored = screening.encode_candidate(clean, specs["bf16"])
    assert metadata["format"] == "bf16_bits" and len(payload) == 2 * clean.size
    assert np.array_equal(restored, clean)
    assert np.array_equal(screening.decode_candidate(payload, metadata), clean)
    payload16, metadata16, restored16 = screening.encode_candidate(clean, specs["fp16"])
    assert metadata16["format"] == "fp16" and len(payload16) == 2 * clean.size
    assert np.array_equal(screening.decode_candidate(payload16, metadata16), restored16)
    assert np.array_equal(restored16, clean.astype(np.float16).astype(np.float32))
    with pytest.raises(screening.GlobalStopError):
        screening.decode_candidate(payload, {"format": "int2", "shape": list(clean.shape)})


# ---------------------------------------------------------------- metrics and rules
def test_relative_l2_and_error_metrics():
    reference = np.array([3.0, 4.0], dtype=np.float32)
    candidate = np.array([3.0, 4.5], dtype=np.float32)
    assert screening.relative_l2(reference, candidate) == pytest.approx(0.1)
    metrics = screening.error_metrics(reference, candidate)
    assert metrics["changed_element_count"] == 1 and metrics["changed_element_fraction"] == 0.5 and not metrics["bit_exact"]
    assert screening.error_metrics(reference, reference)["bit_exact"]
    with pytest.raises(screening.GlobalStopError):
        screening.relative_l2(np.zeros(3, dtype=np.float32), candidate[:1].repeat(3))


def test_pair_eligibility_rules():
    # X ratio exactly 2 is eligible; above 2 is not; zero X never eligible; equal X not orderable
    assert screening.evaluate_pair(1.0, 0.0, 2.0, 0.0)["eligible"]
    assert not screening.evaluate_pair(1.0, 0.0, 2.0000001, 0.0)["eligible"]
    assert not screening.evaluate_pair(0.0, 0.0, 1.0, 0.0)["eligible"]
    equal = screening.evaluate_pair(1.0, 5.0, 1.0, 0.1)
    assert equal["eligible"] and not equal["orderable"] and not equal["meaningful"]


def test_meaningful_reversal_requires_ratio_and_floor():
    # lower-X candidate has the larger Y
    assert screening.evaluate_pair(1.0, 2e-3, 1.5, 1e-4)["meaningful"]
    # ratio satisfied, floor not: 2e-9 vs 1e-9
    verdict = screening.evaluate_pair(1.0, 2e-9, 1.5, 1e-9)
    assert verdict["reversal"] and not verdict["meaningful"]
    # floor satisfied, ratio not: 0.15 vs 0.1
    verdict = screening.evaluate_pair(1.0, 0.15, 1.5, 0.1)
    assert verdict["reversal"] and not verdict["meaningful"]
    # exact boundary values count
    assert screening.evaluate_pair(1.0, 2e-4, 1.5, 1e-4)["meaningful"]
    # ordering preserved -> no reversal regardless of magnitude
    assert not screening.evaluate_pair(1.0, 0.1, 1.5, 0.9)["reversal"]
    # orientation symmetric
    assert screening.evaluate_pair(1.5, 1e-4, 1.0, 2e-3)["meaningful"]


def _meaningful(prompt, step, pair):
    return {"prompt_id": prompt, "step": step, "pair_type": pair, "meaningful": True}


def test_decision_rule_thresholds():
    p8 = "int8:int8_group4|int8_group8"
    p4 = "int4:int4_group4|int4_group8"
    rows = [_meaningful(f"p{i}", 10, p8) for i in range(4)] + [_meaningful("p0", 20, p4)]
    assert screening.decide(rows)["decision"] == "CONTINUE"
    assert screening.decide(rows[:4])["decision"] == "NO_GO"  # one step, one pair type
    assert screening.decide([_meaningful(f"p{i}", 10, p8) for i in range(3)] + [_meaningful("p0", 20, p4)])["decision"] == "NO_GO"
    assert screening.decide([_meaningful(f"p{i}", 10 + 10 * (i % 2), p8) for i in range(6)])["decision"] == "NO_GO"  # one pair type
    assert screening.decide([])["decision"] == "NO_GO"


def test_pair_type_is_order_independent_and_rules_frozen():
    assert screening.pair_type("int8", "b", "a") == screening.pair_type("int8", "a", "b") == "int8:a|b"
    rules = {"primary_tiers": ["int8", "int4"], "x_ratio_max": 2.0, "y_ratio_min": 2.0, "y_abs_floor": 1e-4, "min_distinct_prompts": 4, "min_distinct_steps": 2, "min_distinct_pair_types": 2}
    assert screening.rules_frozen(rules)
    assert not screening.rules_frozen({**rules, "y_abs_floor": 1e-3})


# ---------------------------------------------------------------- trusted bindings (local roots)
def test_trusted_phase1_binding_and_schedule():
    config = _config()
    if not _trusted_roots_present(config):
        pytest.skip("trusted result roots not present")
    phase1 = screening.trusted_phase1(config)
    assert set(phase1["inputs"]) == set(screening.ANCHOR_TRAJECTORIES)
    assert phase1["expected_outputs"]["PLUS1"]["canonical_identity"] == phase1["expected_outputs"]["HISTORICAL_PLUS14"]["canonical_identity"]
    assert phase1["expected_outputs"]["CLEAN"]["canonical_identity"] != phase1["expected_outputs"]["PLUS1"]["canonical_identity"]
    frozen = screening.schedule(config)
    assert screening.loc.timestep_matches(phase1["resume_timestep"], frozen[10], frozen)
    bad = copy.deepcopy(config)
    bad["trusted_phase1"]["trace_manifest_sha256"] = "00" * 32
    with pytest.raises(screening.GlobalStopError):
        screening.trusted_phase1(bad)


def test_trusted_v3_manifest_binding():
    config = _config()
    if not _trusted_roots_present(config):
        pytest.skip("trusted result roots not present")
    _, manifest = screening.load_v3_manifest(config, "recovery_008_9234")
    for step in (9, 10, 19, 20, 29, 30):
        screening.v3_state_record(manifest, step)
    with pytest.raises(screening.GlobalStopError):
        screening.v3_state_record(manifest, 11)
    bad = copy.deepcopy(config)
    bad["trusted_v3"]["config_hash"] = "00" * 32
    with pytest.raises(screening.GlobalStopError):
        screening.load_v3_manifest(bad, "recovery_008_9234")


def test_one_step_sampling_params_request_bounded_execution():
    pytest.importorskip("torch")
    try:
        sampling = screening.one_step_sampling_params(_config(), seed=1, label="x", artifact_dir=Path("/tmp/x"), latents=__import__("torch").zeros(1), step_index=10)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"vllm_omni sampling params unavailable: {exc}")
    assert sampling.extra_args["execution_step_limit"] == 1
    assert sampling.extra_args["skip_vae_decode"] is True
    assert sampling.extra_args["trajectory_probe"]["capture_steps"] == [0, 1]
    assert sampling.step_index == 10


def test_pipeline_bounded_execution_helpers():
    try:
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"pipeline import unavailable: {exc}")

    class _Params:
        def __init__(self, extra):
            self.extra_args = extra

    class _Req:
        def __init__(self, extra):
            self.sampling_params = _Params(extra)

    assert Wan22Pipeline._resolve_execution_step_limit(_Req({}), 30) is None
    assert Wan22Pipeline._resolve_execution_step_limit(_Req({"execution_step_limit": 1}), 30) == 1
    for bad in ({"execution_step_limit": 0}, {"execution_step_limit": 31}, {"execution_step_limit": True}, {"execution_step_limit": "1"}):
        with pytest.raises(ValueError):
            Wan22Pipeline._resolve_execution_step_limit(_Req(bad), 30)
    assert Wan22Pipeline._resolve_skip_vae_decode(_Req({})) is False
    assert Wan22Pipeline._resolve_skip_vae_decode(_Req({"skip_vae_decode": True})) is True
    with pytest.raises(ValueError):
        Wan22Pipeline._resolve_skip_vae_decode(_Req({"skip_vae_decode": 1}))


# ---------------------------------------------------------------- synthetic end-to-end analysis
_SMALL_SHAPE = (1, 16, 1, 2, 2)


def _bf16(values):
    return screening.base.cast_runtime_bf16(np.asarray(values, dtype=np.float32))


def _candidate_state(clean, target_x, rng):
    """A BF16-exact state at approximately relative L2 `target_x` from `clean` (exact X recomputed later)."""
    direction = rng.standard_normal(clean.shape).astype(np.float32)
    direction /= np.linalg.norm(direction)
    return _bf16(clean + target_x * np.linalg.norm(clean) * direction)


def _next_state(reference_next, target_y, rng):
    """Float32 next state at relative L2 `target_y` from the reference.

    Deliberately NOT rounded to BF16: on a 64-element fixture BF16 rounding of O(1) values
    would perturb a 1e-2-scale offset by tens of percent and scramble the intended ordering.
    The analyser only requires candidate INPUT states to be BF16-exact.
    """
    if target_y == 0.0:
        return reference_next.copy()
    direction = rng.standard_normal(reference_next.shape).astype(np.float32)
    direction /= np.linalg.norm(direction)
    return (reference_next + np.float32(target_y * np.linalg.norm(reference_next)) * direction).astype(np.float32)


def _synthetic_screening(tmp_path, monkeypatch, *, y_plan):
    """Build a complete synthetic result root.

    `y_plan(cell_key, candidate_name, x_value) -> target Y` controls the next-state error of each
    candidate so tests can inject or withhold ordering reversals.
    """
    config = _config()
    rng = np.random.default_rng(1234)
    prov = {"provenance_hash": "p" * 64, "git_commit": "deadbeef", "source_dirty_entries": []}
    specs = {row["name"]: row for row in config["candidates"]}
    prompts = {f"prompt_{index:02d}": {"trajectory_id": f"prompt_{index:02d}_1", "generation_seed": 1, "prompt": "x"} for index in range(12)}
    cells = {}
    trace = {"provenance_hash": prov["provenance_hash"], "manifest_sha256": None, "cells": {}}
    cleans = {}
    x_targets = {"int8_per_tensor": 0.012, "int8_group4": 0.011, "int8_group8": 0.0105, "int8_group16": 0.010,
                 "int4_per_tensor": 0.21, "int4_group4": 0.195, "int4_group8": 0.19, "int4_group16": 0.18}
    for prompt_id in prompts:
        for step in screening.CHECKPOINT_STEPS:
            key = screening.cell_key(prompt_id, step)
            clean = _bf16(rng.standard_normal(_SMALL_SHAPE).astype(np.float32))
            cleans[key] = clean
            reference_next = _bf16(clean + 0.05 * rng.standard_normal(_SMALL_SHAPE).astype(np.float32))
            cell_dir = f"screening/{prompt_id}/step{step:03d}"
            reference_record = screening.save_array(tmp_path, f"{cell_dir}/reference/next_state.npy", reference_next)
            candidates = {}
            runs = {}
            for name in screening.CANDIDATE_NAMES:
                if name == "bf16":
                    state = clean.copy()
                elif name == "fp16":
                    state = _bf16(clean.astype(np.float16).astype(np.float32))
                else:
                    state = _candidate_state(clean, x_targets[name], rng)
                x = screening.error_metrics(clean, state)
                state_record = screening.save_array(tmp_path, f"candidates/{prompt_id}/step{step:03d}/{name}.runtime_state.npy", state)
                candidates[name] = {"name": name, "tier": specs[name]["tier"], "serialized_bytes": 1, "byte_fraction_vs_bf16_runtime": 1.0, "runtime_state": state_record, "x": x}
                target_y = 0.0 if name == "bf16" else y_plan(key, name, x["relative_l2"])
                next_state = _next_state(reference_next, target_y, rng)
                run_record = screening.save_array(tmp_path, f"{cell_dir}/{name}/next_state.npy", next_state)
                runs[name] = {"next_state": run_record, "semantics": {"valid": True, "observed_timestep": screening.schedule(config)[step]}}
            cells[key] = {"prompt_id": prompt_id, "trajectory_id": prompts[prompt_id]["trajectory_id"], "generation_seed": 1, "step": step,
                          "clean_state": {"canonical_identity": screening.identity(clean)}, "candidates": candidates}
            trace["cells"][key] = {"reference": {"next_state": reference_record, "semantics": {"valid": True, "observed_timestep": screening.schedule(config)[step]}}, "runs": runs}
    manifest = {
        "provenance_hash": prov["provenance_hash"],
        "frozen_rules": {"primary_tiers": ["int8", "int4"], "x_ratio_max": 2.0, "y_ratio_min": 2.0, "y_abs_floor": 1e-4,
                         "min_distinct_prompts": 4, "min_distinct_steps": 2, "min_distinct_pair_types": 2},
        "execution": {"frozen_schedule": screening.schedule(config)},
        "prompts": prompts,
        "cells": cells,
        "manifest_sha256": "m" * 64,
    }
    trace["manifest_sha256"] = manifest["manifest_sha256"]
    validation = {"anchor": {name: {} for name in screening.ANCHOR_TRAJECTORIES}, "transitions": {key: {} for key in cells}}
    monkeypatch.setattr(screening, "load_cell_clean", lambda config_, cell: cleans[screening.cell_key(cell["prompt_id"], cell["step"])])
    return config, prov, manifest, trace, validation


def _monotone_plan(key, name, x):
    return x * 1.5  # Y ordering follows X ordering exactly


def test_synthetic_analysis_no_go_when_ordering_is_preserved(tmp_path, monkeypatch):
    config, prov, manifest, trace, validation = _synthetic_screening(tmp_path, monkeypatch, y_plan=_monotone_plan)
    result = screening.analyze_artifacts(tmp_path, config, prov, manifest, trace, validation)
    assert [row["name"].split(" ", 1)[0] for row in result["gates"]] == list(screening.ANALYZE_REQUIRED_GATES)
    assert all(row["status"] == "PASS" for row in result["gates"])
    assert result["decision"] == "NO_GO"
    assert result["descriptive"]["eligible_pairs"] == 432 and result["descriptive"]["reversals_any"] == 0
    assert result["descriptive"]["controls"]["bf16_next_state_bit_exact_all_cells"]
    assert len(result["rows"]) == 360
    assert all(row["descriptive_prediction_path_relative_l2"] is not None for row in result["rows"])


def _reversal_plan(reversal_cells):
    def plan(key, name, x):
        if key in reversal_cells and name in ("int8_group16", "int4_group16"):
            return x * 6.0  # finest granularity (lowest X) gets a much worse next-state error
        return x * 1.5
    return plan


def test_synthetic_analysis_continue_when_reversals_are_broad(tmp_path, monkeypatch):
    cells = {screening.cell_key(f"prompt_{i:02d}", 10) for i in range(4)} | {screening.cell_key("prompt_00", 20)}
    config, prov, manifest, trace, validation = _synthetic_screening(tmp_path, monkeypatch, y_plan=_reversal_plan(cells))
    result = screening.analyze_artifacts(tmp_path, config, prov, manifest, trace, validation)
    assert all(row["status"] == "PASS" for row in result["gates"])
    assert result["decision"] == "CONTINUE"
    summary = result["meaningful_reversal_summary"]
    assert len(summary["distinct_prompts"]) == 4 and summary["distinct_steps"] == [10, 20] and len(summary["distinct_pair_types"]) >= 2


def test_synthetic_analysis_no_go_when_reversals_are_narrow(tmp_path, monkeypatch):
    cells = {screening.cell_key(f"prompt_{i:02d}", 10) for i in range(6)}  # six prompts but a single step
    config, prov, manifest, trace, validation = _synthetic_screening(tmp_path, monkeypatch, y_plan=_reversal_plan(cells))
    result = screening.analyze_artifacts(tmp_path, config, prov, manifest, trace, validation)
    assert result["decision"] == "NO_GO"
    assert len(result["meaningful_reversal_summary"]["distinct_prompts"]) == 6
    assert result["meaningful_reversal_summary"]["distinct_steps"] == [10]


def test_synthetic_analysis_reversal_below_floor_is_not_meaningful(tmp_path, monkeypatch):
    def plan(key, name, x):
        if name in ("int8_group16", "int4_group16"):
            return 1e-6 if name.startswith("int8") else 3e-5  # reversed vs neighbours but tiny
        return 1e-9
    config, prov, manifest, trace, validation = _synthetic_screening(tmp_path, monkeypatch, y_plan=plan)
    result = screening.analyze_artifacts(tmp_path, config, prov, manifest, trace, validation)
    assert result["descriptive"]["reversals_any"] > 0
    assert result["meaningful_reversal_summary"]["count"] == 0 and result["decision"] == "NO_GO"


def test_synthetic_analysis_fails_closed_on_tampering(tmp_path, monkeypatch):
    config, prov, manifest, trace, validation = _synthetic_screening(tmp_path, monkeypatch, y_plan=_monotone_plan)
    record = trace["cells"]["prompt_03/step020"]["runs"]["int4_group8"]["next_state"]
    array = np.load(tmp_path / record["relative_path"], allow_pickle=False)
    array.flat[0] = np.float32(0.5)
    np.save(tmp_path / record["relative_path"], array, allow_pickle=False)
    with pytest.raises(screening.GlobalStopError, match="file hash mismatch"):
        screening.analyze_artifacts(tmp_path, config, prov, manifest, trace, validation)


def test_synthetic_analysis_determinism_gate_fails_when_bf16_repeat_differs(tmp_path, monkeypatch):
    def plan(key, name, x):
        return x * 1.5
    config, prov, manifest, trace, validation = _synthetic_screening(tmp_path, monkeypatch, y_plan=plan)
    key = "prompt_05/step030"
    reference = np.load(tmp_path / trace["cells"][key]["reference"]["next_state"]["relative_path"], allow_pickle=False)
    altered = reference.copy()
    altered.flat[3] = _bf16(np.array([altered.flat[3] + 1e-2], dtype=np.float32))[0]
    trace["cells"][key]["runs"]["bf16"]["next_state"] = screening.save_array(tmp_path, f"screening/prompt_05/step030/bf16/next_state.npy", altered)
    result = screening.analyze_artifacts(tmp_path, config, prov, manifest, trace, validation)
    statuses = {row["name"].split(" ", 1)[0]: row["status"] for row in result["gates"]}
    assert statuses["S-A7"] == "FAIL"
    assert not result["descriptive"]["controls"]["bf16_next_state_bit_exact_all_cells"]


def test_synthetic_analysis_rejects_unbound_trace(tmp_path, monkeypatch):
    config, prov, manifest, trace, validation = _synthetic_screening(tmp_path, monkeypatch, y_plan=_monotone_plan)
    trace["manifest_sha256"] = "z" * 64
    with pytest.raises(screening.GlobalStopError, match="not bound"):
        screening.analyze_artifacts(tmp_path, config, prov, manifest, trace, validation)


def test_prediction_path_metric_isolates_update_change():
    clean = np.array([1.0, 2.0], dtype=np.float32)
    reference_next = np.array([1.5, 2.5], dtype=np.float32)      # update (0.5, 0.5)
    candidate = np.array([1.1, 2.0], dtype=np.float32)
    same_update = candidate + (reference_next - clean)             # identical update -> 0
    assert screening.prediction_path_relative_l2(clean, reference_next, candidate, same_update) == 0.0
    changed = candidate + np.array([0.5, 1.0], dtype=np.float32)   # update differs by (0, 0.5)
    assert screening.prediction_path_relative_l2(clean, reference_next, candidate, changed) == pytest.approx(0.5 / np.sqrt(0.5))
    assert screening.prediction_path_relative_l2(clean, clean, candidate, candidate) is None
