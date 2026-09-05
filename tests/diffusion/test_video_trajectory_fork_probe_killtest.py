"""CPU contracts for the Round 4A trajectory-conditioned forkability probe kill test."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments import video_trajectory_fork_probe_killtest as probe

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
CONFIG_PATH = Path("experiments/video_trajectory_fork_probe_killtest_config.yaml")
SOURCE_PRESENT = (probe.SOURCE_ROOT / "summary.json").exists() and (probe.SOURCE_ROOT / "preflight.json").exists()


def _config():
    return probe.load_config(CONFIG_PATH)


def test_frozen_sets_and_prompts():
    config = _config()
    assert tuple(config["seeds"]) == (2345, 3456, 4567) and tuple(config["k_values"]) == (10, 15, 20, 25)
    assert config["prompts"]["old"] == probe.OLD_PROMPT and config["prompts"]["new"] == probe.NEW_PROMPT
    assert probe.ORDINAL == {"NEW": 2, "MIXED_NEW_DOMINANT": 1, "MIXED_BALANCED": 0, "MIXED_OLD_DOMINANT": -1, "OLD": -2}
    assert probe.AUDIT_POINT == (3456, 15) and probe.PRIMARY_K == 15 and probe.SUPPORT_K == 20


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("seeds", [1234, 3456, 4567]),
        lambda c: c.__setitem__("k_values", [5, 10, 15, 20, 25]),
        lambda c: c["probe"].__setitem__("direction", "lower_probe_means_more_forkable"),
        lambda c: c["probe"].__setitem__("primary_signal", "response_cosine"),
        lambda c: c["probe"].__setitem__("eps", 1e-6),
        lambda c: c["gate"].__setitem__("low_forkability_seed", 3456),
        lambda c: c["gate"].__setitem__("primary_k", 20),
        lambda c: c["determinism_audit"].__setitem__("point", [2345, 15]),
        lambda c: c["determinism_audit"].__setitem__("tolerance", 1e-3),
        lambda c: c["ordinal"].__setitem__("MIXED_BALANCED", 1),
    ],
)
def test_config_mutations_fail_closed(tmp_path, mutate):
    config = json.loads(CONFIG_PATH.read_text())
    mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        probe.load_config(path)


def test_output_isolation():
    probe.validate_output_path(probe.REPO_ROOT / "results" / "video_trajectory_fork_probe_killtest")
    for bad in ("video_trajectory_fork_confirmatory", "video_trajectory_fork_killtest", "video_runtime_state_discovery_v3_corrected"):
        with pytest.raises(ValueError):
            probe.validate_output_path(probe.REPO_ROOT / "results" / bad)


def test_expected_key_set_24_plus_audit_and_duplicates():
    keys = probe.expected_keys()
    assert len(keys) == 26 and sum(1 for k in keys if k[3] == "primary") == 24
    assert {k for k in keys if k[3] == "audit"} == {(3456, 15, "old", "audit"), (3456, 15, "new", "audit")}
    rows = [{"seed": s, "k": k, "condition": c, "role": r} for s, k, c, r in sorted(keys)]
    probe.validate_key_set(rows, keys)
    with pytest.raises(probe.GateError, match="duplicates"):
        probe.validate_key_set(rows + [rows[0]], keys)
    with pytest.raises(probe.GateError, match="missing"):
        probe.validate_key_set(rows[:-1], keys)


# ------------------------------------------------------------------ formulas
def test_signal_formulas_on_synthetic_tensors():
    p_old = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    p_new = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)
    x_k = np.array([[1.0, 2.0, 2.0]], dtype=np.float32)
    sig = probe.signals(x_k, p_old, p_new, x_k + p_old, x_k + p_new)
    assert sig["relative_response_l2"] == pytest.approx(5.0 / 5.0)
    assert sig["response_delta_rms"] == pytest.approx(np.sqrt(25.0 / 3.0))
    assert sig["response_cosine"] == pytest.approx(25.0 / (5.0 * np.sqrt(50.0)))
    assert sig["differing_element_fraction"] == pytest.approx(1.0 / 3.0)
    assert sig["relative_step_effect"] == pytest.approx(5.0 / 3.0)
    assert sig["delta_over_response_rms"] == pytest.approx(np.sqrt(25.0 / 3.0) / np.sqrt(25.0 / 3.0))
    identical = probe.signals(None, p_old, p_old.copy(), None, None)
    assert identical["relative_response_l2"] == 0.0 and identical["differing_element_fraction"] == 0.0 and identical["response_cosine"] == pytest.approx(1.0, abs=1e-9)
    assert "relative_step_effect" not in identical


def test_spearman_with_ties_and_degenerate_input():
    assert probe.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert probe.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert probe.spearman([1, 2, 3, 4], [1, 1, 1, 1]) is None  # constant -> undefined
    assert probe.spearman([1, 2], [1, 2]) is None
    # ties get average ranks
    assert probe.spearman([1, 2, 3, 4], [1, 2, 2, 3]) == pytest.approx(0.9486833, abs=1e-6)


# ------------------------------------------------------------------ gates
def test_same_k_examples_from_preregistration():
    a = probe.same_k_pass({2345: 0.10, 3456: 0.20, 4567: 0.30})
    assert a["pass"] is True and a["min_gap"] == pytest.approx(0.10)
    b = probe.same_k_pass({2345: 0.25, 3456: 0.20, 4567: 0.30})
    assert b["pass"] is False and b["comparisons"]["3456"]["pass"] is False and b["comparisons"]["4567"]["pass"] is True
    tie = probe.same_k_pass({2345: 0.20, 3456: 0.20, 4567: 0.30})
    assert tie["pass"] is False  # strict inequality required


def test_within_k_concordance():
    c = probe.within_k_concordance({2345: 0.1, 3456: 0.3, 4567: 0.2}, {2345: 0, 3456: 2, 4567: 2})
    assert c["concordant"] == 2 and c["discordant"] == 0 and c["tied_label_pairs"] == 1 and c["contradicts_labels"] is False
    d = probe.within_k_concordance({2345: 0.5, 3456: 0.3, 4567: 0.2}, {2345: 0, 3456: 2, 4567: 2})
    assert d["discordant"] == 2 and d["contradicts_labels"] is True


def _decide(**over):
    base = dict(valid=True, invalid_reason=None, same_k15=probe.same_k_pass({2345: 0.10, 3456: 0.20, 4567: 0.30}), same_k20=probe.same_k_pass({2345: 0.05, 3456: 0.08, 4567: 0.09}), determinism_pass=True, repeat_abs_diff=0.0, rho_all=0.6, rho_without_k25=0.5)
    base.update(over)
    return probe.decide(**base)


def test_decision_examples():
    assert _decide()["decision"] == "PROMISING" and _decide()["ROUND4B_ELIGIBLE"] is True
    nogo = _decide(same_k15=probe.same_k_pass({2345: 0.25, 3456: 0.20, 4567: 0.30}))
    assert nogo["decision"] == "NO-GO" and nogo["ROUND4B_ELIGIBLE"] is False
    assert _decide(same_k20=probe.same_k_pass({2345: 0.09, 3456: 0.08, 4567: 0.09}))["decision"] == "WEAK-PASS"
    assert _decide(rho_all=-0.1)["decision"] == "WEAK-PASS"
    assert _decide(rho_without_k25=0.0)["decision"] == "WEAK-PASS"
    assert _decide(repeat_abs_diff=0.02)["decision"] == "WEAK-PASS"  # min gap 0.10 <= 10 x 0.02
    assert _decide(valid=False, invalid_reason="checkpoint mismatch")["decision"] == "INVALID"
    assert _decide(determinism_pass=False)["decision"] == "WEAK-PASS"  # determinism failure is also raised as INVALID upstream


def test_secondary_metrics_cannot_rescue_primary_failure():
    verdict = _decide(same_k15=probe.same_k_pass({2345: 0.25, 3456: 0.20, 4567: 0.30}), rho_all=0.95, rho_without_k25=0.9)
    assert verdict["decision"] == "NO-GO"


# ------------------------------------------------------------------ source binding + preregistration (needs synced confirmatory artifacts)
@pytest.mark.skipif(not SOURCE_PRESENT, reason="confirmatory artifacts not present")
def test_source_binding_and_frozen_labels():
    source = probe.load_source(_config())
    assert source["decision"] == "GO"
    assert source["labels"]["2345"] == {"10": "NEW", "15": "MIXED_BALANCED", "20": "MIXED_OLD_DOMINANT", "25": "OLD"}
    assert source["labels"]["3456"] == {"10": "NEW", "15": "NEW", "20": "MIXED_BALANCED", "25": "OLD"}
    assert source["labels"]["4567"] == {"10": "NEW", "15": "NEW", "20": "MIXED_BALANCED", "25": "MIXED_OLD_DOMINANT"}
    assert set(source["checkpoint_hashes"]) == {"2345", "3456", "4567"} and all(len(v) == 64 for s in source["checkpoint_hashes"].values() for v in s.values())
    bad = json.loads(CONFIG_PATH.read_text()); bad["source"]["summary_sha256"] = "00" * 32
    with pytest.raises(probe.GateError, match="summary"):
        probe.load_source(bad)


@pytest.mark.skipif(not SOURCE_PRESENT, reason="confirmatory artifacts not present")
def test_preregistration_immutability_and_content():
    out = probe.REPO_ROOT / "results" / "video_trajectory_fork_probe_killtest" / "_pytest_tmp"
    shutil.rmtree(out, ignore_errors=True)
    try:
        result = probe.run_cpu(_config(), CONFIG_PATH, out)
        assert result["status"] == "FROZEN" and result["expected_local_responses"] == 26
        prov = probe.build_provenance(CONFIG_PATH)
        doc = probe.require_preregistration(out, prov)
        assert doc["probe"]["direction"] == "higher_probe_means_more_forkable" and doc["gate"]["primary_k"] == 15
        assert doc["frozen_labels"]["2345"]["15"] == "MIXED_BALANCED" and doc["determinism_audit"]["point"] == [3456, 15]
        assert len(doc["expected_keys"]) == 26 and doc["source"]["decision"] == "GO"
        path = out / "preregistration.json"; d = json.loads(path.read_text()); d["gate"]["primary_k"] = 20; path.write_text(json.dumps(d))
        with pytest.raises(probe.GateError, match="modified"):
            probe.require_preregistration(out, prov)
    finally:
        shutil.rmtree(out, ignore_errors=True)
