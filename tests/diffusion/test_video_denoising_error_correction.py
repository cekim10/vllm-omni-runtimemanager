from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import video_denoising_error_correction_killtest as killtest
from experiments.video_exact_resume_repeat_control import classify_control
from experiments.video_denoising_error_correction_killtest import (
    ERROR_FAMILIES,
    build_perturbations,
    contraction_statistics,
    latent_error,
    smoke_gate,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _latent() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(7)
    return torch.randn((1, 4, 9, 12, 16), generator=generator)


def test_perturbations_are_iso_error_matched() -> None:
    latent = _latent()
    perturbations = build_perturbations(latent, random_seed=11, matching_tolerance=0.05)

    assert len(perturbations) == 8
    assert {item.family for item in perturbations} == set(ERROR_FAMILIES)
    for strength in ("small", "medium"):
        selected = [item for item in perturbations if item.strength == strength]
        target = selected[0].target_normalized_l2
        target_mse = selected[0].target_mse
        assert all(item.relative_mismatch <= 0.05 for item in selected)
        assert all(item.actual_normalized_l2 == pytest.approx(target, rel=0.05) for item in selected)
        assert all(item.actual_mse == pytest.approx(target_mse, rel=0.05) for item in selected)
    small = next(item for item in perturbations if item.family == "quantization" and item.strength == "small")
    medium = next(item for item in perturbations if item.family == "quantization" and item.strength == "medium")
    assert small.quantization_bits == 8
    assert medium.quantization_bits == 4
    assert medium.actual_normalized_l2 > small.actual_normalized_l2


def test_latent_error_reports_normalized_l2_and_cosine() -> None:
    reference = torch.ones((1, 1, 2, 2, 2))
    candidate = reference + 0.25

    metrics = latent_error(reference, candidate)

    assert metrics["mse"] == pytest.approx(0.25**2)
    assert metrics["normalized_l2"] == pytest.approx(0.25)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)


def test_contraction_statistics_requires_persistent_half_life() -> None:
    monotonic = contraction_statistics([1.0, 0.8, 0.49, 0.3])
    unstable = contraction_statistics([1.0, 0.4, 0.8, 0.3])

    assert monotonic["contraction_fraction"] == pytest.approx(0.7)
    assert monotonic["first_half_step"] == 2
    assert monotonic["persistent_half_life_steps"] == 2
    assert monotonic["correctability_class"] == "highly_correctable"
    assert unstable["first_half_step"] == 1
    assert unstable["persistent_half_life_steps"] is None


def test_smoke_gate_rejects_equal_behavior_and_accepts_type_dependence() -> None:
    def rows(quant_contraction: float, structural_contraction: float, quant_dynamic: float, structural_dynamic: float):
        output = []
        for family in ERROR_FAMILIES:
            structured = family in {"spatial_lowpass", "temporal_lowpass"}
            output.append(
                {
                    "prompt_id": "correction_000",
                    "checkpoint_step": 20,
                    "error_strength": "small",
                    "error_family": family,
                    "relative_error_mismatch": 0.01,
                    "relative_mse_mismatch": 0.01,
                    "contraction_fraction": structural_contraction if structured else quant_contraction,
                    "temporal_dynamic_quality_vs_exact_resume": (
                        structural_dynamic if structured else quant_dynamic
                    ),
                }
            )
        return output

    rejected = smoke_gate(rows(0.20, 0.20, 0.90, 0.90), 0.05)
    accepted = smoke_gate(rows(0.65, 0.20, 0.98, 0.80), 0.05)

    assert rejected["passing_cells"] == 0
    assert accepted["passing_cells"] == 1


def test_quality_metrics_accept_numpy_video_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    temporal = {
        "temporal_shape_composite": 1.0,
        "temporal_dynamic_composite": 1.0,
        "motion_energy_ratio": 1.0,
        "flow_magnitude_cosine": 1.0,
        "flow_magnitude_ratio": 1.0,
        "flow_direction_cosine": 1.0,
        "flicker_similarity": 1.0,
    }
    monkeypatch.setattr(killtest.preflight, "_temporal_metrics", lambda video, baseline: temporal)
    monkeypatch.setattr(killtest.preflight, "_spatial_metric", lambda video, baseline: 1.0)
    baseline = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    video = np.ones_like(baseline) * 2

    metrics = killtest._quality_metrics(video, baseline, "prompt", None, float("nan"))

    assert metrics["video_mse_vs_uninterrupted"] == pytest.approx(4.0)
    assert metrics["video_max_abs_diff_vs_uninterrupted"] == pytest.approx(2.0)


def test_probe_sampling_propagates_euler_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    sampling = SimpleNamespace(extra_args={"flow_shift": 12.0})
    monkeypatch.setattr(
        killtest.protection,
        "_build_probe_sampling_params",
        lambda *args, **kwargs: sampling,
    )
    args = argparse.Namespace(
        num_inference_steps=40,
        fps=16.0,
        flow_shift=12.0,
        sample_solver="euler",
    )

    result = killtest._probe_sampling(
        args,
        seed=1234,
        latents=None,
        checkpoint_step=0,
        artifact_dir=killtest.Path("/tmp/probe"),
        label="euler-probe",
    )

    assert result.extra_args["sample_solver"] == "euler"


def test_require_exact_resume_rejects_trajectory_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Exact-resume validation failed"):
        killtest._require_exact_resume(
            [{"normalized_l2": 0.0}, {"normalized_l2": 0.1}],
            {"video_mse_vs_uninterrupted": 0.0},
            tolerance=1e-6,
        )

    killtest._require_exact_resume(
        [{"normalized_l2": 0.0}],
        {"video_mse_vs_uninterrupted": 0.0},
        tolerance=1e-6,
    )


def test_exact_repeat_control_distinguishes_resume_mismatch_from_noise_floor() -> None:
    stable = classify_control(
        [1e-7, 2e-7, 1.5e-7],
        small_initial_error=0.01,
        medium_initial_error=0.20,
        uninterrupted_resume_error=0.12,
    )
    noisy = classify_control(
        [0.10, 0.12, 0.11],
        small_initial_error=0.01,
        medium_initial_error=0.20,
        uninterrupted_resume_error=0.12,
    )

    assert stable["diagnosis"] == "RESUME-PATH MISMATCH"
    assert stable["small_error_materially_confounded"] is False
    assert noisy["diagnosis"] == "SMALL-ERROR NOISE FLOOR DOMINATES"
    assert noisy["small_error_noise_floor_dominates"] is True
