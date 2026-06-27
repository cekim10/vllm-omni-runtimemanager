from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diffusion_state_recovery_smoke import _preflight_real_model_args


def _args(*, backend: str, model: str) -> SimpleNamespace:
    return SimpleNamespace(
        backend=backend,
        model=model,
        output_dir=Path("/tmp/out"),
        disk_path=Path("/tmp/checkpoints"),
        failure_step=10,
        num_inference_steps=20,
    )


def test_preflight_rejects_known_small_model_without_step_execution() -> None:
    args = _args(backend="real-model", model="Tongyi-MAI/Z-Image-Turbo")

    with pytest.raises(ValueError, match="step execution support"):
        _preflight_real_model_args(args)


def test_preflight_allows_validated_real_model_target() -> None:
    args = _args(backend="real-model", model="Qwen/Qwen-Image")

    _preflight_real_model_args(args)


def test_preflight_skips_stub_backend() -> None:
    args = _args(backend="stub", model="Tongyi-MAI/Z-Image-Turbo")

    _preflight_real_model_args(args)
